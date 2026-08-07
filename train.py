"""Single-device PyTorch reproduction of the promoted LILAC training run."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import LILACConfig, learning_rates
from lilac.codec import LILAC
from lilac.dataset import HiFiTTS2Batcher, ShardCursor
from lilac.discriminator import Discriminator
from lilac.loss import (
    SpectralLoss,
    adversarial_scale,
    discriminator_loss,
    generator_loss,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Orthogonal inverses are part of the scientific contract.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _log_json(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _checkpoint_payload(
    *,
    step: int,
    config: LILACConfig,
    data_roots: list[str],
    batcher: HiFiTTS2Batcher,
    generator: LILAC,
    discriminator: Discriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
) -> dict:
    return {
        "format": "lilac-train-checkpoint-v1",
        "step": step,
        "config": config.to_dict(),
        "data_roots": data_roots,
        "data_cursor": batcher.state_dict(),
        "rng": _rng_state(),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
    }


def _save_training_state(
    run_dir: Path,
    *,
    step: int,
    config: LILACConfig,
    data_roots: list[str],
    batcher: HiFiTTS2Batcher,
    generator: LILAC,
    discriminator: Discriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
) -> None:
    payload = _checkpoint_payload(
        step=step,
        config=config,
        data_roots=data_roots,
        batcher=batcher,
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
    )
    _atomic_torch_save(payload, run_dir / "latest.pt")
    snapshot = run_dir / f"generator_{step:07d}.pt"
    _atomic_torch_save(
        {
            "format": "lilac-generator-snapshot-v1",
            "step": step,
            "config": config.to_dict(),
            "generator": generator.state_dict(),
        },
        snapshot,
    )

    # Only the final ten generator snapshots are needed for the promoted
    # uniform average. The complete resumable state remains in latest.pt.
    snapshots = sorted(run_dir.glob("generator_[0-9]*.pt"))
    for old_snapshot in snapshots[: -config.swa_window]:
        old_snapshot.unlink()


def _build_final_swa(run_dir: Path, config: LILACConfig) -> Path:
    snapshots = sorted(run_dir.glob("generator_[0-9]*.pt"))
    if len(snapshots) != config.swa_window:
        raise RuntimeError(
            f"need {config.swa_window} terminal snapshots, found {len(snapshots)}"
        )
    records = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in snapshots
    ]
    steps = [int(record["step"]) for record in records]
    expected = list(
        range(
            config.max_steps - (config.swa_window - 1) * config.save_interval,
            config.max_steps + 1,
            config.save_interval,
        )
    )
    if steps != expected:
        raise RuntimeError(f"SWA window mismatch: expected {expected}, found {steps}")

    state_dicts = [record["generator"] for record in records]
    average: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        values = [state[key] for state in state_dicts]
        if values[0].is_floating_point():
            average[key] = torch.stack(
                [value.to(dtype=torch.float64) for value in values]
            ).mean(0).to(dtype=values[0].dtype)
        else:
            average[key] = values[-1]
    destination = run_dir / "model_swa10.pt"
    _atomic_torch_save(
        {
            "format": "lilac-generator-swa10-v1",
            "config": config.to_dict(),
            "steps": steps,
            "generator": average,
        },
        destination,
    )
    return destination


def _load_resume(
    path: Path,
    *,
    config: LILACConfig,
    data_roots: list[str],
    generator: LILAC,
    discriminator: Discriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
) -> tuple[int, ShardCursor, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "lilac-train-checkpoint-v1":
        raise ValueError(f"{path}: unsupported checkpoint format")
    if checkpoint["config"] != config.to_dict():
        raise ValueError("resume config differs from the checkpoint")
    if checkpoint["data_roots"] != data_roots:
        raise ValueError("resume data roots differ from the checkpoint")
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    return (
        int(checkpoint["step"]),
        ShardCursor.from_dict(checkpoint["data_cursor"]),
        checkpoint["rng"],
    )


def train(
    name: str,
    data: list[str],
    *,
    device: str | None = None,
    resume: str | None = None,
    micro_batch_size: int | None = None,
) -> None:
    config = LILACConfig()
    if micro_batch_size is not None:
        config = replace(config, micro_batch_size=micro_batch_size)
    config.validate()
    accumulation_steps = config.global_batch_size // config.micro_batch_size
    data_roots = [str(Path(path).expanduser().resolve()) for path in data]
    run_dir = Path(config.output_dir).resolve() / name
    resume_path = Path(resume).expanduser().resolve() if resume else None
    if resume_path is None:
        if run_dir.exists():
            raise FileExistsError(
                f"refusing to reuse output namespace {run_dir}; choose a fresh name"
            )
        run_dir.mkdir(parents=True)
    else:
        if resume_path.parent != run_dir or not resume_path.is_file():
            raise ValueError("--resume must name this run's existing latest.pt")

    _set_seed(config.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    generator = LILAC(config).to(dev)
    discriminator = Discriminator().to(dev)
    spectral_loss = SpectralLoss(config).to(dev)
    optimizer_g = torch.optim.AdamW(
        generator.parameters(),
        lr=config.lr_generator,
        betas=config.adam_betas,
        weight_decay=config.weight_decay,
        fused=dev.type == "cuda",
    )
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config.lr_discriminator,
        betas=config.adam_betas,
        weight_decay=config.weight_decay,
        fused=dev.type == "cuda",
    )

    step = 0
    cursor = None
    restored_rng = None
    if resume_path is not None:
        step, cursor, restored_rng = _load_resume(
            resume_path,
            config=config,
            data_roots=data_roots,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
        )
    batcher = HiFiTTS2Batcher(data_roots, config, cursor=cursor)
    if restored_rng is not None:
        _restore_rng_state(restored_rng)

    log_path = run_dir / "train.jsonl"
    started = time.monotonic()
    print(
        f"LILAC | device={dev} | step={step} | "
        f"batch={config.micro_batch_size}x{accumulation_steps}="
        f"{config.global_batch_size}",
        flush=True,
    )

    while step < config.max_steps:
        next_step = step + 1
        learning_rate_g, learning_rate_d = learning_rates(config, next_step)
        _set_learning_rate(optimizer_g, learning_rate_g)
        _set_learning_rate(optimizer_d, learning_rate_d)
        cpu_batches = [
            torch.from_numpy(batcher.next_batch(config.micro_batch_size))
            for _ in range(accumulation_steps)
        ]

        # Discriminator update. The no-grad generator pass also accumulates the
        # three global-batch Frobenius norms needed by the generator objective.
        optimizer_d.zero_grad(set_to_none=True)
        loss_d_value = 0.0
        global_numerator = torch.zeros(3, device=dev)
        global_denominator = torch.zeros(3, device=dev)
        d_scale = adversarial_scale(next_step, config.adv_warmup_steps)
        for cpu_batch in cpu_batches:
            x = cpu_batch.unsqueeze(1).to(dev)
            with torch.no_grad():
                x_hat = generator(x)
                numerator, denominator = spectral_loss.global_squared_norms(x, x_hat)
                global_numerator += numerator
                global_denominator += denominator
            real = discriminator(x)
            fake = discriminator(x_hat)
            loss_d = d_scale * discriminator_loss(real[0], fake[0])
            (loss_d / accumulation_steps).backward()
            loss_d_value += float(loss_d.detach()) / accumulation_steps
        torch.nn.utils.clip_grad_norm_(
            discriminator.parameters(), config.grad_clip
        )
        optimizer_d.step()

        # Generator update against the newly updated discriminator, replaying
        # the identical 256 crops. Discriminator parameters are constants, but
        # its input gradient remains live.
        optimizer_g.zero_grad(set_to_none=True)
        _set_requires_grad(discriminator, False)
        metric_sums = {
            "mel": 0.0,
            "mrstft": 0.0,
            "adv": 0.0,
            "feat": 0.0,
            "total": 0.0,
        }
        try:
            for cpu_batch in cpu_batches:
                x = cpu_batch.unsqueeze(1).to(dev)
                x_hat = generator(x)
                with torch.no_grad():
                    real = discriminator(x)
                fake = discriminator(x_hat)
                loss_g, metrics = generator_loss(
                    config,
                    spectral_loss,
                    x,
                    x_hat,
                    real,
                    fake,
                    next_step,
                    global_squares=(global_numerator, global_denominator),
                    micro_batches=accumulation_steps,
                )
                (loss_g / accumulation_steps).backward()
                for key, value in metrics.items():
                    metric_sums[key] += float(value) / accumulation_steps
        finally:
            _set_requires_grad(discriminator, True)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip)
        optimizer_g.step()
        step = next_step

        if step % config.log_interval == 0:
            elapsed = time.monotonic() - started
            record = {
                "step": step,
                "loss_d": loss_d_value,
                **{f"loss_{key}": value for key, value in metric_sums.items()},
                "lr_generator": learning_rate_g,
                "lr_discriminator": learning_rate_d,
                "steps_per_second": step / elapsed,
            }
            _log_json(log_path, record)
            print(
                f"[{step:>7d}] mel={record['loss_mel']:.4f} "
                f"mrstft={record['loss_mrstft']:.4f} "
                f"adv={record['loss_adv']:.4f} feat={record['loss_feat']:.4f} "
                f"d={loss_d_value:.4f} lr_g={learning_rate_g:.3e}",
                flush=True,
            )

        if step % config.save_interval == 0 or step == config.max_steps:
            _save_training_state(
                run_dir,
                step=step,
                config=config,
                data_roots=data_roots,
                batcher=batcher,
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
            )

    destination = _build_final_swa(run_dir, config)
    print(f"complete: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the frozen LILAC 0.75 kb/s reproduction"
    )
    parser.add_argument("name", help="fresh run namespace below runs/")
    parser.add_argument(
        "--data",
        action="append",
        required=True,
        help="local HiFiTTS-2 shard directory; repeat for multiple directories",
    )
    parser.add_argument("--device", help="for example cuda, cuda:1, or cpu")
    parser.add_argument("--resume", help="this run's existing runs/NAME/latest.pt")
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        help="device batch; must divide the frozen global batch of 256",
    )
    arguments = parser.parse_args()
    train(
        arguments.name,
        arguments.data,
        device=arguments.device,
        resume=arguments.resume,
        micro_batch_size=arguments.micro_batch_size,
    )


if __name__ == "__main__":
    main()
