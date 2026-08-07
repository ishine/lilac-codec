"""CPU-only contract checks for the minimal LILAC reproduction."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from config import LILACConfig, learning_rates
from lilac.codec import LILAC
from lilac.dataset import HiFiTTS2Batcher, ShardCursor, augment
from lilac.discriminator import Discriminator
from lilac.loss import SpectralLoss

EXPECTED_GENERATOR_PARAMETERS = 58_462_122


def _write_tiny_shard(root: Path, config: LILACConfig) -> None:
    length = 2 * config.segment_length
    time = np.arange(length, dtype=np.float32) / config.sample_rate
    first = 0.35 * np.sin(2 * np.pi * 220 * time)
    second = 0.30 * np.sin(2 * np.pi * 330 * time)
    pcm = np.round(np.concatenate((first, second)) * 32767).astype("<i2")
    pcm.tofile(root / "shard_000000.bin")
    metadata = {
        "sr": config.sample_rate,
        "dtype": "int16",
        "source": "hifitts2",
        "clips": [
            {
                "id": "a",
                "reader": "r",
                "book": "b",
                "offset_samples": 0,
                "n_samples": length,
            },
            {
                "id": "b",
                "reader": "r",
                "book": "b",
                "offset_samples": length,
                "n_samples": length,
            },
        ],
    }
    with (root / "shard_000000.idx.json").open("w") as handle:
        json.dump(metadata, handle)


def _verify_data(config: LILACConfig) -> None:
    with tempfile.TemporaryDirectory(prefix="lilac-verify-") as directory:
        root = Path(directory)
        _write_tiny_shard(root, config)
        first = HiFiTTS2Batcher(
            [root], config, draws_per_shard=7, augment_audio=True
        )
        batch_a = first.next_batch(2)
        cursor = first.state_dict()
        expected = first.next_batch(2)
        resumed = HiFiTTS2Batcher(
            [root],
            config,
            cursor=ShardCursor.from_dict(cursor),
            draws_per_shard=7,
            augment_audio=True,
        )
        actual = resumed.next_batch(2)
        assert np.array_equal(actual, expected)
        assert batch_a.shape == (2, config.segment_length)
        assert float(np.abs(batch_a).max()) <= 1.0

    crop = np.linspace(-1, 1, config.segment_length, dtype=np.float32)
    one = augment(crop, np.random.default_rng(17))
    two = augment(crop, np.random.default_rng(17))
    assert np.array_equal(one, two)


def _verify_global_spectral_gradient(config: LILACConfig) -> None:
    loss = SpectralLoss(config)
    torch.manual_seed(11)
    target = torch.randn(2, 1, config.hop_length)
    full_prediction = torch.randn_like(target, requires_grad=True)
    mel, mrstft = loss(target, full_prediction)
    (mel + mrstft).backward()
    full_gradient = full_prediction.grad.detach().clone()

    split_prediction = full_prediction.detach().clone().requires_grad_(True)
    with torch.no_grad():
        stats = [
            loss.global_squared_norms(target[index : index + 1], split_prediction[index : index + 1])
            for index in range(2)
        ]
        totals = (
            sum(item[0] for item in stats),
            sum(item[1] for item in stats),
        )
    for index in range(2):
        split_mel, split_mrstft = loss(
            target[index : index + 1],
            split_prediction[index : index + 1],
            global_squares=totals,
            micro_batches=2,
        )
        ((split_mel + split_mrstft) / 2).backward()
    torch.testing.assert_close(
        split_prediction.grad, full_gradient, rtol=2e-4, atol=2e-5
    )


def main() -> None:
    config = LILACConfig()
    config.validate()
    assert config.bitrate == 750.0
    assert learning_rates(config, 1) == (0.0, 0.0)
    assert learning_rates(config, 848_001) == (5e-5, 2.5e-5)
    _verify_data(config)
    _verify_global_spectral_gradient(config)

    torch.manual_seed(7)
    model = LILAC(config).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == EXPECTED_GENERATOR_PARAMETERS, parameters
    codes = torch.randint(
        0, config.n_fsq_levels, (1, config.n_fsq_channels, 1)
    )
    with torch.inference_mode():
        waveform = model.decode(codes)
        recovered = model.encode(waveform)
    assert waveform.shape == (1, 1, config.hop_length)
    assert torch.equal(recovered, codes)

    discriminator = Discriminator().eval()
    spectral = SpectralLoss(config)
    with torch.inference_mode():
        logits, features = discriminator(waveform)
        mel, mrstft = spectral(torch.zeros_like(waveform), waveform)
    assert len(logits) == len(features) == 10
    assert all(torch.isfinite(logit).all() for logit in logits)
    assert torch.isfinite(mel) and torch.isfinite(mrstft)
    print(
        "PASS | "
        f"{parameters:,} generator parameters | "
        f"{config.bitrate:.0f} bit/s | "
        "code idempotence | exact shard resume | finite objectives"
    )


if __name__ == "__main__":
    main()
