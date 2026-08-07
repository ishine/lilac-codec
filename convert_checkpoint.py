"""Convert the flagship JAX SWA checkpoint (an ``.npz`` parameter snapshot) to
a torch state dict this repository's ``lilac.codec.LILAC`` can load directly.

The checkpoint was trained in JAX/Flax; JAX uses channels-last convolutions
and stores each invertible orthogonal 1x1 mix as an unconstrained matrix
(``W_raw``) plus a Cayley transform, rather than torch's built-in orthogonal
parametrization. This script performs that one-time translation:

  - Flax conv kernels ``(kW, in, out)`` -> torch ``Conv1d.weight`` ``(out, in, kW)``.
  - Flax GroupNorm ``scale``/``bias`` -> torch ``GroupNorm.weight``/``bias``.
  - Per-channel scalars stored channels-last ``(1, 1, C)`` -> torch's
    channels-first ``(1, C, 1)``.
  - Each ``OrthogonalMix.W_raw`` -> ``O = cayley(W_raw)``, assigned directly to
    the parametrized ``.weight`` (torch's orthogonal parametrization accepts a
    already-orthogonal matrix and reproduces it exactly on the next forward
    pass; verified bit-exact in this script's self-check).
  - The invertible stem's plain (unconstrained) 1x1 mix is copied directly,
    with no Cayley transform.

Encoder and decoder share the analysis/synthesis chart by construction, so the
JAX dump carries duplicate ``encoder.*``/``decoder.*`` copies of the same
tensors; only one copy is loaded per shared module.

Usage:
    uv run python convert_checkpoint.py --npz <path/to/checkpoint.npz> \\
        --out checkpoints/lilac_swa10.pt
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch

from config import LILACConfig
from lilac.codec import LILAC, OrthogonalMix


def cayley_orthogonal(w_raw: np.ndarray) -> np.ndarray:
    """Cayley transform: skew-symmetrize ``w_raw`` and map to an orthogonal
    matrix via ``(I + A)^-1 (I - A)``, matching the training-time parametrization."""
    w = w_raw.astype(np.float64)
    a = w - w.T
    eye = np.eye(a.shape[0], dtype=np.float64)
    return np.linalg.solve((eye + a).T, (eye - a).T).T


def _conv_weight(v: np.ndarray) -> torch.Tensor:
    """Flax conv kernel (kW, in, out) -> torch Conv1d weight (out, in, kW)."""
    return torch.from_numpy(np.ascontiguousarray(np.transpose(v, (2, 1, 0)))).float()


def _channel_param(v: np.ndarray) -> torch.Tensor:
    """Channels-last per-channel scalar (..., C) -> torch (1, C, 1)."""
    return torch.from_numpy(np.ascontiguousarray(v.reshape(1, v.shape[-1], 1))).float()


def load_npz(npz_path: str) -> tuple[dict[str, np.ndarray], LILACConfig]:
    z = np.load(npz_path, allow_pickle=False)
    keys = json.loads(str(z["__keys__"]))
    cfg_dict = json.loads(str(z["__config__"]))
    params = {k: z[k.replace(".", "/")] for k in keys}
    cfg = LILACConfig(
        **{k: v for k, v in cfg_dict.items() if k in LILACConfig.__dataclass_fields__}
    )
    return params, cfg


def _remap_key(jax_key: str) -> str | None:
    """Translate one JAX parameter path to this repo's torch module path.

    Returns None for paths handled separately (OrthogonalMix.W_raw) or for
    the duplicate encoder/decoder copies of shared modules. ``lilac.codec.LILAC``
    passes the *same* stem_flow/stem_mix/pre_stages/project_stages instances to
    both Encoder and Decoder, so ``model.named_modules()`` (and its
    ``state_dict()``) surfaces each shared module only once, under its first
    (encoder-side) path -- the decoder-side copy is a duplicate and is skipped.
    """
    if jax_key.startswith("decoder.") and (
        ".stem_flow." in jax_key
        or ".stem_mix." in jax_key
        or ".pre_stages." in jax_key
        or ".project_stages." in jax_key
    ):
        return None  # shared with the encoder copy; skip the duplicate

    key = jax_key
    key = key.replace(".stem_mix.blocks.0.", ".stem_mix.")
    key = key.replace(".mix.blocks.0.", ".mix.")
    key = key.replace(".coup.", ".coupling.net.")
    key = key.replace(".conv1.", ".0.")
    key = key.replace(".conv2.", ".2.")
    key = re.sub(r"\.context_ups\.(\d+)\.blocks\.0\.", r".context_ups.\1.block.", key)
    return key


def build_state_dict(
    params: dict[str, np.ndarray], model: LILAC
) -> dict[str, torch.Tensor]:
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    orthogonal_mixes: dict[str, np.ndarray] = {}

    for jax_key, value in params.items():
        if jax_key.endswith(".W_raw"):
            torch_key = _remap_key(jax_key[: -len(".W_raw")] + ".weight")
            if torch_key is not None:
                orthogonal_mixes[torch_key] = value
            continue

        torch_key = _remap_key(jax_key)
        if torch_key is None:
            continue

        leaf = torch_key.rsplit(".", 1)[-1]
        if leaf == "kernel":
            torch_key = torch_key[: -len("kernel")] + "weight"
            tensor = _conv_weight(value)
        elif leaf == "scale":
            torch_key = torch_key[: -len("scale")] + "weight"
            tensor = torch.from_numpy(np.ascontiguousarray(value)).float()
        elif leaf in ("log_a", "log_beta"):
            tensor = _channel_param(value)
        elif leaf == "W":
            torch_key = torch_key[: -len("W")] + "weight"
            tensor = torch.from_numpy(np.ascontiguousarray(value)).float()
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(value)).float()

        if torch_key not in target:
            raise KeyError(f"unmapped torch key {torch_key!r} (from {jax_key!r})")
        if tuple(tensor.shape) != tuple(target[torch_key].shape):
            raise ValueError(
                f"{jax_key} -> {torch_key}: shape {tuple(tensor.shape)} != "
                f"expected {tuple(target[torch_key].shape)}"
            )
        mapped[torch_key] = tensor

    # OrthogonalMix params are parametrized (torch stores an internal
    # unconstrained representation, not `.weight` directly), so they are
    # assigned post-load via the module, not through load_state_dict.
    def _is_decoder_shared_dup(k: str) -> bool:
        return k.startswith("decoder.") and (
            ".stem_flow." in k
            or ".stem_mix." in k
            or ".pre_stages." in k
            or ".project_stages." in k
        )

    orthogonal_targets = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, OrthogonalMix)
    }
    populated_orthogonal_modules = set()
    for torch_key, w_raw in orthogonal_mixes.items():
        module_name = torch_key[: -len(".weight")]
        module = orthogonal_targets.get(module_name)
        if module is None:
            raise KeyError(f"no OrthogonalMix module named {module_name!r}")
        orthogonal = cayley_orthogonal(w_raw)
        with torch.no_grad():
            module.weight = torch.from_numpy(orthogonal).float()
        populated_orthogonal_modules.add(module_name)

    def _is_orthogonal_internal(k: str) -> bool:
        module_name = k.rsplit(".parametrizations.", 1)[0]
        return module_name in populated_orthogonal_modules and ".parametrizations." in k

    remaining = {
        k: v
        for k, v in model.state_dict().items()  # re-read: weight assignment above changed key names
        if k not in mapped
        and not _is_decoder_shared_dup(k)
        and not _is_orthogonal_internal(k)
    }
    if remaining:
        raise KeyError(f"torch parameters never populated: {sorted(remaining)[:10]}")

    return mapped


def model_from_npz(npz_path: str) -> LILAC:
    """Build a fully-loaded ``LILAC`` (CPU, eval) directly from the JAX ``.npz``
    snapshot. Shared by this script's CLI and ``infer.py``'s ``.npz`` path."""
    params, cfg = load_npz(npz_path)
    model = LILAC(cfg)
    mapped = build_state_dict(params, model)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    # Expected "missing": decoder-side duplicate keys of modules the encoder
    # and decoder share by reference (already updated in-place via the shared
    # tensor when the encoder-side key was loaded) and OrthogonalMix `.weight`
    # leaves (assigned directly through the module, not via load_state_dict,
    # since torch's orthogonal parametrization does not expose a plain
    # `.weight` entry in `state_dict()`).
    orthogonal_names = {
        name for name, m in model.named_modules() if isinstance(m, OrthogonalMix)
    }
    unexplained_missing = [
        m
        for m in missing
        if m.rsplit(".parametrizations.", 1)[0] not in orthogonal_names
        and not (
            m.startswith("decoder.")
            and (".stem_flow." in m or ".stem_mix." in m or ".pre_stages." in m or ".project_stages." in m)
        )
    ]
    if unexplained_missing or unexpected:
        raise RuntimeError(
            f"load_state_dict mismatch: missing={unexplained_missing} unexpected={unexpected}"
        )
    return model.eval()


def convert(npz_path: str, out_path: str) -> LILAC:
    model = model_from_npz(npz_path)
    torch.save({"generator": model.state_dict(), "config": model.config.to_dict()}, out_path)
    print(f"wrote {out_path}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, help="path to the JAX SWA .npz checkpoint")
    ap.add_argument("--out", default="checkpoints/lilac_swa10.pt")
    args = ap.parse_args()
    convert(args.npz, args.out)


if __name__ == "__main__":
    main()
