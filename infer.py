"""Single-file inference CLI for the LILAC 0.75 kb/s codec.

Loads either a torch checkpoint (``{"generator": state_dict, "config": dict}``,
as produced by ``convert_checkpoint.py`` or ``train.py``) or the original JAX
SWA ``.npz`` snapshot directly, resamples an arbitrary input wav to 24 kHz
mono, encodes it to integer FSQ codes, and writes the reconstruction as 24 kHz
PCM16.

Usage:
    uv run python infer.py --checkpoint checkpoints/lilac_swa10.pt \\
        --input in.wav --output out.wav

Idempotence demo (decode -> re-encode K times, checking bit-exact codes):
    uv run python infer.py --checkpoint checkpoints/lilac_swa10.pt \\
        --input in.wav --output out.wav --cycles 10
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
import torch
import torchaudio

from config import LILACConfig
from lilac.codec import LILAC

SAMPLE_RATE = 24_000


def load_model(checkpoint: str, device: torch.device) -> LILAC:
    if checkpoint.endswith(".npz"):
        from convert_checkpoint import model_from_npz

        model = model_from_npz(checkpoint)
    else:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = LILACConfig(**ckpt["config"])
        model = LILAC(cfg)
        model.load_state_dict(ckpt["generator"], strict=True)
    return model.to(device).eval()


def load_wav(path: str, device: torch.device) -> torch.Tensor:
    """Load an arbitrary wav file, downmix to mono, resample to 24 kHz.

    Returns a ``[1, 1, T]`` float32 tensor on ``device``.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
    wav = torch.from_numpy(audio.T)  # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.unsqueeze(0).to(device)  # [1, 1, T]


def pad_to_hop(x: torch.Tensor, hop_length: int) -> torch.Tensor:
    length = x.shape[-1]
    pad = (hop_length - length % hop_length) % hop_length
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x


def save_wav(path: str, wav: torch.Tensor) -> None:
    """Write a ``[1, 1, T]`` (or ``[T]``) float tensor as 24 kHz PCM16."""
    samples = wav.detach().float().cpu().reshape(-1).clamp(-1.0, 1.0).numpy()
    pcm16 = (samples * 32767.0).round().astype(np.int16)
    sf.write(path, pcm16, SAMPLE_RATE, subtype="PCM_16")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="torch .pt or JAX SWA .npz")
    ap.add_argument("--input", required=True, help="input wav (any sample rate/channel count)")
    ap.add_argument("--output", required=True, help="output wav, written at 24 kHz PCM16")
    ap.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="decode->re-encode this many times, reporting per-cycle code "
        "bit-exactness against cycle 1 (idempotence demo); final audio is "
        "written from the last cycle's decode",
    )
    ap.add_argument("--codes-out", help="optional path to save cycle-1 integer codes (.npy)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.cycles < 1:
        raise ValueError("--cycles must be >= 1")

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    hop = model.config.hop_length

    original_length = None
    wav = load_wav(args.input, device)
    original_length = wav.shape[-1]
    wav = pad_to_hop(wav, hop)

    codes_first = None
    decoded = None
    for cycle in range(1, args.cycles + 1):
        codes = model.encode(wav)
        decoded = model.decode(codes)
        if codes_first is None:
            codes_first = codes
            if args.codes_out:
                np.save(args.codes_out, codes.cpu().numpy())
        exact = bool(torch.equal(codes, codes_first))
        print(f"cycle {cycle}: codes {'bit-exact' if exact else 'DIFFER'} vs cycle 1")
        wav = decoded

    output = decoded[..., :original_length]
    save_wav(args.output, output)
    rms = float(output.pow(2).mean().sqrt())
    duration = output.shape[-1] / SAMPLE_RATE
    print(f"wrote {args.output} | {duration:.2f}s | rms={rms:.4f}")


if __name__ == "__main__":
    main()
