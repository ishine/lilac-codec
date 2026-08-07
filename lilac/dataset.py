"""Deterministic local reader for the public HiFiTTS-2 training shards.

Locked shard format:

``shard_*.bin``
    Concatenated little-endian int16, mono, 24 kHz PCM.
``shard_*.idx.json``
    ``{"sr":24000,"dtype":"int16","source":"hifitts2","clips":[...]}``.

The four-integer cursor is sufficient for exact batch continuation. Crop and
augmentation randomness is derived from that cursor, never ambient RNG state.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from config import LILACConfig

INT16_MAX = 32767.0
KURT_MAX_TRIES = 8
KURT_THRESHOLD = 40.0
RMS_FLOOR = 0.02
SILENCE_CEILING = 0.5
AUGMENT_SALT = 104_729


@dataclass
class ShardCursor:
    perm_seed: int = 0
    epoch: int = 0
    shard_idx: int = 0
    draws_in_shard: int = 0

    @classmethod
    def from_dict(cls, value: dict | None) -> "ShardCursor":
        if value is None:
            return cls()
        return cls(
            **{
                key: int(value[key])
                for key in cls.__dataclass_fields__
                if key in value
            }
        )


@dataclass(frozen=True)
class _Shard:
    bin_path: Path
    clips: tuple[tuple[int, int], ...]


def _load_index(index_path: Path) -> _Shard:
    with index_path.open() as handle:
        metadata = json.load(handle)
    if (
        metadata.get("sr") != 24_000
        or metadata.get("dtype") != "int16"
        or metadata.get("source") != "hifitts2"
    ):
        raise ValueError(
            f"{index_path}: expected 24k/int16/hifitts2, got "
            f"{metadata.get('sr')}/{metadata.get('dtype')}/{metadata.get('source')}"
        )
    bin_path = index_path.with_name(index_path.name.removesuffix(".idx.json") + ".bin")
    if not bin_path.is_file():
        raise FileNotFoundError(f"missing PCM partner for {index_path}: {bin_path}")
    clips = tuple(
        (int(clip["offset_samples"]), int(clip["n_samples"]))
        for clip in metadata["clips"]
    )
    if any(offset < 0 or length <= 0 for offset, length in clips):
        raise ValueError(f"{index_path}: invalid clip offset or length")
    if any(a[0] > b[0] for a, b in zip(clips, clips[1:])):
        raise ValueError(f"{index_path}: clips must be sorted by offset")
    n_samples = bin_path.stat().st_size // np.dtype("<i2").itemsize
    if clips and max(offset + length for offset, length in clips) > n_samples:
        raise ValueError(f"{index_path}: clip extends beyond {bin_path.name}")
    return _Shard(bin_path, clips)


def _excess_kurtosis(x: np.ndarray) -> float:
    centered = x - x.mean()
    square = centered * centered
    variance = float(square.mean())
    if variance <= 0:
        return -np.inf
    return float((square * square).mean() / (variance * variance) - 3.0)


def crop_and_normalize(
    clip: np.ndarray,
    segment_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Quality-gated random crop followed by per-crop peak normalization."""
    clip = np.asarray(clip, dtype=np.float32)
    if clip.shape[0] < segment_length:
        crop = np.pad(clip, (0, segment_length - clip.shape[0]))
    else:
        crop = clip[:segment_length]
        for _ in range(KURT_MAX_TRIES):
            offset = int(rng.integers(0, clip.shape[0] - segment_length + 1))
            crop = clip[offset : offset + segment_length]
            kurtosis = _excess_kurtosis(crop)
            rms = float(np.sqrt(np.mean(crop.astype(np.float64) ** 2)))
            silence = float(np.mean(np.abs(crop) < 1e-3))
            if (
                not np.isneginf(kurtosis)
                and kurtosis < KURT_THRESHOLD
                and rms > RMS_FLOOR
                and silence < SILENCE_CEILING
            ):
                break
    peak = float(np.abs(crop).max())
    if peak > 0:
        crop = crop / peak
    return np.asarray(crop, dtype=np.float32)


def _fix_length(x: np.ndarray, length: int) -> np.ndarray:
    if x.shape[0] > length:
        return x[:length]
    if x.shape[0] < length:
        return np.pad(x, (0, length - x.shape[0]))
    return x


def _resample_roundtrip(x: np.ndarray, sample_rate: int, middle_rate: int) -> np.ndarray:
    common = gcd(middle_rate, sample_rate)
    down = resample_poly(x, middle_rate // common, sample_rate // common)
    common = gcd(sample_rate, middle_rate)
    return resample_poly(
        down, sample_rate // common, middle_rate // common
    ).astype(np.float32)


def _lowpass(x: np.ndarray, cutoff: float, sample_rate: int, order: int) -> np.ndarray:
    normalized = min(cutoff / (sample_rate / 2), 0.999)
    return sosfiltfilt(
        butter(order, normalized, btype="low", output="sos"), x
    ).astype(np.float32)


def augment(
    crop: np.ndarray,
    rng: np.random.Generator,
    sample_rate: int = 24_000,
) -> np.ndarray:
    """The flagship target-preserving, attenuation-biased augmentation.

    The returned crop is used as both model input and reconstruction target.
    """
    length = crop.shape[0]
    category = float(rng.random())
    out = crop
    if category < 0.25:
        out = _resample_roundtrip(crop, sample_rate, 16_000)
    elif category < 0.35:
        middle_rate = 22_050 if rng.random() < 0.5 else 12_000
        out = _resample_roundtrip(crop, sample_rate, middle_rate)
    elif category < 0.45:
        out = _lowpass(crop, float(rng.uniform(4_000, 10_000)), sample_rate, 8)
    elif category < 0.55:
        shelf_hz = float(rng.uniform(5_000, 7_000))
        low = _lowpass(crop, shelf_hz, sample_rate, 4)
        high = crop - low
        high_gain = 10.0 ** (float(rng.uniform(-12.0, -4.0)) / 20.0)
        tilt = 10.0 ** (float(rng.uniform(-2.0, 0.0)) / 20.0)
        out = (low + high_gain * high) * tilt
    if rng.random() < 0.10:
        out = out * float(rng.uniform(0.3, 0.9))
    out = _fix_length(np.asarray(out, dtype=np.float32), length)
    peak = float(np.abs(out).max())
    return out / peak if peak > 1.0 else out


class HiFiTTS2Batcher:
    """Single-process, exact-resume batch stream over local HiFiTTS-2 shards."""

    def __init__(
        self,
        roots: Iterable[str | Path],
        config: LILACConfig,
        *,
        seed: int | None = None,
        cursor: ShardCursor | None = None,
        augment_audio: bool = True,
        draws_per_shard: int | None = None,
    ):
        index_paths: list[Path] = []
        for root in roots:
            root_path = Path(root).expanduser().resolve()
            if not root_path.is_dir():
                raise FileNotFoundError(f"shard directory does not exist: {root_path}")
            index_paths.extend(
                sorted(root_path.glob("shard_*.idx.json"), key=lambda path: path.name)
            )
        if len(index_paths) != len(set(index_paths)):
            raise ValueError("the same shard directory was supplied more than once")
        self.index_paths = index_paths
        if not self.index_paths:
            raise ValueError("no shard_*.idx.json files found")

        self.config = config
        self.augment_audio = augment_audio
        self.cursor = cursor or ShardCursor(perm_seed=config.seed if seed is None else seed)
        self._global_id = {path: i for i, path in enumerate(self.index_paths)}
        self._draws_per_shard = (
            int(draws_per_shard)
            if draws_per_shard is not None
            else self._estimate_mean_clips()
        )
        if self._draws_per_shard < 1:
            raise ValueError("draws_per_shard must be positive")
        self._loaded_path: Path | None = None
        self._loaded_shard: _Shard | None = None
        self._memmap: np.memmap | None = None

    def _estimate_mean_clips(self) -> int:
        paths = self.index_paths
        if len(paths) > 48:
            stride = len(paths) // 48
            paths = paths[::stride][:48]
        counts = [len(_load_index(path).clips) for path in paths]
        return max(1, sum(counts) // len(counts))

    def state_dict(self) -> dict:
        return asdict(self.cursor)

    def load_state_dict(self, state: dict) -> None:
        self.cursor = ShardCursor.from_dict(state)
        self._close_shard()

    def _epoch_order(self) -> list[Path]:
        rng = np.random.default_rng((self.cursor.perm_seed, self.cursor.epoch, 0))
        order = rng.permutation(len(self.index_paths))
        return [self.index_paths[int(index)] for index in order]

    def _open_shard(self, path: Path) -> tuple[_Shard, np.memmap]:
        if path != self._loaded_path:
            self._close_shard()
            self._loaded_path = path
            self._loaded_shard = _load_index(path)
            self._memmap = np.memmap(
                self._loaded_shard.bin_path, dtype="<i2", mode="r"
            )
        assert self._loaded_shard is not None and self._memmap is not None
        return self._loaded_shard, self._memmap

    def _close_shard(self) -> None:
        self._memmap = None
        self._loaded_shard = None
        self._loaded_path = None

    def _emit_plan(self, shard: _Shard, epoch: int, global_id: int) -> list[int]:
        rng = np.random.default_rng((self.cursor.perm_seed, epoch, global_id))
        if not shard.clips:
            return []
        parent_indices = rng.integers(
            0, len(shard.clips), size=self._draws_per_shard
        )
        offsets: list[int] = []
        for parent_index in parent_indices:
            offset, length = shard.clips[int(parent_index)]
            start = (
                int(rng.integers(0, length - self.config.segment_length + 1))
                if length > self.config.segment_length
                else 0
            )
            offsets.append(offset + start)
        shuffle_rng = np.random.default_rng(
            (self.cursor.perm_seed, epoch, global_id, 7_919)
        )
        return [offsets[int(i)] for i in shuffle_rng.permutation(len(offsets))]

    def _advance_shard(self) -> None:
        self.cursor.shard_idx += 1
        self.cursor.draws_in_shard = 0
        if self.cursor.shard_idx >= len(self.index_paths):
            self.cursor.epoch += 1
            self.cursor.shard_idx = 0
        self._close_shard()

    def _next_crop(self) -> np.ndarray:
        while True:
            order = self._epoch_order()
            index_path = order[self.cursor.shard_idx]
            shard, memmap = self._open_shard(index_path)
            global_id = self._global_id[index_path]
            plan = self._emit_plan(shard, self.cursor.epoch, global_id)
            if not plan or self.cursor.draws_in_shard >= len(plan):
                self._advance_shard()
                continue

            draw_index = self.cursor.draws_in_shard
            planned_offset = plan[draw_index]
            clip_starts = [clip[0] for clip in shard.clips]
            parent_index = bisect.bisect_right(clip_starts, planned_offset) - 1
            parent_offset, parent_length = shard.clips[parent_index]
            clip = (
                np.asarray(
                    memmap[parent_offset : parent_offset + parent_length],
                    dtype=np.float32,
                )
                / INT16_MAX
            )
            crop_rng = np.random.default_rng(
                (
                    self.cursor.perm_seed,
                    self.cursor.epoch,
                    global_id,
                    draw_index,
                )
            )
            crop = crop_and_normalize(
                clip, self.config.segment_length, crop_rng
            )
            if self.augment_audio:
                augment_rng = np.random.default_rng(
                    (
                        self.cursor.perm_seed,
                        self.cursor.epoch,
                        global_id,
                        draw_index,
                        AUGMENT_SALT,
                    )
                )
                crop = augment(crop, augment_rng, self.config.sample_rate)

            self.cursor.draws_in_shard += 1
            if self.cursor.draws_in_shard >= len(plan):
                self._advance_shard()
            return np.asarray(crop, dtype=np.float32)

    def next_batch(self, batch_size: int) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return np.stack([self._next_crop() for _ in range(batch_size)], axis=0)
