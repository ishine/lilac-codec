"""Frozen configuration for the public LILAC 0.75 kb/s reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LILACConfig:
    # Audio and rate: 20 coordinates × 4 bits × 24,000 / 2,560 = 750 bit/s.
    sample_rate: int = 24_000
    hop_length: int = 2_560
    segment_length: int = 25_600
    n_fsq_channels: int = 20
    n_fsq_levels: int = 16

    # The published analysis/synthesis chart and deterministic fill network.
    conv_hidden: int = 256
    conv_depth: int = 4
    stem_flow_blocks: int = 3
    stem_flow_hidden: int = 32
    stem_log_scale_bound: float = 0.5

    # One optimizer step represents exactly 256 crops. A single-device PyTorch
    # run realizes this with gradient accumulation.
    global_batch_size: int = 256
    micro_batch_size: int = 1
    max_steps: int = 898_000
    save_interval: int = 1_000
    log_interval: int = 50
    grad_clip: float = 0.3

    # Original run through 848k, followed by the promoted constant-LR tail.
    lr_generator: float = 2e-4
    lr_discriminator: float = 1e-4
    lr_warmup_steps: int = 1_000
    lr_decay_rate: float = 0.999996
    tail_start_step: int = 848_001
    tail_lr_generator: float = 5e-5
    tail_lr_discriminator: float = 2.5e-5
    adam_betas: tuple[float, float] = (0.8, 0.99)
    weight_decay: float = 1e-4

    lambda_mel: float = 15.0
    lambda_mrstft: float = 1.0
    lambda_feat: float = 2.0
    lambda_adv: float = 1.0
    adv_warmup_steps: int = 5_000

    # The shipping artifact is the uniform mean of checkpoints 889k--898k.
    swa_window: int = 10
    output_dir: str = "runs"
    seed: int = 0

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def bitrate(self) -> float:
        return (
            self.n_fsq_channels
            * (self.n_fsq_levels.bit_length() - 1)
            * self.frame_rate
        )

    def validate(self) -> None:
        if self.hop_length != 2_560:
            raise ValueError("the frozen chart requires hop_length=2560")
        if self.segment_length % self.hop_length:
            raise ValueError("segment_length must be divisible by hop_length")
        if self.n_fsq_channels != 20 or self.n_fsq_levels != 16:
            raise ValueError("the frozen 0.75 kb/s model uses FSQ 20x16")
        if self.global_batch_size % self.micro_batch_size:
            raise ValueError("global_batch_size must be divisible by micro_batch_size")
        if self.save_interval != 1_000 or self.swa_window != 10:
            raise ValueError("the shipping SWA window requires 1k saves and 10 snapshots")
        if self.tail_start_step != 848_001 or self.max_steps != 898_000:
            raise ValueError("the promoted tail is frozen at updates 848001--898000")

    def to_dict(self) -> dict:
        return asdict(self)


def learning_rates(config: LILACConfig, step: int) -> tuple[float, float]:
    """Learning rates for the one-indexed optimizer update ``step``."""
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if step >= config.tail_start_step:
        return config.tail_lr_generator, config.tail_lr_discriminator
    count = step - 1
    if count < config.lr_warmup_steps:
        warmup = count / config.lr_warmup_steps
        return config.lr_generator * warmup, config.lr_discriminator * warmup
    decay = config.lr_decay_rate ** (count - config.lr_warmup_steps)
    return config.lr_generator * decay, config.lr_discriminator * decay
