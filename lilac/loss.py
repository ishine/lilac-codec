"""Frozen reconstruction and adversarial objectives for LILAC."""

from __future__ import annotations

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from config import LILACConfig
from lilac.discriminator import DiscOutput


class SpectralLoss(nn.Module):
    """Three-resolution mel L1 plus global-Frobenius MR-STFT."""

    def __init__(self, config: LILACConfig):
        super().__init__()
        self.resolutions = (
            (512, 128, 64),
            (1024, 256, 128),
            (2048, 512, 256),
        )
        for n_fft, _, n_mels in self.resolutions:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft))
            filterbank = torchaudio.functional.melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                f_min=0.0,
                f_max=config.sample_rate / 2,
                n_mels=n_mels,
                sample_rate=config.sample_rate,
                norm=None,
                mel_scale="htk",
            )
            self.register_buffer(f"mel_{n_fft}", filterbank)

    def global_squared_norms(
        self, x: torch.Tensor, x_hat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-resolution squared norms used to recover global-batch SC."""
        x = x.squeeze(1)
        x_hat = x_hat.squeeze(1)
        numerators: list[torch.Tensor] = []
        denominators: list[torch.Tensor] = []
        for n_fft, hop, _ in self.resolutions:
            window = getattr(self, f"window_{n_fft}")
            magnitude = torch.stft(
                x, n_fft, hop_length=hop, window=window, return_complex=True
            ).abs()
            magnitude_hat = torch.stft(
                x_hat, n_fft, hop_length=hop, window=window, return_complex=True
            ).abs()
            numerators.append((magnitude - magnitude_hat).square().sum())
            denominators.append(magnitude.square().sum())
        return torch.stack(numerators), torch.stack(denominators)

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        *,
        global_squares: tuple[torch.Tensor, torch.Tensor] | None = None,
        micro_batches: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.squeeze(1)
        x_hat = x_hat.squeeze(1)
        mel_loss = x.new_zeros(())
        stft_loss = x.new_zeros(())
        for n_fft, hop, _ in self.resolutions:
            window = getattr(self, f"window_{n_fft}")
            filterbank = getattr(self, f"mel_{n_fft}")
            magnitude = torch.stft(
                x,
                n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                return_complex=True,
            ).abs()
            magnitude_hat = torch.stft(
                x_hat,
                n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                return_complex=True,
            ).abs()

            # The global denominator is load-bearing: equal-weight per-example
            # convergence is unstable when a batch contains a near-silent crop.
            numerator_square = (magnitude - magnitude_hat).square().sum()
            denominator_square = magnitude.square().sum()
            if global_squares is None:
                convergence = numerator_square.sqrt() / denominator_square.sqrt().clamp_min(1e-7)
            else:
                index = next(
                    i
                    for i, resolution in enumerate(self.resolutions)
                    if resolution[0] == n_fft
                )
                global_num = global_squares[0][index].detach().clamp_min(1e-20)
                global_den = global_squares[1][index].detach().clamp_min(1e-20)
                global_value = global_num.sqrt() / global_den.sqrt().clamp_min(1e-7)
                # With the caller's 1/K backward scaling, this surrogate has
                # exactly the gradient of sqrt(sum_i N_i)/sqrt(sum_i D_i).
                gradient_term = (
                    micro_batches
                    * numerator_square
                    / (2 * global_num.sqrt() * global_den.sqrt().clamp_min(1e-7))
                )
                convergence = gradient_term + (global_value - gradient_term.detach())
            log_magnitude = F.l1_loss(
                torch.log(magnitude_hat + 1e-7),
                torch.log(magnitude + 1e-7),
            )
            stft_loss = stft_loss + convergence + log_magnitude

            mel = torch.matmul(magnitude.transpose(-1, -2), filterbank)
            mel_hat = torch.matmul(magnitude_hat.transpose(-1, -2), filterbank)
            mel_loss = mel_loss + F.l1_loss(mel_hat, mel)
        return mel_loss, stft_loss


def discriminator_loss(real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]) -> torch.Tensor:
    pairs = [
        F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
        for real, fake in zip(real_logits, fake_logits)
    ]
    return sum(pairs) / len(pairs)


def feature_matching_loss(
    real_features: list[list[torch.Tensor]],
    fake_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    pairs = [
        F.l1_loss(fake, real.detach())
        for real_group, fake_group in zip(real_features, fake_features)
        for real, fake in zip(real_group, fake_group)
    ]
    return sum(pairs) / len(pairs)


def adversarial_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    return sum(-logit.mean() for logit in fake_logits) / len(fake_logits)


def adversarial_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(step / warmup_steps, 1.0)


def generator_loss(
    config: LILACConfig,
    spectral_loss: SpectralLoss,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    real: DiscOutput,
    fake: DiscOutput,
    step: int,
    *,
    global_squares: tuple[torch.Tensor, torch.Tensor] | None = None,
    micro_batches: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mel, mrstft = spectral_loss(
        x,
        x_hat,
        global_squares=global_squares,
        micro_batches=micro_batches,
    )
    adversarial = adversarial_loss(fake[0])
    feature = feature_matching_loss(real[1], fake[1])
    scale = adversarial_scale(step, config.adv_warmup_steps)
    total = (
        config.lambda_mel * mel
        + config.lambda_mrstft * mrstft
        + scale
        * (
            config.lambda_adv * adversarial
            + config.lambda_feat * feature
        )
    )
    return total, {
        "mel": mel.detach(),
        "mrstft": mrstft.detach(),
        "adv": adversarial.detach(),
        "feat": feature.detach(),
        "total": total.detach(),
    }
