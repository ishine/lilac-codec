"""The fixed multi-period plus multi-resolution STFT discriminator."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

DiscOutput = tuple[list[torch.Tensor], list[list[torch.Tensor]]]


def _conv1d(*args, **kwargs) -> nn.Module:
    return weight_norm(nn.Conv1d(*args, **kwargs))


def _conv2d(*args, **kwargs) -> nn.Module:
    return weight_norm(nn.Conv2d(*args, **kwargs))


class PeriodSubDiscriminator(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        channels = (32, 64, 128, 256)
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.append(
                _conv1d(in_channels, out_channels, 5, stride=3, padding=2)
            )
            in_channels = out_channels
        self.convs = nn.ModuleList(layers)
        self.conv_post = _conv1d(in_channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        _, _, length = x.shape
        padding = (self.period - length % self.period) % self.period
        if padding:
            x = F.pad(x, (0, padding))
        batch, _, padded = x.shape
        x = (
            x.view(batch, 1, padded // self.period, self.period)
            .permute(0, 1, 3, 2)
            .reshape(batch, 1, padded)
        )
        features: list[torch.Tensor] = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x.flatten(1), features


class STFTSubDiscriminator(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        channels = (1, 16, 32, 64, 128)
        self.convs = nn.ModuleList(
            _conv2d(
                channels[index],
                channels[index + 1],
                (3, 9),
                stride=(2, 1) if index < len(channels) - 2 else (1, 1),
                padding=(1, 4),
            )
            for index in range(len(channels) - 1)
        )
        self.conv_post = _conv2d(channels[-1], 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = torch.stft(
            x.squeeze(1),
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        ).abs().unsqueeze(1)
        features: list[torch.Tensor] = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x.flatten(1), features


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.periods = nn.ModuleList(
            PeriodSubDiscriminator(period) for period in (2, 3, 5, 7, 11)
        )
        self.stfts = nn.ModuleList(
            STFTSubDiscriminator(*resolution)
            for resolution in (
                (2048, 512, 2048),
                (1024, 256, 1024),
                (512, 128, 512),
                (256, 64, 256),
                (128, 32, 128),
            )
        )

    def forward(self, x: torch.Tensor) -> DiscOutput:
        logits: list[torch.Tensor] = []
        features: list[list[torch.Tensor]] = []
        for discriminator in (*self.periods, *self.stfts):
            logit, feature_map = discriminator(x)
            logits.append(logit)
            features.append(feature_map)
        return logits, features
