"""The complete LILAC 0.75 kb/s generator.

The encoder is an exactly invertible analysis chart followed by finite scalar
quantization. The decoder predicts only the discarded chart coordinates, then
runs that same chart backward. Consequently, decoded integer codes re-encode to
the same integer codes (up to the deliberately audited fp32 numerical margin).
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint as torch_checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from config import LILACConfig


def _squeeze(x: torch.Tensor, factor: int) -> torch.Tensor:
    """[B,C,T] -> [B,factor*C,T/factor], with adjacent phases in channels."""
    if x.ndim != 3 or x.shape[-1] % factor:
        raise ValueError(
            f"expected [B,C,T] with T divisible by {factor}, got {tuple(x.shape)}"
        )
    b, c, t = x.shape
    return (
        x.view(b, c, t // factor, factor)
        .permute(0, 3, 1, 2)
        .reshape(b, factor * c, t // factor)
    )


def _unsqueeze(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Exact inverse of :func:`_squeeze`."""
    if x.ndim != 3 or x.shape[1] % factor:
        raise ValueError(
            f"expected [B,C,T] with C divisible by {factor}, got {tuple(x.shape)}"
        )
    b, fc, t = x.shape
    c = fc // factor
    return (
        x.view(b, factor, c, t)
        .permute(0, 2, 3, 1)
        .reshape(b, c, t * factor)
    )


class OrthogonalMix(nn.Module):
    """Channels-first orthogonal 1x1 mix; its inverse is its transpose."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, channels))
        nn.init.orthogonal_(self.weight)
        nn.utils.parametrizations.orthogonal(self, "weight")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.weight[:, :, None])

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.weight.transpose(0, 1)[:, :, None])


class Snake(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.log_a = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.log_a.clamp(-20.0, 20.0).exp()
        return x + torch.sin(a * x).square() / a


class ConvNeXt1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, checkpoint: bool = False):
        super().__init__()
        hidden = 4 * channels
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv1d(channels, hidden, 1)
        self.act = Snake(hidden)
        self.pw2 = nn.Conv1d(hidden, channels, 1)
        self.checkpoint = checkpoint

    def _body(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.checkpoint
            and self.training
            and torch.is_grad_enabled()
            and x.requires_grad
            and x.shape[-1] >= 512
        ):
            return torch_checkpoint.checkpoint(self._body, x, use_reentrant=False)
        return self._body(x)


class ConvResidualNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: LILACConfig,
        *,
        checkpoint: bool = False,
    ):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, config.conv_hidden, 1)
        self.blocks = nn.ModuleList(
            ConvNeXt1D(config.conv_hidden, checkpoint=checkpoint)
            for _ in range(config.conv_depth)
        )
        self.head = nn.Conv1d(config.conv_hidden, out_channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class RevMixBlock(nn.Module):
    """Orthogonal mixing plus a NICE additive coupling pair."""

    def __init__(self, channels: int, config: LILACConfig):
        super().__init__()
        left = channels // 2
        self.left_channels = left
        self.mix_in = OrthogonalMix(channels)
        self.f = ConvResidualNet(left, channels - left, config, checkpoint=True)
        self.g = ConvResidualNet(channels - left, left, config, checkpoint=True)
        self.mix_out = OrthogonalMix(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mix_in(x)
        a, b = x[:, : self.left_channels], x[:, self.left_channels :]
        b = b + self.f(a)
        a = a + self.g(b)
        return self.mix_out(torch.cat((a, b), dim=1))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mix_out.inverse(x)
        a, b = x[:, : self.left_channels], x[:, self.left_channels :]
        a = a - self.g(b)
        b = b - self.f(a)
        return self.mix_in.inverse(torch.cat((a, b), dim=1))


class Invertible1x1(nn.Module):
    """General five-phase mix, allowing attenuation as well as rotation."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,bjt->bit", self.weight, x)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        inverse = torch.linalg.inv(self.weight)
        return torch.einsum("ij,bjt->bit", inverse, x)


class AffineCoupling(nn.Module):
    def __init__(
        self,
        channels: int,
        split: int,
        hidden: int,
        log_scale_bound: float,
    ):
        super().__init__()
        self.split = split
        self.log_scale_bound = log_scale_bound
        self.net = nn.Sequential(
            nn.Conv1d(split, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, 2 * (channels - split), 5, padding=2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _scale_shift(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_scale, shift = self.net(a).chunk(2, dim=1)
        return self.log_scale_bound * raw_scale.tanh(), shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x[:, : self.split], x[:, self.split :]
        log_scale, shift = self._scale_shift(a)
        return torch.cat((a, b * log_scale.exp() + shift), dim=1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x[:, : self.split], x[:, self.split :]
        log_scale, shift = self._scale_shift(a)
        return torch.cat((a, (b - shift) * (-log_scale).exp()), dim=1)


class StemFlowBlock(nn.Module):
    def __init__(self, config: LILACConfig):
        super().__init__()
        self.mix = Invertible1x1(5)
        self.coupling = AffineCoupling(
            5,
            split=2,
            hidden=config.stem_flow_hidden,
            log_scale_bound=config.stem_log_scale_bound,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coupling(self.mix(x))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix.inverse(self.coupling.inverse(x))


class StemFlow(nn.Module):
    """Trainable invertible anti-imaging transform over the five audio phases."""

    def __init__(self, config: LILACConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            StemFlowBlock(config) for _ in range(config.stem_flow_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        for block in reversed(self.blocks):
            x = block.inverse(x)
        return x


class SqueezeMixStage(nn.Module):
    def __init__(self, in_channels: int, config: LILACConfig):
        super().__init__()
        self.mix = RevMixBlock(2 * in_channels, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(_squeeze(x, 2))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return _unsqueeze(self.mix.inverse(x), 2)


class ProjectStage(nn.Module):
    def __init__(self, in_channels: int, keep_channels: int, config: LILACConfig):
        super().__init__()
        out_channels = 2 * in_channels
        self.keep_channels = keep_channels
        self.free_channels = out_channels - keep_channels
        self.mix = RevMixBlock(out_channels, config)

    def analyze(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.mix(_squeeze(x, 2))
        return y[:, : self.keep_channels], y[:, self.keep_channels :]

    def synthesize(self, keep: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
        return _unsqueeze(self.mix.inverse(torch.cat((keep, free), dim=1)), 2)


class CodeContext(nn.Module):
    def __init__(self, config: LILACConfig):
        super().__init__()
        self.stem = nn.Conv1d(config.n_fsq_channels, config.conv_hidden, 1)
        kernels = (7, 9, 11)
        self.blocks = nn.ModuleList(
            ConvNeXt1D(config.conv_hidden, kernel_size=kernels[i % len(kernels)])
            for i in range(config.conv_depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x


class ContextUpsample(nn.Module):
    def __init__(self, config: LILACConfig):
        super().__init__()
        self.block = ConvNeXt1D(config.conv_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x.repeat_interleave(2, dim=-1))


class FreePredictor(nn.Module):
    def __init__(
        self,
        keep_channels: int,
        free_channels: int,
        config: LILACConfig,
    ):
        super().__init__()
        self.net = ConvResidualNet(
            keep_channels + config.conv_hidden, free_channels, config
        )
        self.log_beta = nn.Parameter(torch.full((1, free_channels, 1), -2.0))

    def forward(self, keep: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        value = self.net(torch.cat((keep, context), dim=1))
        return F.softplus(self.log_beta) * value.tanh()


class FSQ(nn.Module):
    def __init__(self, levels: int):
        super().__init__()
        if levels < 2:
            raise ValueError("FSQ requires at least two levels")
        self.levels = levels
        self.half_levels = (levels - 1) / 2

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        bounded = z.clamp(-1.0, 1.0)
        quantized = self.dequantize(self.codes(bounded))
        return bounded + (quantized - bounded).detach()

    def codes(self, z: torch.Tensor) -> torch.Tensor:
        return torch.round((z.clamp(-1.0, 1.0) + 1.0) * self.half_levels).long()

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.numel() and (codes.min() < 0 or codes.max() >= self.levels):
            raise ValueError(f"FSQ codes must be in [0,{self.levels - 1}]")
        return codes.to(dtype=torch.float32) / self.half_levels - 1.0


class Encoder(nn.Module):
    def __init__(
        self,
        stem_flow: StemFlow,
        stem_mix: RevMixBlock,
        pre_stages: list[SqueezeMixStage],
        project_stages: list[ProjectStage],
    ):
        super().__init__()
        self.stem_flow = stem_flow
        self.stem_mix = stem_mix
        self.pre_stages = nn.ModuleList(pre_stages)
        self.project_stages = nn.ModuleList(project_stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with parametrize.cached():
            x = self.stem_flow.inverse(_squeeze(x, 5))
            x = self.stem_mix(x)
            for stage in self.pre_stages:
                x = stage(x)
            for stage in self.project_stages:
                x, _ = stage.analyze(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        stem_flow: StemFlow,
        stem_mix: RevMixBlock,
        pre_stages: list[SqueezeMixStage],
        project_stages: list[ProjectStage],
        context: CodeContext,
        predictors: list[FreePredictor],
        context_ups: list[ContextUpsample],
    ):
        super().__init__()
        self.stem_flow = stem_flow
        self.stem_mix = stem_mix
        self.pre_stages = nn.ModuleList(pre_stages)
        self.project_stages = nn.ModuleList(project_stages)
        self.context = context
        self.predictors = nn.ModuleList(predictors)
        self.context_ups = nn.ModuleList(context_ups)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        with parametrize.cached():
            y = z
            context = self.context(z)
            upsamplers: list[ContextUpsample | None] = [*self.context_ups, None]
            for stage, predictor, upsample in zip(
                reversed(self.project_stages),
                reversed(self.predictors),
                upsamplers,
            ):
                y = stage.synthesize(y, predictor(y, context))
                if upsample is not None:
                    context = upsample(context)
            for stage in reversed(self.pre_stages):
                y = stage.inverse(y)
            y = self.stem_mix.inverse(y)
            return _unsqueeze(self.stem_flow(y), 5)


class LILAC(nn.Module):
    """LILAC generator with waveform, latent, and integer-code interfaces."""

    def __init__(self, config: LILACConfig | None = None):
        super().__init__()
        self.config = config or LILACConfig()
        self.config.validate()

        stem_flow = StemFlow(self.config)
        stem_mix = RevMixBlock(5, self.config)
        pre_stages = [
            SqueezeMixStage(channels, self.config)
            for channels in (5, 10, 20, 40)
        ]
        project_stages = [
            ProjectStage(80, keep, self.config) for keep in (80, 80, 80, 80, 20)
        ]
        context = CodeContext(self.config)
        predictors = [
            FreePredictor(stage.keep_channels, stage.free_channels, self.config)
            for stage in project_stages
        ]
        context_ups = [ContextUpsample(self.config) for _ in range(4)]

        self.encoder = Encoder(
            stem_flow, stem_mix, pre_stages, project_stages
        )
        self.fsq = FSQ(self.config.n_fsq_levels)
        self.decoder = Decoder(
            stem_flow,
            stem_mix,
            pre_stages,
            project_stages,
            context,
            predictors,
            context_ups,
        )

    def _validate_waveform(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"waveform must have shape [B,1,T], got {tuple(x.shape)}")
        if x.shape[-1] % self.config.hop_length:
            raise ValueError(
                f"waveform length must be divisible by {self.config.hop_length}"
            )
        if not x.is_floating_point():
            raise TypeError("waveform must be floating point")

    def analyze(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_waveform(x)
        return self.encoder(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode waveform ``[B,1,T]`` to integer codes ``[B,20,T/2560]``."""
        return self.fsq.codes(self.analyze(x))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode integer codes ``[B,20,N]`` to waveform ``[B,1,2560*N]``."""
        if codes.ndim != 3 or codes.shape[1] != self.config.n_fsq_channels:
            raise ValueError(
                f"codes must have shape [B,{self.config.n_fsq_channels},N], "
                f"got {tuple(codes.shape)}"
            )
        if codes.is_floating_point():
            raise TypeError("decode expects integer FSQ codes")
        return self.decoder(self.fsq.dequantize(codes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fsq(self.analyze(x)))
