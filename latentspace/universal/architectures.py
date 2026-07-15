"""Decoder architectures for the universal solver.

The one modality-specific element the universal algorithm allows: the
decoder network's shape may match its output (convolutions for images,
1-D convolutions for signals, dense as the fallback), exactly as the
fitness function already does. Evolution's operators never change.

Architectures are plain callables `builder(latent, output_shape) ->
nn.Module` whose module maps a (B, latent) tensor to (B, prod(output_shape))
logits (pre-sigmoid). Register your own with `register_architecture`.
"""
from __future__ import annotations

import math
from typing import Callable

import torch.nn as nn

Builder = Callable[[int, tuple], nn.Module]

_REGISTRY: dict[str, Builder] = {}


def register_architecture(name: str, builder: Builder) -> None:
    """Make `name` usable as the `architecture` argument of `solve`."""
    _REGISTRY[name] = builder


def build_mlp(latent: int, output_shape: tuple, hidden: int = 64) -> nn.Module:
    dim = int(math.prod(output_shape))
    return nn.Sequential(
        nn.Linear(latent, hidden), nn.LeakyReLU(),
        nn.Linear(hidden, dim),
    )


class _Conv2d(nn.Module):
    """latent -> small feature map -> upsample+conv stages -> H x W logits."""

    def __init__(self, latent: int, output_shape: tuple, channels: int = 16):
        super().__init__()
        height, width = output_shape
        doublings = max(1, min(
            3,
            int(math.log2(max(height, 4) // 4)),
            int(math.log2(max(width, 4) // 4)),
        ))
        self.base = (max(1, height >> doublings), max(1, width >> doublings))
        self.channels = channels
        self.fc = nn.Linear(latent, channels * self.base[0] * self.base[1])
        blocks: list[nn.Module] = []
        for _ in range(doublings):
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.LeakyReLU(),
            ]
        blocks += [
            nn.Upsample(size=(height, width), mode="nearest"),
            nn.Conv2d(channels, 1, 3, padding=1),
        ]
        self.convs = nn.Sequential(*blocks)

    def forward(self, z):
        x = self.fc(z).view(-1, self.channels, *self.base)
        return self.convs(x).flatten(1)


class _Conv1d(nn.Module):
    """latent -> short feature track -> upsample+conv stages -> dim logits."""

    def __init__(self, latent: int, output_shape: tuple, channels: int = 16):
        super().__init__()
        (dim,) = output_shape
        doublings = max(1, min(4, int(math.log2(max(dim, 16) // 16))))
        self.base = max(2, dim >> doublings)
        self.channels = channels
        self.fc = nn.Linear(latent, channels * self.base)
        blocks: list[nn.Module] = []
        for _ in range(doublings):
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(channels, channels, 5, padding=2),
                nn.LeakyReLU(),
            ]
        blocks += [
            nn.Upsample(size=dim, mode="nearest"),
            nn.Conv1d(channels, 1, 5, padding=2),
        ]
        self.convs = nn.Sequential(*blocks)

    def forward(self, z):
        x = self.fc(z).view(-1, self.channels, self.base)
        return self.convs(x).flatten(1)


register_architecture("mlp", build_mlp)
register_architecture("conv2d", _Conv2d)
register_architecture("conv1d", _Conv1d)


def resolve(architecture, latent: int, output_shape: tuple) -> Callable[[], nn.Module]:
    """Turn `architecture` (name, "auto", or builder) into a zero-arg factory."""
    output_shape = tuple(int(s) for s in output_shape)
    if callable(architecture):
        return lambda: architecture(latent, output_shape)
    if architecture == "auto":
        if len(output_shape) == 2 and min(output_shape) >= 8:
            architecture = "conv2d"
        elif len(output_shape) == 1 and output_shape[0] >= 32:
            architecture = "conv1d"
        else:
            architecture = "mlp"
    if architecture not in _REGISTRY:
        raise ValueError(
            f"unknown architecture {architecture!r}; "
            f"registered: {sorted(_REGISTRY)}"
        )
    builder = _REGISTRY[architecture]
    return lambda: builder(latent, output_shape)
