"""Tests for latentspace.universal — runnable with pytest or directly:
    python3 tests/test_universal.py
"""
from __future__ import annotations

import numpy as np
import torch

from latentspace.universal import (
    ExplorerConfig,
    LatentSpace,
    distill,
    register_architecture,
    resolve,
    solve,
)


def _curve_fitness(target):
    def fitness(phenotypes: torch.Tensor):
        t = torch.as_tensor(target, device=phenotypes.device,
                            dtype=phenotypes.dtype)
        return -torch.mean((phenotypes.flatten(1) - t) ** 2, dim=1)
    return fitness


def test_solve_1d_improves_and_respects_budget():
    rng = np.random.default_rng(0)
    target = (np.sin(np.linspace(0, 3 * np.pi, 64)) * 0.4 + 0.5)
    result = solve(_curve_fitness(target), output_shape=(64,), budget=600,
                   latent=8, device="cpu", seed=0)
    assert result.evaluations == 600
    assert len(result.history) == 600
    assert result.best_phenotype.shape == (64,)
    assert 0.0 <= result.best_phenotype.min() <= result.best_phenotype.max() <= 1.0
    assert result.best_fitness > result.history[31]  # improved after init gen
    assert result.explore_evaluations < result.evaluations  # exploit phase ran


def test_solve_2d_auto_architecture_is_conv():
    factory = resolve("auto", 8, (16, 16))
    names = [type(m).__name__ for m in factory().modules()]
    assert any("Conv2d" in n for n in names)


def test_fixed_split_and_lineage_cap():
    target = np.full(32, 0.25)
    result = solve(_curve_fitness(target), output_shape=(32,), budget=400,
                   latent=4, device="cpu", explore_fraction=0.5,
                   lineage_cap=3, seed=1)
    assert result.evaluations == 400
    assert result.explore_evaluations <= 200 + 32  # fixed split honoured


def test_distill_roundtrip():
    rng = np.random.default_rng(2)
    base = rng.random((50, 16)) * 0.5 + 0.25
    space = distill(base, latent=4, output_shape=(16,), device="cpu")
    decoded = space.decode(np.zeros((1, 4)))
    assert decoded.shape == (1, 16)
    # zero latent decodes near the corpus mean
    assert np.abs(decoded.numpy().mean() - base.mean()) < 0.1


def test_custom_architecture_registration():
    calls = []

    def tiny(latent, output_shape):
        calls.append((latent, output_shape))
        dim = int(np.prod(output_shape))
        return torch.nn.Linear(latent, dim)

    register_architecture("tiny-test", tiny)
    factory = resolve("tiny-test", 4, (8,))
    net = factory()
    assert isinstance(net, torch.nn.Linear)
    assert calls == [(4, (8,))]


def test_deterministic_with_seed():
    target = np.full(32, 0.7)
    a = solve(_curve_fitness(target), output_shape=(32,), budget=300,
              latent=4, device="cpu", seed=7)
    b = solve(_curve_fitness(target), output_shape=(32,), budget=300,
              latent=4, device="cpu", seed=7)
    assert a.best_fitness == b.best_fitness


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all tests passed")
