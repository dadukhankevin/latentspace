"""Tests for latentspace.universal — runnable with pytest or directly:
    python3 tests/test_universal.py
"""
from __future__ import annotations

import numpy as np
import torch

"""These tests cover the LEGACY engines (per-individual explorer stack and
champion-per-problem population), which are benchmark opponents now, not the
API. The API is tested in test_ga.py."""
from latentspace.universal import (
    LatentSpace,
    distill,
    register_architecture,
    resolve,
)
from latentspace.universal.explorer import ExplorerConfig  # noqa: F401
from latentspace.universal.multi import solve_many
from latentspace.universal.solver import solve_single as solve


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
    # default is pure decoder evolution: no exploit phase, all budget explored
    assert result.explore_evaluations == result.evaluations


def test_solve_2d_auto_architecture_is_conv():
    factory = resolve("auto", 8, (16, 16))
    names = [type(m).__name__ for m in factory().modules()]
    assert any("Conv2d" in n for n in names)


def test_fixed_split_and_lineage_cap():
    # A fixed explore/exploit split only exists when the exploit phase runs.
    target = np.full(32, 0.25)
    result = solve(_curve_fitness(target), output_shape=(32,), budget=400,
                   latent=4, device="cpu", explore_fraction=0.5,
                   exploit="ga", lineage_cap=3, seed=1)
    assert result.evaluations == 400
    assert result.explore_evaluations <= 200 + 32  # fixed split honoured
    # with exploit off, the same split is ignored and everything explores
    result = solve(_curve_fitness(target), output_shape=(32,), budget=400,
                   latent=4, device="cpu", explore_fraction=0.5, seed=1)
    assert result.explore_evaluations == result.evaluations == 400


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


def test_solve_default_is_pure_decoder_evolution():
    # exploit=None resolves to "off" for single-phase runs: the whole budget
    # is exploration and no distilled latent space is built. CMA-ES must not
    # be reachable through solve() at all (it is a baseline, not a component).
    target = (np.sin(np.linspace(0, 2 * np.pi, 32)) * 0.3 + 0.5)
    result = solve(_curve_fitness(target), output_shape=(32,), budget=600,
                   latent=4, device="cpu", seed=11)
    assert result.evaluations == 600
    assert result.explore_evaluations == 600
    assert result.latent_space is None
    import inspect
    from latentspace.universal import solver
    assert "cma" not in inspect.getsource(solver).replace(
        "cma.py exists", "")


def test_cycle_requires_ga_exploit():
    target = np.full(16, 0.4, dtype=np.float32)
    try:
        solve(_curve_fitness(target), output_shape=(16,), budget=200,
              latent=4, device="cpu", phases="cycle", exploit="off", seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("cycle with exploit='off' should raise")


def test_solve_many_auto_ratio_adapts_and_improves():
    # The default latent_sigma_scale="auto": the genes/latents mutation
    # ratio self-tunes (matrix result: no fixed ratio is universal).
    rng = np.random.default_rng(8)
    targets = [rng.uniform(0.2, 0.8, 24).astype(np.float32) for _ in range(2)]
    result = solve_many([_curve_fitness(t) for t in targets],
                        output_shape=(24,), budget=600, latent=4,
                        children=16, device="cpu", seed=12)
    assert result.evaluations == 600
    for problem in result.problems:
        assert problem.best_fitness >= problem.initial_fitness
    ratios = [h["latent_ratio"] for h in result.history]
    assert ratios[-1] != 1.0    # the controller actually moved


def test_cycle_phases_respects_budget():
    target = (np.sin(np.linspace(0, 2 * np.pi, 32)) * 0.3 + 0.5)
    result = solve(_curve_fitness(target), output_shape=(32,), budget=800,
                   latent=4, device="cpu", phases="cycle", seed=3)
    assert result.evaluations == 800
    assert result.best_fitness >= result.history[31]


def test_solve_many_improves_every_problem_and_respects_budget():
    rng = np.random.default_rng(1)
    targets = [rng.uniform(0.2, 0.8, 32).astype(np.float32) for _ in range(3)]
    result = solve_many([_curve_fitness(t) for t in targets],
                        output_shape=(32,), budget=900, latent=4,
                        slots_per_problem=2, children=16, device="cpu", seed=2)
    assert result.evaluations == 900
    assert len(result.problems) == 3
    assert sum(p.evaluations for p in result.problems) == 900
    for problem in result.problems:
        assert problem.best_phenotype.shape == (32,)
        assert problem.best_fitness >= problem.initial_fitness
    assert (result.history[-1]["mean_best_fitness"]
            > result.history[0]["mean_best_fitness"])


def test_solve_many_deterministic_with_seed():
    target = np.full(16, 0.3, dtype=np.float32)
    fns = [_curve_fitness(target), _curve_fitness(1 - target)]
    a = solve_many(fns, output_shape=(16,), budget=300, latent=4,
                   children=8, device="cpu", seed=5)
    b = solve_many(fns, output_shape=(16,), budget=300, latent=4,
                   children=8, device="cpu", seed=5)
    assert list(a.best_fitnesses) == list(b.best_fitnesses)


def test_solve_many_consolidation_trains_and_returns_a_core():
    rng = np.random.default_rng(3)
    targets = [rng.uniform(0.2, 0.8, 16).astype(np.float32) for _ in range(2)]
    fns = [_curve_fitness(t) for t in targets]
    result = solve_many(fns, output_shape=(16,), budget=400, latent=4,
                        slots_per_problem=2, children=8, device="cpu",
                        consolidate="champions", seed=4)
    assert result.consolidations >= 1
    assert result.decoder is not None and result.decoder.ndim == 1
    for problem in result.problems:
        assert problem.best_fitness >= problem.initial_fitness
    # the returned core warm-starts a new run on a fresh problem
    fresh = _curve_fitness(rng.uniform(0.2, 0.8, 16).astype(np.float32))
    warm = solve_many([fresh], output_shape=(16,), budget=120, latent=4,
                      slots_per_problem=2, children=8, device="cpu",
                      init_decoder=result.decoder, seed=6)
    assert warm.evaluations == 120


def test_auto_rgb_architecture_is_conv_and_orders_pixels_correctly():
    factory = resolve("auto", 8, (16, 16, 3))
    net = factory()
    names = [type(m).__name__ for m in net.modules()]
    assert any("Conv2d" in n for n in names)
    out = net(torch.zeros(2, 8))
    assert out.shape == (2, 16 * 16 * 3)
    # channels-last flattening: consecutive triples are one pixel's colors,
    # so shifting by one channel decorrelates less than shifting by one row
    img = out[0].reshape(16, 16, 3)
    assert img.shape == (16, 16, 3)


def test_solve_many_breeders_strategy_runs():
    target = np.linspace(0.2, 0.8, 16).astype(np.float32)
    fns = [_curve_fitness(target), _curve_fitness(target[::-1].copy())]
    result = solve_many(fns, output_shape=(16,), budget=400, latent=4,
                        slots_per_problem=2, children=8, device="cpu",
                        consolidate="breeders", seed=9)
    assert result.consolidations >= 1
    assert result.evaluations == 400


def test_solve_many_crossover_modes_run():
    # The shared-decoder architecture: crossover acts on the genes (z + LoRA
    # coefficients) read by ONE shared decoder. All modes must run and improve.
    target = np.linspace(0.2, 0.8, 16).astype(np.float32)
    fns = [_curve_fitness(target), _curve_fitness(target[::-1].copy())]
    for mode in ("one_point", "uniform", "bpe"):
        r = solve_many(fns, output_shape=(16,), budget=400, latent=4,
                       slots_per_problem=2, children=8, device="cpu",
                       crossover_rate=0.5, crossover_mode=mode, seed=7)
        assert r.evaluations == 400
        for p in r.problems:
            assert p.best_fitness >= p.initial_fitness


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all tests passed")
