"""Round 1: deceptive, hierarchical, and rugged objectives.

The TSP studies established that the latent decoder loses badly where direct
locality is already perfect. This suite tests the opposite terrain — problems
whose structure punishes direct bitwise search:

  * trap5_50    — ten concatenated 5-bit deceptive traps (optimum: all ones);
  * hiff64      — 64-bit hierarchical-if-and-only-if (optima: all 0s / all 1s);
  * nk32        — an N=32, K=4 rugged NK landscape, one fixed instance;
  * rastrigin64 — the 64-dimensional version of the one objective family where
                  latent variants beat direct search at 16 dimensions.

Two baselines missing from earlier studies are added: a direct bitstring GA
(the fair conventional method for binary problems) and a from-scratch
(mu/mu_w, lambda)-CMA-ES following Hansen's tutorial, with clip-to-bounds
repair. pip cannot install pycma in this environment, so the implementation
is validated by `--self-test` (sphere must reach 1e-6 in 5,000 evaluations).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from latentspace import Evolver
from latentspace.training import GuardedTrainer

from benchmarks.compare import (
    BenchmarkConfig,
    Objective,
    Rastrigin,
    TrackedFitness,
    _finish_result,
    _rank_probabilities,
    _require_mps,
    _seed_everything,
    _warm_mps,
    print_summary,
    run_direct_ga,
    run_latent,
    run_random_search,
    summarize,
)
from benchmarks.decoder_training import make_trainer


class BinaryObjective(Objective):
    """Loss is defined on thresholded bits; tensors reuse the NumPy path."""

    def bits(self, phenotypes: np.ndarray) -> np.ndarray:
        return np.asarray(phenotypes) > 0.5

    def loss_tensor(self, phenotypes: torch.Tensor) -> torch.Tensor:
        losses = self.loss_numpy(phenotypes.detach().cpu().numpy())
        return torch.as_tensor(losses, device=phenotypes.device, dtype=phenotypes.dtype)


class Trap5(BinaryObjective):
    name = "trap5_50"
    metric_name = "trap_loss"

    def __init__(self, blocks: int = 10):
        self.blocks = blocks
        self.dimension = blocks * 5

    def loss_numpy(self, phenotypes):
        bits = self.bits(phenotypes).reshape(len(phenotypes), self.blocks, 5)
        unitation = bits.sum(axis=2)
        score = np.where(unitation == 5, 5, 4 - unitation)
        return (5 * self.blocks - score.sum(axis=1)).astype(np.float32)


class HIFF(BinaryObjective):
    name = "hiff64"
    metric_name = "hiff_loss"

    def __init__(self, bits: int = 64):
        if bits & (bits - 1):
            raise ValueError("HIFF size must be a power of two")
        self.dimension = bits
        self.max_value = bits * (int(np.log2(bits)) + 1)

    def loss_numpy(self, phenotypes):
        bits = self.bits(phenotypes)
        batch, n = bits.shape
        value = np.full(batch, n, dtype=np.int64)
        current = bits.astype(np.int8)
        uniform = np.ones_like(current, dtype=bool)
        size = 1
        while size < n:
            left, right = current[:, 0::2], current[:, 1::2]
            uniform = uniform[:, 0::2] & uniform[:, 1::2] & (left == right)
            current = left
            size *= 2
            value += uniform.sum(axis=1) * size
        return (self.max_value - value).astype(np.float32)


class NKLandscape(BinaryObjective):
    name = "nk32"
    metric_name = "nk_loss"

    def __init__(self, n: int = 32, k: int = 4, instance_seed: int = 2026):
        self.dimension = n
        self.k = k
        rng = np.random.default_rng(instance_seed)
        neighbors = np.empty((n, k), dtype=np.int64)
        for site in range(n):
            others = np.delete(np.arange(n), site)
            neighbors[site] = rng.choice(others, size=k, replace=False)
        self.sites = np.concatenate([np.arange(n)[:, None], neighbors], axis=1)
        self.tables = rng.random((n, 2 ** (k + 1)), dtype=np.float32)
        self.powers = (2 ** np.arange(k + 1)).astype(np.int64)

    def loss_numpy(self, phenotypes):
        bits = self.bits(phenotypes).astype(np.int64)
        codes = bits[:, self.sites] @ self.powers
        contributions = self.tables[np.arange(self.dimension), codes]
        return (1.0 - contributions.mean(axis=1)).astype(np.float32)


class Rastrigin64(Rastrigin):
    name = "rastrigin64"

    def __init__(self):
        super().__init__(dimension=64)


OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "trap5_50": Trap5,
    "hiff64": HIFF,
    "nk32": NKLandscape,
    "rastrigin64": Rastrigin64,
}

BINARY_OBJECTIVES = {"trap5_50", "hiff64", "nk32"}


def run_direct_bit_ga(objective, seed, config):
    """Rank-selected bitstring GA: uniform crossover, 1/n bit flips."""
    rng = np.random.default_rng(seed)
    tracker = TrackedFitness(objective)
    started = time.perf_counter()
    dimension = objective.dimension
    population = (rng.random((config.population, dimension)) < 0.5).astype(np.float32)
    fitness = tracker.evaluate_numpy(population)
    generations = 0

    while tracker.evaluations < config.evaluation_budget:
        amount = min(config.offspring, config.evaluation_budget - tracker.evaluations)
        order = np.argsort(-fitness)
        ranked = population[order]
        probabilities = _rank_probabilities(len(ranked))
        parent_indices = rng.choice(
            len(ranked), size=(amount, 2), replace=True, p=probabilities
        )
        first, second = ranked[parent_indices[:, 0]], ranked[parent_indices[:, 1]]
        crossover_mask = rng.random((amount, dimension)) < 0.5
        children = np.where(crossover_mask, first, second)
        flip_mask = rng.random((amount, dimension)) < (1.0 / dimension)
        empty = np.flatnonzero(~flip_mask.any(axis=1))
        if len(empty):
            flip_mask[empty, rng.integers(0, dimension, len(empty))] = True
        children = np.where(flip_mask, 1.0 - children, children).astype(np.float32)
        child_fitness = tracker.evaluate_numpy(children)
        population = np.concatenate([population, children])
        fitness = np.concatenate([fitness, child_fitness])
        survivors = np.argsort(-fitness)[: config.population]
        population, fitness = population[survivors], fitness[survivors]
        generations += 1

    return _finish_result(
        objective, "direct_bit_ga", seed, config, tracker, started, generations
    )


def run_cmaes(objective, seed, config, lam=None, label="cmaes"):
    """(mu/mu_w, lambda)-CMA-ES per Hansen's tutorial, clip-to-bounds repair.

    The final partial generation is evaluated to honour the budget but not used
    for a distribution update. `lam=None` uses Hansen's default population.
    """
    rng = np.random.default_rng(seed)
    tracker = TrackedFitness(objective)
    started = time.perf_counter()
    dim = objective.dimension

    lam = lam if lam is not None else 4 + int(3 * np.log(dim))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights**2)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

    mean = np.full(dim, 0.5)
    sigma = 0.3
    covariance = np.eye(dim)
    ps = np.zeros(dim)
    pc = np.zeros(dim)
    generations = 0

    while tracker.evaluations < config.evaluation_budget:
        covariance = (covariance + covariance.T) / 2
        eigenvalues, basis = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        scales = np.sqrt(eigenvalues)
        inv_sqrt_c = (basis / scales) @ basis.T

        remaining = config.evaluation_budget - tracker.evaluations
        n_sample = min(lam, remaining)
        z = rng.standard_normal((n_sample, dim))
        y = z @ (basis * scales).T
        x = np.clip(mean + sigma * y, 0.0, 1.0)
        losses = -tracker.evaluate_numpy(x.astype(np.float32))
        if n_sample < lam:
            break

        y = (x - mean) / sigma  # repair: update from the points actually scored
        order = np.argsort(losses)
        selected = y[order[:mu]]
        y_weighted = weights @ selected
        mean = mean + sigma * y_weighted
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_c @ y_weighted)
        generations += 1
        hsig = (
            np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * generations)) / chi_n
            < 1.4 + 2 / (dim + 1)
        )
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_weighted
        covariance = (
            (1 - c1 - cmu) * covariance
            + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * covariance)
            + cmu * (selected.T * weights) @ selected
        )
        sigma = min(
            sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1)), 1.0
        )

    return _finish_result(
        objective, label, seed, config, tracker, started, generations
    )


def run_latent_guarded(objective, seed, config):
    """Guarded random non-RL mixture — the project's robust trainer default."""
    _require_mps()
    _seed_everything(seed)
    tracker = TrackedFitness(objective)
    trainer = make_trainer("guarded_random_non_rl", config, seed=seed)
    families = max(1, config.offspring // 4)
    evolver = Evolver(
        tracker,
        output_shape=(objective.dimension,),
        device="mps",
        latent=config.latent,
        population=config.population,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        lr=config.decoder_lr,
        mutation_rate=config.mutation_rate,
        mutation_sigma=config.mutation_sigma,
        refine_every=config.refine_every,
        refine_percent=config.refine_percent,
        pressure=1.8,
        scheme="linear",
        families=families,
        children=4,
        n_points=4,
        offspring_only_mutation=config.offspring_only_mutation,
        trainer=trainer,
    )
    parameter_devices = {p.device.type for p in evolver.decoder.parameters()}
    if parameter_devices != {"mps"}:
        raise RuntimeError(f"decoder parameters are not exclusively on MPS: {parameter_devices}")

    started = time.perf_counter()
    while tracker.evaluations < config.evaluation_budget:
        evolver.solve(1, verbose_every=0)
    torch.mps.synchronize()
    result = _finish_result(
        objective,
        "latent_guarded",
        seed,
        config,
        tracker,
        started,
        generations=evolver.env.generation,
        neural_device="mps",
        trainer_acceptance=(
            tuple(trainer.acceptance_history)
            if isinstance(trainer, GuardedTrainer)
            else None
        ),
    )
    torch.mps.empty_cache()
    return result


STRATEGIES: dict[str, Callable] = {
    "random_search": run_random_search,
    "direct_ga": run_direct_ga,
    "direct_bit_ga": run_direct_bit_ga,
    "cmaes": run_cmaes,
    "latent_fixed": lambda o, s, c: run_latent(o, s, c, "latent_fixed"),
    "latent_gradient": lambda o, s, c: run_latent(o, s, c, "latent_gradient"),
    "latent_decoder_es": lambda o, s, c: run_latent(o, s, c, "latent_decoder_es"),
    "latent_guarded": run_latent_guarded,
}


def self_test():
    """Objective invariants plus CMA-ES validation on a 16-d sphere."""
    trap = Trap5()
    ones = np.ones((1, trap.dimension), dtype=np.float32)
    zeros = np.zeros((1, trap.dimension), dtype=np.float32)
    assert trap.loss_numpy(ones)[0] == 0.0, "trap optimum must be all ones"
    assert trap.loss_numpy(zeros)[0] == 10.0, "trap all-zeros must be the deceptive trap"

    hiff = HIFF()
    ones = np.ones((1, 64), dtype=np.float32)
    zeros = np.zeros((1, 64), dtype=np.float32)
    alternating = np.tile([0.0, 1.0], 32).astype(np.float32)[None]
    assert hiff.loss_numpy(ones)[0] == 0.0
    assert hiff.loss_numpy(zeros)[0] == 0.0
    assert hiff.loss_numpy(alternating)[0] == float(hiff.max_value - 64)

    nk = NKLandscape()
    samples = (np.random.default_rng(0).random((100, nk.dimension)) < 0.5).astype(np.float32)
    losses = nk.loss_numpy(samples)
    assert np.all((losses > 0) & (losses < 1))

    class Sphere(Objective):
        name = "sphere"
        metric_name = "sphere"
        dimension = 16

        def loss_numpy(self, phenotypes):
            return np.sum((np.asarray(phenotypes) - 0.5) ** 2, axis=1)

    config = BenchmarkConfig(evaluation_budget=5_000)
    result = run_cmaes(Sphere(), 0, config)
    assert result.metric_at_budget < 1e-6, f"CMA-ES sphere check failed: {result.metric_at_budget}"
    print(f"self-test ok (cmaes sphere loss {result.metric_at_budget:.3g})")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES)
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    config = BenchmarkConfig(evaluation_budget=args.budget)
    has_neural = any(name.startswith("latent_") for name in args.strategies)
    if has_neural:
        _require_mps()
    results = []
    for objective_name in args.objectives:
        if has_neural:
            _warm_mps(OBJECTIVES[objective_name](), config)
        for strategy_name in args.strategies:
            if strategy_name == "direct_bit_ga" and objective_name not in BINARY_OBJECTIVES:
                continue
            for seed in args.seeds:
                objective = OBJECTIVES[objective_name]()
                print(
                    f"run objective={objective_name:<12} strategy={strategy_name:<19} "
                    f"seed={seed} budget={config.evaluation_budget}",
                    flush=True,
                )
                result = STRATEGIES[strategy_name](objective, seed, config)
                print(
                    f"  {result.metric}={result.metric_at_budget:.6g} "
                    f"evals_run={result.evaluations_run} "
                    f"device={result.neural_device or 'numpy/cpu'}",
                    flush=True,
                )
                results.append(result)
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "torch_version": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "runs": [asdict(result) for result in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
