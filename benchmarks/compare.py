"""Evaluation-budgeted comparison of latent evolution and direct baselines.

Neural decoder variants are required to run on Apple's MPS backend. Direct
baselines use NumPy because they contain no neural networks to accelerate.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from latentspace import EvolveDecoder, Evolver, MLPDecoder


class Objective:
    name: str
    metric_name: str
    dimension: int

    def loss_numpy(self, phenotypes: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def loss_tensor(self, phenotypes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TargetMatch(Objective):
    name = "target_match"
    metric_name = "mse"

    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        self.target = np.linspace(0, 1, dimension, dtype=np.float32)

    def loss_numpy(self, phenotypes):
        return np.mean((phenotypes - self.target) ** 2, axis=1)

    def loss_tensor(self, phenotypes):
        target = torch.linspace(
            0, 1, self.dimension, device=phenotypes.device, dtype=phenotypes.dtype
        )
        return torch.mean((phenotypes - target) ** 2, dim=1)


class Rastrigin(Objective):
    name = "rastrigin"
    metric_name = "rastrigin"

    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def loss_numpy(self, phenotypes):
        values = phenotypes * 10.24 - 5.12
        return 10 * self.dimension + np.sum(
            values**2 - 10 * np.cos(2 * np.pi * values), axis=1
        )

    def loss_tensor(self, phenotypes):
        values = phenotypes * 10.24 - 5.12
        return 10 * self.dimension + torch.sum(
            values**2 - 10 * torch.cos(2 * torch.pi * values), dim=1
        )


class TravelingSalesperson(Objective):
    name = "tsp"
    metric_name = "tour_length"

    def __init__(self, dimension: int = 12, instance_seed: int = 2026):
        self.dimension = dimension
        rng = np.random.default_rng(instance_seed)
        self.cities = rng.random((dimension, 2), dtype=np.float32)
        delta = self.cities[:, None, :] - self.cities[None, :, :]
        self.distances = np.sqrt(np.sum(delta**2, axis=-1)).astype(np.float32)
        self._tensor_cache: dict[str, torch.Tensor] = {}

    def loss_numpy(self, phenotypes):
        routes = np.argsort(phenotypes, axis=1)
        next_cities = np.roll(routes, -1, axis=1)
        return self.distances[routes, next_cities].sum(axis=1)

    def loss_tensor(self, phenotypes):
        key = str(phenotypes.device)
        distances = self._tensor_cache.get(key)
        if distances is None:
            distances = torch.as_tensor(
                self.distances, device=phenotypes.device, dtype=phenotypes.dtype
            )
            self._tensor_cache[key] = distances
        routes = torch.argsort(phenotypes, dim=1)
        next_cities = torch.roll(routes, -1, dims=1)
        return distances[routes, next_cities].sum(dim=1)


OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "target_match": TargetMatch,
    "rastrigin": Rastrigin,
    "tsp": TravelingSalesperson,
}


class TrackedFitness:
    """Fitness adapter that records best loss after every objective evaluation."""

    def __init__(self, objective: Objective):
        self.objective = objective
        self.evaluations = 0
        self.best_loss = float("inf")
        self.best_phenotype: np.ndarray | None = None
        self.trace: list[float] = []

    def _record(self, phenotypes: np.ndarray, losses: np.ndarray):
        for phenotype, loss in zip(phenotypes, losses):
            value = float(loss)
            self.evaluations += 1
            if value < self.best_loss:
                self.best_loss = value
                self.best_phenotype = np.asarray(phenotype, dtype=np.float32).copy()
            self.trace.append(self.best_loss)

    def __call__(self, phenotypes: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            losses = self.objective.loss_tensor(phenotypes)
        self._record(
            phenotypes.detach().cpu().numpy(),
            losses.detach().cpu().numpy(),
        )
        return -losses

    def evaluate_numpy(self, phenotypes: np.ndarray) -> np.ndarray:
        phenotypes = np.asarray(phenotypes, dtype=np.float32)
        losses = self.objective.loss_numpy(phenotypes)
        self._record(phenotypes, losses)
        return -losses

    def best_at(self, evaluation_budget: int) -> float:
        if len(self.trace) < evaluation_budget:
            raise RuntimeError(
                f"only {len(self.trace)} evaluations recorded; need {evaluation_budget}"
            )
        return self.trace[evaluation_budget - 1]


@dataclass(frozen=True)
class BenchmarkConfig:
    evaluation_budget: int = 5_000
    population: int = 64
    offspring: int = 64
    latent: int = 32
    hidden_size: int = 128
    num_layers: int = 2
    decoder_lr: float = 1e-3
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.12
    refine_every: int = 5
    refine_percent: float = 0.4
    decoder_es_candidates: int = 4
    decoder_es_percent: float = 0.25
    decoder_es_sigma: float = 1e-3
    offspring_only_mutation: bool = False

    def __post_init__(self):
        if self.evaluation_budget < self.population:
            raise ValueError("evaluation_budget must be at least population")


@dataclass(frozen=True)
class RunResult:
    objective: str
    metric: str
    strategy: str
    seed: int
    evaluation_budget: int
    metric_at_budget: float
    evaluations_run: int
    elapsed_seconds: float
    generations: int | None
    neural_device: str | None
    trainer_choices: tuple[str, ...] | None = None
    trainer_acceptance: tuple[bool, ...] | None = None
    trainer_rewards: tuple[float, ...] | None = None
    trainer_final_probabilities: dict[str, float] | None = None


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _require_mps():
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is required for latent neural benchmarks but is not available"
        )


def _warm_mps(objective: Objective, config: BenchmarkConfig):
    """Pay one-time MPS kernel setup before any neural strategy is timed."""
    decoder = MLPDecoder(
        input_length=config.latent,
        output_shape=(objective.dimension,),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        lr=config.decoder_lr,
        device="mps",
    )
    genes = torch.randn(config.population, config.latent, device="mps")
    output = decoder(genes)
    objective.loss_tensor(output.detach())
    decoder.opt.zero_grad()
    output.square().mean().backward()
    decoder.opt.step()
    torch.mps.synchronize()
    del decoder, genes, output
    torch.mps.empty_cache()


def _finish_result(
    objective: Objective,
    strategy: str,
    seed: int,
    config: BenchmarkConfig,
    tracker: TrackedFitness,
    started: float,
    generations: int | None = None,
    neural_device: str | None = None,
    trainer_choices: tuple[str, ...] | None = None,
    trainer_acceptance: tuple[bool, ...] | None = None,
    trainer_rewards: tuple[float, ...] | None = None,
    trainer_final_probabilities: dict[str, float] | None = None,
) -> RunResult:
    return RunResult(
        objective=objective.name,
        metric=objective.metric_name,
        strategy=strategy,
        seed=seed,
        evaluation_budget=config.evaluation_budget,
        metric_at_budget=tracker.best_at(config.evaluation_budget),
        evaluations_run=tracker.evaluations,
        elapsed_seconds=time.perf_counter() - started,
        generations=generations,
        neural_device=neural_device,
        trainer_choices=trainer_choices,
        trainer_acceptance=trainer_acceptance,
        trainer_rewards=trainer_rewards,
        trainer_final_probabilities=trainer_final_probabilities,
    )


def run_random_search(objective, seed, config):
    rng = np.random.default_rng(seed)
    tracker = TrackedFitness(objective)
    started = time.perf_counter()
    while tracker.evaluations < config.evaluation_budget:
        amount = min(config.offspring, config.evaluation_budget - tracker.evaluations)
        tracker.evaluate_numpy(rng.random((amount, objective.dimension), dtype=np.float32))
    return _finish_result(objective, "random_search", seed, config, tracker, started)


def _rank_probabilities(size: int, pressure: float = 2.0) -> np.ndarray:
    weights = np.exp(-pressure * np.arange(size, dtype=np.float64) / size)
    return weights / weights.sum()


def run_direct_ga(objective, seed, config):
    rng = np.random.default_rng(seed)
    tracker = TrackedFitness(objective)
    started = time.perf_counter()
    population = rng.random(
        (config.population, objective.dimension), dtype=np.float32
    )
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
        crossover_mask = rng.random((amount, objective.dimension)) < 0.5
        children = np.where(crossover_mask, first, second).astype(np.float32)
        mutation_mask = (
            rng.random((amount, objective.dimension)) < config.mutation_rate
        )
        empty = np.flatnonzero(~mutation_mask.any(axis=1))
        if len(empty):
            mutation_mask[empty, rng.integers(0, objective.dimension, len(empty))] = True
        noise = rng.normal(
            0, config.mutation_sigma, size=(amount, objective.dimension)
        ).astype(np.float32)
        children = np.clip(children + noise * mutation_mask, 0, 1)
        child_fitness = tracker.evaluate_numpy(children)
        population = np.concatenate([population, children])
        fitness = np.concatenate([fitness, child_fitness])
        survivors = np.argsort(-fitness)[: config.population]
        population, fitness = population[survivors], fitness[survivors]
        generations += 1

    return _finish_result(
        objective, "direct_ga", seed, config, tracker, started, generations
    )


def run_mu_plus_lambda_es(objective, seed, config):
    """A bounded direct (mu+lambda)-ES with intermediate recombination."""
    rng = np.random.default_rng(seed)
    tracker = TrackedFitness(objective)
    started = time.perf_counter()
    population = rng.random(
        (config.population, objective.dimension), dtype=np.float32
    )
    fitness = tracker.evaluate_numpy(population)
    sigma = config.mutation_sigma
    generations = 0

    while tracker.evaluations < config.evaluation_budget:
        amount = min(config.offspring, config.evaluation_budget - tracker.evaluations)
        order = np.argsort(-fitness)
        parents = population[order]
        parent_fitness = fitness[order]
        chosen = rng.integers(0, len(parents), size=(amount, 2))
        centers = (parents[chosen[:, 0]] + parents[chosen[:, 1]]) / 2
        children = np.clip(
            centers
            + rng.normal(0, sigma, size=(amount, objective.dimension)).astype(np.float32),
            0,
            1,
        )
        child_fitness = tracker.evaluate_numpy(children)
        reference = np.maximum(
            parent_fitness[chosen[:, 0]], parent_fitness[chosen[:, 1]]
        )
        success_rate = float(np.mean(child_fitness > reference))
        sigma = float(np.clip(sigma * np.exp((success_rate - 0.2) / 3), 1e-3, 0.5))
        population = np.concatenate([population, children])
        fitness = np.concatenate([fitness, child_fitness])
        survivors = np.argsort(-fitness)[: config.population]
        population, fitness = population[survivors], fitness[survivors]
        generations += 1

    return _finish_result(
        objective,
        "mu_plus_lambda_es",
        seed,
        config,
        tracker,
        started,
        generations,
    )


def run_differential_evolution(objective, seed, config):
    """SciPy DE/rand/1/bin, interrupted at the exact evaluation budget."""
    from scipy.optimize import differential_evolution

    tracker = TrackedFitness(objective)
    started = time.perf_counter()

    class BudgetReached(Exception):
        pass

    def scipy_objective(phenotype):
        loss = -tracker.evaluate_numpy(np.asarray(phenotype)[None])[0]
        if tracker.evaluations >= config.evaluation_budget:
            raise BudgetReached
        return float(loss)

    # SciPy defines population size as popsize * dimension.
    popsize = max(4, round(config.population / objective.dimension))
    try:
        differential_evolution(
            scipy_objective,
            bounds=[(0.0, 1.0)] * objective.dimension,
            strategy="best1bin",
            maxiter=config.evaluation_budget,
            popsize=popsize,
            tol=0,
            atol=0,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=seed,
            polish=False,
            updating="immediate",
            workers=1,
        )
    except BudgetReached:
        pass
    return _finish_result(
        objective,
        "differential_evolution",
        seed,
        config,
        tracker,
        started,
    )


def run_latent(objective, seed, config, strategy):
    _require_mps()
    _seed_everything(seed)
    tracker = TrackedFitness(objective)
    decoder_update = None
    refine_every = config.refine_every
    if strategy == "latent_fixed":
        refine_every = None
    elif strategy == "latent_decoder_es":
        refine_every = None
        decoder_update = EvolveDecoder(
            tracker,
            every=config.refine_every,
            n_candidates=config.decoder_es_candidates,
            percent=config.decoder_es_percent,
            sigma=config.decoder_es_sigma,
        )
    elif strategy != "latent_gradient":
        raise ValueError(strategy)

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
        refine_every=refine_every,
        refine_percent=config.refine_percent,
        pressure=1.8,
        scheme="linear",
        families=families,
        children=4,
        n_points=4,
        offspring_only_mutation=config.offspring_only_mutation,
        decoder_update=decoder_update,
    )
    parameter_devices = {parameter.device.type for parameter in evolver.decoder.parameters()}
    if parameter_devices != {"mps"}:
        raise RuntimeError(f"decoder parameters are not exclusively on MPS: {parameter_devices}")

    started = time.perf_counter()
    while tracker.evaluations < config.evaluation_budget:
        evolver.solve(1, verbose_every=0)
    torch.mps.synchronize()
    result = _finish_result(
        objective,
        strategy,
        seed,
        config,
        tracker,
        started,
        generations=evolver.env.generation,
        neural_device="mps",
    )
    torch.mps.empty_cache()
    return result


STRATEGIES: dict[str, Callable] = {
    "random_search": run_random_search,
    "direct_ga": run_direct_ga,
    "mu_plus_lambda_es": run_mu_plus_lambda_es,
    "differential_evolution": run_differential_evolution,
    "latent_fixed": lambda objective, seed, config: run_latent(
        objective, seed, config, "latent_fixed"
    ),
    "latent_gradient": lambda objective, seed, config: run_latent(
        objective, seed, config, "latent_gradient"
    ),
    "latent_decoder_es": lambda objective, seed, config: run_latent(
        objective, seed, config, "latent_decoder_es"
    ),
}


def summarize(results: Iterable[RunResult]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[RunResult]] = {}
    for result in results:
        groups.setdefault(
            (result.objective, result.metric, result.strategy), []
        ).append(result)
    summary = []
    for (objective, metric, strategy), runs in sorted(groups.items()):
        values = [run.metric_at_budget for run in runs]
        times = [run.elapsed_seconds for run in runs]
        summary.append(
            {
                "objective": objective,
                "metric": metric,
                "strategy": strategy,
                "seeds": len(runs),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "mean_seconds": statistics.fmean(times),
            }
        )
    return summary


def print_summary(summary: Iterable[dict]):
    print("\nmetric at the exact evaluation budget (lower is better)")
    print(f"{'objective':<14} {'strategy':<21} {'mean':>11} {'stdev':>11} {'sec':>8}")
    print("-" * 70)
    for row in summary:
        print(
            f"{row['objective']:<14} {row['strategy']:<21} "
            f"{row['mean']:>11.5g} {row['stdev']:>11.5g} "
            f"{row['mean_seconds']:>8.3f}"
        )


def run_suite(objective_names, strategy_names, seeds, config):
    has_neural = any(name.startswith("latent_") for name in strategy_names)
    if has_neural:
        _require_mps()
    results = []
    for objective_name in objective_names:
        if has_neural:
            _warm_mps(OBJECTIVES[objective_name](), config)
        for strategy_name in strategy_names:
            for seed in seeds:
                objective = OBJECTIVES[objective_name]()
                print(
                    f"run objective={objective_name:<12} strategy={strategy_name:<21} "
                    f"seed={seed} budget={config.evaluation_budget}"
                )
                result = STRATEGIES[strategy_name](objective, seed, config)
                print(
                    f"  {result.metric}={result.metric_at_budget:.6g} "
                    f"evals_run={result.evaluations_run} "
                    f"device={result.neural_device or 'numpy/cpu'}"
                )
                results.append(result)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES)
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = BenchmarkConfig(evaluation_budget=args.budget)
    results = run_suite(args.objectives, args.strategies, args.seeds, config)
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
