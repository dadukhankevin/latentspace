"""Long-run, evaluation-matched TSP progress curves.

This compares the project's generic direct GA with its strongest current
24-city latent configuration: a frozen, shallow MLP decoder and float genes.
The neural decoder is required to run on Apple's MPS backend.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from latentspace import Evolver

from .compare import (
    BenchmarkConfig,
    TrackedFitness,
    TravelingSalesperson,
    _rank_probabilities,
    _require_mps,
    _seed_everything,
    _warm_mps,
)


@dataclass(frozen=True)
class LongRunConfig:
    cities: int = 24
    evaluation_budget: int = 100_000
    population: int = 64
    offspring: int = 64
    checkpoint_step: int = 250
    instance_seed: int = 2026
    direct_mutation_rate: float = 0.1
    direct_mutation_sigma: float = 0.12
    latent: int = 32
    latent_hidden_size: int = 128
    latent_num_layers: int = 1
    latent_mutation_rate: float = 0.1
    latent_mutation_sigma: float = 0.25
    latent_n_points: int = 8
    latent_pressure: float = 1.8

    def __post_init__(self):
        if self.cities < 3:
            raise ValueError("cities must be at least 3")
        if self.evaluation_budget < self.population:
            raise ValueError("evaluation_budget must be at least population")
        if self.checkpoint_step < 1:
            raise ValueError("checkpoint_step must be at least 1")


@dataclass(frozen=True)
class ProgressRun:
    method: str
    seed: int
    evaluations_run: int
    generations: int
    elapsed_seconds: float
    neural_device: str | None
    evaluations: list[int]
    best_tour_length: list[float]


def checkpoints(config: LongRunConfig) -> list[int]:
    values = list(
        range(config.checkpoint_step, config.evaluation_budget + 1, config.checkpoint_step)
    )
    values.extend([config.population, config.evaluation_budget])
    return sorted({value for value in values if value <= config.evaluation_budget})


def sample_trace(tracker: TrackedFitness, points: list[int]) -> list[float]:
    return [float(tracker.best_at(point)) for point in points]


def run_direct(seed: int, config: LongRunConfig, points: list[int]) -> ProgressRun:
    """Run the same rank-selected, uniform-crossover GA as compare.py."""
    rng = np.random.default_rng(seed)
    objective = TravelingSalesperson(config.cities, config.instance_seed)
    tracker = TrackedFitness(objective)
    started = time.perf_counter()
    population = rng.random((config.population, config.cities), dtype=np.float32)
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
        first = ranked[parent_indices[:, 0]]
        second = ranked[parent_indices[:, 1]]
        crossover_mask = rng.random((amount, config.cities)) < 0.5
        children = np.where(crossover_mask, first, second).astype(np.float32)
        mutation_mask = (
            rng.random((amount, config.cities)) < config.direct_mutation_rate
        )
        empty = np.flatnonzero(~mutation_mask.any(axis=1))
        if len(empty):
            mutation_mask[empty, rng.integers(0, config.cities, len(empty))] = True
        noise = rng.normal(
            0, config.direct_mutation_sigma, size=(amount, config.cities)
        ).astype(np.float32)
        children = np.clip(children + noise * mutation_mask, 0, 1)
        child_fitness = tracker.evaluate_numpy(children)
        population = np.concatenate([population, children])
        fitness = np.concatenate([fitness, child_fitness])
        survivors = np.argsort(-fitness)[: config.population]
        population = population[survivors]
        fitness = fitness[survivors]
        generations += 1

    return ProgressRun(
        method="direct_ga",
        seed=seed,
        evaluations_run=tracker.evaluations,
        generations=generations,
        elapsed_seconds=time.perf_counter() - started,
        neural_device=None,
        evaluations=points,
        best_tour_length=sample_trace(tracker, points),
    )


def run_latent(seed: int, config: LongRunConfig, points: list[int]) -> ProgressRun:
    """Run the best current 24-city latent recipe with a frozen MLP decoder."""
    _require_mps()
    _seed_everything(seed)
    objective = TravelingSalesperson(config.cities, config.instance_seed)
    tracker = TrackedFitness(objective)
    evolver = Evolver(
        tracker,
        output_shape=(config.cities,),
        device="mps",
        latent=config.latent,
        population=config.population,
        hidden_size=config.latent_hidden_size,
        num_layers=config.latent_num_layers,
        lr=1e-5,
        binary=False,
        mutation_rate=config.latent_mutation_rate,
        mutation_sigma=config.latent_mutation_sigma,
        refine_every=None,
        pressure=config.latent_pressure,
        scheme="linear",
        families=max(1, config.offspring // 4),
        children=4,
        n_points=config.latent_n_points,
        offspring_only_mutation=False,
    )
    devices = {parameter.device.type for parameter in evolver.decoder.parameters()}
    if devices != {"mps"}:
        raise RuntimeError(f"decoder parameters are not exclusively on MPS: {devices}")

    started = time.perf_counter()
    while tracker.evaluations < config.evaluation_budget:
        evolver.solve(1, verbose_every=0)
    torch.mps.synchronize()
    run = ProgressRun(
        method="latent_ga_fixed_mlp",
        seed=seed,
        evaluations_run=tracker.evaluations,
        generations=evolver.env.generation,
        elapsed_seconds=time.perf_counter() - started,
        neural_device="mps",
        evaluations=points,
        best_tour_length=sample_trace(tracker, points),
    )
    torch.mps.empty_cache()
    return run


def aggregate(runs: list[ProgressRun]) -> dict[str, list[dict[str, float | int]]]:
    result: dict[str, list[dict[str, float | int]]] = {}
    methods = sorted({run.method for run in runs})
    for method in methods:
        selected = [run for run in runs if run.method == method]
        rows = []
        for index, evaluation in enumerate(selected[0].evaluations):
            values = [run.best_tour_length[index] for run in selected]
            rows.append(
                {
                    "evaluation": evaluation,
                    "mean": statistics.fmean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
            )
        result[method] = rows
    return result


def milestone_summary(
    aggregate_rows: dict[str, list[dict[str, float | int]]],
    milestones: list[int],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for method, rows in aggregate_rows.items():
        by_evaluation = {int(row["evaluation"]): float(row["mean"]) for row in rows}
        summary[method] = {
            str(milestone): by_evaluation[milestone] for milestone in milestones
        }
        if len(milestones) >= 2:
            start, end = milestones[-2:]
            improvement = by_evaluation[start] - by_evaluation[end]
            summary[method][f"improvement_{start}_to_{end}"] = improvement
            summary[method][f"slope_per_1000_{start}_to_{end}"] = (
                -improvement / ((end - start) / 1000)
            )
    return summary


def run_suite(config: LongRunConfig, seeds: list[int]) -> dict:
    _require_mps()
    points = checkpoints(config)
    warm_config = BenchmarkConfig(
        evaluation_budget=config.evaluation_budget,
        population=config.population,
        offspring=config.offspring,
        latent=config.latent,
        hidden_size=config.latent_hidden_size,
        num_layers=config.latent_num_layers,
    )
    _warm_mps(
        TravelingSalesperson(config.cities, config.instance_seed), warm_config
    )

    runs = []
    for seed in seeds:
        direct = run_direct(seed, config, points)
        runs.append(direct)
        print(
            f"method={direct.method:<20} seed={seed} "
            f"length={direct.best_tour_length[-1]:.6f} "
            f"seconds={direct.elapsed_seconds:.3f}"
        )
        latent = run_latent(seed, config, points)
        runs.append(latent)
        print(
            f"method={latent.method:<20} seed={seed} "
            f"length={latent.best_tour_length[-1]:.6f} "
            f"seconds={latent.elapsed_seconds:.3f} device=mps"
        )

    aggregate_rows = aggregate(runs)
    milestones = sorted(
        {
            value
            for value in [5_000, 10_000, 25_000, 50_000, config.evaluation_budget]
            if value <= config.evaluation_budget and value in points
        }
    )
    return {
        "study": "tsp_long_run_progress",
        "comparison_axis": "objective_evaluations",
        "metric": "best_tour_length_so_far",
        "lower_is_better": True,
        "config": asdict(config),
        "seeds": seeds,
        "torch_version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "methods": {
            "direct_ga": {
                "representation": "24 random keys",
                "selection": "rank",
                "crossover": "uniform",
                "survivors": "mu_plus_lambda",
            },
            "latent_ga_fixed_mlp": {
                "representation": "32 float latent genes",
                "decoder": "32-128-24 MLP with LeakyReLU; frozen",
                "selection": "linear rank",
                "crossover": "8-point",
                "neural_device": "mps",
            },
        },
        "runs": [asdict(run) for run in runs],
        "aggregate": aggregate_rows,
        "milestones": milestone_summary(aggregate_rows, milestones),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", type=int, default=24)
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--checkpoint-step", type=int, default=250)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/mps_tsp_long_run_100000.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = LongRunConfig(
        cities=args.cities,
        evaluation_budget=args.budget,
        checkpoint_step=args.checkpoint_step,
    )
    payload = run_suite(config, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.output}")
    print(json.dumps(payload["milestones"], indent=2))


if __name__ == "__main__":
    main()
