"""Scale TSP size under a fixed evaluation budget: Finch versus direct GA."""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .compare import (
    BenchmarkConfig,
    TravelingSalesperson,
    _require_mps,
    _warm_mps,
    run_direct_ga,
)
from .decoder_training import run_trainer


DEFAULT_SIZES = (8, 12, 16, 24, 32)
METHODS = ("direct_ga", "finch_guarded_non_rl")


@dataclass(frozen=True)
class ScalingRun:
    cities: int
    method: str
    seed: int
    evaluation_budget: int
    tour_length: float
    normalized_length: float
    evaluations_run: int
    elapsed_seconds: float
    generations: int | None
    neural_device: str | None


def held_karp_optimum(distances: np.ndarray) -> float:
    """Exact anchored TSP optimum, intended for reference instances <= 16."""
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix")
    cities = len(distances)
    if cities < 2:
        return 0.0
    if cities > 20:
        raise ValueError("exact Held-Karp reference is limited to 20 cities")

    # Only subsets containing city 0 are used. dp[mask, j] is the shortest
    # path from 0 through mask ending at j.
    states = 1 << cities
    dp = np.full((states, cities), np.inf, dtype=np.float64)
    dp[1, 0] = 0.0
    for mask in range(1, states, 2):
        remaining = mask & ~1
        while remaining:
            bit = remaining & -remaining
            city = bit.bit_length() - 1
            previous_mask = mask ^ bit
            predecessors = previous_mask & ~1
            if previous_mask == 1:
                dp[mask, city] = distances[0, city]
            else:
                best = np.inf
                while predecessors:
                    predecessor_bit = predecessors & -predecessors
                    predecessor = predecessor_bit.bit_length() - 1
                    best = min(
                        best,
                        dp[previous_mask, predecessor]
                        + distances[predecessor, city],
                    )
                    predecessors ^= predecessor_bit
                dp[mask, city] = best
            remaining ^= bit
    full = states - 1
    return float(
        min(dp[full, city] + distances[city, 0] for city in range(1, cities))
    )


def _convert(result, cities: int, method: str) -> ScalingRun:
    return ScalingRun(
        cities=cities,
        method=method,
        seed=result.seed,
        evaluation_budget=result.evaluation_budget,
        tour_length=result.metric_at_budget,
        normalized_length=result.metric_at_budget / np.sqrt(cities),
        evaluations_run=result.evaluations_run,
        elapsed_seconds=result.elapsed_seconds,
        generations=result.generations,
        neural_device=result.neural_device,
    )


def summarize(runs: list[ScalingRun], exact_optima: dict[int, float]):
    rows = []
    for cities in sorted({run.cities for run in runs}):
        for method in METHODS:
            selected = [
                run for run in runs
                if run.cities == cities and run.method == method
            ]
            values = [run.tour_length for run in selected]
            normalized = [run.normalized_length for run in selected]
            times = [run.elapsed_seconds for run in selected]
            optimum = exact_optima.get(cities)
            rows.append(
                {
                    "cities": cities,
                    "method": method,
                    "seeds": len(selected),
                    "mean_tour_length": statistics.fmean(values),
                    "stdev_tour_length": (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    ),
                    "mean_normalized_length": statistics.fmean(normalized),
                    "stdev_normalized_length": (
                        statistics.stdev(normalized) if len(normalized) > 1 else 0.0
                    ),
                    "mean_seconds": statistics.fmean(times),
                    "exact_optimum": optimum,
                    "mean_optimality_gap_percent": (
                        100 * (statistics.fmean(values) / optimum - 1)
                        if optimum is not None else None
                    ),
                }
            )
    comparisons = []
    for cities in sorted({run.cities for run in runs}):
        direct = next(
            row for row in rows
            if row["cities"] == cities and row["method"] == "direct_ga"
        )
        finch = next(
            row for row in rows
            if row["cities"] == cities
            and row["method"] == "finch_guarded_non_rl"
        )
        comparisons.append(
            {
                "cities": cities,
                "finch_vs_direct_gap_percent": 100
                * (finch["mean_tour_length"] / direct["mean_tour_length"] - 1),
                "finch_vs_direct_time_ratio": (
                    finch["mean_seconds"] / direct["mean_seconds"]
                ),
            }
        )
    return rows, comparisons


def run_suite(sizes, seeds, config):
    _require_mps()
    runs = []
    exact_optima = {}
    for cities in sizes:
        reference = TravelingSalesperson(dimension=cities)
        _warm_mps(reference, config)
        if cities <= 16:
            exact_optima[cities] = held_karp_optimum(reference.distances)
        for seed in seeds:
            direct = run_direct_ga(
                TravelingSalesperson(dimension=cities), seed, config
            )
            runs.append(_convert(direct, cities, "direct_ga"))
            print(
                f"cities={cities:<3} method={'direct_ga':<24} seed={seed} "
                f"length={direct.metric_at_budget:.6g}"
            )

            finch = run_trainer(
                TravelingSalesperson(dimension=cities),
                seed,
                config,
                "guarded_random_non_rl",
            )
            runs.append(_convert(finch, cities, "finch_guarded_non_rl"))
            print(
                f"cities={cities:<3} method={'finch_guarded_non_rl':<24} "
                f"seed={seed} length={finch.metric_at_budget:.6g} device=mps"
            )
    return runs, exact_optima


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument(
        "--offspring-only-mutation",
        action="store_true",
        help="preserve incumbent parents and mutate only newly bred children",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(set(args.sizes)) != len(args.sizes):
        raise ValueError("sizes must be unique")
    if any(size < 2 for size in args.sizes):
        raise ValueError("every TSP size must be at least 2")
    config = BenchmarkConfig(
        evaluation_budget=args.budget,
        offspring_only_mutation=args.offspring_only_mutation,
    )
    runs, exact_optima = run_suite(args.sizes, args.seeds, config)
    summary, comparisons = summarize(runs, exact_optima)
    print("\nmean tour length (lower is better)")
    for row in summary:
        print(
            f"cities={row['cities']:<3} method={row['method']:<24} "
            f"mean={row['mean_tour_length']:.6g} "
            f"sd={row['stdev_tour_length']:.5g} "
            f"sec={row['mean_seconds']:.4f}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "study": "tsp_scaling",
            "config": asdict(config),
            "sizes": list(args.sizes),
            "seeds": list(args.seeds),
            "instance_seed": 2026,
            "torch_version": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "exact_optima": {str(key): value for key, value in exact_optima.items()},
            "runs": [asdict(run) for run in runs],
            "summary": summary,
            "comparisons": comparisons,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
