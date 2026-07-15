"""Ablate the original GeneSpace TSP recipe under an exact evaluation budget.

Frozen variants isolate search-space geometry, decoder architecture, and
population settings. Rare-training variants then test guarded decoder updates
without returning to the rapidly moving five-generation schedule. Every neural
decoder is required to reside on Apple's MPS backend.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from latentspace import Evolver

from .compare import (
    BenchmarkConfig,
    RunResult,
    TrackedFitness,
    TravelingSalesperson,
    _finish_result,
    _require_mps,
    _seed_everything,
    _warm_mps,
    run_direct_ga,
)
from .decoder_training import make_trainer


@dataclass(frozen=True)
class LatentVariant:
    latent: int
    population: int
    hidden_size: int
    num_layers: int
    binary: bool
    mutation_rate: float
    mutation_sigma: float
    pressure: float
    scheme: str
    n_points: int
    offspring: int = 64
    decoder_lr: float = 1e-5
    refine_every: int | None = None
    trainer_name: str | None = None
    offspring_only_mutation: bool = True


VARIANTS = {
    "compact_float": LatentVariant(
        latent=32,
        population=64,
        hidden_size=128,
        num_layers=2,
        binary=False,
        mutation_rate=0.1,
        mutation_sigma=0.12,
        pressure=1.8,
        scheme="linear",
        n_points=4,
    ),
    "binary250_w128": LatentVariant(
        latent=250,
        population=64,
        hidden_size=128,
        num_layers=1,
        binary=True,
        mutation_rate=0.1,
        mutation_sigma=0.0,
        pressure=1.8,
        scheme="linear",
        n_points=8,
    ),
    "binary250_w512": LatentVariant(
        latent=250,
        population=64,
        hidden_size=512,
        num_layers=1,
        binary=True,
        mutation_rate=0.1,
        mutation_sigma=0.0,
        pressure=1.8,
        scheme="linear",
        n_points=8,
    ),
    "binary250_w2000": LatentVariant(
        latent=250,
        population=64,
        hidden_size=2000,
        num_layers=1,
        binary=True,
        mutation_rate=0.1,
        mutation_sigma=0.0,
        pressure=1.8,
        scheme="linear",
        n_points=8,
    ),
    "genespace_scale1": LatentVariant(
        latent=250,
        population=200,
        hidden_size=2000,
        num_layers=1,
        binary=True,
        mutation_rate=0.1,
        mutation_sigma=0.0,
        pressure=20.0,
        scheme="exp",
        n_points=8,
    ),
    "compact_guarded20": LatentVariant(
        latent=32,
        population=64,
        hidden_size=128,
        num_layers=2,
        binary=False,
        mutation_rate=0.1,
        mutation_sigma=0.12,
        pressure=1.8,
        scheme="linear",
        n_points=4,
        decoder_lr=1e-3,
        refine_every=20,
        trainer_name="guarded_random_non_rl",
    ),
    "binary250_w2000_guarded20": LatentVariant(
        latent=250,
        population=64,
        hidden_size=2000,
        num_layers=1,
        binary=True,
        mutation_rate=0.1,
        mutation_sigma=0.0,
        pressure=1.8,
        scheme="linear",
        n_points=8,
        refine_every=20,
        trainer_name="guarded_random_non_rl",
    ),
}


def run_latent_variant(
    objective: TravelingSalesperson,
    seed: int,
    evaluation_budget: int,
    name: str,
    variant: LatentVariant,
) -> RunResult:
    _require_mps()
    _seed_everything(seed)
    tracker = TrackedFitness(objective)
    families = max(1, variant.offspring // 4)
    config = BenchmarkConfig(
        evaluation_budget=evaluation_budget,
        population=variant.population,
        offspring=variant.offspring,
        latent=variant.latent,
        hidden_size=variant.hidden_size,
        num_layers=variant.num_layers,
        decoder_lr=variant.decoder_lr,
        mutation_rate=variant.mutation_rate,
        mutation_sigma=variant.mutation_sigma,
        refine_every=variant.refine_every or 5,
    )
    trainer = (
        make_trainer(variant.trainer_name, config, seed=seed)
        if variant.trainer_name is not None
        else None
    )
    evolver = Evolver(
        tracker,
        output_shape=(objective.dimension,),
        device="mps",
        latent=variant.latent,
        population=variant.population,
        hidden_size=variant.hidden_size,
        num_layers=variant.num_layers,
        lr=variant.decoder_lr,
        binary=variant.binary,
        mutation_rate=variant.mutation_rate,
        mutation_sigma=variant.mutation_sigma,
        refine_every=variant.refine_every,
        pressure=variant.pressure,
        scheme=variant.scheme,
        families=families,
        children=4,
        n_points=variant.n_points,
        offspring_only_mutation=variant.offspring_only_mutation,
        trainer=trainer,
    )
    parameter_devices = {
        parameter.device.type for parameter in evolver.decoder.parameters()
    }
    if parameter_devices != {"mps"}:
        raise RuntimeError(
            f"decoder parameters are not exclusively on MPS: {parameter_devices}"
        )

    started = time.perf_counter()
    while tracker.evaluations < evaluation_budget:
        evolver.solve(1, verbose_every=0)
    torch.mps.synchronize()
    result = _finish_result(
        objective,
        name,
        seed,
        config,
        tracker,
        started,
        generations=evolver.env.generation,
        neural_device="mps",
    )
    torch.mps.empty_cache()
    return result


def summarize(results: list[RunResult]) -> list[dict]:
    rows = []
    for cities in sorted({result.objective for result in results}):
        # Objective names are replaced with tsp_<cities> in run_suite.
        selected_city = [result for result in results if result.objective == cities]
        for strategy in sorted({result.strategy for result in selected_city}):
            selected = [
                result for result in selected_city if result.strategy == strategy
            ]
            values = [result.metric_at_budget for result in selected]
            rows.append(
                {
                    "cities": int(cities.removeprefix("tsp_")),
                    "strategy": strategy,
                    "seeds": len(selected),
                    "mean_tour_length": statistics.fmean(values),
                    "stdev_tour_length": (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    ),
                    "mean_seconds": statistics.fmean(
                        result.elapsed_seconds for result in selected
                    ),
                    "mean_generations": statistics.fmean(
                        result.generations for result in selected
                        if result.generations is not None
                    ),
                }
            )
    return sorted(rows, key=lambda row: (row["cities"], row["mean_tour_length"]))


def run_suite(sizes, seeds, evaluation_budget, variant_names):
    _require_mps()
    results = []
    direct_config = BenchmarkConfig(evaluation_budget=evaluation_budget)
    for cities in sizes:
        _warm_mps(TravelingSalesperson(dimension=cities), direct_config)
        for seed in seeds:
            objective = TravelingSalesperson(dimension=cities)
            direct = run_direct_ga(objective, seed, direct_config)
            direct = RunResult(
                **{
                    **asdict(direct),
                    "objective": f"tsp_{cities}",
                }
            )
            results.append(direct)
            print(
                f"cities={cities:<3} variant={'direct_ga':<20} seed={seed} "
                f"length={direct.metric_at_budget:.6g}"
            )
            for name in variant_names:
                objective = TravelingSalesperson(dimension=cities)
                result = run_latent_variant(
                    objective, seed, evaluation_budget, name, VARIANTS[name]
                )
                result = RunResult(
                    **{
                        **asdict(result),
                        "objective": f"tsp_{cities}",
                    }
                )
                results.append(result)
                print(
                    f"cities={cities:<3} variant={name:<20} seed={seed} "
                    f"length={result.metric_at_budget:.6g} device=mps"
                )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 12, 16, 24, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_suite(args.sizes, args.seeds, args.budget, args.variants)
    summary = summarize(results)
    print("\nmean tour length (lower is better)")
    for row in summary:
        print(
            f"cities={row['cities']:<3} variant={row['strategy']:<20} "
            f"mean={row['mean_tour_length']:.6g} "
            f"sd={row['stdev_tour_length']:.5g} "
            f"gens={row['mean_generations']:.1f}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "study": "tsp_genespace_ablation",
            "evaluation_budget": args.budget,
            "sizes": list(args.sizes),
            "seeds": list(args.seeds),
            "instance_seed": 2026,
            "variants": {
                name: asdict(VARIANTS[name]) for name in args.variants
            },
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
