"""TSP decoder-objective, learning-rate, and optimizer ablation on MPS.

The latent GA uses the strongest current 24-city search configuration. Every
fitness probe used for guarded or backtracking updates is counted against the
same objective-evaluation budget as evolutionary search.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from latentspace import (
    BacktrackingTrainer,
    DistillationTrainer,
    Evolver,
    MLPDecoder,
    PermutationTrainer,
    TrainMode,
)

from .compare import (
    BenchmarkConfig,
    TrackedFitness,
    TravelingSalesperson,
    _require_mps,
    _seed_everything,
    _warm_mps,
    run_direct_ga,
)


OBJECTIVES = (
    "raw_mse",
    "permutation",
    "permutation_anchor",
    "permutation_anchor_backtrack",
)
LEARNING_RATES = (3e-5, 1e-4, 3e-4, 1e-3)
OPTIMIZERS = ("adam", "adam_beta2_095", "sgd_momentum")


@dataclass(frozen=True)
class StudyConfig:
    cities: int = 24
    evaluation_budget: int = 50_000
    population: int = 64
    offspring: int = 64
    latent: int = 32
    hidden_size: int = 128
    num_layers: int = 1
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.25
    refine_every: int = 10
    refine_percent: float = 0.4
    n_points: int = 8
    pressure: float = 1.8
    instance_seed: int = 2026


@dataclass(frozen=True)
class Variant:
    objective: str
    learning_rate: float
    optimizer: str = "adam"

    @property
    def name(self):
        return (
            f"{self.objective}__{self.optimizer}__"
            f"lr{self.learning_rate:g}"
        )


@dataclass(frozen=True)
class StudyRun:
    variant: str
    objective: str
    optimizer: str
    learning_rate: float
    seed: int
    tour_length: float
    evaluation_budget: int
    evaluations_run: int
    elapsed_seconds: float
    generations: int
    decoder_updates: int
    neural_device: str | None
    backtracking_probe_evaluations: int = 0
    backtracking_factors: tuple[float, ...] = ()


def make_optimizer(decoder: MLPDecoder, name: str, learning_rate: float):
    if name == "adam":
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    elif name == "adam_beta2_095":
        optimizer = torch.optim.Adam(
            decoder.parameters(), lr=learning_rate, betas=(0.9, 0.95)
        )
    elif name == "sgd_momentum":
        optimizer = torch.optim.SGD(
            decoder.parameters(), lr=learning_rate, momentum=0.9
        )
    else:
        raise ValueError(name)
    decoder.optimizer = optimizer
    decoder.opt = optimizer


def make_trainer(objective: str, percent: float):
    if objective == "raw_mse":
        return DistillationTrainer(
            mode=TrainMode.GOOD_TO_BEST,
            percent=percent,
        )
    if objective == "permutation":
        return PermutationTrainer(
            percent=percent,
            temperature=0.1,
        )
    if objective in {
        "permutation_anchor",
        "permutation_anchor_backtrack",
    }:
        trainer = PermutationTrainer(
            percent=percent,
            temperature=0.1,
            anchor_weight=1.0,
            anchor_percent=0.1,
        )
        if objective.endswith("backtrack"):
            return BacktrackingTrainer(
                trainer,
                probe_percent=0.25,
                factors=(1.0, 0.5, 0.25, 0.125),
            )
        return trainer
    raise ValueError(objective)


def run_variant(
    variant: Variant,
    seed: int,
    config: StudyConfig,
) -> StudyRun:
    _require_mps()
    _seed_everything(seed)
    objective = TravelingSalesperson(config.cities, config.instance_seed)
    tracker = TrackedFitness(objective)
    decoder = MLPDecoder(
        input_length=config.latent,
        output_shape=(config.cities,),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        lr=variant.learning_rate,
        device="mps",
    )
    make_optimizer(decoder, variant.optimizer, variant.learning_rate)
    trainer = make_trainer(variant.objective, config.refine_percent)
    evolver = Evolver(
        tracker,
        output_shape=(config.cities,),
        device="mps",
        latent=config.latent,
        population=config.population,
        binary=False,
        mutation_rate=config.mutation_rate,
        mutation_sigma=config.mutation_sigma,
        refine_every=config.refine_every,
        pressure=config.pressure,
        scheme="linear",
        families=max(1, config.offspring // 4),
        children=4,
        n_points=config.n_points,
        offspring_only_mutation=False,
        decoder=decoder,
        trainer=trainer,
    )
    devices = {parameter.device.type for parameter in decoder.parameters()}
    if devices != {"mps"}:
        raise RuntimeError(
            f"decoder parameters are not exclusively on MPS: {devices}"
        )

    started = time.perf_counter()
    while tracker.evaluations < config.evaluation_budget:
        evolver.solve(1, verbose_every=0)
    torch.mps.synchronize()
    backtracking = (
        trainer if isinstance(trainer, BacktrackingTrainer) else None
    )
    result = StudyRun(
        variant=variant.name,
        objective=variant.objective,
        optimizer=variant.optimizer,
        learning_rate=variant.learning_rate,
        seed=seed,
        tour_length=tracker.best_at(config.evaluation_budget),
        evaluation_budget=config.evaluation_budget,
        evaluations_run=tracker.evaluations,
        elapsed_seconds=time.perf_counter() - started,
        generations=evolver.env.generation,
        decoder_updates=decoder.version,
        neural_device="mps",
        backtracking_probe_evaluations=(
            backtracking.probe_evaluations if backtracking else 0
        ),
        backtracking_factors=(
            tuple(backtracking.factor_history) if backtracking else ()
        ),
    )
    torch.mps.empty_cache()
    return result


def run_frozen(seed: int, config: StudyConfig) -> StudyRun:
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
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        lr=1e-5,
        binary=False,
        mutation_rate=config.mutation_rate,
        mutation_sigma=config.mutation_sigma,
        refine_every=None,
        pressure=config.pressure,
        scheme="linear",
        families=max(1, config.offspring // 4),
        children=4,
        n_points=config.n_points,
        offspring_only_mutation=False,
    )
    started = time.perf_counter()
    while tracker.evaluations < config.evaluation_budget:
        evolver.solve(1, verbose_every=0)
    torch.mps.synchronize()
    result = StudyRun(
        variant="frozen",
        objective="frozen",
        optimizer="none",
        learning_rate=0.0,
        seed=seed,
        tour_length=tracker.best_at(config.evaluation_budget),
        evaluation_budget=config.evaluation_budget,
        evaluations_run=tracker.evaluations,
        elapsed_seconds=time.perf_counter() - started,
        generations=evolver.env.generation,
        decoder_updates=0,
        neural_device="mps",
    )
    torch.mps.empty_cache()
    return result


def run_direct(seed: int, config: StudyConfig) -> StudyRun:
    direct_config = BenchmarkConfig(
        evaluation_budget=config.evaluation_budget,
        population=config.population,
        offspring=config.offspring,
        mutation_rate=0.1,
        mutation_sigma=0.12,
    )
    result = run_direct_ga(
        TravelingSalesperson(config.cities, config.instance_seed),
        seed,
        direct_config,
    )
    return StudyRun(
        variant="direct_ga",
        objective="direct_ga",
        optimizer="none",
        learning_rate=0.0,
        seed=seed,
        tour_length=result.metric_at_budget,
        evaluation_budget=config.evaluation_budget,
        evaluations_run=result.evaluations_run,
        elapsed_seconds=result.elapsed_seconds,
        generations=result.generations or 0,
        decoder_updates=0,
        neural_device=None,
    )


def summarize(runs: list[StudyRun]):
    rows = []
    for variant in sorted({run.variant for run in runs}):
        selected = [run for run in runs if run.variant == variant]
        values = [run.tour_length for run in selected]
        factors = Counter(
            factor
            for run in selected
            for factor in run.backtracking_factors
        )
        rows.append(
            {
                "variant": variant,
                "objective": selected[0].objective,
                "optimizer": selected[0].optimizer,
                "learning_rate": selected[0].learning_rate,
                "seeds": len(selected),
                "mean_tour_length": statistics.fmean(values),
                "stdev_tour_length": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
                "min_tour_length": min(values),
                "max_tour_length": max(values),
                "mean_seconds": statistics.fmean(
                    run.elapsed_seconds for run in selected
                ),
                "mean_generations": statistics.fmean(
                    run.generations for run in selected
                ),
                "mean_decoder_updates": statistics.fmean(
                    run.decoder_updates for run in selected
                ),
                "mean_backtracking_probe_evaluations": statistics.fmean(
                    run.backtracking_probe_evaluations for run in selected
                ),
                "backtracking_factor_counts": {
                    str(factor): count for factor, count in sorted(factors.items())
                },
            }
        )
    return sorted(rows, key=lambda row: row["mean_tour_length"])


def print_run(run: StudyRun):
    print(
        f"variant={run.variant:<58} seed={run.seed} "
        f"length={run.tour_length:.6f} sec={run.elapsed_seconds:.3f} "
        f"updates={run.decoder_updates}",
        flush=True,
    )


def run_objective_stage(config: StudyConfig, seeds, learning_rates):
    runs = []
    for seed in seeds:
        direct = run_direct(seed, config)
        runs.append(direct)
        print_run(direct)
        frozen = run_frozen(seed, config)
        runs.append(frozen)
        print_run(frozen)
    for objective in OBJECTIVES:
        for learning_rate in learning_rates:
            variant = Variant(objective, learning_rate, "adam")
            for seed in seeds:
                result = run_variant(variant, seed, config)
                runs.append(result)
                print_run(result)
    return runs


def winning_objective(runs: list[StudyRun]) -> str:
    eligible = [
        row for row in summarize(runs)
        if row["objective"] in OBJECTIVES
    ]
    if not eligible:
        raise ValueError("objective-stage results are required")
    return eligible[0]["objective"]


def run_optimizer_stage(
    objective: str,
    config: StudyConfig,
    seeds,
    learning_rates,
):
    runs = []
    for optimizer in OPTIMIZERS[1:]:
        for learning_rate in learning_rates:
            variant = Variant(objective, learning_rate, optimizer)
            for seed in seeds:
                result = run_variant(variant, seed, config)
                runs.append(result)
                print_run(result)
    return runs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("pilot", "objectives", "confirm", "all"),
        default="all",
    )
    parser.add_argument("--cities", type=int, default=24)
    parser.add_argument("--budget", type=int, default=50_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=list(LEARNING_RATES),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/mps_tsp_training_objectives_50000.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = StudyConfig(
        cities=args.cities,
        evaluation_budget=args.budget,
    )
    _require_mps()
    _warm_mps(
        TravelingSalesperson(config.cities, config.instance_seed),
        BenchmarkConfig(
            evaluation_budget=config.evaluation_budget,
            population=config.population,
            offspring=config.offspring,
            latent=config.latent,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        ),
    )
    started = time.perf_counter()
    if args.phase == "pilot":
        variants = [
            Variant("permutation_anchor_backtrack", 3e-4, "adam")
        ]
        runs = [
            run_variant(variant, seed, config)
            for variant in variants
            for seed in args.seeds
        ]
        for run in runs:
            print_run(run)
        winner = None
    elif args.phase == "confirm":
        finalists = [
            Variant("raw_mse", 3e-4, "sgd_momentum"),
            Variant("raw_mse", 1e-4, "adam"),
            Variant("permutation_anchor_backtrack", 3e-5, "adam"),
        ]
        runs = []
        for seed in args.seeds:
            direct = run_direct(seed, config)
            runs.append(direct)
            print_run(direct)
            frozen = run_frozen(seed, config)
            runs.append(frozen)
            print_run(frozen)
        for variant in finalists:
            for seed in args.seeds:
                result = run_variant(variant, seed, config)
                runs.append(result)
                print_run(result)
        winner = None
    else:
        runs = run_objective_stage(
            config, args.seeds, args.learning_rates
        )
        winner = winning_objective(runs)
        print(f"\nwinning objective stage: {winner}\n", flush=True)
        if args.phase == "all":
            runs.extend(
                run_optimizer_stage(
                    winner,
                    config,
                    args.seeds,
                    args.learning_rates,
                )
            )

    summary = summarize(runs)
    payload = {
        "study": "tsp_training_objectives",
        "config": asdict(config),
        "phase": args.phase,
        "learning_rates": list(args.learning_rates),
        "seeds": list(args.seeds),
        "winning_objective": winner,
        "torch_version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "elapsed_seconds": time.perf_counter() - started,
        "runs": [asdict(run) for run in runs],
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("\nmean tour length (lower is better)")
    for row in summary:
        print(
            f"{row['variant']:<58} "
            f"{row['mean_tour_length']:.6f} ± "
            f"{row['stdev_tour_length']:.6f}"
        )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
