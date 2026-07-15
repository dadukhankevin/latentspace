"""Controlled TSP ablation of latent encoding and GeneSpace operator schedules.

The study separates representation (float versus binary), mutation strength,
crossover granularity, and generation schedule. Neural decoders are frozen and
must run on Apple's MPS backend. A deliberately faithful legacy-alias variant is
included only to measure GeneSpace's original in-place/reference behavior.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from latentspace import Evolver, GenePool, LatentIndividual, MLPDecoder
from latentspace.core import Environment, Layer
from latentspace.layers import Cap, Crossover, DecodeAndEvaluate, Populate, Sort
from latentspace.selection import RankSelection

from .compare import (
    BenchmarkConfig,
    TrackedFitness,
    TravelingSalesperson,
    _require_mps,
    _seed_everything,
    _warm_mps,
    run_direct_ga,
)


@dataclass(frozen=True)
class OperatorVariant:
    latent: int = 32
    binary: bool = False
    hidden_size: int = 128
    num_layers: int = 1
    population: int = 64
    offspring: int = 64
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.12
    n_points: int = 8
    pressure: float = 1.8
    scheme: str = "linear"
    operator_schedule: str = "single_stage"
    offspring_only_mutation: bool = False
    mutation_with_replacement: bool = True
    ensure_mutation: bool = True
    legacy_alias: bool = False


F32 = OperatorVariant()
B32 = replace(F32, binary=True, mutation_sigma=0.0)
F250 = replace(F32, latent=250)
B250 = replace(B32, latent=250)

VARIANTS = {
    # Matched float/binary representations and mutation-strength sweeps.
    "float32_s05_all8": replace(F32, mutation_sigma=0.05),
    "float32_s12_all8": F32,
    "float32_s25_all8": replace(F32, mutation_sigma=0.25),
    "binary32_r02_all8": replace(B32, mutation_rate=0.02),
    "binary32_r05_all8": replace(B32, mutation_rate=0.05),
    "binary32_r10_all8": B32,
    "float250_s05_all8": replace(F250, mutation_sigma=0.05),
    "float250_s12_all8": F250,
    "float250_s25_all8": replace(F250, mutation_sigma=0.25),
    "binary250_r02_all8": replace(B250, mutation_rate=0.02),
    "binary250_r05_all8": replace(B250, mutation_rate=0.05),
    "binary250_r10_all8": B250,
    # Connect the controlled shallow study to the previous compact benchmark.
    "compact_current": replace(F32, num_layers=2, n_points=4),
    # One-stage all-population versus offspring-only mutation and crossover count.
    "float32_s12_all4": replace(F32, n_points=4),
    "binary250_r05_all4": replace(B250, mutation_rate=0.05, n_points=4),
    "float32_s12_offspring8": replace(F32, offspring_only_mutation=True),
    "binary250_r05_offspring8": replace(
        B250, mutation_rate=0.05, offspring_only_mutation=True
    ),
    # Clean GeneSpace-style two-stage schedules.
    "float32_s12_two_wor8": replace(
        F32,
        operator_schedule="two_stage",
        mutation_with_replacement=False,
    ),
    "float32_s12_two_wr8": replace(F32, operator_schedule="two_stage"),
    "float32_s12_two_wr8_noforce": replace(
        F32,
        operator_schedule="two_stage",
        ensure_mutation=False,
    ),
    "binary250_r05_two_wor8": replace(
        B250,
        mutation_rate=0.05,
        operator_schedule="two_stage",
        mutation_with_replacement=False,
    ),
    "binary250_r05_two_wr8": replace(
        B250,
        mutation_rate=0.05,
        operator_schedule="two_stage",
    ),
    "binary250_r05_two_wr4": replace(
        B250,
        mutation_rate=0.05,
        operator_schedule="two_stage",
        n_points=4,
    ),
    # Faithful diagnostic: mutate selected objects in place and append aliases.
    "binary250_r05_legacy8": replace(
        B250,
        mutation_rate=0.05,
        operator_schedule="two_stage",
        ensure_mutation=False,
        legacy_alias=True,
    ),
    "genespace_scale1_legacy": OperatorVariant(
        latent=250,
        binary=True,
        hidden_size=2000,
        num_layers=1,
        population=200,
        offspring=64,
        mutation_rate=0.1,
        mutation_sigma=0.0,
        n_points=8,
        pressure=20.0,
        scheme="exp",
        operator_schedule="two_stage",
        ensure_mutation=False,
        legacy_alias=True,
    ),
}


@dataclass(frozen=True)
class OperatorRun:
    cities: int
    variant: str
    seed: int
    evaluation_budget: int
    tour_length: float
    evaluations_run: int
    elapsed_seconds: float
    generations: int
    neural_device: str | None


def _mutate_in_place(individual, variant: OperatorVariant):
    genes = individual.genes
    mask = np.random.random(genes.shape) < variant.mutation_rate
    if variant.ensure_mutation and variant.mutation_rate > 0 and not mask.any():
        mask.flat[np.random.randint(mask.size)] = True
    if mask.any():
        if variant.binary:
            genes[mask] = 1 - genes[mask]
        else:
            genes[mask] += (
                np.random.randn(int(mask.sum())) * variant.mutation_sigma
            ).astype(genes.dtype)
    # GeneSpace marked every selected object modified even when no bit flipped.
    individual.evaluated_at = -1


class LegacyAliasMutation(Layer):
    """Reproduce GeneSpace's mutation sampling and object aliasing exactly."""

    def __init__(self, variant: OperatorVariant):
        super().__init__()
        self.variant = variant

    def __call__(self, pop):
        indices = np.random.choice(len(pop), size=len(pop), replace=True)
        selected = [pop[index] for index in indices]
        for individual in selected:
            _mutate_in_place(individual, self.variant)
        return pop + selected


def _build_legacy_environment(
    fitness,
    objective,
    variant: OperatorVariant,
):
    decoder = MLPDecoder(
        input_length=variant.latent,
        output_shape=(objective.dimension,),
        hidden_size=variant.hidden_size,
        num_layers=variant.num_layers,
        lr=1e-5,
        device="mps",
    )
    genepool = GenePool(variant.latent, binary=variant.binary)
    families = max(1, variant.offspring // 4)
    environment = Environment(
        layers=[
            Populate(variant.population),
            DecodeAndEvaluate(fitness),
            Sort(),
            Crossover(
                RankSelection(
                    pressure=variant.pressure,
                    scheme=variant.scheme,
                ),
                families=families,
                children=4,
                n_points=variant.n_points,
            ),
            DecodeAndEvaluate(fitness),
            Sort(),
            Cap(variant.population),
            LegacyAliasMutation(variant),
            DecodeAndEvaluate(fitness),
            Sort(),
            Cap(variant.population),
        ],
        genepool=genepool,
        decoder=decoder,
    ).compile()
    return environment, decoder


def run_variant(objective, seed, budget, name, variant):
    _require_mps()
    _seed_everything(seed)
    fitness = TrackedFitness(objective)
    if variant.legacy_alias:
        environment, decoder = _build_legacy_environment(
            fitness, objective, variant
        )
        solve_one = lambda: environment.evolve(1, verbose_every=0)
    else:
        evolver = Evolver(
            fitness,
            output_shape=(objective.dimension,),
            device="mps",
            latent=variant.latent,
            population=variant.population,
            hidden_size=variant.hidden_size,
            num_layers=variant.num_layers,
            lr=1e-5,
            binary=variant.binary,
            mutation_rate=variant.mutation_rate,
            mutation_sigma=variant.mutation_sigma,
            refine_every=None,
            pressure=variant.pressure,
            scheme=variant.scheme,
            families=max(1, variant.offspring // 4),
            children=4,
            n_points=variant.n_points,
            offspring_only_mutation=variant.offspring_only_mutation,
            operator_schedule=variant.operator_schedule,
            mutation_children=variant.offspring,
            mutation_with_replacement=variant.mutation_with_replacement,
            ensure_mutation=variant.ensure_mutation,
        )
        environment, decoder = evolver.env, evolver.decoder
        solve_one = lambda: evolver.solve(1, verbose_every=0)

    parameter_devices = {parameter.device.type for parameter in decoder.parameters()}
    if parameter_devices != {"mps"}:
        raise RuntimeError(
            f"decoder parameters are not exclusively on MPS: {parameter_devices}"
        )
    started = time.perf_counter()
    while fitness.evaluations < budget:
        solve_one()
    torch.mps.synchronize()
    result = OperatorRun(
        cities=objective.dimension,
        variant=name,
        seed=seed,
        evaluation_budget=budget,
        tour_length=fitness.best_at(budget),
        evaluations_run=fitness.evaluations,
        elapsed_seconds=time.perf_counter() - started,
        generations=environment.generation,
        neural_device="mps",
    )
    torch.mps.empty_cache()
    return result


def summarize(runs: list[OperatorRun]):
    rows = []
    for cities in sorted({run.cities for run in runs}):
        for variant in sorted({run.variant for run in runs if run.cities == cities}):
            selected = [
                run for run in runs
                if run.cities == cities and run.variant == variant
            ]
            rows.append(
                {
                    "cities": cities,
                    "variant": variant,
                    "seeds": len(selected),
                    "mean_tour_length": statistics.fmean(
                        run.tour_length for run in selected
                    ),
                    "stdev_tour_length": (
                        statistics.stdev(run.tour_length for run in selected)
                        if len(selected) > 1 else 0.0
                    ),
                    "mean_generations": statistics.fmean(
                        run.generations for run in selected
                    ),
                    "mean_evaluations_run": statistics.fmean(
                        run.evaluations_run for run in selected
                    ),
                    "mean_seconds": statistics.fmean(
                        run.elapsed_seconds for run in selected
                    ),
                }
            )
    return sorted(rows, key=lambda row: (row["cities"], row["mean_tour_length"]))


def run_suite(sizes, seeds, budget, variant_names):
    _require_mps()
    runs = []
    direct_config = BenchmarkConfig(evaluation_budget=budget)
    for cities in sizes:
        _warm_mps(TravelingSalesperson(dimension=cities), direct_config)
        for seed in seeds:
            direct = run_direct_ga(
                TravelingSalesperson(dimension=cities), seed, direct_config
            )
            runs.append(
                OperatorRun(
                    cities=cities,
                    variant="direct_ga",
                    seed=seed,
                    evaluation_budget=budget,
                    tour_length=direct.metric_at_budget,
                    evaluations_run=direct.evaluations_run,
                    elapsed_seconds=direct.elapsed_seconds,
                    generations=direct.generations or 0,
                    neural_device=None,
                )
            )
            print(
                f"cities={cities:<3} variant={'direct_ga':<34} seed={seed} "
                f"length={direct.metric_at_budget:.6g}"
            )
            for name in variant_names:
                result = run_variant(
                    TravelingSalesperson(dimension=cities),
                    seed,
                    budget,
                    name,
                    VARIANTS[name],
                )
                runs.append(result)
                print(
                    f"cities={cities:<3} variant={name:<34} seed={seed} "
                    f"length={result.tour_length:.6g} device=mps"
                )
    return runs


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
    random.seed(0)
    runs = run_suite(args.sizes, args.seeds, args.budget, args.variants)
    summary = summarize(runs)
    print("\nmean tour length (lower is better)")
    for row in summary:
        print(
            f"cities={row['cities']:<3} variant={row['variant']:<34} "
            f"mean={row['mean_tour_length']:.6g} "
            f"sd={row['stdev_tour_length']:.5g} "
            f"gens={row['mean_generations']:.1f}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "study": "tsp_operator_ablation",
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
            "runs": [asdict(run) for run in runs],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
