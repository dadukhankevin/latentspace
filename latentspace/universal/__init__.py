"""One universal genetic algorithm: solve(fitness_fns, output_shape, epochs).

    from latentspace.universal import solve

    result = solve(fitness_fn, output_shape=(32, 32), epochs=1_000)
    result.best_phenotype                # the best solution found

    result = solve([fit_a, fit_b, ...], output_shape=(32, 32),
                   epochs=10_000)
    result.problems[1].best_phenotype    # per-function best-ever

The design (Daniel's specification, 2026-07-21, in ga.py): random
founders per fitness function (16 by default — the founding count is the
run's entire coverage of the space, and two was measured too few,
2026-07-27); genes and latents as permanently distinct spaces with their own
crossover and mutation operators; a capped population with extinction
allowed; speciation re-assigning individuals across fitness functions over
time; and one shared decoder that CONSOLIDATES by distillation: on
multi-function runs the base is periodically gradient-trained to reproduce
each function's best-ever phenotype from its genes, then every
per-individual modifier decays (measured: 10/10 seeds, t=+16.7, -30% MSE
on 8 co-resident problems; the arithmetic fold it replaces was searched at
every budget and substrate and never helped). Every operator is a
replaceable function. `init_decoder=` warm-starts the shared decoder from
a previous run's `GAResult.decoder`.

The per-individual modifier's FORM is a first-class choice: low-rank
gating of frozen random directions (default) or sparse weight patches
(`directions="sparse"`, see sparse.py) — one shared decoder either way.

The retired engines (the per-individual explorer stack that holds the
single-fitness records, the champion-per-problem population, distill, and
the CMA-ES baseline) live in benchmarks/legacy_engines/ — benchmark
opponents the new design has to beat, no longer part of the library.
"""
from .agentic import AgenticGA
from .architectures import build_mlp, register_architecture, resolve
from .serve import live_progress
from .ga import (
    Distillation,
    GAResult,
    ProblemResult,
    register_substrate,
    coin_flip_latent_inheritance,
    fitness_shares,
    make_gaussian_mutation,
    make_random_speciation,
    make_species_selection,
    one_point_gene_crossover,
    share_selection,
    solve,
    uniform_selection,
)

__all__ = [
    "solve", "GAResult", "ProblemResult",
    "make_species_selection", "share_selection", "uniform_selection",
    "fitness_shares",
    "one_point_gene_crossover",
    "coin_flip_latent_inheritance", "make_gaussian_mutation",
    "make_random_speciation",
    "register_substrate", "Distillation",
    "register_architecture", "resolve", "build_mlp",
    "AgenticGA", "live_progress",
]
