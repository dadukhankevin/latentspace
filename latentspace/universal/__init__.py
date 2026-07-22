"""One universal genetic algorithm: solve(fitness_fns, output_shape, epochs).

    from latentspace.universal import solve

    result = solve(fitness_fn, output_shape=(32, 32), epochs=1_000)
    result.best_phenotype                # the best solution found

    result = solve([fit_a, fit_b, ...], output_shape=(32, 32),
                   epochs=10_000)
    result.problems[1].best_phenotype    # per-function best-ever

The design (Daniel's specification, 2026-07-21, in ga.py): two random
founders; genes and latents as permanently distinct spaces with their own
crossover and mutation operators; a capped population with extinction
allowed; speciation re-assigning individuals across fitness functions over
time; and one shared decoder that discoveries are folded into. Every
operator is a replaceable function.

The retired engines (the per-individual explorer stack that holds the
single-fitness records, the champion-per-problem population, distill, and
the CMA-ES baseline) live in benchmarks/legacy_engines/ — benchmark
opponents the new design has to beat, no longer part of the library.
"""
from .architectures import build_mlp, register_architecture, resolve
from .ga import (
    GAResult,
    ProblemResult,
    coin_flip_latent_inheritance,
    fitness_shares,
    largest_niche_champion_fold_selection,
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
    "make_random_speciation", "largest_niche_champion_fold_selection",
    "register_architecture", "resolve", "build_mlp",
]
