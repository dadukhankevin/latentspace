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

Legacy engines (the per-individual-weights explorer stack in solver.py /
explorer.py, and the champion-per-problem population in multi.py) are no
longer part of the API. They remain importable from their modules ONLY as
benchmark opponents — the records they hold are the bar the new design has
to clear — alongside cma.py, the CMA-ES baseline.
"""
from .architectures import build_mlp, register_architecture, resolve
from .distill import LatentSpace, distill
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
    "distill", "LatentSpace",
]
