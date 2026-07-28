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
time; and one shared decoder that discoveries are folded into — by a
share-weighted SIGN VOTE across the population by default, which ties
absorbing the single champion on average and halves run-to-run variance.
Every operator is a replaceable function. `init_decoder=` warm-starts the
shared decoder from a previous run's `GAResult.decoder`.

The per-individual modifier's FORM is a first-class choice: low-rank
gating of frozen random directions (default) or sparse weight patches
(`directions="sparse"`, see sparse.py) — one shared decoder either way.

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
    sign_vote_fold_selection,
    rank_weighted_fold_selection,
    centered_rank_fold_selection,
    natural_gradient_fold_selection,
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
    "sign_vote_fold_selection", "rank_weighted_fold_selection",
    "centered_rank_fold_selection", "natural_gradient_fold_selection",
    "register_architecture", "resolve", "build_mlp",
]
