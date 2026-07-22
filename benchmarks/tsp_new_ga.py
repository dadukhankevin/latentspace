"""TSP through the redesigned universal GA — random-keys encoding.

The honest framing: the new solve() has never touched TSP, and the
anchor-field grammar that actually WON TSP (round 25/33, beat the tour GA
at 100+ cities) is a legacy-only decoder never ported here. This measures
the PLAIN universal engine with its default decoder via random keys — the
decoder emits N priorities in [0, 1], argsort gives the tour — which round
21 flagged as the decoder GA's weak domain (index-space priors are
negative knowledge for permutations).

Reference arms (all legacy helpers): nearest-neighbor construction (the
strong non-evolutionary baseline), the traditional tour GA with
segment-reversal mutation, and direct CMA-ES on the same random keys.
Epochs are chosen so the new GA spends ~`budget` evaluations, matched to
the reference arms' budget.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from benchmarks.round21_tsp import (
    direct_cma,
    nearest_neighbor_length,
    tour_lengths_np,
    traditional_tour_ga,
)
from latentspace.universal import solve


def make_cities(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((n, 2)).astype(np.float32)


def new_ga_tour(cities, epochs, children, seed, directions, latents):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pts = torch.as_tensor(cities, device=device)

    def fitness(phenotypes):
        priorities = phenotypes.reshape(len(phenotypes), -1)
        ordered = pts[torch.argsort(priorities, dim=1)]
        return -(ordered - ordered.roll(-1, dims=1)).norm(dim=2).sum(dim=1)

    result = solve(fitness, output_shape=(len(cities),), epochs=epochs,
                   children=children, directions=directions,
                   latents=latents, seed=seed)
    return float(-result.best_fitness), result.evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", type=int, default=50)
    parser.add_argument("--budget", type=int, default=20_000)
    parser.add_argument("--children", type=int, default=32)
    parser.add_argument("--directions", default="frozen",
                        choices=("frozen", "sparse"))
    parser.add_argument("--latents", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    args = parser.parse_args()

    epochs = max(1, args.budget // args.children)
    rows = {"new_ga": [], "tour_ga": [], "cma": [], "nn": []}
    for seed in args.seeds:
        cities = make_cities(args.cities, seed)
        ga_len, evals = new_ga_tour(
            cities, epochs, args.children, seed, args.directions,
            args.latents)
        rows["new_ga"].append(ga_len)
        rows["tour_ga"].append(traditional_tour_ga(cities, args.budget, seed))
        rows["cma"].append(direct_cma(cities, args.budget, seed))
        rows["nn"].append(nearest_neighbor_length(cities))
        print(f"  {args.cities}c seed {seed}: new_ga {ga_len:6.3f} "
              f"({evals} ev)  tour_ga {rows['tour_ga'][-1]:6.3f}  "
              f"cma {rows['cma'][-1]:6.3f}  nn {rows['nn'][-1]:6.3f}",
              flush=True)

    print(f"\n{args.cities} cities, {len(args.seeds)} seeds, "
          f"~{args.budget} evals ({args.directions}):")
    for arm, label in (("new_ga", "new universal GA"),
                       ("tour_ga", "traditional tour GA"),
                       ("cma", "direct CMA-ES"),
                       ("nn", "nearest-neighbor")):
        v = np.array(rows[arm])
        print(f"  {label:<22} {v.mean():7.3f}  (lower is better)")


if __name__ == "__main__":
    main()
