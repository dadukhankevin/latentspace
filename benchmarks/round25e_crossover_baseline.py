"""Round 25e: is the anchor-field win robust to a stronger GA baseline?

The tour GA beaten in rounds 25/25d is mutation-only (segment reversal).
The standard strengthening for permutation GAs is order crossover (OX):
a child inherits a random slice from one parent and the remaining cities
in the order the second parent visits them, followed by the same
inversion mutation. This script races that stronger baseline against the
anchor-field solver's already-recorded results at 50 and 100 cities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.compare import _seed_everything
from benchmarks.round21_tsp import (make_instance, tour_lengths_np,
                                    traditional_tour_ga)


def order_crossover(a: np.ndarray, b: np.ndarray, rng) -> np.ndarray:
    n = len(a)
    i, j = np.sort(rng.integers(0, n, 2))
    child = np.full(n, -1)
    child[i:j + 1] = a[i:j + 1]
    used = set(child[i:j + 1].tolist())
    fill = [c for c in b if c not in used]
    slots = [k for k in range(n) if child[k] < 0]
    child[slots] = fill
    return child


def crossover_tour_ga(cities: np.ndarray, budget: int, seed: int,
                      population: int = 32, elite: int = 16) -> float:
    rng = np.random.default_rng(seed)
    n = len(cities)
    tours = np.stack([rng.permutation(n) for _ in range(population)])
    lengths = tour_lengths_np(cities, tours)
    spent = len(tours)
    best = float(lengths.min())
    while spent < budget:
        order = np.argsort(lengths)[:elite]
        tours, lengths = tours[order], lengths[order]
        k = min(population, budget - spent)
        children = []
        for _ in range(k):
            pa, pb = rng.integers(0, len(tours), 2)
            child = order_crossover(tours[pa], tours[pb], rng)
            i, j = np.sort(rng.integers(0, n, 2))
            child[i:j + 1] = child[i:j + 1][::-1]
            children.append(child)
        children = np.stack(children)
        child_lengths = tour_lengths_np(cities, children)
        spent += k
        best = min(best, float(child_lengths.min()))
        tours = np.concatenate([tours, children])
        lengths = np.concatenate([lengths, child_lengths])
    return best


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(10)))
    parser.add_argument("--cities", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for n_cities in args.cities:
        for seed in args.seeds:
            cities = make_instance(seed, n_cities)
            _seed_everything(seed)
            mutation_only = traditional_tour_ga(cities, args.budget, seed)
            _seed_everything(seed)
            with_crossover = crossover_tour_ga(cities, args.budget, seed)
            rows.append({"cities": n_cities, "seed": seed,
                         "mutation_only": mutation_only,
                         "with_crossover": with_crossover})
            print(f"cities {n_cities} seed {seed}: mutation-only "
                  f"{mutation_only:.4f}, +crossover {with_crossover:.4f}",
                  flush=True)
        mo = [r["mutation_only"] for r in rows if r["cities"] == n_cities]
        xo = [r["with_crossover"] for r in rows if r["cities"] == n_cities]
        print(f"cities {n_cities} means: mutation-only {np.mean(mo):.4f}, "
              f"+crossover {np.mean(xo):.4f}", flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
