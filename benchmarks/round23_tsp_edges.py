"""Round 23: the edge-matrix encoding for TSP — paths as edges, not orderings.

Rounds 21-22 showed random keys (decoder emits per-city priorities, the
fitness function argsorts them into a tour) is hostile to the whole
universal stack: mutations are either inert or discontinuous, and
averaging key vectors does not average tours, so distillation is
structurally powerless. This round changes the ENCODING, which is part
of the problem interface: the decoder now emits a 50x50 edge-score
matrix ("how much do I want city j to follow city i?") and the fitness
function constructs the tour by a greedy walk — start at city 0,
repeatedly move to the unvisited city with the highest edge score from
the current one — then returns its negative length.

Why this should fit the machinery: raising one edge's score changes the
constructed tour locally (mutation locality restored), and the average
of many good tours' edge matrices is an edge-frequency map — exactly the
pheromone matrix ant colony optimization has used for thirty years
(distillation-by-averaging restored). Evolution still only touches
genomes and decoder weights.

Arms (same instances, seeds, and 5,000-evaluation budget as rounds 21-22):
  traditional_tour_ga — the standing phenotype-operator baseline: GA
                        mutating tours by segment reversal (score to
                        beat: 8.00; random keys left decoders at ~16)
  direct_edge_ga      — no decoder: truncation GA mutating the raw edge
                        matrix with Gaussian noise. The is-the-encoding-
                        alone-searchable control. (CMA-ES on the raw
                        2,500-d matrix is omitted: full-covariance CMA
                        needs multi-second eigendecompositions per
                        generation at that dimension.)
  solve_mlp / solve_conv2d — the packaged universal solver emitting edge
                        matrices, dense vs convolutional decoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round21_tsp import (make_instance, nearest_neighbor_length,
                                    tour_lengths_np, traditional_tour_ga)
from benchmarks.legacy_engines.solver import solve_single as solve


def tours_from_edge_scores(scores: np.ndarray) -> np.ndarray:
    """Greedy walk per batch item: from city 0, always move to the
    unvisited city with the highest edge score from the current city."""
    batch, n, _ = scores.shape
    rows = np.arange(batch)
    visited = np.zeros((batch, n), dtype=bool)
    visited[:, 0] = True
    current = np.zeros(batch, dtype=int)
    tours = np.zeros((batch, n), dtype=int)
    for step in range(1, n):
        options = np.where(visited, -np.inf, scores[rows, current])
        current = options.argmax(axis=1)
        tours[:, step] = current
        visited[rows, current] = True
    return tours


def edge_lengths(cities: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    return tour_lengths_np(cities, tours_from_edge_scores(matrices))


def direct_edge_ga(cities: np.ndarray, budget: int, seed: int,
                   population: int = 32, elite: int = 16,
                   sigma: float = 0.1) -> float:
    rng = np.random.default_rng(seed)
    n = len(cities)
    pool = rng.random((population, n, n))
    lengths = edge_lengths(cities, pool)
    spent = population
    best = float(lengths.min())
    while spent < budget:
        order = np.argsort(lengths)[:elite]
        pool, lengths = pool[order], lengths[order]
        k = min(population, budget - spent)
        children = np.clip(
            pool[rng.integers(0, len(pool), k)]
            + rng.normal(0, sigma, (k, n, n)), 0, 1)
        child_lengths = edge_lengths(cities, children)
        spent += k
        best = min(best, float(child_lengths.min()))
        pool = np.concatenate([pool, children])
        lengths = np.concatenate([lengths, child_lengths])
    return best


def solve_arm(cities: np.ndarray, budget: int, seed: int,
              architecture: str) -> tuple[float, int]:
    n = len(cities)

    def fitness(phenotypes: torch.Tensor) -> np.ndarray:
        matrices = phenotypes.reshape(len(phenotypes), n, n)
        matrices = matrices.detach().cpu().numpy().astype(np.float64)
        return -edge_lengths(cities, matrices)

    result = solve(fitness, output_shape=(n, n), budget=budget,
                   architecture=architecture, seed=seed)
    assert result.evaluations == budget
    return float(-result.best_fitness), result.explore_evaluations


ARMS = ("traditional_tour_ga", "direct_edge_ga", "solve_mlp", "solve_conv2d")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--cities", type=int, default=50)
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for seed in args.seeds:
        cities = make_instance(seed, args.cities)
        greedy = nearest_neighbor_length(cities)
        print(f"seed {seed}: nearest-neighbor greedy {greedy:.3f}", flush=True)
        for arm in args.arms:
            _seed_everything(seed)
            explored = None
            if arm == "traditional_tour_ga":
                length = traditional_tour_ga(cities, args.budget, seed)
            elif arm == "direct_edge_ga":
                length = direct_edge_ga(cities, args.budget, seed)
            else:
                length, explored = solve_arm(
                    cities, args.budget, seed,
                    architecture=arm.removeprefix("solve_"))
            rows.append({"arm": arm, "seed": seed, "tour_length": length,
                         "greedy": greedy, "explore_evaluations": explored})
            note = f" (explored {explored})" if explored is not None else ""
            print(f"  {arm:<20} best tour {length:.4f}{note}", flush=True)

    print("\nmeans over seeds:")
    for arm in args.arms:
        vals = [r["tour_length"] for r in rows if r["arm"] == arm]
        print(f"  {arm:<20} {np.mean(vals):.4f} +- {np.std(vals, ddof=1):.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cities": args.cities, "budget": args.budget,
                   "torch_version": torch.__version__, "runs": rows}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
