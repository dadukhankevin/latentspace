"""Round 21: the discrete frontier — traveling salesman with sequence decoders.

Discrete/permutation problems are the one family nothing latent has ever
won in this campaign. The universal encoding tested here is random keys:
the decoder outputs one priority value per city and the FITNESS FUNCTION
argsorts the priorities into a tour and returns its negative length. No
operator ever touches a tour — evolution stays entirely on genomes and
decoder weights — and the modality-matched architectures under test are
the sequence decoders (GRU, LSTM, transformer) against the dense MLP.

Arms (same city instance and evaluation budget per seed):
  traditional_tour_ga — the phenotype-operator baseline: a truncation GA
      mutating tours directly by segment reversal (inversion, the standard
      strong TSP mutation). This is what universality has to beat.
  direct_cma          — CMA-ES directly on the raw priority vector, no
      decoder (the round-19c control: direct CMA owns low-d continuous
      spaces, so it may own low-d random keys too).
  solve_mlp / solve_gru / solve_lstm / solve_transformer
                      — the packaged universal solver with the decoder
      architecture varying.

References printed per instance: mean random-tour length and the
nearest-neighbor greedy tour length.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.legacy_engines.solver import solve_single as solve
from benchmarks.legacy_engines.cma import cma_minimize


def make_instance(seed: int, n_cities: int) -> np.ndarray:
    rng = np.random.default_rng(10_000 + seed)
    return rng.random((n_cities, 2)).astype(np.float32)


def tour_lengths_np(cities: np.ndarray, tours: np.ndarray) -> np.ndarray:
    pts = cities[tours]                                   # (B, N, 2)
    return np.linalg.norm(pts - np.roll(pts, -1, axis=1), axis=2).sum(axis=1)


def nearest_neighbor_length(cities: np.ndarray) -> float:
    n = len(cities)
    dist = np.linalg.norm(cities[:, None] - cities[None], axis=2)
    tour = [0]
    unvisited = np.ones(n, dtype=bool)
    unvisited[0] = False
    while unvisited.any():
        d = dist[tour[-1]].copy()
        d[~unvisited] = np.inf
        nxt = int(d.argmin())
        tour.append(nxt)
        unvisited[nxt] = False
    return float(tour_lengths_np(cities, np.asarray([tour]))[0])


def traditional_tour_ga(cities: np.ndarray, budget: int, seed: int,
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
        children = tours[rng.integers(0, len(tours), k)].copy()
        for child in children:
            i, j = np.sort(rng.integers(0, n, 2))
            child[i:j + 1] = child[i:j + 1][::-1]
        child_lengths = tour_lengths_np(cities, children)
        spent += k
        best = min(best, float(child_lengths.min()))
        tours = np.concatenate([tours, children])
        lengths = np.concatenate([lengths, child_lengths])
    return best


def direct_cma(cities: np.ndarray, budget: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    best = math.inf

    def evaluate(priorities: np.ndarray) -> np.ndarray:
        nonlocal best
        lengths = tour_lengths_np(cities, np.argsort(priorities, axis=1))
        best = min(best, float(lengths.min()))
        return lengths

    cma_minimize(evaluate, dim=len(cities), budget_evaluations=budget,
                 evaluations_done=0, rng=rng,
                 mean0=np.zeros(len(cities)), sigma0=0.5)
    return best


def solve_arm(cities: np.ndarray, budget: int, seed: int,
              architecture: str) -> tuple[float, int]:
    cache: dict[str, torch.Tensor] = {}

    def fitness(phenotypes: torch.Tensor) -> torch.Tensor:
        priorities = phenotypes.reshape(len(phenotypes), -1)
        key = str(priorities.device)
        if key not in cache:
            cache[key] = torch.as_tensor(cities, device=priorities.device)
        pts = cache[key][torch.argsort(priorities, dim=1)]   # (B, N, 2)
        return -(pts - pts.roll(-1, dims=1)).norm(dim=2).sum(dim=1)

    result = solve(fitness, output_shape=(len(cities),), budget=budget,
                   architecture=architecture, seed=seed)
    assert result.evaluations == budget
    return float(-result.best_fitness), result.explore_evaluations


ARMS = ("traditional_tour_ga", "direct_cma",
        "solve_mlp", "solve_gru", "solve_lstm", "solve_transformer")


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
        rng = np.random.default_rng(seed)
        random_mean = float(tour_lengths_np(
            cities, np.stack([rng.permutation(args.cities)
                              for _ in range(256)])).mean())
        greedy = nearest_neighbor_length(cities)
        print(f"seed {seed}: random tour mean {random_mean:.3f}, "
              f"nearest-neighbor greedy {greedy:.3f}", flush=True)
        for arm in args.arms:
            _seed_everything(seed)
            explored = None
            if arm == "traditional_tour_ga":
                best = traditional_tour_ga(cities, args.budget, seed)
            elif arm == "direct_cma":
                best = direct_cma(cities, args.budget, seed)
            else:
                best, explored = solve_arm(cities, args.budget, seed,
                                           architecture=arm.removeprefix("solve_"))
            rows.append({"arm": arm, "seed": seed, "tour_length": best,
                         "random_mean": random_mean, "greedy": greedy,
                         "explore_evaluations": explored})
            note = f" (explored {explored})" if explored is not None else ""
            print(f"  {arm:<22} best tour {best:.4f}{note}", flush=True)

    print("\nmeans over seeds:")
    for arm in args.arms:
        vals = [r["tour_length"] for r in rows if r["arm"] == arm]
        print(f"  {arm:<22} {np.mean(vals):.4f} +- {np.std(vals, ddof=1):.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cities": args.cities, "budget": args.budget,
                   "torch_version": torch.__version__, "runs": rows}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
