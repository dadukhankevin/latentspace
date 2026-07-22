"""Round 22: ordering-aware distillation for TSP — the original package's insight, ported.

Round 21 (50-city TSP under random keys: the decoder emits one priority
per city, the fitness function argsorts priorities into a tour) left every
decoder arm at ~16 while a tour-mutating GA reached 8 and CMA-ES directly
on raw priorities reached 9. The original latentspace package predicted
the failure mode in its PermutationTrainer docstring: raw key VALUES are
meaningless (any monotone transform preserves the tour), so distilling
them by Euclidean PCA copies noise, not tours.

This round ports that insight to the universal stack's distill phase.
Nothing here uses the answer: the elites being distilled are vetted by
the fitness function alone, exactly as before. Only their representation
at the distillation step changes.

Arms (identical exploration: MLP decoder, stall-adaptive, then distill +
CMA-ES with the remaining budget):
  value_distill          — the status quo: PCA over the elites' raw key
                           values in logit space (round-21 reproduction)
  rank_distill           — each elite's keys replaced by their normalized
                           ranks before PCA. Monotone-invariant: elites
                           with the same tour become the same vector.
                           Generic to any argsort-interpreted output.
  canonical_rank_distill — additionally canonicalizes each elite's route
                           across cyclic rotation and reversal (an
                           undirected-cycle symmetry of the ENCODING, not
                           of any particular instance), reusing
                           PermutationTrainer.canonical_route from the
                           original package.

References re-run per seed for pairing: the traditional tour GA and
direct CMA-ES on raw priorities from round 21.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round21_tsp import (direct_cma, make_instance,
                                    nearest_neighbor_length,
                                    tour_lengths_np, traditional_tour_ga)
from latentspace.training import PermutationTrainer
from latentspace.universal.architectures import resolve
from benchmarks.legacy_engines.cma import cma_minimize
from benchmarks.legacy_engines.distill import distill
from benchmarks.legacy_engines.explorer import ExplorerConfig, PerIndividualExplorer

LATENT = 64
DISTILL_TOP = 200


def rank_transform(keys: np.ndarray) -> np.ndarray:
    """Keys -> normalized ranks in (0, 1); monotone-invariant."""
    ranks = np.empty(len(keys))
    ranks[np.argsort(keys)] = np.arange(len(keys))
    return (ranks + 0.5) / len(keys)


def canonical_rank_transform(keys: np.ndarray) -> np.ndarray:
    """Keys -> ranks of the rotation/reversal-canonicalized route."""
    route = PermutationTrainer.canonical_route(keys)
    ranks = np.empty(len(route))
    for rank, city in enumerate(route):
        ranks[city] = rank
    return (ranks + 0.5) / len(route)


TRANSFORMS = {
    "value_distill": None,
    "rank_distill": rank_transform,
    "canonical_rank_distill": canonical_rank_transform,
}


def run_stack(cities: np.ndarray, budget: int, seed: int,
              transform) -> tuple[float, int, int]:
    """Explore -> (transform) -> distill -> CMA. Returns (best tour length,
    explore evaluations, unique canonical routes among distilled elites)."""
    device = "mps"
    rng = np.random.default_rng(seed)
    cities_t = torch.as_tensor(cities, device=device)
    spent = 0
    best = math.inf

    def evaluate_losses(phenotypes: torch.Tensor) -> np.ndarray:
        nonlocal spent, best
        priorities = phenotypes.reshape(len(phenotypes), -1)
        pts = cities_t[torch.argsort(priorities, dim=1)]
        lengths = (pts - pts.roll(-1, dims=1)).norm(dim=2).sum(dim=1)
        lengths = lengths.cpu().numpy().astype(np.float64)
        spent += len(lengths)
        best = min(best, float(lengths.min()))
        return lengths

    builder = resolve("mlp", LATENT, (len(cities),))
    explorer = PerIndividualExplorer(builder, LATENT, device, ExplorerConfig())
    archive = explorer.run(evaluate_losses, rng,
                           stop_after=lambda _n: budget - spent,
                           adaptive=True, reserve=10 * LATENT)
    explored = spent

    idx = archive.select(DISTILL_TOP)
    elites = archive.phenotypes[idx]
    unique_routes = len({PermutationTrainer.canonical_route(e)
                         for e in elites})
    if transform is not None:
        elites = np.stack([transform(e) for e in elites])
    space = distill(elites, LATENT, (len(cities),), device=device)
    cma_minimize(lambda z: evaluate_losses(space.decode(z)),
                 dim=LATENT, budget_evaluations=budget,
                 evaluations_done=spent, rng=rng,
                 mean0=np.zeros(LATENT), sigma0=1.0)
    assert spent == budget
    return best, explored, unique_routes


ARMS = ("traditional_tour_ga", "direct_cma", *TRANSFORMS)


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
            explored = unique_routes = None
            if arm == "traditional_tour_ga":
                length = traditional_tour_ga(cities, args.budget, seed)
            elif arm == "direct_cma":
                length = direct_cma(cities, args.budget, seed)
            else:
                length, explored, unique_routes = run_stack(
                    cities, args.budget, seed, TRANSFORMS[arm])
            rows.append({"arm": arm, "seed": seed, "tour_length": length,
                         "greedy": greedy, "explore_evaluations": explored,
                         "unique_elite_routes": unique_routes})
            note = (f" (explored {explored}, {unique_routes} unique "
                    f"elite routes)" if explored is not None else "")
            print(f"  {arm:<24} best tour {length:.4f}{note}", flush=True)

    print("\nmeans over seeds:")
    for arm in args.arms:
        vals = [r["tour_length"] for r in rows if r["arm"] == arm]
        print(f"  {arm:<24} {np.mean(vals):.4f} +- {np.std(vals, ddof=1):.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cities": args.cities, "budget": args.budget,
                   "latent": LATENT, "distill_top": DISTILL_TOP,
                   "torch_version": torch.__version__, "runs": rows}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
