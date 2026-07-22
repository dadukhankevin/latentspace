"""Round 34: does the anchor grammar still help now that mutation self-tunes?

Daniel's question. The anchor field (round 25) and win-rate step control
(rounds 30-31) attack the same underlying problem — mutation-to-fitness
locality — from two directions. Anchors fix WHERE a mutation lands: the
64 genes are read as 8 spatial sources, so mutating one anchor's genes
redraws one region of the tour instead of scrambling all of it. The
controller fixes HOW BIG the mutation is. If the controller subsumes the
grammar, the anchors are now dead weight and the decoder gets simpler.

Three decoders, identical evolution (pure decoder GA, win-rate control,
no distill, no CMA), decomposing the two tier-2 ideas:

  * mlp_decoder    — tier 0: never sees the city coordinates at all. The
                     genome goes through a dense net to per-city
                     priorities. Round 21 measured 15.7 at 50 cities.
  * city_context   — tier 2, GLOBAL genome: a transformer reading city
                     coordinates as tokens, with the genome added to
                     every token as one context vector. Every mutation
                     shifts all city priorities at once. Round 24: 11.1.
  * anchor_field   — tier 2, SPATIAL genome: the same transformer, but
                     the genome enters as 8 anchors with locations and
                     messages, so each city reads only the anchors near
                     it. Round 25: 7.85, the first discrete win.

Old numbers above were measured with the falsified fixed mutation
constant AND the distill->CMA stack, so all three are rerun clean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round21_tsp import make_instance, traditional_tour_ga
from benchmarks.round24_city_conditioned import CityConditionedTransformer
from benchmarks.round25_anchor_field import AnchorFieldTransformer
from latentspace.universal import solve

DECODERS = {
    "mlp_decoder": lambda cities: "mlp",
    "city_context": lambda cities: (
        lambda l, s: CityConditionedTransformer(l, s, cities)),
    "anchor_field": lambda cities: (
        lambda l, s: AnchorFieldTransformer(l, s, cities)),
}
ARMS = ("traditional_tour_ga",) + tuple(DECODERS)


def solve_arm(cities: np.ndarray, budget: int, seed: int, arm: str) -> dict:
    cache: dict[str, torch.Tensor] = {}

    def fitness(phenotypes: torch.Tensor) -> torch.Tensor:
        priorities = phenotypes.reshape(len(phenotypes), -1)
        key = str(priorities.device)
        if key not in cache:
            cache[key] = torch.as_tensor(cities, device=priorities.device)
        pts = cache[key][torch.argsort(priorities, dim=1)]
        return -(pts - pts.roll(-1, dims=1)).norm(dim=2).sum(dim=1)

    result = solve(fitness, output_shape=(len(cities),), budget=budget,
                   architecture=DECODERS[arm](cities),
                   explore_fraction=1.0, seed=seed)
    assert result.evaluations == budget
    assert result.explore_evaluations == budget, "CMA must not run here"
    return {"tour_length": float(-result.best_fitness),
            "after_first_generation": float(-result.history[31])}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--cities", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for n_cities in args.cities:
        print(f"\n########## {n_cities} CITIES ##########", flush=True)
        for seed in args.seeds:
            cities = make_instance(seed, n_cities)
            for arm in args.arms:
                _seed_everything(seed)
                if arm == "traditional_tour_ga":
                    row = {"tour_length": traditional_tour_ga(
                        cities, args.budget, seed)}
                else:
                    row = solve_arm(cities, args.budget, seed, arm)
                row.update({"cities": n_cities, "arm": arm, "seed": seed})
                rows.append(row)
                prior = (f"  (untrained prior "
                         f"{row['after_first_generation']:.2f})"
                         if "after_first_generation" in row else "")
                print(f"  seed {seed} {arm:<20} {row['tour_length']:8.3f}"
                      f"{prior}", flush=True)
        print(f"  --- {n_cities} cities: means over {len(args.seeds)} seeds ---",
              flush=True)
        for arm in args.arms:
            vals = [r["tour_length"] for r in rows
                    if r["arm"] == arm and r["cities"] == n_cities]
            print(f"  {arm:<20} {np.mean(vals):8.3f} +- "
                  f"{np.std(vals, ddof=1) if len(vals) > 1 else 0:.3f}",
                  flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "torch_version": torch.__version__,
             "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
