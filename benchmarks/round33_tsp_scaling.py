"""Round 33: does pure decoder evolution beat CMA-ES on tours, and how far up does it scale?

Two gaps round 32 left open.

(1) NO CMA BASELINE ON LARGE TOURS. Every TSP claim so far races the
traditional tour GA. The only direct-CMA-on-tours number is 50 cities
(8.97 vs the tour GA's 8.00 — CMA lost), so CMA was quietly assumed
irrelevant at 100+. Under the standing ruling that CMA-ES is a BASELINE
and not a component, that arm has to be on the board.

(2) NO DATA ABOVE 100 CITIES. The advantage compounds with problem size
(GA wins at 20, we win 27% at 100), so the interesting question is
whether that keeps going.

Arms, at each city count:

  * traditional_tour_ga  — segment-reversal mutation on tours directly.
  * direct_cma           — CMA-ES on the raw priority vector (dim = cities);
                           the fitness function argsorts it into a tour.
  * anchor_evolution     — PURE decoder evolution, no distill, no CMA:
                           the honest deliverable under the ruling.
  * anchor_stack         — explore -> distill -> CMA-ES, the configuration
                           the old recorded TSP numbers used. Included to
                           track how much of any win CMA is carrying.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round21_tsp import (direct_cma, make_instance,
                                    nearest_neighbor_length,
                                    traditional_tour_ga)
from benchmarks.round25_anchor_field import AnchorFieldTransformer
from benchmarks.legacy_engines.solver import solve_single as solve


def solve_arm(cities: np.ndarray, budget: int, seed: int,
              pure: bool) -> dict:
    cache: dict[str, torch.Tensor] = {}

    def fitness(phenotypes: torch.Tensor) -> torch.Tensor:
        priorities = phenotypes.reshape(len(phenotypes), -1)
        key = str(priorities.device)
        if key not in cache:
            cache[key] = torch.as_tensor(cities, device=priorities.device)
        pts = cache[key][torch.argsort(priorities, dim=1)]
        return -(pts - pts.roll(-1, dims=1)).norm(dim=2).sum(dim=1)

    result = solve(
        fitness, output_shape=(len(cities),), budget=budget,
        architecture=lambda l, s: AnchorFieldTransformer(l, s, cities),
        explore_fraction=1.0 if pure else "auto", seed=seed)
    assert result.evaluations == budget
    if pure:
        assert result.explore_evaluations == budget, "CMA must not run here"
    explored = result.explore_evaluations
    return {"tour_length": float(-result.best_fitness),
            "explore_evaluations": explored,
            "after_explore": float(-result.history[explored - 1])}


ARMS = ("traditional_tour_ga", "direct_cma", "anchor_evolution",
        "anchor_stack")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--cities", nargs="+", type=int,
                        default=[100, 200, 400])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for n_cities in args.cities:
        print(f"\n########## {n_cities} CITIES (budget {args.budget}) ##########",
              flush=True)
        for seed in args.seeds:
            cities = make_instance(seed, n_cities)
            greedy = nearest_neighbor_length(cities)
            for arm in args.arms:
                _seed_everything(seed)
                if arm == "traditional_tour_ga":
                    row = {"tour_length": traditional_tour_ga(
                        cities, args.budget, seed)}
                elif arm == "direct_cma":
                    row = {"tour_length": direct_cma(
                        cities, args.budget, seed)}
                else:
                    row = solve_arm(cities, args.budget, seed,
                                    pure=(arm == "anchor_evolution"))
                row.update({"cities": n_cities, "arm": arm, "seed": seed,
                            "greedy": greedy})
                rows.append(row)
                extra = ""
                if "after_explore" in row:
                    extra = (f" (explore {row['after_explore']:.2f} "
                             f"@{row['explore_evaluations']})")
                print(f"  seed {seed} {arm:<20} {row['tour_length']:8.3f}"
                      f"{extra}", flush=True)
        print(f"  --- {n_cities} cities: means over {len(args.seeds)} seeds "
              f"(greedy nearest-neighbor {greedy:.2f}) ---", flush=True)
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
