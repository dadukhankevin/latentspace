"""Round 20: should the solver re-enter exploration, or is one-way enough?

Daniel's observation: the current stack is a one-way conveyor (explore ->
distill -> exploit to the end), but the two forces pay off at different
moments — an elegant system would hand the budget back and forth. The
cycle mode built for this round: exploitation gets the same stall rule
exploration has (20 CMA generations without 1% relative improvement);
when it fires with budget left, exploration re-enters with half its
population warm-started from the current distilled knowledge decompressed
into decoder weights plus noise (round 12's off-manifold escape channel,
aimed at the representation floor) and half fresh (round 15c's error-
independence guardrail); the archive is cumulative and each cycle
re-distills a richer space.

Arms, both through `latentspace.universal.solve` (10 paired seeds):

  * single — the round-18 one-way stack (phases="single");
  * cycle  — the alternating version (phases="cycle").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from benchmarks.compare import Objective, _require_mps, _seed_everything
from benchmarks.round3_structure import SmoothTarget
from benchmarks.round8_mlp_pretrain import BlobImage2D
from latentspace.universal import solve

OBJECTIVES: dict[str, tuple[Callable[[], Objective], tuple]] = {
    "smooth1d_256": (SmoothTarget, (256,)),
    "blob2d_1024": (BlobImage2D, (32, 32)),
}


def run_one(objective_name, phases, seed, budget):
    objective_cls, shape = OBJECTIVES[objective_name]
    _seed_everything(seed)
    objective = objective_cls()

    def fitness(phenotypes, o=objective):
        return -o.loss_tensor(phenotypes.flatten(1))

    result = solve(fitness, output_shape=shape, budget=budget,
                   phases=phases, seed=seed)
    assert result.evaluations == budget
    return float(-result.best_fitness), result.explore_evaluations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES,
                        default=list(OBJECTIVES))
    parser.add_argument("--phases", nargs="+",
                        choices=("single", "cycle"),
                        default=["single", "cycle"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(10)))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for objective_name in args.objectives:
        for phases in args.phases:
            losses = []
            for seed in args.seeds:
                loss, explored = run_one(objective_name, phases, seed,
                                         args.budget)
                losses.append(loss)
                rows.append({
                    "objective": objective_name, "strategy": phases,
                    "seed": seed, "mse": loss,
                    "explore_evaluations": explored,
                })
            print(f"{objective_name:<14} {phases:<7} "
                  f"mean={np.mean(losses):.5f} "
                  f"stdev={np.std(losses, ddof=1):.5f}", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "budget": args.budget,
            "torch_version": torch.__version__,
            "runs": rows,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
