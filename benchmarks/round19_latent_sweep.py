"""Round 19: is 32 the right latent size for the universal solver?

The number 32 was inherited from the pretraining era and never swept for
the explore -> distill -> exploit stack. It plays two roles there, forced
equal: the genome size every private decoder reads during exploration, and
the dimension of the distilled space CMA-ES searches during exploitation.
Suspicion cuts both ways: the distilled space is fit from only ~200
solutions (large latents fit noise directions), but too-small latents may
not span what exploration found.

Sweep latent in {8, 16, 32, 64, 128} on both standing problems, 10 paired
seeds, THROUGH THE PACKAGED `latentspace.universal.solve` — so this round
also dogfoods the public API against the benchmark harness.
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
from benchmarks.legacy_engines.solver import solve_single as solve

OBJECTIVES: dict[str, tuple[Callable[[], Objective], tuple]] = {
    "smooth1d_256": (SmoothTarget, (256,)),
    "blob2d_1024": (BlobImage2D, (32, 32)),
}

LATENTS = (8, 16, 32, 64, 128)


def run_one(objective_name, latent, seed, budget):
    objective_cls, shape = OBJECTIVES[objective_name]
    _seed_everything(seed)
    objective = objective_cls()

    def fitness(phenotypes, o=objective):
        return -o.loss_tensor(phenotypes.flatten(1))

    result = solve(fitness, output_shape=shape, budget=budget,
                   latent=latent, seed=seed)
    assert result.evaluations == budget
    return float(-result.best_fitness), result.explore_evaluations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES,
                        default=list(OBJECTIVES))
    parser.add_argument("--latents", nargs="+", type=int, default=list(LATENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for objective_name in args.objectives:
        for latent in args.latents:
            losses = []
            for seed in args.seeds:
                loss, explored = run_one(objective_name, latent, seed,
                                         args.budget)
                losses.append(loss)
                rows.append({
                    "objective": objective_name, "latent": latent,
                    "seed": seed, "mse": loss,
                    "explore_evaluations": explored,
                })
            print(f"{objective_name:<14} latent={latent:<4} "
                  f"mean={np.mean(losses):.5f} stdev={np.std(losses, ddof=1):.5f}",
                  flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "budget": args.budget, "latents": args.latents,
            "torch_version": torch.__version__,
            "runs": rows,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
