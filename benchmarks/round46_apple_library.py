"""Round 46: the apple photo, rerun through the shipped library.

Every previous apple record came from a benchmark-local loop. This run uses
`latentspace.universal.solve` exactly as a user would — one fitness function,
one call — now that the framework itself carries the campaign's full stack:
crossover with averaged decoder inheritance (rounds 42-45), 8 survivors
(round 43), win-rate step control (rounds 29-31), batched vmap decoding.

Apple ladder to beat, all at 150k evaluations on the 96x96 RGB photo:

    0.120026  pixel GA (published)
    0.011163  distill->CMA hybrid (published)
    0.004929  pure conv evolution, fixed sigma (published)
    0.004005  pure conv evolution, win-rate control (round 31 record)

Frames of the best-so-far phenotype are captured inside the fitness function
(which sees every decoded phenotype anyway), so the animation costs nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round27_apple_no_cma import DEMO, load_apple
from benchmarks.round28_anchor_conv import ConvRGB
from benchmarks.round31_apple_animated import _png
from latentspace.universal import solve

SHAPE = (3, 96, 96)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    target = load_apple()
    refs = json.loads(DEMO.read_text())["D"]["finalMse"]
    print(f"apple {target.size} values; ladder: GA {refs['ga']}, hybrid "
          f"{refs['stack']}, conv fixed {refs['cf']}, record 0.004005",
          flush=True)

    target_t = torch.as_tensor(target, device="mps")
    frames: list[dict] = []
    state = {"spent": 0, "best": np.inf, "next_frame": 0}
    every = max(1, args.budget // max(args.frames, 1))

    def fitness(phenotypes: torch.Tensor) -> torch.Tensor:
        mse = ((phenotypes.flatten(1) - target_t) ** 2).mean(dim=1)
        state["spent"] += len(mse)
        lo = int(mse.argmin())
        if float(mse[lo]) < state["best"]:
            state["best"] = float(mse[lo])
            state["pheno"] = phenotypes[lo].detach().cpu().numpy()
        if state["spent"] >= state["next_frame"]:
            frames.append({"e": state["spent"], "m": state["best"],
                           "p": _png(state["pheno"].reshape(-1))})
            state["next_frame"] += every
            print(f"  {state['spent']:>7} evals  best {state['best']:.6f}",
                  flush=True)
        return -mse

    _seed_everything(args.seed)
    result = solve(fitness, output_shape=SHAPE, budget=args.budget,
                   architecture=lambda latent, shape: ConvRGB(latent, shape),
                   explore_fraction=1.0, seed=args.seed)
    final = float(-result.best_fitness)
    frames.append({"e": result.evaluations, "m": final,
                   "p": _png(result.best_phenotype.reshape(-1))})
    print(f"\nFINAL {final:.6f} at {result.evaluations} evaluations "
          f"({len(frames)} frames)")
    print(f"vs record 0.004005: {'NEW RECORD' if final < 0.004005 else 'no'}"
          f" ({0.004005 / final:.3f}x)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "seed": args.seed,
             "published_references": refs, "record_previous": 0.004005,
             "final_mse": final,
             "history": result.history[::50],
             "frames": frames,
             "torch_version": torch.__version__}) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
