"""Evolve any photograph with the shipped library, capturing animation frames.

Generic version of the apple demo (rounds 46/50): point it at an image file,
it downscales to 96x96 RGB, runs `solve()` with the current defaults
(crossover + averaged decoders, fitness-signed mutation memory, win-rate
step control), and writes a frames JSON plus an animated GIF. The solver
sees only MSE fitness scores — never the target.

    python3 -m benchmarks.demo_target_image path/to/photo.jpg \
        --budget 150000 --gif out.gif --output out.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round28_anchor_conv import ConvRGB
from benchmarks.round31_apple_animated import _png
from latentspace.universal import solve

SHAPE = (3, 96, 96)


def load_target(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((96, 96), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1).reshape(-1)   # (3*96*96,) in [0,1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--budget", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--gif-size", type=int, default=288)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    target = load_target(args.image)
    target_t = torch.as_tensor(target, device="mps")

    frames: list[dict] = []
    state = {"spent": 0, "best": np.inf, "next": 0}
    every = max(1, args.budget // max(args.frames, 1))

    def fitness(phenotypes: torch.Tensor) -> torch.Tensor:
        mse = ((phenotypes.flatten(1) - target_t) ** 2).mean(dim=1)
        state["spent"] += len(mse)
        lo = int(mse.argmin())
        if float(mse[lo]) < state["best"]:
            state["best"] = float(mse[lo])
            state["pheno"] = phenotypes[lo].detach().cpu().numpy()
        if state["spent"] >= state["next"]:
            frames.append({"e": state["spent"], "m": state["best"],
                           "p": _png(state["pheno"].reshape(-1))})
            state["next"] += every
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
    print(f"\nFINAL {final:.6f} at {result.evaluations} evaluations")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"image": str(args.image), "budget": args.budget,
             "seed": args.seed, "final_mse": final,
             "history": result.history[::50], "frames": frames,
             "torch_version": torch.__version__}) + "\n")
        print(f"wrote {args.output}")

    if args.gif:
        size = (args.gif_size, args.gif_size)
        imgs = [Image.open(io.BytesIO(base64.b64decode(f["p"].split(",", 1)[1])))
                .convert("RGB").resize(size, Image.NEAREST) for f in frames]
        imgs.append(imgs[-1])
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:],
                     duration=[60] * (len(imgs) - 1) + [2000], loop=0,
                     optimize=True)
        print(f"wrote {args.gif}")


if __name__ == "__main__":
    main()
