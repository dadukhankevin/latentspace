"""The genes/latents operator matrix for solve_many.

Genes = z (the decoder's input, the universal genotype). Latents = the LoRA
coefficients gating the shared decoder's directions. The library historically
treated the concatenation [z | coefficients] as one undifferentiated
chromosome in all three operators (mutation sigma, crossover cut,
compatibility distance). Each arm here changes exactly one operator to
respect the boundary; "base" is the shipped boundary-blind default.

Benchmarks: `image` — 16 native-32x32 CIFAR targets through the conv
conditional decoder; `curve` — 8 random smooth curves, output (64,), through
the generic LoRA decoder. Metric: mean percent of founder error removed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.legacy_engines.multi import solve_many

ARMS = {
    "base": {},
    "compat_genes": {"compat_distance": "genes"},
    "compat_latents": {"compat_distance": "latents"},
    "cuts_separate": {"crossover_cuts": "separate"},
    "cuts_genes": {"crossover_cuts": "genes_only"},
    "cuts_latents": {"crossover_cuts": "latents_only"},
    "sigma_quarter": {"latent_sigma_scale": 0.25},
    "sigma_2x": {"latent_sigma_scale": 2.0},
    "sigma_4x": {"latent_sigma_scale": 4.0},
    "sigma_8x": {"latent_sigma_scale": 8.0},
    "sigma_16x": {"latent_sigma_scale": 16.0},
    "sigma_alt": {"latent_sigma_scale": "alt"},
    "sigma_auto": {"latent_sigma_scale": "auto"},
    # combination arms, run after the single-axis reads:
    "combo_harness": {"compat_distance": "genes", "crossover_cuts": "separate"},
    "combo_sigma4_compat": {"latent_sigma_scale": 4.0,
                            "compat_distance": "genes"},
    "default_new": {"latent_sigma_scale": "auto", "compat_distance": "genes"},
}


def image_problem(args):
    from PIL import Image
    files = sorted(Path(args.targets_dir).glob("*.png"))[:args.count]
    targets = [np.asarray(Image.open(f), dtype=np.float32) / 255.0
               for f in files]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    fns = []
    for t in targets:
        flat = torch.as_tensor(t.reshape(-1), device=device)
        fns.append(lambda ph, f=flat: -((ph - f) ** 2).mean(dim=1))
    return fns, (32, 32, 3), [f.stem for f in files]


def curve_problem(args):
    rng = np.random.default_rng(123)   # fixed target set across arms/seeds
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = np.linspace(0, 1, 64)
    fns, names = [], []
    for i in range(args.count):
        freq = rng.uniform(1, 4, 3)
        phase = rng.uniform(0, 2 * np.pi, 3)
        amp = rng.uniform(0.05, 0.15, 3)
        t = 0.5 + sum(a * np.sin(2 * np.pi * f * x + p)
                      for f, p, a in zip(freq, phase, amp))
        flat = torch.as_tensor(np.clip(t, 0, 1).astype(np.float32),
                               device=device)
        fns.append(lambda ph, f=flat: -((ph - f) ** 2).mean(dim=1))
        names.append(f"curve_{i}")
    return fns, (64,), names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("image", "curve"),
                        default="image")
    parser.add_argument("--arm", choices=sorted(ARMS), default="base")
    parser.add_argument("--targets-dir",
                        default="/tmp/latentspace_cifar100_scaling_1024")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=24_000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fns, shape, names = (image_problem(args) if args.benchmark == "image"
                         else curve_problem(args))
    start = time.time()
    result = solve_many(fns, output_shape=shape, budget=args.budget,
                        seed=args.seed, **ARMS[args.arm])
    elapsed = time.time() - start

    removed = {}
    for name, problem in zip(names, result.problems):
        init_mse, best_mse = -problem.initial_fitness, -problem.best_fitness
        removed[name] = float(100 * (1 - best_mse / init_mse))
    values = np.array(list(removed.values()))
    out = {
        "benchmark": args.benchmark,
        "arm": args.arm,
        "arm_kwargs": ARMS[args.arm],
        "count": len(fns),
        "budget": args.budget,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 1),
        "torch_version": torch.__version__,
        "mean_removed_pct": float(values.mean()),
        "worst_removed_pct": float(values.min()),
        "removed_pct": removed,
    }
    print(f"{args.benchmark}/{args.arm} seed {args.seed}: "
          f"mean {values.mean():.1f}%  worst {values.min():.1f}%  "
          f"({elapsed:.0f}s)")
    if args.output:
        args.output.write_text(json.dumps(out, indent=1))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
