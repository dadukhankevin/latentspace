"""Live window for the LIBRARY's unified solve() — many images, one call.

Same view as demo_image_lazy_population's --live, but everything on screen
is `latentspace.universal.solve([fitness_fns...])` with shipped defaults:
one shared conditional decoder, rare compatibility-gated crossover, and
the self-tuning genes/latents mutation ratio. The window stays open after
the run; close it to exit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from benchmarks.demo_image_species_vector import ReferenceSpeciesView
from latentspace.universal import solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-dir",
                        default="/tmp/latentspace_cifar100_scaling_1024")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=48_000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--live-targets", type=int, default=16,
                        help="how many problems the window tracks; the run "
                             "always solves --count of them")
    args = parser.parse_args()

    from PIL import Image
    files = sorted(Path(args.targets_dir).glob("*.png"))[:args.count]
    targets = np.stack([np.asarray(Image.open(f), dtype=np.float32) / 255.0
                        for f in files])                    # (N, 32, 32, 3)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    fns = []
    for t in targets:
        flat = torch.as_tensor(t.reshape(-1), device=device)
        fns.append(lambda ph, f=flat: -((ph - f) ** 2).mean(dim=1))
    names = [f.stem for f in files]
    n_view = min(args.live_targets, len(names))
    initial = {}

    view = ReferenceSpeciesView(
        names[:n_view], targets[:n_view].transpose(0, 3, 1, 2), args.budget)

    def progress(spent, budget, phenos, fits):
        for name, fit in zip(names, fits):
            initial.setdefault(name, -float(fit))
        hall = [{
            "image": phenos[i].reshape(32, 32, 3).transpose(2, 0, 1),
            "score": float(fits[i]),
        } for i in range(n_view)]
        view.update(spent, hall)
        removed = np.mean([100 * (1 - (-fits[i]) / initial[names[i]])
                           for i in range(len(names))])
        print(f"  {spent:>7} evals  mean removed {removed:.1f}% "
              f"(all {len(names)})", flush=True)

    result = solve(fns, output_shape=(32, 32, 3), budget=args.budget,
                   seed=args.seed, progress=progress)
    removed = [100 * (1 - (-p.best_fitness) / -p.initial_fitness)
               for p in result.problems]
    print(f"final: mean {np.mean(removed):.1f}%  "
          f"best {max(removed):.1f}%  hardest {min(removed):.1f}%")
    ratios = [h["latent_ratio"] for h in result.history]
    print(f"latents/genes mutation ratio self-tuned to {ratios[-1]:.2f}")
    print("run finished — close the live window to exit")
    view.plt.show(block=True)


if __name__ == "__main__":
    main()
