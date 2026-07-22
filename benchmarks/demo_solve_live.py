"""Live window for THE universal GA — solve(fitness_fns, output_shape, epochs).

Everything on screen is the redesigned engine with shipped defaults: seeded
species, fitness shares, assortative selection with rare outcrossing, the
two-space mutation dials, and the Adam fold absorbing species' consensus
into the one shared decoder. Tiles show each function's best-ever solution
from the bookkeeping archive. The window stays open after the run; close it
to exit.
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
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--live-targets", type=int, default=16)
    parser.add_argument("--directions", default="frozen",
                        choices=("frozen", "sparse", "individual", "evolve"))
    parser.add_argument("--latents", type=int, default=64,
                        help="latent size; with --directions sparse this is "
                             "the weight-patch size K")
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
        names[:n_view], targets[:n_view].transpose(0, 3, 1, 2), args.epochs)

    def progress(epoch, epochs, spent, phenos, scores):
        for name, sc in zip(names, scores):
            if np.isfinite(sc):
                initial.setdefault(name, -float(sc))
        hall = [{
            "image": (np.zeros((3, 32, 32), dtype=np.float32)
                      if phenos[i] is None
                      else phenos[i].reshape(32, 32, 3).transpose(2, 0, 1)),
            "score": float(scores[i]),
        } for i in range(n_view)]
        view.update(epoch, hall)
        known = [i for i, name in enumerate(names) if name in initial]
        removed = np.mean([100 * (1 - (-scores[i]) / initial[names[i]])
                           for i in known]) if known else 0.0
        print(f"  epoch {epoch:>5}  {spent:>7} evals  "
              f"mean removed {removed:.1f}%", flush=True)

    result = solve(fns, output_shape=(32, 32, 3), epochs=args.epochs,
                   seed=args.seed, directions=args.directions,
                   latents=args.latents, progress=progress)
    removed = [100 * (1 - (-p.best_fitness) / -p.initial_fitness)
               for p in result.problems if p.best_phenotype is not None]
    print(f"final: mean {np.mean(removed):.1f}%  best {max(removed):.1f}%  "
          f"hardest {min(removed):.1f}%  ({result.evaluations} evals)")
    print("run finished — close the live window to exit")
    view.plt.show(block=True)


if __name__ == "__main__":
    main()
