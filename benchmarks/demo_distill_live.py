"""Live window for the distillation loop (experimental_distill_loop.run).

Watch the shared decoder's baseline rise as gradient distillation absorbs
evolution's vetted phenotypes. Tiles show each species' best-ever image.
Close the window to exit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from benchmarks.demo_image_species_vector import ReferenceSpeciesView
from benchmarks.experimental_distill_loop import run, Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-distill", action="store_true")
    args = parser.parse_args()

    from PIL import Image
    files = sorted(Path("/tmp/latentspace_cifar100_scaling_1024")
                   .glob("*.png"))[:args.count]
    targets = np.stack([np.asarray(Image.open(f), dtype=np.float32) / 255.0
                        for f in files])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    names = [f.stem for f in files]
    view = ReferenceSpeciesView(names, targets.transpose(0, 3, 1, 2),
                                args.epochs)
    initial = {}

    def progress(epoch, epochs, phenos, scores):
        for i, name in enumerate(names):
            if np.isfinite(scores[i]):
                initial.setdefault(name, -float(scores[i]))
        hall = [{
            "image": (np.zeros((3, 32, 32), np.float32) if phenos[i] is None
                      else phenos[i].reshape(32, 32, 3).transpose(2, 0, 1)),
            "score": float(scores[i]),
        } for i in range(len(names))]
        view.update(epoch, hall)
        known = [i for i, n in enumerate(names) if n in initial]
        removed = (np.mean([100 * (1 - (-scores[i]) / initial[names[i]])
                            for i in known]) if known else 0.0)
        print(f"  epoch {epoch:>5}  mean removed {removed:.1f}%", flush=True)

    cfg = Config(epochs=args.epochs)
    mean, _ = run([targets[i] for i in range(len(names))], (32, 32, 3), cfg,
                  seed=args.seed, device=device, distill=not args.no_distill,
                  progress=progress)
    print(f"final mean error removed: {mean:.1f}%")
    print("run finished — close the live window to exit")
    view.plt.show(block=True)


if __name__ == "__main__":
    main()
