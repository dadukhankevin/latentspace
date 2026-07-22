"""The apple, live, through THE universal GA — one fitness function.

The flagship single-fitness benchmark: a 96x96 photo, MSE fitness. The
retired per-individual engine's records are the bar: 0.004566 at 120k
evaluations (round 31), 0.00178 at 150k (round 50, the all-time record).
This run shows what the redesigned engine — genes + latents on one shared
decoder, Adam fold — does on the same target.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.demo_image_species_vector import ReferenceSpeciesView
from latentspace.universal import solve

DEMO = Path(__file__).resolve().parent.parent / "demo/apple_demo_recovered.json"


def load_apple() -> np.ndarray:
    from PIL import Image
    payload = json.loads(DEMO.read_text())
    encoded = payload["imgs"]["target"].split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0         # (96, 96, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=9_000)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    target = load_apple()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    flat = torch.as_tensor(target.reshape(-1), device=device)

    def fitness(phenotypes):
        return -((phenotypes.reshape(len(phenotypes), -1) - flat) ** 2
                 ).mean(dim=1)

    view = ReferenceSpeciesView(
        ["apple"], target.transpose(2, 0, 1)[None], args.epochs)

    def progress(epoch, epochs, spent, phenos, scores):
        if phenos[0] is None:
            return
        view.update(epoch, [{
            "image": phenos[0].reshape(96, 96, 3).transpose(2, 0, 1),
            "score": float(scores[0]),
        }])
        print(f"  epoch {epoch:>6}  {spent:>7} evals  "
              f"mse {-scores[0]:.6f}", flush=True)

    result = solve(fitness, output_shape=(96, 96, 3), epochs=args.epochs,
                   seed=args.seed, progress=progress)
    mse = -result.best_fitness
    print(f"final: mse {mse:.6f} in {result.evaluations} evaluations")
    print("legacy bars: 0.004566 @ 120k (round 31), "
          "0.001780 @ 150k (round 50, all-time)")
    print("run finished — close the live window to exit")
    view.plt.show(block=True)


if __name__ == "__main__":
    main()
