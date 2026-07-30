"""The apple, live, through THE universal GA — one fitness function.

The flagship single-fitness benchmark: a 96x96 photo, MSE fitness. The
retired per-individual engine's records are the bar: 0.004566 at 120k
evaluations (round 31), 0.00178 at 150k (round 50, the all-time record).
This run shows what the current engine — genes + latents on one shared
decoder, library defaults (sparse-shared patches; distillation is
multi-function-only, so it does not fire here) — does on the same target.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch

from latentspace.universal import solve

DEMO = Path(__file__).resolve().parent.parent / "demo/apple_demo_recovered.json"


def load_apple() -> np.ndarray:
    from PIL import Image
    payload = json.loads(DEMO.read_text())
    encoded = payload["imgs"]["target"].split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0         # (96, 96, 3)


class AppleView:
    """Target and evolved side by side, MSE curve underneath. One image,
    one curve, no dead space."""

    def __init__(self, target):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig = plt.figure(figsize=(9, 5.5), layout="constrained")
        self.fig.canvas.manager.set_window_title("apple — live")
        grid = self.fig.add_gridspec(2, 2, height_ratios=[3, 2])
        ax_t = self.fig.add_subplot(grid[0, 0])
        ax_t.imshow(target)
        ax_t.set_title("target", fontsize=11)
        self.ax_e = self.fig.add_subplot(grid[0, 1])
        self.im = self.ax_e.imshow(target * 0)
        self.ax_e.set_title("evolved", fontsize=11)
        for ax in (ax_t, self.ax_e):
            ax.set_xticks([]); ax.set_yticks([])
        self.ax_c = self.fig.add_subplot(grid[1, :])
        self.xs, self.ys = [], []
        plt.show(block=False)

    def update(self, spent, image, mse):
        self.im.set_data(image.clip(0, 1))
        self.ax_e.set_title(f"evolved — mse {mse:.4f}", fontsize=11)
        self.xs.append(spent); self.ys.append(mse)
        self.ax_c.clear()
        self.ax_c.semilogy(self.xs, self.ys, color="#c2703a", lw=1.6)
        self.ax_c.set_xlabel("evaluations")
        self.ax_c.set_ylabel("best MSE (log)")
        for side in ("top", "right"):
            self.ax_c.spines[side].set_visible(False)
        self.plt.pause(0.001)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=9_000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--directions", default=None,
                        choices=("frozen", "sparse", "sparse-shared",
                                 "individual", "evolve"),
                        help="omit to use the library default (sparse-shared)")
    parser.add_argument("--latents", type=int, default=None,
                        help="omit for the library default (per substrate: "
                             "patch K=2048 sparse, 64 low-rank)")
    args = parser.parse_args()

    target = load_apple()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    flat = torch.as_tensor(target.reshape(-1), device=device)

    def fitness(phenotypes):
        return -((phenotypes.reshape(len(phenotypes), -1) - flat) ** 2
                 ).mean(dim=1)

    view = AppleView(target)

    def progress(epoch, epochs, spent, phenos, scores):
        if phenos[0] is None:
            return
        view.update(spent, phenos[0].reshape(96, 96, 3), -float(scores[0]))
        print(f"  epoch {epoch:>6}  {spent:>7} evals  "
              f"mse {-scores[0]:.6f}", flush=True)

    overrides = {}
    if args.directions is not None:
        overrides["directions"] = args.directions
    if args.latents is not None:
        overrides["latents"] = args.latents
    result = solve(fitness, output_shape=(96, 96, 3), epochs=args.epochs,
                   seed=args.seed, progress=progress, **overrides)
    mse = -result.best_fitness
    print(f"final: mse {mse:.6f} in {result.evaluations} evaluations")
    print("legacy bars: 0.004566 @ 120k (round 31), "
          "0.001780 @ 150k (round 50, all-time)")
    print("run finished — close the live window to exit")
    view.plt.show(block=True)


if __name__ == "__main__":
    main()
