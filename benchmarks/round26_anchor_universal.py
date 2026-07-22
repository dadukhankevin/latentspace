"""Round 26: is the anchor field a universal genome grammar, or a TSP trick?

Round 25 won 50-city TSP by changing how the genome ENTERS the decoder:
64 genes read as 8 anchors, each with a location in the space the solution
lives in (2 genes through a sigmoid) and a message (6 genes); every city
draws conditioning from the anchors near it (softmax over negative squared
distance, bandwidth 0.15). Mutating one anchor edits one region.

Strip the word "city" out of that sentence and nothing about it is about
TSP. This round tests whether the SAME grammar — same 64 genes, same 8
anchors, same 0.15 bandwidth — drives the two problems the campaign
already won with a different decoder:

  * blob2d_1024  — 32x32 image; champion decoder is conv2d;
  * smooth1d_256 — low-frequency curve; champion decoder is conv1d.

If one grammar works on tours, images and curves, the universal genetic
code stops being "a flat vector we agree to mutate the same way" and
becomes a code with loci: genes 0-1 mean WHERE, genes 2-7 mean WHAT, in
every problem. If it loses here, locality has to be expressed per
modality (convolution for images, anchors for tours) and there is no
single grammar — which is just as sharp a result.

The decoders are a direct translation of round 25's, not a redesign:
round 25 built city tokens as `embed(coords) + anchor_conditioning`, mixed
them with a transformer, and read one logit per city with a shared head.
Here every pixel (or sample) gets `embed(coords) + anchor_conditioning`,
a convolutional trunk mixes them, and a shared head reads one logit per
site. Only the trunk is modality-shaped, which is the allowance the
campaign already grants (conv for images, attention for sets).

Arms: direct_ga (traditional baseline), conv_decoder (the current
champion, genome enters through a dense layer touching every cell),
anchor_field (this round: genome enters through localized anchors).

Honest confounds, to be resolved separately if this wins:
  * blob2d's target IS three Gaussian blobs, and an anchor field is
    literally a sum of localized sources — the prior may fit this family
    for reasons that will not transfer to a photograph;
  * the anchor decoder has far fewer weights than conv2d (no dense
    latent->feature-map layer), which independently helps weight
    evolution. Locality and parameter count cannot be separated here;
    they are the same design decision.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from benchmarks.compare import (BenchmarkConfig, _require_mps,
                                _seed_everything, run_direct_ga)
from benchmarks.round3_structure import SmoothTarget
from benchmarks.round8_mlp_pretrain import BlobImage2D
from latentspace.universal import solve

CHANNELS = 16
ANCHORS = 8
BANDWIDTH = 0.15

OBJECTIVES = {
    "smooth1d_256": (SmoothTarget, (256,), "conv1d"),
    "blob2d_1024": (BlobImage2D, (32, 32), "conv2d"),
}


def _site_grid(output_shape: tuple) -> np.ndarray:
    """Cell-centre coordinates of every output site, in [0, 1]^D."""
    axes = [(np.arange(n) + 0.5) / n for n in output_shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], axis=1).astype(np.float32)


class AnchorField(nn.Module):
    """The candidate universal genome grammar.

    The genome's genes are read as K anchors. Each anchor has a location
    in the space the solution lives in (one gene per spatial dimension,
    through a sigmoid) and a message (the remaining genes). Every site in
    that space draws a conditioning vector from the anchors near it,
    softmax-weighted by negative squared distance. Mutating one anchor's
    message edits one region of the solution; mutating its location moves
    that region.

    Round 25 applied this to cities in the plane. Nothing in it is about
    cities: `coords` is any set of site positions.
    """

    def __init__(self, latent: int, coords: np.ndarray, width: int,
                 anchors: int = ANCHORS, bandwidth: float = BANDWIDTH):
        super().__init__()
        dims = coords.shape[1]
        per_anchor = latent // anchors
        assert per_anchor > dims, "each anchor needs a location and a message"
        self.anchors = anchors
        self.dims = dims
        self.features = per_anchor - dims
        self.bandwidth = bandwidth
        self.register_buffer(
            "coords", torch.as_tensor(coords, dtype=torch.float32))
        self.proj = nn.Linear(self.features, width)

    def forward(self, z):
        batch = z.shape[0]
        used = self.anchors * (self.dims + self.features)
        genes = z[:, :used].view(batch, self.anchors, self.dims + self.features)
        positions = torch.sigmoid(genes[..., :self.dims])       # (B, K, D)
        messages = self.proj(genes[..., self.dims:])            # (B, K, W)
        sq_dist = ((self.coords[None, :, None, :]
                    - positions[:, None, :, :]) ** 2).sum(-1)   # (B, N, K)
        weights = torch.softmax(-sq_dist / self.bandwidth**2, dim=-1)
        return weights @ messages                               # (B, N, W)


class _AnchorConv(nn.Module):
    """Site coordinates embedded and conditioned by the anchor field, then
    mixed by a convolutional trunk at full output resolution.

    The champion conv decoder reaches its smooth prior by painting a small
    feature map from a dense layer and upsampling it; here the anchor
    field paints the feature map directly at full resolution — a morphogen
    gradient is already smooth — and the trunk only refines it. That is
    why there is no upsampling stage: the two decoders reach a
    locally-coherent prior by different routes, which is the comparison.
    """

    def __init__(self, latent: int, output_shape: tuple, depth: int,
                 kernel: int, anchors: int = ANCHORS,
                 bandwidth: float = BANDWIDTH, channels: int = CHANNELS):
        super().__init__()
        self.output_shape = output_shape
        conv_cls = nn.Conv2d if len(output_shape) == 2 else nn.Conv1d
        coords = _site_grid(output_shape)
        self.field = AnchorField(latent, coords, channels, anchors, bandwidth)
        self.embed = nn.Linear(coords.shape[1], channels)
        blocks: list[nn.Module] = []
        for _ in range(depth):
            blocks += [conv_cls(channels, channels, kernel,
                                padding=kernel // 2), nn.LeakyReLU()]
        blocks += [conv_cls(channels, 1, kernel, padding=kernel // 2)]
        self.convs = nn.Sequential(*blocks)

    def forward(self, z):
        sites = self.embed(self.field.coords)[None] + self.field(z)  # (B,N,C)
        grid = sites.transpose(1, 2).reshape(
            z.shape[0], -1, *self.output_shape)
        return self.convs(grid).flatten(1)


def build_anchor(latent: int, output_shape: tuple) -> nn.Module:
    """Modality picks the trunk (2-D convs for images, 1-D for signals);
    the genome grammar is identical either way."""
    if len(output_shape) == 2:
        return _AnchorConv(latent, output_shape, depth=3, kernel=3)
    return _AnchorConv(latent, output_shape, depth=4, kernel=5)


def solve_arm(objective, output_shape, budget, seed, architecture,
              explore_fraction="auto") -> dict:
    def fitness(phenotypes: torch.Tensor) -> torch.Tensor:
        return -objective.loss_tensor(phenotypes.reshape(len(phenotypes), -1))

    result = solve(fitness, output_shape=output_shape, budget=budget,
                   architecture=architecture, seed=seed,
                   explore_fraction=explore_fraction)
    assert result.evaluations == budget
    explored = result.explore_evaluations
    return {
        "mse": float(objective.loss_numpy(
            result.best_phenotype.reshape(1, -1))[0]),
        "explore_evaluations": explored,
        "after_first_generation": float(-result.history[31]),
        "after_explore": float(-result.history[explored - 1]),
    }


ARMS = ("direct_ga", "conv_decoder", "anchor_field")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES,
                        default=list(OBJECTIVES))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument(
        "--explore-fraction", default="auto",
        help="'auto' lets each arm's stall detector pick its own split; a "
             "float fixes the same split for every arm, which is what "
             "isolates the genome grammar from the scheduler.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.explore_fraction != "auto":
        args.explore_fraction = float(args.explore_fraction)
    return args


def main():
    args = parse_args()
    _require_mps()
    config = BenchmarkConfig(evaluation_budget=args.budget)
    rows = []
    for name in args.objectives:
        factory, output_shape, conv_name = OBJECTIVES[name]
        print(f"\n=== {name} (shape {output_shape}, budget {args.budget}) ===",
              flush=True)
        for seed in args.seeds:
            for arm in args.arms:
                objective = factory()
                _seed_everything(seed)
                if arm == "direct_ga":
                    result = run_direct_ga(objective, seed, config)
                    row = {"mse": float(result.metric_at_budget)}
                else:
                    architecture = (conv_name if arm == "conv_decoder"
                                    else build_anchor)
                    row = solve_arm(objective, output_shape, args.budget,
                                    seed, architecture, args.explore_fraction)
                row.update({"objective": name, "arm": arm, "seed": seed})
                rows.append(row)
                if "after_explore" in row:
                    print(f"  seed {seed} {arm:<14} mse {row['mse']:.6g} "
                          f"(prior {row['after_first_generation']:.4g} -> "
                          f"explore {row['after_explore']:.4g} "
                          f"@{row['explore_evaluations']} -> "
                          f"final {row['mse']:.4g})", flush=True)
                else:
                    print(f"  seed {seed} {arm:<14} mse {row['mse']:.6g}",
                          flush=True)

        print(f"  --- means over {len(args.seeds)} seeds ---")
        for arm in args.arms:
            vals = [r["mse"] for r in rows
                    if r["arm"] == arm and r["objective"] == name]
            spread = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            print(f"  {arm:<14} {np.mean(vals):.6g} +- {spread:.3g}",
                  flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": asdict(config), "anchors": ANCHORS,
                   "bandwidth": BANDWIDTH, "channels": CHANNELS,
                   "explore_fraction": args.explore_fraction,
                   "torch_version": torch.__version__, "runs": rows}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
