"""Round 28: can the anchor grammar take the photo crown from conv?

Round 27 settled the CMA question (pure decoder evolution beats every
CMA variant on the apple at 150k) but left conv on top: pure conv
evolution's recorded 0.0049 vs pure anchor evolution's 0.0080. The
hypothesis for the gap: a photograph's fine texture rewards
convolution's translation-invariant local filters, and round 27's
anchor decoder painted its conditioning field at full resolution with a
thin trunk (~7.5k weights) — no multiscale texture machinery at all.

This round tests whether the gap is the GRAMMAR's fault or the TRUNK's,
by keeping the genome's entry point identical (64 genes = 8 anchors,
locations + messages, bandwidth 0.15, unchanged since round 25) and
varying only what processes the painted field. All arms are pure
per-individual evolution — no distill, no CMA (round 27's ruling).

  * anchor_flat    — round 27's decoder verbatim (16 channels, depth 3,
                     full resolution): the 0.0080 incumbent, rerun only
                     at stage-1 budget for trajectory comparison.
  * anchor_wide    — same shape, twice the channels (32) and depth 4:
                     is it just capacity?
  * anchor_pyramid — anchors paint a coarse 24x24 field that upsample+
                     conv stages refine to 96x96: conv's multiscale
                     texture machinery, genome still entering through
                     anchors only.
  * conv_rgb       — the conv champion rebuilt for RGB in this harness
                     (dense latent -> 6x6 feature map -> upsample+conv
                     to 96x96): validates the recorded 0.0049 under
                     identical evolution code, and is the score to beat.

Stage 1 races all arms at a short budget to rank trajectories; the
winners go to 150k against the recorded references.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round26_anchor_universal import AnchorField, _site_grid
from benchmarks.round27_apple_no_cma import DEMO, AnchorRGB, load_apple
from latentspace.universal import solve

LATENT = 64


class AnchorPyramid(nn.Module):
    """Anchors paint a coarse feature field; upsample+conv stages refine
    it to full resolution. The genome's entry point is unchanged — only
    the trunk gains conv's multiscale texture machinery."""

    def __init__(self, latent: int, output_shape: tuple, base: int = 24,
                 channels: int = 16):
        super().__init__()
        colors, height, width = output_shape
        assert height % base == 0 and width % base == 0
        coords = _site_grid((base, base))
        self.base = base
        self.channels = channels
        self.field = AnchorField(latent, coords, channels)
        self.embed = nn.Linear(2, channels)
        doublings = int(np.log2(height // base))
        blocks: list[nn.Module] = []
        for _ in range(doublings):
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(channels, channels, 3, padding=1),
                       nn.LeakyReLU()]
        blocks += [nn.Conv2d(channels, colors, 3, padding=1)]
        self.convs = nn.Sequential(*blocks)

    def forward(self, z):
        sites = self.embed(self.field.coords)[None] + self.field(z)
        grid = sites.transpose(1, 2).reshape(
            z.shape[0], self.channels, self.base, self.base)
        return self.convs(grid).flatten(1)


class ConvRGB(nn.Module):
    """The conv champion's shape for an RGB target: dense latent -> small
    feature map -> upsample+conv stages. The genome enters through a
    dense layer touching every feature-map cell — no anchors."""

    def __init__(self, latent: int, output_shape: tuple, base: int = 6,
                 channels: int = 16):
        super().__init__()
        colors, height, width = output_shape
        self.base = base
        self.channels = channels
        self.fc = nn.Linear(latent, channels * base * base)
        doublings = int(np.log2(height // base))
        blocks: list[nn.Module] = []
        for _ in range(doublings):
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(channels, channels, 3, padding=1),
                       nn.LeakyReLU()]
        blocks += [nn.Conv2d(channels, colors, 3, padding=1)]
        self.convs = nn.Sequential(*blocks)

    def forward(self, z):
        grid = self.fc(z).view(-1, self.channels, self.base, self.base)
        return self.convs(grid).flatten(1)


BUILDERS = {
    "anchor_flat": lambda latent, shape: AnchorRGB(latent, shape),
    "anchor_wide": lambda latent, shape: AnchorRGB(latent, shape,
                                                   channels=32, depth=4),
    "anchor_pyramid": AnchorPyramid,
    "conv_rgb": ConvRGB,
}


def run_arm(arm: str, target: np.ndarray, budget: int, seed: int) -> dict:
    target_t = torch.as_tensor(target, device="mps")

    def fitness(phenotypes: torch.Tensor) -> torch.Tensor:
        return -((phenotypes.flatten(1) - target_t) ** 2).mean(dim=1)

    _seed_everything(seed)
    result = solve(fitness, output_shape=(3, 96, 96), budget=budget,
                   architecture=BUILDERS[arm], latent=LATENT,
                   explore_fraction=1.0, seed=seed)
    assert result.evaluations == budget
    assert result.explore_evaluations == budget, "CMA must never run here"
    return {"mse": float(-result.best_fitness),
            "history": [float(-h) for h in result.history]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=BUILDERS,
                        default=list(BUILDERS))
    parser.add_argument("--budget", type=int, default=50_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--history-step", type=int, default=300)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    target = load_apple()
    recorded = json.loads(DEMO.read_text())["D"]["finalMse"]
    for arm in args.arms:
        net = BUILDERS[arm](LATENT, (3, 96, 96))
        print(f"{arm}: {sum(p.numel() for p in net.parameters()):,} weights",
              flush=True)
    print(f"recorded 150k refs: conv evolution {recorded['cf']}, hand-off "
          f"stack {recorded['stack']}, GA {recorded['ga']}", flush=True)

    rows = []
    for seed in args.seeds:
        for arm in args.arms:
            outcome = run_arm(arm, target, args.budget, seed)
            curve = outcome.pop("history")
            checkpoints = {str(i): curve[i]
                           for i in range(0, len(curve), args.history_step)}
            checkpoints[str(len(curve) - 1)] = curve[-1]
            for mark in (5_000, 25_000, 50_000, 100_000):
                if mark <= len(curve):
                    print(f"  seed {seed} {arm:<15} best mse at {mark:>7}: "
                          f"{curve[mark - 1]:.6f}", flush=True)
            print(f"  seed {seed} {arm:<15} FINAL {outcome['mse']:.6f} "
                  f"({args.budget} evaluations)", flush=True)
            rows.append({"arm": arm, "seed": seed, **outcome,
                         "checkpoints": checkpoints})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"budget": args.budget, "latent": LATENT,
                   "recorded_150k_references": recorded,
                   "torch_version": torch.__version__, "runs": rows}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
