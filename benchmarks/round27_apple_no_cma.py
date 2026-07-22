"""Round 27: the apple at 150k — pure anchor evolution vs CMA-as-baseline.

Daniel's goal ruling for this round and onward: the deliverable is a
neural-decoder GA that BEATS CMA-ES, not a stack that uses it. CMA-ES
appears here strictly as an opponent arm.

The published apple note (loseylabs.ai) recorded, at 150,000 evaluations
on a 96x96 RGB photo: pure conv decoder evolution 0.004929, the
distill->CMA hand-off stack 0.011163 (CMA sprints ~25k evals then
flattens at its frozen gene space's ceiling), traditional GA 0.120026.
The ceiling had a concrete cause: the leaf region never entered the
elite archive, so the PCA gene space literally could not express it.

The anchor genome is NOT distilled from an archive — it is structural
(8 sources with positions and messages), defined before any evolution.
An anchor can move to the leaf. So round 27 separates two hypotheses the
hand-off result confounded: was the ceiling CMA's fault, or the frozen
archive-derived gene space's?

Arms (same target, same budget, same MSE as the recorded run):

  * anchor_evolution — the champion: per-individual anchor decoders
    (genome + private weights, mutate both), NO distill, NO CMA-ES.
  * cma_anchor_genes — the baseline: one UNTRAINED anchor decoder with
    weights frozen at initialization; CMA-ES searches its 64 genes.
    If this also flattens near 25k, the ceiling follows CMA; if it keeps
    descending, the apple ceiling was the frozen archive's, and the
    anchor grammar gives even CMA room to move.

Recorded references (deterministic same target, from the demo JSON) are
written into the output for the scoreboard; they are single-run, as this
round's arms are.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round26_anchor_universal import AnchorField, _site_grid
from latentspace.universal import solve
from latentspace.universal.cma import cma_minimize

CHANNELS = 16
LATENT = 64
DEMO = Path(__file__).resolve().parent.parent / "demo/apple_demo_recovered.json"


def load_apple() -> np.ndarray:
    """The demo target as a flat (3*96*96,) float array in [0, 1],
    channels-first to match the decoder's output layout."""
    from PIL import Image

    payload = json.loads(DEMO.read_text())
    encoded = payload["imgs"]["target"].split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0        # (96, 96, 3)
    return array.transpose(2, 0, 1).reshape(-1)


class AnchorRGB(nn.Module):
    """Round 26's anchor grammar (unchanged constants) with a conv trunk
    reading three logits per pixel instead of one."""

    def __init__(self, latent: int, output_shape: tuple,
                 channels: int = CHANNELS, depth: int = 3):
        super().__init__()
        colors, height, width = output_shape
        coords = _site_grid((height, width))
        self.plane = (height, width)
        self.field = AnchorField(latent, coords, channels)
        self.embed = nn.Linear(2, channels)
        blocks: list[nn.Module] = []
        for _ in range(depth):
            blocks += [nn.Conv2d(channels, channels, 3, padding=1),
                       nn.LeakyReLU()]
        blocks += [nn.Conv2d(channels, colors, 3, padding=1)]
        self.convs = nn.Sequential(*blocks)

    def forward(self, z):
        sites = self.embed(self.field.coords)[None] + self.field(z)
        grid = sites.transpose(1, 2).reshape(z.shape[0], -1, *self.plane)
        return self.convs(grid).flatten(1)


def run_anchor_evolution(target: np.ndarray, budget: int, seed: int) -> dict:
    target_t = torch.as_tensor(target, device="mps")

    def fitness(phenotypes: torch.Tensor) -> torch.Tensor:
        return -((phenotypes.flatten(1) - target_t) ** 2).mean(dim=1)

    result = solve(fitness, output_shape=(3, 96, 96), budget=budget,
                   architecture=lambda latent, shape: AnchorRGB(latent, shape),
                   latent=LATENT, explore_fraction=1.0, seed=seed)
    assert result.evaluations == budget
    assert result.explore_evaluations == budget, "CMA must never run here"
    return {"mse": float(-result.best_fitness),
            "history": [float(-h) for h in result.history]}


def run_cma_anchor_genes(target: np.ndarray, budget: int, seed: int) -> dict:
    _seed_everything(seed)
    net = AnchorRGB(LATENT, (3, 96, 96)).to("mps")
    for p in net.parameters():
        p.requires_grad_(False)
    target_t = torch.as_tensor(target, device="mps")
    rng = np.random.default_rng(seed)
    history: list[float] = []
    best = np.inf

    def evaluate_batch(zs: np.ndarray) -> np.ndarray:
        nonlocal best
        genes = torch.as_tensor(zs.astype(np.float32), device="mps")
        with torch.no_grad():
            phenotypes = torch.sigmoid(net(genes))
        losses = ((phenotypes - target_t) ** 2).mean(dim=1).cpu().numpy()
        for loss in losses:
            best = min(best, float(loss))
            history.append(best)
        return losses.astype(np.float64)

    cma_minimize(evaluate_batch, dim=LATENT, budget_evaluations=budget,
                 evaluations_done=0, rng=rng,
                 mean0=np.zeros(LATENT), sigma0=1.0)
    return {"mse": best, "history": history}


ARMS = {"anchor_evolution": run_anchor_evolution,
        "cma_anchor_genes": run_cma_anchor_genes}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--budget", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-step", type=int, default=300)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    target = load_apple()
    recorded = json.loads(DEMO.read_text())["D"]["finalMse"]
    print(f"apple target loaded ({target.size} values); recorded 150k refs: "
          f"conv evolution {recorded['cf']}, hand-off stack "
          f"{recorded['stack']}, traditional GA {recorded['ga']}", flush=True)

    rows = []
    for arm in args.arms:
        _seed_everything(args.seed)
        outcome = ARMS[arm](target, args.budget, args.seed)
        curve = outcome.pop("history")
        checkpoints = {str(i): curve[i]
                       for i in range(0, len(curve), args.history_step)}
        checkpoints[str(len(curve) - 1)] = curve[-1]
        for mark in (5_000, 25_000, 50_000, 100_000):
            if mark <= len(curve):
                print(f"  {arm:<18} best mse at {mark:>7}: "
                      f"{curve[mark - 1]:.6f}", flush=True)
        print(f"  {arm:<18} FINAL {outcome['mse']:.6f} "
              f"({args.budget} evaluations)", flush=True)
        rows.append({"arm": arm, "seed": args.seed, **outcome,
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
