"""Round 25: spatially local genome influence — anchors instead of a global context.

Round 24's problem-conditioned transformer (city coordinates as tokens,
genome as one global context vector) moved TSP from 15.7 to 11.1 but
decomposed badly: the untrained prior supplied ~13 and evolution added
only ~2, because every genome mutation shifts ALL city priorities at
once — the opposite of the segment-reversal locality that wins the tour
GA its 8.0.

This round restructures how the genome enters the decoder so that
mutations become spatially local. The 64 genes are read as 8 ANCHORS,
each with a position in the unit square (2 genes, through sigmoid) and a
6-d feature vector. Every city receives a conditioning vector from the
anchors near it (softmax over negative squared distance, bandwidth 0.15),
added to its coordinate embedding before the attention layers. Mutating
one anchor's features now edits priorities only in that anchor's region
— a local tour edit; mutating its position moves the region. Everything
else (fitness, arms, budget, instances) matches rounds 21-24.

Every solver arm's trajectory is decomposed into three checkpoints:
best after the first generation (32 evaluations — the prior), best when
exploration ended, and the final answer — so prior quality and climb
rate are measured separately.

Arms: traditional_tour_ga (8.00 to beat), solve_city_transformer
(round-24 champion: global genome context), solve_anchor_field (this
round: localized genome context).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round21_tsp import (make_instance, nearest_neighbor_length,
                                    traditional_tour_ga)
from benchmarks.round24_city_conditioned import CityConditionedTransformer
from latentspace.universal import solve


class AnchorFieldTransformer(nn.Module):
    """City-coordinate tokens conditioned by genome-defined spatial
    anchors: each city draws features from the anchors nearest to it, so
    a mutation to one anchor's genes perturbs one region of the tour."""

    def __init__(self, latent: int, output_shape: tuple, cities: np.ndarray,
                 width: int = 32, heads: int = 4, depth: int = 2,
                 anchors: int = 8, bandwidth: float = 0.15):
        super().__init__()
        assert output_shape == (len(cities),)
        assert latent % anchors == 0 and latent // anchors > 2
        self.anchors = anchors
        self.features = latent // anchors - 2
        self.bandwidth = bandwidth
        self.register_buffer(
            "coords", torch.as_tensor(cities, dtype=torch.float32))
        self.embed = nn.Linear(2, width)
        self.feature_proj = nn.Linear(self.features, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=2 * width,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(width, 1)

    def forward(self, z):
        batch = z.shape[0]
        genes = z.view(batch, self.anchors, 2 + self.features)
        positions = torch.sigmoid(genes[..., :2])          # (B, K, 2)
        projected = self.feature_proj(genes[..., 2:])      # (B, K, W)
        sq_dist = ((self.coords[None, :, None, :]
                    - positions[:, None, :, :]) ** 2).sum(-1)  # (B, N, K)
        weights = torch.softmax(-sq_dist / self.bandwidth**2, dim=-1)
        conditioning = weights @ projected                 # (B, N, W)
        tokens = self.embed(self.coords)[None] + conditioning
        return self.head(self.encoder(tokens)).flatten(1)


def solve_arm(cities: np.ndarray, budget: int, seed: int,
              architecture) -> dict:
    cache: dict[str, torch.Tensor] = {}

    def fitness(phenotypes: torch.Tensor) -> torch.Tensor:
        priorities = phenotypes.reshape(len(phenotypes), -1)
        key = str(priorities.device)
        if key not in cache:
            cache[key] = torch.as_tensor(cities, device=priorities.device)
        pts = cache[key][torch.argsort(priorities, dim=1)]
        return -(pts - pts.roll(-1, dims=1)).norm(dim=2).sum(dim=1)

    result = solve(fitness, output_shape=(len(cities),), budget=budget,
                   architecture=architecture, seed=seed)
    assert result.evaluations == budget
    explored = result.explore_evaluations
    return {
        "tour_length": float(-result.best_fitness),
        "explore_evaluations": explored,
        "after_first_generation": float(-result.history[31]),
        "after_explore": float(-result.history[explored - 1]),
    }


ARMS = ("traditional_tour_ga", "solve_city_transformer", "solve_anchor_field")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--cities", type=int, default=50)
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for seed in args.seeds:
        cities = make_instance(seed, args.cities)
        greedy = nearest_neighbor_length(cities)
        print(f"seed {seed}: nearest-neighbor greedy {greedy:.3f}", flush=True)

        builders = {
            "solve_city_transformer": lambda latent, shape, c=cities:
                CityConditionedTransformer(latent, shape, c),
            "solve_anchor_field": lambda latent, shape, c=cities:
                AnchorFieldTransformer(latent, shape, c),
        }
        for arm in args.arms:
            _seed_everything(seed)
            if arm == "traditional_tour_ga":
                row = {"tour_length": traditional_tour_ga(
                    cities, args.budget, seed)}
            else:
                row = solve_arm(cities, args.budget, seed, builders[arm])
            row.update({"arm": arm, "seed": seed, "greedy": greedy})
            rows.append(row)
            if "after_explore" in row:
                print(f"  {arm:<24} best tour {row['tour_length']:.4f} "
                      f"(prior {row['after_first_generation']:.2f} -> "
                      f"explore {row['after_explore']:.2f} "
                      f"@{row['explore_evaluations']} -> "
                      f"final {row['tour_length']:.2f})", flush=True)
            else:
                print(f"  {arm:<24} best tour {row['tour_length']:.4f}",
                      flush=True)

    print("\nmeans over seeds:")
    for arm in args.arms:
        vals = [r["tour_length"] for r in rows if r["arm"] == arm]
        print(f"  {arm:<24} {np.mean(vals):.4f} +- {np.std(vals, ddof=1):.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cities": args.cities, "budget": args.budget,
                   "torch_version": torch.__version__, "runs": rows}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
