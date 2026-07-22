"""Round 24: the problem-conditioned decoder — city coordinates as decoder input.

Rounds 21-23 established that no index-space decoder can search
permutations: exploration sticks at ~16 against the tour GA's 8 because
decoder mutations lack mutation-to-fitness locality (the tour GA's
segment reversal changes exactly two edge lengths). This round moves one
tier up the interface ladder: the decoder architecture now READS THE
SAME INSTANCE DATA THE FITNESS FUNCTION DOES — a transformer whose input
tokens are the 50 city coordinates, with the genome injected as context,
emitting one priority per city (random keys, argsorted by the fitness
function exactly as in round 21).

What this does and does not change. Evolution still only mutates genomes
and decoder weights; no operator touches a tour; the decoder never sees
a good tour or any answer — coordinates are public problem data, already
consumed by the fitness function. The interface ladder: tier 0, fitness
function only (rounds 21-23); tier 1, modality-shaped architecture (conv
on images, +24x); tier 2, problem-conditioned architecture (this round).

The mechanism being tested is the deep-image-prior analogue: an
UNTRAINED network reading coordinates computes spatially smooth
functions of position, so nearby cities get similar priorities from
generation zero ("visit clusters together"), and weight mutations
perturb a spatial rule rather than 50 unrelated numbers. The script
therefore first measures the prior in isolation: best tour among 32
untrained (genome, weights) samples, coordinate-conditioned vs plain
MLP random keys.

Arms (same instances, seeds, budget as rounds 21-23):
  traditional_tour_ga    — the standing baseline (8.00 to beat)
  solve_mlp              — round-21 reproduction, tier-0 reference
  solve_city_transformer — the tier-2 decoder through the packaged solver
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
                                    tour_lengths_np, traditional_tour_ga)
from latentspace.universal import solve
from latentspace.universal.explorer import _Template
from latentspace.universal.architectures import resolve


class CityConditionedTransformer(nn.Module):
    """Tokens are the city coordinates; the genome is a context vector
    added to every token; attention mixes them; a shared head emits one
    priority per city. Coordinates live in a buffer, not a parameter, so
    evolution cannot touch them."""

    def __init__(self, latent: int, output_shape: tuple, cities: np.ndarray,
                 width: int = 32, heads: int = 4, depth: int = 2):
        super().__init__()
        assert output_shape == (len(cities),)
        self.register_buffer(
            "coords", torch.as_tensor(cities, dtype=torch.float32))
        self.embed = nn.Linear(2, width)
        self.context = nn.Linear(latent, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=2 * width,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(width, 1)

    def forward(self, z):
        tokens = self.embed(self.coords)[None] + self.context(z)[:, None, :]
        return self.head(self.encoder(tokens)).flatten(1)


def untrained_prior_best(cities: np.ndarray, builder, seed: int,
                         samples: int = 32, latent: int = 64) -> float:
    """Best tour among `samples` completely untrained (genome, weights)
    draws — the architecture's prior, before any evolution."""
    rng = np.random.default_rng(seed)
    template = _Template(builder, "mps")
    best = np.inf
    for _ in range(samples):
        theta = template.init_theta(int(rng.integers(0, 2**31)))
        z = rng.standard_normal(latent).astype(np.float32)
        keys = template.decode(theta, z).cpu().numpy()[None]
        best = min(best, float(tour_lengths_np(
            cities, np.argsort(keys, axis=1))[0]))
    return best


def solve_arm(cities: np.ndarray, budget: int, seed: int,
              architecture) -> tuple[float, int]:
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
    return float(-result.best_fitness), result.explore_evaluations


ARMS = ("traditional_tour_ga", "solve_mlp", "solve_city_transformer")


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

        def city_builder(latent, output_shape, c=cities):
            return CityConditionedTransformer(latent, output_shape, c)

        _seed_everything(seed)
        prior_mlp = untrained_prior_best(
            cities, resolve("mlp", 64, (args.cities,)), seed)
        prior_city = untrained_prior_best(
            cities, lambda: city_builder(64, (args.cities,)), seed)
        print(f"seed {seed}: greedy {greedy:.3f} | untrained prior "
              f"best-of-32: mlp {prior_mlp:.3f}, "
              f"city-transformer {prior_city:.3f}", flush=True)

        for arm in args.arms:
            _seed_everything(seed)
            explored = None
            if arm == "traditional_tour_ga":
                length = traditional_tour_ga(cities, args.budget, seed)
            elif arm == "solve_mlp":
                length, explored = solve_arm(cities, args.budget, seed, "mlp")
            else:
                length, explored = solve_arm(cities, args.budget, seed,
                                             city_builder)
            rows.append({"arm": arm, "seed": seed, "tour_length": length,
                         "greedy": greedy, "explore_evaluations": explored,
                         "prior_mlp": prior_mlp, "prior_city": prior_city})
            note = f" (explored {explored})" if explored is not None else ""
            print(f"  {arm:<24} best tour {length:.4f}{note}", flush=True)

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
