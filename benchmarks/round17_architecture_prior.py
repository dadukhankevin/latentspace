"""Round 17: modality lives in the decoder's ARCHITECTURE, not the operators.

Daniel's ruling: the universal algorithm always evolves (genome, decoder
weights) with the same tensor operators — but the decoder network itself
may be shaped like its output modality (CNNs for images, 1-D convolutions
for signals, transformers for sequences, ...), just as the fitness function
already is. This round measures what that buys with ONLY a fitness function
— no practice problems, no pretraining.

The mechanism being bought: an UNTRAINED convolutional network is already
biased toward locally-coherent, smooth outputs (the "deep image prior"), so
a population of per-individual conv decoders starts inside plausible-output
territory and its weight mutations move phenotypes in structured, local
ways. The dense-MLP baseline spends its budget discovering what the conv
architecture gets for free.

Arms (identical evolution — per-individual decoder mutation, same rates):

  * mlp_decoder  — private decoders are dense MLPs (round-15 baseline);
  * conv_decoder — private decoders are small upsampling conv nets shaped
                   like the output (2-D for the image, 1-D for the curve);
  * direct_ga    — traditional GA reference (modality-specific mutation).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from benchmarks.compare import (
    BenchmarkConfig,
    Objective,
    TrackedFitness,
    _finish_result,
    _require_mps,
    _seed_everything,
    print_summary,
    run_direct_ga,
    summarize,
)
from benchmarks.round3_structure import SmoothTarget
from benchmarks.round8_mlp_pretrain import BlobImage2D
from benchmarks.round15_individual_decoders import (
    ELITE,
    HIDDEN,
    LATENT,
    POPULATION,
    _mutate_theta,
    _mutate_z,
)

OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "blob2d_1024": BlobImage2D,
}

CHANNELS = 16


def _build_mlp(dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(LATENT, HIDDEN), nn.LeakyReLU(),
        nn.Linear(HIDDEN, dim),
    )


class _Conv2dDecoder(nn.Module):
    """latent -> 4x4 feature map -> upsample+conv to side x side logits."""

    def __init__(self, dim: int):
        super().__init__()
        side = int(math.isqrt(dim))
        assert side * side == dim
        doublings = int(math.log2(side // 4))
        self.fc = nn.Linear(LATENT, CHANNELS * 4 * 4)
        blocks: list[nn.Module] = []
        for _ in range(doublings):
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1),
                nn.LeakyReLU(),
            ]
        blocks += [nn.Conv2d(CHANNELS, 1, 3, padding=1)]
        self.convs = nn.Sequential(*blocks)

    def forward(self, z):
        x = self.fc(z).view(-1, CHANNELS, 4, 4)
        return self.convs(x).flatten(1)


class _Conv1dDecoder(nn.Module):
    """latent -> 16-point feature track -> upsample+conv to dim logits."""

    def __init__(self, dim: int):
        super().__init__()
        doublings = int(math.log2(dim // 16))
        self.fc = nn.Linear(LATENT, CHANNELS * 16)
        blocks: list[nn.Module] = []
        for _ in range(doublings):
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(CHANNELS, CHANNELS, 5, padding=2),
                nn.LeakyReLU(),
            ]
        blocks += [nn.Conv1d(CHANNELS, 1, 5, padding=2)]
        self.convs = nn.Sequential(*blocks)

    def forward(self, z):
        x = self.fc(z).view(-1, CHANNELS, 16)
        return self.convs(x).flatten(1)


def _build_conv(objective_name: str, dim: int) -> nn.Module:
    if objective_name == "blob2d_1024":
        return _Conv2dDecoder(dim)
    return _Conv1dDecoder(dim)


class _ArchTemplate:
    """One reusable net; per-individual weights are flat vectors loaded in."""

    def __init__(self, builder: Callable[[], nn.Module], device: str):
        self.builder = builder
        self.net = builder().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.n_params = sum(p.numel() for p in self.net.parameters())

    def init_theta(self, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        fresh = self.builder()
        return nn.utils.parameters_to_vector(
            fresh.parameters()).detach().numpy().astype(np.float32)

    def decode(self, theta: np.ndarray, z: np.ndarray) -> torch.Tensor:
        nn.utils.vector_to_parameters(
            torch.as_tensor(theta, device=self.device),
            self.net.parameters())
        genes = torch.as_tensor(z[None].astype(np.float32), device=self.device)
        return torch.sigmoid(self.net(genes))[0]


def run_arm(objective, objective_name, seed, config, arm):
    _require_mps()
    _seed_everything(seed)

    if arm == "direct_ga":
        return run_direct_ga(objective, seed, config)

    rng = np.random.default_rng(seed)
    dim = objective.dimension
    if arm == "mlp_decoder":
        template = _ArchTemplate(lambda: _build_mlp(dim), "mps")
    elif arm == "conv_decoder":
        template = _ArchTemplate(lambda: _build_conv(objective_name, dim),
                                 "mps")
    else:
        raise ValueError(arm)

    tracker = TrackedFitness(objective)
    started = time.perf_counter()

    def evaluate(z_batch, theta_batch) -> np.ndarray:
        phenotypes = torch.stack([
            template.decode(t, z) for z, t in zip(z_batch, theta_batch)])
        return (-tracker(phenotypes)).detach().cpu().numpy()

    zs = rng.standard_normal((POPULATION, LATENT)).astype(np.float32)
    thetas = np.stack([
        template.init_theta(int(rng.integers(0, 2**31)))
        for _ in range(POPULATION)])
    n = min(POPULATION, config.evaluation_budget)
    loss = evaluate(zs[:n], thetas[:n])
    zs, thetas = zs[:n], thetas[:n]
    while tracker.evaluations < config.evaluation_budget:
        order = np.argsort(loss)[:ELITE]
        zs, thetas, loss = zs[order], thetas[order], loss[order]
        n = min(POPULATION, config.evaluation_budget - tracker.evaluations)
        parent = rng.integers(0, len(zs), n)
        child_z = np.stack([_mutate_z(zs[p], rng) for p in parent])
        child_theta = np.stack([_mutate_theta(thetas[p], rng) for p in parent])
        child_loss = evaluate(child_z, child_theta)
        zs = np.concatenate([zs, child_z])
        thetas = np.concatenate([thetas, child_theta])
        loss = np.concatenate([loss, child_loss])

    torch.mps.synchronize()
    result = _finish_result(objective, arm, seed, config, tracker, started,
                            neural_device="mps")
    torch.mps.empty_cache()
    return result


STRATEGIES = ("direct_ga", "mlp_decoder", "conv_decoder")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES)
    )
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                        default=list(STRATEGIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(3)))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = BenchmarkConfig(evaluation_budget=args.budget)
    _require_mps()
    results = []
    for objective_name in args.objectives:
        for arm in args.strategies:
            for seed in args.seeds:
                objective = OBJECTIVES[objective_name]()
                print(
                    f"run objective={objective_name:<14} arm={arm:<13} "
                    f"seed={seed} budget={config.evaluation_budget}",
                    flush=True,
                )
                result = run_arm(objective, objective_name, seed, config, arm)
                print(f"  {result.metric}={result.metric_at_budget:.6g}",
                      flush=True)
                results.append(result)
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "channels": CHANNELS,
            "torch_version": torch.__version__,
            "runs": [asdict(r) for r in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
