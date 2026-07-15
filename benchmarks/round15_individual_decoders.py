"""Round 15: per-individual decoders — evolution only on universal substrates.

Daniel's constraint: a universal algorithm may never evolve the phenotype
directly, because phenotype operators (mutation, crossover) are modality-
specific — pixels have them, 3D meshes and programs do not. Latent vectors
and decoder WEIGHTS are always plain tensors, so evolution restricted to
(z, theta) is modality-blind by construction; the phenotype is only ever
computed, never operated on.

The proposal under test: every individual carries its own genotype z AND
its own private decoder weights theta. The population searches over
genotype-to-phenotype maps, not just genotypes — escaping round 1's death
(one frozen random manifold) because the manifold itself is under
selection. Round 12's calibration says decoder weight space is evolvable
(43% of small perturbations improve outputs at sigma 0.003).

Arms (mu+lambda, truncation selection, identical budgets):

  * zw_mut  — children mutate a parent's z (Gaussian, original evolver
              rates) and theta (per-child log-uniform sigma in round-12's
              viable regime, scaled by the parent's weight std);
  * zw_pca  — same, but half the children are made by Daniel's weight-
              crossover: PCA over the parent population's theta vectors,
              children mix parent COEFFICIENTS in that subspace (dodges
              the permuted-neuron problem of naive weight splicing), plus
              small coefficient noise;
  * direct_ga — the traditional bar (modality-specific by definition;
              here only as the reference the explorer must approach).

Frozen-single-decoder plateau for context (round 14, same seeds/config):
smooth 0.0795, blob 0.1011 — the number to beat is the gap between that
plateau and direct_ga (0.0146 / 0.0562).
"""

from __future__ import annotations

import argparse
import json
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

OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "blob2d_1024": BlobImage2D,
}

LATENT = 32
HIDDEN = 64                     # per-individual nets are deliberately small
POPULATION = 32
ELITE = 16
SIGMA_Z_RATE, SIGMA_Z = 0.1, 0.12          # original evolver's z mutation
SIGMA_W_LOW, SIGMA_W_HIGH = 0.003, 0.02    # round-12 viable weight regime
PCA_FRACTION = 0.5              # zw_pca: fraction of children from crossover
PCA_COEF_NOISE = 0.1


class _Template:
    """One reusable net; per-individual weights are flat vectors loaded in."""

    def __init__(self, dim: int, device: str):
        self.net = nn.Sequential(
            nn.Linear(LATENT, HIDDEN), nn.LeakyReLU(),
            nn.Linear(HIDDEN, dim),
        ).to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.n_params = sum(p.numel() for p in self.net.parameters())

    def init_theta(self, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        fresh = nn.Sequential(
            nn.Linear(LATENT, HIDDEN), nn.LeakyReLU(),
            nn.Linear(HIDDEN, self.net[-1].out_features),
        )
        return nn.utils.parameters_to_vector(
            fresh.parameters()).detach().numpy().astype(np.float32)

    def decode(self, theta: np.ndarray, z: np.ndarray) -> torch.Tensor:
        nn.utils.vector_to_parameters(
            torch.as_tensor(theta, device=self.device),
            self.net.parameters())
        genes = torch.as_tensor(z[None].astype(np.float32), device=self.device)
        return torch.sigmoid(self.net(genes))[0]


def _mutate_z(z: np.ndarray, rng) -> np.ndarray:
    mask = rng.random(z.shape) < SIGMA_Z_RATE
    if not mask.any():
        mask[rng.integers(0, len(z))] = True
    return (z + mask * rng.normal(0, SIGMA_Z, z.shape)).astype(np.float32)


def _mutate_theta(theta: np.ndarray, rng) -> np.ndarray:
    sigma_w = float(np.exp(rng.uniform(
        np.log(SIGMA_W_LOW), np.log(SIGMA_W_HIGH))))
    scale = max(float(theta.std()), 1e-3)
    return (theta + rng.normal(0, sigma_w * scale, theta.shape)
            ).astype(np.float32)


def _pca_children(thetas: np.ndarray, count: int, rng) -> np.ndarray:
    """Daniel's weight crossover: mix parent COEFFICIENTS in the parent
    population's own principal subspace instead of splicing raw weights."""
    mean = thetas.mean(axis=0)
    centered = thetas - mean
    # rank <= ELITE-1; SVD in the small dimension is cheap
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    coefs = u * s                            # (parents, comps)
    idx_a = rng.integers(0, len(thetas), count)
    idx_b = rng.integers(0, len(thetas), count)
    alpha = rng.random((count, 1)).astype(np.float32)
    mixed = alpha * coefs[idx_a] + (1 - alpha) * coefs[idx_b]
    mixed = mixed + rng.normal(
        0, PCA_COEF_NOISE * (np.abs(s) + 1e-8), mixed.shape)
    return (mean + mixed @ vt).astype(np.float32)


def run_arm(objective, seed, config, arm):
    _require_mps()
    _seed_everything(seed)

    if arm == "direct_ga":
        return run_direct_ga(objective, seed, config)

    rng = np.random.default_rng(seed)
    template = _Template(objective.dimension, "mps")
    tracker = TrackedFitness(objective)
    started = time.perf_counter()

    zs = rng.standard_normal((POPULATION, LATENT)).astype(np.float32)
    thetas = np.stack([template.init_theta(seed * 100_000 + i)
                       for i in range(POPULATION)])

    def evaluate(z_batch, theta_batch) -> np.ndarray:
        phenotypes = torch.stack([
            template.decode(t, z) for z, t in zip(z_batch, theta_batch)])
        return (-tracker(phenotypes)).detach().cpu().numpy()

    n = min(POPULATION, config.evaluation_budget)
    loss = evaluate(zs[:n], thetas[:n])
    zs, thetas = zs[:n], thetas[:n]

    while tracker.evaluations < config.evaluation_budget:
        order = np.argsort(loss)[:ELITE]
        zs, thetas, loss = zs[order], thetas[order], loss[order]

        n = min(POPULATION, config.evaluation_budget - tracker.evaluations)
        n_pca = int(n * PCA_FRACTION) if arm == "zw_pca" else 0
        parent = rng.integers(0, ELITE, n)
        child_z = np.stack([_mutate_z(zs[p], rng) for p in parent])
        child_theta = np.empty((n, template.n_params), dtype=np.float32)
        if n_pca:
            child_theta[:n_pca] = _pca_children(thetas, n_pca, rng)
        for i in range(n_pca, n):
            child_theta[i] = _mutate_theta(thetas[parent[i]], rng)
        child_loss = evaluate(child_z, child_theta)

        zs = np.concatenate([zs, child_z])
        thetas = np.concatenate([thetas, child_theta])
        loss = np.concatenate([loss, child_loss])

    torch.mps.synchronize()
    result = _finish_result(objective, arm, seed, config, tracker, started,
                            neural_device="mps")
    torch.mps.empty_cache()
    return result


STRATEGIES = ("direct_ga", "zw_mut", "zw_pca")


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
                    f"run objective={objective_name:<14} arm={arm:<10} "
                    f"seed={seed} budget={config.evaluation_budget}",
                    flush=True,
                )
                result = run_arm(objective, seed, config, arm)
                print(f"  {result.metric}={result.metric_at_budget:.6g}",
                      flush=True)
                results.append(result)
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "latent": LATENT, "hidden": HIDDEN,
            "population": POPULATION, "elite": ELITE,
            "sigma_z": [SIGMA_Z_RATE, SIGMA_Z],
            "sigma_w": [SIGMA_W_LOW, SIGMA_W_HIGH],
            "pca_fraction": PCA_FRACTION, "pca_coef_noise": PCA_COEF_NOISE,
            "torch_version": torch.__version__,
            "runs": [asdict(r) for r in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
