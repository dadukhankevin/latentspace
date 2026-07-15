"""Round 15c: the self-teaching loop — can the universal explorer bootstrap?

Round 15b found that pretraining on practice elites harvested by the
modality-blind explorer (per-individual decoder evolution: each individual
owns its genotype AND its decoder weights, children mutate both, no
phenotype operator anywhere) preserves the scaling law's direction but with
teachers too weak to compete: practice elites at mean loss 0.062 vs the
traditional GA's 0.040, giving test solves of 0.035 vs 0.004.

The observation that motivates this round: the FIRST-generation decoder
already solves fresh instances (0.035) better than the raw data it was
trained on (0.062) — the student outperforms its teacher's examples. So
iterate: re-harvest every practice instance with the explorer's private
decoders initialized from the CURRENT shared decoder's weights plus noise
(instead of random), refit the shared decoder on the better elites, and
repeat. Weight noise remains the off-manifold escape channel (round 12
measured its viable regime), so each harvest can leave the current
decoder's span; instance-specific fitness keeps every elite externally
vetted. Every substrate touched is a tensor. If the loop climbs toward the
traditional-GA-taught curve (test 0.004), universality costs nothing at
the system level; if it stalls, the explorer itself is the bottleneck.

Bridging detail: the shared decoder is a linear PCA map, the private
decoders are small MLPs, so at the start of each round after the first the
PCA map is distilled into one MLP (supervised regression on PCA-generated
latent/logit pairs) whose weights seed every individual.
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
    summarize,
)
from benchmarks.round3_structure import SmoothTarget
from benchmarks.round4_latent_cma import _cma_minimize
from benchmarks.round6_learned_structure import fit_pca_decoder
from benchmarks.round8_mlp_pretrain import BlobImage2D
from benchmarks.round10_online_refine import _to_logits
from benchmarks.round15_individual_decoders import (
    ELITE,
    HIDDEN,
    LATENT,
    POPULATION,
    _mutate_theta,
    _mutate_z,
    _Template,
)

OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "blob2d_1024": BlobImage2D,
}

ROUNDS = 3
FAMILY_K = 128
PER_INSTANCE_EVALUATIONS = 2_000
ELITES_PER_INSTANCE = 10
DISTILL_SAMPLES = 5_000
DISTILL_STEPS = 4_000
WARM_INIT_SIGMA = 0.01          # per-individual noise around the warm weights


def harvest(objective, rng, budget, template, warm_theta):
    """One practice instance solved by per-individual decoder evolution.
    Private decoders start at warm_theta + noise (or random if None)."""
    tracker = TrackedFitness(objective)
    archive_x: list[np.ndarray] = []
    archive_loss: list[float] = []

    def evaluate(z_batch, theta_batch) -> np.ndarray:
        phenotypes = torch.stack([
            template.decode(t, z) for z, t in zip(z_batch, theta_batch)])
        losses = (-tracker(phenotypes)).detach().cpu().numpy()
        archive_x.extend(phenotypes.detach().cpu().numpy())
        archive_loss.extend(losses.tolist())
        return losses

    zs = rng.standard_normal((POPULATION, LATENT)).astype(np.float32)
    if warm_theta is None:
        thetas = np.stack([
            template.init_theta(int(rng.integers(0, 2**31)))
            for _ in range(POPULATION)])
    else:
        scale = max(float(warm_theta.std()), 1e-3)
        thetas = np.stack([
            warm_theta + rng.normal(
                0, WARM_INIT_SIGMA * scale, warm_theta.shape
            ).astype(np.float32)
            for _ in range(POPULATION)])

    n = min(POPULATION, budget)
    loss = evaluate(zs[:n], thetas[:n])
    zs, thetas = zs[:n], thetas[:n]
    while tracker.evaluations < budget:
        order = np.argsort(loss)[:ELITE]
        zs, thetas, loss = zs[order], thetas[order], loss[order]
        n = min(POPULATION, budget - tracker.evaluations)
        parent = rng.integers(0, len(zs), n)
        child_z = np.stack([_mutate_z(zs[p], rng) for p in parent])
        child_theta = np.stack([_mutate_theta(thetas[p], rng) for p in parent])
        child_loss = evaluate(child_z, child_theta)
        zs = np.concatenate([zs, child_z])
        thetas = np.concatenate([thetas, child_theta])
        loss = np.concatenate([loss, child_loss])
    return np.asarray(archive_x), np.asarray(archive_loss)


def distill_pca_to_theta(pca, dim, rng, device="mps") -> np.ndarray:
    """Supervised regression of the PCA decoder into one MLP weight vector."""
    z = rng.standard_normal((DISTILL_SAMPLES, LATENT)).astype(np.float32)
    samples = pca.decode(z).detach().cpu().numpy().reshape(DISTILL_SAMPLES, -1)
    logits = _to_logits(samples)

    net = nn.Sequential(
        nn.Linear(LATENT, HIDDEN), nn.LeakyReLU(),
        nn.Linear(HIDDEN, dim),
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    z_t = torch.as_tensor(z, device=device)
    y_t = torch.as_tensor(logits, dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(int(rng.integers(0, 2**31)))
    for _ in range(DISTILL_STEPS):
        index = torch.randint(0, DISTILL_SAMPLES, (128,), generator=generator)
        idx = index.to(device)
        optimizer.zero_grad()
        loss = loss_fn(net(z_t[idx]), y_t[idx])
        loss.backward()
        optimizer.step()
    return nn.utils.parameters_to_vector(
        net.parameters()).detach().cpu().numpy().astype(np.float32)


def run_seed(objective, seed, config):
    _require_mps()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = _Template(objective.dimension, "mps")

    results, quality = [], []
    warm_theta = None
    for bootstrap_round in range(1, ROUNDS + 1):
        pooled_x: list[np.ndarray] = []
        pooled_loss: list[float] = []
        for instance_seed in range(100, 100 + FAMILY_K):
            practice = type(objective)(instance_seed=instance_seed)
            archive_x, archive_loss = harvest(
                practice, rng, PER_INSTANCE_EVALUATIONS, template, warm_theta)
            order = np.argsort(archive_loss)[:ELITES_PER_INSTANCE]
            pooled_x.extend(archive_x[order])
            pooled_loss.extend(archive_loss[order])

        pca = fit_pca_decoder(
            np.asarray(pooled_x), np.asarray(pooled_loss), config.latent,
            "mps", top=len(pooled_x))

        tracker = TrackedFitness(objective)
        started = time.perf_counter()
        generations = _cma_minimize(
            lambda z: -tracker(pca.decode(z)).detach().cpu().numpy(),
            dim=config.latent, budget_evaluations=config.evaluation_budget,
            evaluations_done=0, rng=rng,
            mean0=np.zeros(config.latent), sigma0=1.0,
        )
        torch.mps.synchronize()
        label = f"bootstrap_round{bootstrap_round}"
        result = _finish_result(objective, label, seed, config, tracker,
                                started, generations=generations,
                                neural_device="mps")
        corpus_mean = float(np.mean(pooled_loss))
        print(f"  round {bootstrap_round}: corpus_mean_loss={corpus_mean:.5f}"
              f"  test_{result.metric}={result.metric_at_budget:.6g}",
              flush=True)
        results.append(result)
        quality.append({
            "objective": objective.name, "seed": seed,
            "bootstrap_round": bootstrap_round,
            "corpus_mean_loss": corpus_mean,
        })
        if bootstrap_round < ROUNDS:
            warm_theta = distill_pca_to_theta(pca, objective.dimension, rng)
        torch.mps.empty_cache()
    return results, quality


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES,
        default=["smooth1d_256"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(3)))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = BenchmarkConfig(evaluation_budget=args.budget)
    _require_mps()
    results, quality = [], []
    for objective_name in args.objectives:
        for seed in args.seeds:
            objective = OBJECTIVES[objective_name]()
            print(f"run {objective_name} bootstrap seed={seed} "
                  f"rounds={ROUNDS} K={FAMILY_K}", flush=True)
            seed_results, seed_quality = run_seed(objective, seed, config)
            results.extend(seed_results)
            quality.extend(seed_quality)
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "rounds": ROUNDS, "family_k": FAMILY_K,
            "per_instance_evaluations": PER_INSTANCE_EVALUATIONS,
            "elites_per_instance": ELITES_PER_INSTANCE,
            "warm_init_sigma": WARM_INIT_SIGMA,
            "distill": [DISTILL_SAMPLES, DISTILL_STEPS],
            "torch_version": torch.__version__,
            "runs": [asdict(r) for r in results],
            "corpus_quality": quality,
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
