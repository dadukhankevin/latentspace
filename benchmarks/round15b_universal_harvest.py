"""Round 15b: can a modality-blind explorer feed the pretraining scaling law?

Round 15a showed per-individual decoder evolution (each individual = its own
genotype + its own decoder weights; children mutate both; no phenotype
operator anywhere) beats a frozen random decoder but stays well behind the
traditional GA within a run. For Daniel's universal-algorithm goal the
explorer's real job is different: generate externally-vetted practice
elites good enough to PRETRAIN the shared decoder. If the scaling law
survives with this harvester, the whole pipeline touches phenotypes only to
score them — modality-blind end to end.

Protocol is round 7's exactly: K practice instances (seeds 100..100+K-1),
2,000 evaluations each, top 10 elites pooled, PCA-32 decoder fit on the
pool, frozen, CMA-ES in the latent on the fresh test instance (5,000
evaluations). The ONLY difference between arms is who harvests:

  * harvest_direct_K     — traditional GA (round 7's teacher);
  * harvest_universal_K  — per-individual decoder evolution (mutation-only
                           variant; the weight-PCA crossover was neutral).

CMA-ES operates on the latent vector — a universal substrate — so it is
allowed here; the modality-specific thing being replaced is the direct
phenotype-space GA.
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
from benchmarks.round4_latent_cma import _cma_minimize
from benchmarks.round6_learned_structure import _bootstrap_direct_ga, fit_pca_decoder
from benchmarks.round8_mlp_pretrain import BlobImage2D
from benchmarks.round15_individual_decoders import (
    ELITE,
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

FAMILY_SIZES = (16, 128)
PER_INSTANCE_EVALUATIONS = 2_000
ELITES_PER_INSTANCE = 10


def harvest_universal(objective, rng, budget):
    """Archive of (phenotype, loss) from per-individual decoder evolution."""
    template = _Template(objective.dimension, "mps")
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
    thetas = np.stack([
        template.init_theta(int(rng.integers(0, 2**31)))
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


def run_family(objective, seed, config, harvester, k, label):
    _require_mps()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)

    pooled_x: list[np.ndarray] = []
    pooled_loss: list[float] = []
    for instance_seed in range(100, 100 + k):
        pretrain_objective = type(objective)(instance_seed=instance_seed)
        if harvester == "direct":
            pretrain_tracker = TrackedFitness(pretrain_objective)
            archive_x, archive_loss = _bootstrap_direct_ga(
                pretrain_objective, rng, pretrain_tracker, config,
                PER_INSTANCE_EVALUATIONS,
            )
        else:
            archive_x, archive_loss = harvest_universal(
                pretrain_objective, rng, PER_INSTANCE_EVALUATIONS)
        order = np.argsort(archive_loss)[:ELITES_PER_INSTANCE]
        pooled_x.extend(np.asarray(archive_x)[order])
        pooled_loss.extend(np.asarray(archive_loss)[order])

    decoder = fit_pca_decoder(
        np.asarray(pooled_x), np.asarray(pooled_loss), config.latent, "mps",
        top=len(pooled_x),
    )

    tracker = TrackedFitness(objective)
    started = time.perf_counter()
    generations = _cma_minimize(
        lambda z: -tracker(decoder.decode(z)).detach().cpu().numpy(),
        dim=config.latent, budget_evaluations=config.evaluation_budget,
        evaluations_done=0, rng=rng, mean0=np.zeros(config.latent), sigma0=1.0,
    )
    torch.mps.synchronize()
    result = _finish_result(objective, label, seed, config, tracker, started,
                            generations=generations, neural_device="mps")
    torch.mps.empty_cache()
    return result, float(np.mean(pooled_loss))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES,
        default=["smooth1d_256"],
    )
    parser.add_argument("--family-sizes", nargs="+", type=int,
                        default=list(FAMILY_SIZES))
    parser.add_argument("--harvesters", nargs="+",
                        choices=("direct", "universal"),
                        default=["direct", "universal"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(3)))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--include-direct-ga", action="store_true",
                        help="also run the direct GA on the test instance")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = BenchmarkConfig(evaluation_budget=args.budget)
    _require_mps()
    results, corpus_quality = [], []
    for objective_name in args.objectives:
        if args.include_direct_ga:
            for seed in args.seeds:
                objective = OBJECTIVES[objective_name]()
                result = run_direct_ga(objective, seed, config)
                print(f"run {objective_name} direct_ga seed={seed}  "
                      f"{result.metric}={result.metric_at_budget:.6g}",
                      flush=True)
                results.append(result)
        for harvester in args.harvesters:
            for k in args.family_sizes:
                label = f"harvest_{harvester}_{k}"
                for seed in args.seeds:
                    objective = OBJECTIVES[objective_name]()
                    print(f"run {objective_name} {label} seed={seed}",
                          flush=True)
                    result, corpus_mean = run_family(
                        objective, seed, config, harvester, k, label)
                    print(f"  {result.metric}={result.metric_at_budget:.6g}  "
                          f"corpus_mean_loss={corpus_mean:.5f}", flush=True)
                    results.append(result)
                    corpus_quality.append({
                        "objective": objective_name, "strategy": label,
                        "seed": seed, "corpus_mean_loss": corpus_mean,
                    })
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "per_instance_evaluations": PER_INSTANCE_EVALUATIONS,
            "elites_per_instance": ELITES_PER_INSTANCE,
            "torch_version": torch.__version__,
            "runs": [asdict(r) for r in results],
            "corpus_quality": corpus_quality,
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
