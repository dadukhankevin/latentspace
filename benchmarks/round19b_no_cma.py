"""Round 19b: does the exploit phase need CMA-ES, or just *any* genotype GA?

Daniel's challenge: the stack should be evolution over decoder inputs the
whole way. It is — CMA-ES is an evolution strategy over genotypes — but
the fair measurement is to keep everything else identical (adaptive-stall
exploration with per-individual conv decoders, distillation into a learned
decoder) and swap ONLY the final phase's evolution rule:

  * ga_exploit — plain fixed-sigma GA over the distilled decoder's
    genotypes: uniform crossover over an elite pool, per-gene Gaussian
    mutation at the original evolver's rates (0.1 / 0.12), (mu+lambda)
    truncation. No covariance adaptation anywhere.

Compare pairwise against round 18's `adaptive` arm (same seeds, identical
exploration; CMA-ES exploit) in mps_round18_adaptive_10seed.json, and
against exploration-only (round 17 conv arm: no exploit phase at all).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import time
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
    summarize,
)
from benchmarks.round3_structure import SmoothTarget
from benchmarks.round6_learned_structure import fit_pca_decoder
from benchmarks.round8_mlp_pretrain import BlobImage2D
from benchmarks.round17_architecture_prior import _ArchTemplate, _build_conv
from benchmarks.round18_adaptive import DISTILL_TOP, explore_adaptive

OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "blob2d_1024": BlobImage2D,
}

POPULATION, ELITE = 64, 32
MUTATION_RATE, MUTATION_SIGMA = 0.1, 0.12


def _latent_ga(evaluate, dim, budget_left, rng):
    """Fixed-sigma GA over genotypes: uniform crossover + per-gene noise."""
    n = min(POPULATION, budget_left)
    pool = rng.standard_normal((n, dim)).astype(np.float32)
    loss = evaluate(pool)
    budget_left -= n
    while budget_left > 0:
        order = np.argsort(loss)[:ELITE]
        pool, loss = pool[order], loss[order]
        n = min(POPULATION, budget_left)
        idx_a = rng.integers(0, len(pool), n)
        idx_b = rng.integers(0, len(pool), n)
        mask = rng.random((n, dim)) < 0.5
        children = np.where(mask, pool[idx_a], pool[idx_b])
        mutate = rng.random((n, dim)) < MUTATION_RATE
        silent = np.flatnonzero(~mutate.any(axis=1))
        if len(silent):
            mutate[silent, rng.integers(0, dim, len(silent))] = True
        children = (children + mutate * rng.normal(0, MUTATION_SIGMA, (n, dim))
                    ).astype(np.float32)
        child_loss = evaluate(children)
        budget_left -= n
        pool = np.concatenate([pool, children])
        loss = np.concatenate([loss, child_loss])


def run_arm(objective, seed, config):
    _require_mps()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = _ArchTemplate(
        lambda: _build_conv(objective.name, objective.dimension), "mps")
    tracker = TrackedFitness(objective)
    started = time.perf_counter()

    archive_x, archive_loss, archive_lineage = explore_adaptive(
        objective, tracker, rng, config, template, "adaptive")
    explore_spent = tracker.evaluations

    if tracker.evaluations < config.evaluation_budget:
        order = np.argsort(archive_loss)[:DISTILL_TOP]
        decoder = fit_pca_decoder(
            archive_x[order], archive_loss[order], config.latent, "mps",
            top=len(order))
        _latent_ga(
            lambda z: -tracker(decoder.decode(z)).detach().cpu().numpy(),
            dim=config.latent,
            budget_left=config.evaluation_budget - tracker.evaluations,
            rng=rng,
        )

    torch.mps.synchronize()
    result = _finish_result(objective, "ga_exploit", seed, config, tracker,
                            started, neural_device="mps")
    torch.mps.empty_cache()
    return result, explore_spent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES,
                        default=list(OBJECTIVES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = BenchmarkConfig(evaluation_budget=args.budget)
    _require_mps()
    results = []
    for objective_name in args.objectives:
        for seed in args.seeds:
            objective = OBJECTIVES[objective_name]()
            print(f"run {objective_name} ga_exploit seed={seed}", flush=True)
            result, explored = run_arm(objective, seed, config)
            print(f"  {result.metric}={result.metric_at_budget:.6g} "
                  f"explored={explored}", flush=True)
            results.append(result)
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "population": POPULATION, "elite": ELITE,
            "mutation": [MUTATION_RATE, MUTATION_SIGMA],
            "torch_version": torch.__version__,
            "runs": [asdict(r) for r in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
