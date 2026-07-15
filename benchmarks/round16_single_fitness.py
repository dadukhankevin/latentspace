"""Round 16: best fully-universal method given ONLY a fitness function.

Daniel's constraint, final form: a realistic universal GA gets one fitness
function and a budget. No practice problems, no second fitness function, no
operator that touches the solution format (mutation/crossover exist only
for genomes and decoder weights — plain tensors).

Candidate, combining the campaign's two surviving lessons: (1) per-
individual decoder evolution (each individual = genome + private decoder
weights, children mutate both) is the only from-scratch method that beat a
frozen random decoder; (2) pooled solutions teach a good decoder when their
errors are INDEPENDENT — and a population of separate lineages, each with
its own private decoder, has independence built in.

  explore_distill — first EXPLORE_FRACTION of the budget: per-individual
                    decoder evolution, archiving every vetted phenotype;
                    then compress the archive's best DISTILL_TOP solutions
                    into a linear decoder (closed-form PCA fit in logit
                    space) and spend the rest of the budget with CMA-ES in
                    its 32-number latent (a universal operation).

Controls: per-individual evolution for the full budget (the previous
universal best), and the traditional GA (modality-specific reference).
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
from benchmarks.round6_learned_structure import fit_pca_decoder
from benchmarks.round8_mlp_pretrain import BlobImage2D
from benchmarks.round15_individual_decoders import (
    ELITE,
    LATENT,
    POPULATION,
    _mutate_theta,
    _mutate_z,
    _Template,
)
from benchmarks.round17_architecture_prior import _ArchTemplate, _build_conv

OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "blob2d_1024": BlobImage2D,
}

EXPLORE_FRACTION = 0.6
DISTILL_TOP = 200


def explore(objective, tracker, rng, budget, template=None):
    """Per-individual decoder evolution against a shared budget tracker;
    returns the archive of every evaluated phenotype and its loss."""
    if template is None:
        template = _Template(objective.dimension, "mps")
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
    n = min(POPULATION, budget - tracker.evaluations)
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


def run_arm(objective, seed, config, arm):
    _require_mps()
    _seed_everything(seed)

    if arm == "direct_ga":
        return run_direct_ga(objective, seed, config)

    rng = np.random.default_rng(seed)
    tracker = TrackedFitness(objective)
    started = time.perf_counter()

    if arm == "individual_only":
        explore(objective, tracker, rng, config.evaluation_budget)
    elif arm in ("explore_distill", "explore_distill_conv"):
        template = None
        if arm == "explore_distill_conv":
            template = _ArchTemplate(
                lambda: _build_conv(objective.name, objective.dimension),
                "mps")
        explore_budget = int(config.evaluation_budget * EXPLORE_FRACTION)
        archive_x, archive_loss = explore(objective, tracker, rng,
                                          explore_budget, template)
        order = np.argsort(archive_loss)[:DISTILL_TOP]
        decoder = fit_pca_decoder(
            archive_x[order], archive_loss[order], config.latent, "mps",
            top=len(order))
        _cma_minimize(
            lambda z: -tracker(decoder.decode(z)).detach().cpu().numpy(),
            dim=config.latent, budget_evaluations=config.evaluation_budget,
            evaluations_done=tracker.evaluations, rng=rng,
            mean0=np.zeros(config.latent), sigma0=1.0,
        )
    else:
        raise ValueError(arm)

    torch.mps.synchronize()
    result = _finish_result(objective, arm, seed, config, tracker, started,
                            neural_device="mps")
    torch.mps.empty_cache()
    return result


STRATEGIES = ("direct_ga", "individual_only", "explore_distill",
              "explore_distill_conv")


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
                    f"run objective={objective_name:<14} arm={arm:<16} "
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
            "explore_fraction": EXPLORE_FRACTION, "distill_top": DISTILL_TOP,
            "torch_version": torch.__version__,
            "runs": [asdict(r) for r in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
