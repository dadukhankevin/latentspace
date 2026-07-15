"""Round 18: replace the magic 60/40 split with earned, adaptive rules.

The round 16/17 stack (per-individual conv-decoder exploration, then
distill the best 200 into PCA-32, then CMA-ES) beat the traditional GA
10/0 on the image problem but tied with high variance on the curve. Two
suspects, two candidate fixes, both tested here at 10 seeds:

  * The rigid 60/40 phase split → SWITCH ON STALL: exploration ends when
    the best score has improved less than STALL_TOL (relative) over the
    last STALL_WINDOW generations. One safety bound, justified by CMA-ES
    itself rather than tuning: always reserve EXPLOIT_RESERVE = 10 x
    latent evaluations so the exploit phase can converge.
  * Selection concentrates the harvest into few lineages, destroying the
    error-independence the distillation needs (round 15c's law) →
    STRATIFIED HARVEST: each individual carries a lineage id (inherited
    from its parent); no lineage may contribute more than LINEAGE_CAP of
    the DISTILL_TOP distilled solutions (filled back up with the global
    best if the caps leave a shortfall).

Arms: fixed (the round-16/17 stack unchanged), adaptive (stall switch),
adaptive_stratified (stall switch + capped harvest). direct_ga numbers for
the same seeds come from mps_round16c_confirmation_10seed.json.
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
)
from benchmarks.round17_architecture_prior import _ArchTemplate, _build_conv

OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "blob2d_1024": BlobImage2D,
}

EXPLORE_FRACTION = 0.6          # the fixed arm's magic number
DISTILL_TOP = 200
STALL_WINDOW = 10               # generations
STALL_TOL = 0.01                # relative improvement below this = stalled
EXPLOIT_RESERVE = 10 * LATENT   # evals CMA-ES needs to be worth starting
LINEAGE_CAP = 10                # max distilled solutions per lineage


def explore_adaptive(objective, tracker, rng, config, template, arm):
    """Per-individual decoder evolution that decides its own stopping point
    (except the 'fixed' arm). Returns (archive_x, archive_loss, lineages)."""
    budget = config.evaluation_budget
    fixed_end = int(budget * EXPLORE_FRACTION)
    archive_x: list[np.ndarray] = []
    archive_loss: list[float] = []
    archive_lineage: list[int] = []

    def evaluate(z_batch, theta_batch, lineage_batch) -> np.ndarray:
        phenotypes = torch.stack([
            template.decode(t, z) for z, t in zip(z_batch, theta_batch)])
        losses = (-tracker(phenotypes)).detach().cpu().numpy()
        archive_x.extend(phenotypes.detach().cpu().numpy())
        archive_loss.extend(losses.tolist())
        archive_lineage.extend(lineage_batch.tolist())
        return losses

    zs = rng.standard_normal((POPULATION, LATENT)).astype(np.float32)
    thetas = np.stack([
        template.init_theta(int(rng.integers(0, 2**31)))
        for _ in range(POPULATION)])
    lineages = np.arange(POPULATION)
    n = min(POPULATION, budget)
    loss = evaluate(zs[:n], thetas[:n], lineages[:n])
    zs, thetas, lineages = zs[:n], thetas[:n], lineages[:n]

    best_history = [float(loss.min())]
    while True:
        remaining = budget - tracker.evaluations
        if remaining <= 0:
            break
        if arm == "fixed":
            if tracker.evaluations >= fixed_end:
                break
        else:
            if remaining <= EXPLOIT_RESERVE:
                break
            if len(best_history) > STALL_WINDOW:
                then = best_history[-STALL_WINDOW - 1]
                if then - best_history[-1] < STALL_TOL * then:
                    break
        order = np.argsort(loss)[:ELITE]
        zs, thetas, loss = zs[order], thetas[order], loss[order]
        lineages = lineages[order]
        n = min(POPULATION, remaining)
        parent = rng.integers(0, len(zs), n)
        child_z = np.stack([_mutate_z(zs[p], rng) for p in parent])
        child_theta = np.stack([_mutate_theta(thetas[p], rng) for p in parent])
        child_loss = evaluate(child_z, child_theta, lineages[parent])
        zs = np.concatenate([zs, child_z])
        thetas = np.concatenate([thetas, child_theta])
        lineages = np.concatenate([lineages, lineages[parent]])
        loss = np.concatenate([loss, child_loss])
        best_history.append(float(loss.min()))

    return (np.asarray(archive_x), np.asarray(archive_loss),
            np.asarray(archive_lineage))


def select_distill(archive_loss, archive_lineage, arm):
    """Indices of the solutions to compress, most-fit first."""
    order = np.argsort(archive_loss)
    if arm != "adaptive_stratified":
        return order[:DISTILL_TOP]
    counts: dict[int, int] = {}
    chosen: list[int] = []
    for i in order:
        lineage = int(archive_lineage[i])
        if counts.get(lineage, 0) >= LINEAGE_CAP:
            continue
        counts[lineage] = counts.get(lineage, 0) + 1
        chosen.append(int(i))
        if len(chosen) == DISTILL_TOP:
            return np.asarray(chosen)
    for i in order:                       # caps left a shortfall: top up
        if int(i) not in set(chosen):
            chosen.append(int(i))
            if len(chosen) == DISTILL_TOP:
                break
    return np.asarray(chosen)


def run_arm(objective, seed, config, arm):
    _require_mps()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = _ArchTemplate(
        lambda: _build_conv(objective.name, objective.dimension), "mps")
    tracker = TrackedFitness(objective)
    started = time.perf_counter()

    archive_x, archive_loss, archive_lineage = explore_adaptive(
        objective, tracker, rng, config, template, arm)
    explore_spent = tracker.evaluations

    if tracker.evaluations < config.evaluation_budget:
        idx = select_distill(archive_loss, archive_lineage, arm)
        decoder = fit_pca_decoder(
            archive_x[idx], archive_loss[idx], config.latent, "mps",
            top=len(idx))
        _cma_minimize(
            lambda z: -tracker(decoder.decode(z)).detach().cpu().numpy(),
            dim=config.latent, budget_evaluations=config.evaluation_budget,
            evaluations_done=tracker.evaluations, rng=rng,
            mean0=np.zeros(config.latent), sigma0=1.0,
        )

    torch.mps.synchronize()
    result = _finish_result(objective, arm, seed, config, tracker, started,
                            neural_device="mps")
    torch.mps.empty_cache()
    return result, explore_spent


STRATEGIES = ("fixed", "adaptive", "adaptive_stratified")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES)
    )
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                        default=list(STRATEGIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = BenchmarkConfig(evaluation_budget=args.budget)
    _require_mps()
    results, splits = [], []
    for objective_name in args.objectives:
        for arm in args.strategies:
            for seed in args.seeds:
                objective = OBJECTIVES[objective_name]()
                print(
                    f"run objective={objective_name:<14} arm={arm:<20} "
                    f"seed={seed} budget={config.evaluation_budget}",
                    flush=True,
                )
                result, explore_spent = run_arm(objective, seed, config, arm)
                print(
                    f"  {result.metric}={result.metric_at_budget:.6g}  "
                    f"explored={explore_spent}",
                    flush=True,
                )
                results.append(result)
                splits.append({
                    "objective": objective_name, "strategy": arm,
                    "seed": seed, "explore_evaluations": explore_spent,
                })
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "explore_fraction_fixed": EXPLORE_FRACTION,
            "distill_top": DISTILL_TOP,
            "stall_window": STALL_WINDOW, "stall_tol": STALL_TOL,
            "exploit_reserve": EXPLOIT_RESERVE, "lineage_cap": LINEAGE_CAP,
            "torch_version": torch.__version__,
            "runs": [asdict(r) for r in results],
            "explore_splits": splits,
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
