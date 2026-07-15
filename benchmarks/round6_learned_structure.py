"""Round 6: can the decoder LEARN the structure the DCT oracle was given?

Every trainer tested so far regresses the decoder toward its own output on
the current best individual — a self-referential point target that can only
collapse the mapping, never learn the solution manifold. This round replaces
that with the EDA view: the elite set is rich training data (full phenotype
vectors, not one scalar reward), and the decoder should be a generative model
of it. The linear instantiation is PCA in logit space.

Strategies (5,000-evaluation budget unless noted):

  * cmaes_pca32          — 1,000 evals of direct-GA bootstrap, PCA-32 fit on
                           the elites, CMA-ES in the learned latent for the
                           remaining 4,000. All evaluations counted.
  * cmaes_pca32_refit    — same, but the basis is refit from the growing
                           archive every 1,000 evaluations (CMA restarts).
  * cmaes_pca32_transfer — basis fit on elites harvested from a DIFFERENT
                           instance (2,000 pretraining evals, reported
                           separately as amortizable cost), then the full
                           5,000-eval budget on the test instance. Success
                           here and failure on the rough control would be a
                           learned, transferable genetic code.

Reference points from rounds 3-4 on the same objectives and seeds: oracle
DCT+CMA 0.0011, direct CMA-ES 0.00075, direct GA 0.014, random-MLP latents
~0.078 (smooth1d_256, 5k).
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

from latentspace import Decoder

from benchmarks.compare import (
    BenchmarkConfig,
    Objective,
    TrackedFitness,
    _finish_result,
    _rank_probabilities,
    _require_mps,
    _seed_everything,
    print_summary,
    run_direct_ga,
    summarize,
)
from benchmarks.round3_structure import RoughTarget, SmoothTarget
from benchmarks.round4_latent_cma import _cma_minimize

OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "rough1d_256": RoughTarget,
}

PRETRAIN_INSTANCE_SEED = 7  # transfer arm harvests elites from this instance


class PCADecoder(Decoder):
    """sigmoid(mean + z @ basis): a linear generative model of elite logits."""

    def __init__(self, mean: np.ndarray, basis: np.ndarray, device: str = "cpu"):
        super().__init__(basis.shape[0], (mean.shape[0],), device)
        self.mean = torch.as_tensor(mean.astype(np.float32), device=device)
        self.basis_t = torch.as_tensor(basis.astype(np.float32), device=device)

    def decode(self, genes_batch) -> torch.Tensor:
        genes = torch.as_tensor(
            np.asarray(genes_batch, dtype=np.float32), device=self.device
        )
        with torch.no_grad():
            out = torch.sigmoid(self.mean + genes @ self.basis_t)
        return out.view(-1, *self.output_shape)


def fit_pca_decoder(phenotypes, losses, latent, device, top=200):
    """PCA of the top elites in logit space; latent scaled to unit variance."""
    order = np.argsort(losses)[:top]
    elites = np.clip(np.asarray(phenotypes)[order], 1e-3, 1 - 1e-3)
    logits = np.log(elites / (1 - elites))
    mean = logits.mean(axis=0)
    centered = logits - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(latent, len(singular))
    scale = singular[:k] / np.sqrt(max(len(elites) - 1, 1))
    basis = (scale[:, None] * vt[:k]).astype(np.float32)   # (k, dim)
    if k < latent:
        basis = np.vstack([basis, np.zeros((latent - k, mean.size), np.float32)])
    return PCADecoder(mean, basis, device=device)


def _bootstrap_direct_ga(objective, rng, tracker, config, evaluations):
    """Direct-GA warm-up that records every evaluated phenotype."""
    dimension = objective.dimension
    archive_x: list[np.ndarray] = []
    archive_loss: list[float] = []

    def evaluate(batch):
        fitness = tracker.evaluate_numpy(batch)
        archive_x.extend(np.asarray(batch, dtype=np.float32))
        archive_loss.extend((-fitness).tolist())
        return fitness

    population = rng.random((config.population, dimension), dtype=np.float32)
    fitness = evaluate(population)
    target_evaluations = tracker.evaluations - len(population) + evaluations
    while tracker.evaluations < target_evaluations:
        amount = min(config.offspring, target_evaluations - tracker.evaluations)
        order = np.argsort(-fitness)
        ranked = population[order]
        probabilities = _rank_probabilities(len(ranked))
        parents = rng.choice(len(ranked), size=(amount, 2), replace=True, p=probabilities)
        first, second = ranked[parents[:, 0]], ranked[parents[:, 1]]
        mask = rng.random((amount, dimension)) < 0.5
        children = np.where(mask, first, second).astype(np.float32)
        mutation = rng.random((amount, dimension)) < config.mutation_rate
        noise = rng.normal(0, config.mutation_sigma, (amount, dimension)).astype(np.float32)
        children = np.clip(children + noise * mutation, 0, 1)
        child_fitness = evaluate(children)
        population = np.concatenate([population, children])
        fitness = np.concatenate([fitness, child_fitness])
        keep = np.argsort(-fitness)[: config.population]
        population, fitness = population[keep], fitness[keep]
    return np.asarray(archive_x), np.asarray(archive_loss)


def run_cmaes_pca(objective, seed, config, refit_every=None, label="cmaes_pca32",
                  bootstrap_evaluations=1_000):
    """Bootstrap -> PCA decoder -> CMA-ES in the learned latent, one budget."""
    _require_mps()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    tracker = TrackedFitness(objective)
    started = time.perf_counter()

    archive_x, archive_loss = _bootstrap_direct_ga(
        objective, rng, tracker, config, bootstrap_evaluations
    )
    archive_x, archive_loss = list(archive_x), list(archive_loss)
    generations = 0
    while tracker.evaluations < config.evaluation_budget:
        decoder = fit_pca_decoder(
            np.asarray(archive_x), np.asarray(archive_loss), config.latent, "mps"
        )

        def evaluate_batch(latents):
            phenotypes = decoder.decode(latents)
            fitness = tracker(phenotypes).detach().cpu().numpy()
            flat = phenotypes.detach().cpu().numpy().reshape(len(latents), -1)
            archive_x.extend(flat)
            archive_loss.extend((-fitness).tolist())
            return -fitness

        chunk_end = (
            min(tracker.evaluations + refit_every, config.evaluation_budget)
            if refit_every
            else config.evaluation_budget
        )
        generations += _cma_minimize(
            evaluate_batch,
            dim=config.latent,
            budget_evaluations=chunk_end,
            evaluations_done=tracker.evaluations,
            rng=rng,
            mean0=np.zeros(config.latent),
            sigma0=1.0,
        )
    torch.mps.synchronize()
    result = _finish_result(
        objective, label, seed, config, tracker, started,
        generations=generations, neural_device="mps",
    )
    torch.mps.empty_cache()
    return result


def run_cmaes_pca_transfer(objective, seed, config, pretrain_evaluations=2_000):
    """Fit the basis on a different instance's elites; solve this one fresh."""
    _require_mps()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)

    pretrain_objective = type(objective)(instance_seed=PRETRAIN_INSTANCE_SEED)
    pretrain_tracker = TrackedFitness(pretrain_objective)
    archive_x, archive_loss = _bootstrap_direct_ga(
        pretrain_objective, rng, pretrain_tracker, config, pretrain_evaluations
    )
    decoder = fit_pca_decoder(archive_x, archive_loss, config.latent, "mps")

    tracker = TrackedFitness(objective)
    started = time.perf_counter()

    def evaluate_batch(latents):
        return -tracker(decoder.decode(latents)).detach().cpu().numpy()

    generations = _cma_minimize(
        evaluate_batch,
        dim=config.latent,
        budget_evaluations=config.evaluation_budget,
        evaluations_done=0,
        rng=rng,
        mean0=np.zeros(config.latent),
        sigma0=1.0,
    )
    torch.mps.synchronize()
    result = _finish_result(
        objective, "cmaes_pca32_transfer", seed, config, tracker, started,
        generations=generations, neural_device="mps",
    )
    torch.mps.empty_cache()
    return result


FAMILY_INSTANCE_SEEDS = (7, 11, 13, 17, 19, 23, 29, 31)


def run_cmaes_pca_family(objective, seed, config, per_instance_evaluations=1_000,
                         elites_per_instance=40,
                         instance_seeds=FAMILY_INSTANCE_SEEDS,
                         label="cmaes_pca32_family"):
    """Fit the basis on elites pooled across MANY instances of the family.

    A single instance's elite variance is dominated by the direction toward
    that one target; the span of many instances' directions is the family
    manifold itself. Pretraining cost (8 x 1,000 evals) is reported separately
    as amortizable; the test instance gets the full fresh budget.
    """
    _require_mps()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)

    pooled_x: list[np.ndarray] = []
    pooled_loss: list[float] = []
    for instance_seed in instance_seeds:
        pretrain_objective = type(objective)(instance_seed=instance_seed)
        pretrain_tracker = TrackedFitness(pretrain_objective)
        archive_x, archive_loss = _bootstrap_direct_ga(
            pretrain_objective, rng, pretrain_tracker, config,
            per_instance_evaluations,
        )
        order = np.argsort(archive_loss)[:elites_per_instance]
        pooled_x.extend(archive_x[order])
        pooled_loss.extend(archive_loss[order])
    decoder = fit_pca_decoder(
        np.asarray(pooled_x), np.asarray(pooled_loss), config.latent, "mps",
        top=len(pooled_x),
    )

    tracker = TrackedFitness(objective)
    started = time.perf_counter()

    def evaluate_batch(latents):
        return -tracker(decoder.decode(latents)).detach().cpu().numpy()

    generations = _cma_minimize(
        evaluate_batch,
        dim=config.latent,
        budget_evaluations=config.evaluation_budget,
        evaluations_done=0,
        rng=rng,
        mean0=np.zeros(config.latent),
        sigma0=1.0,
    )
    torch.mps.synchronize()
    result = _finish_result(
        objective, label, seed, config, tracker, started,
        generations=generations, neural_device="mps",
    )
    torch.mps.empty_cache()
    return result


STRATEGIES: dict[str, Callable] = {
    "direct_ga": run_direct_ga,
    "cmaes_pca32": lambda o, s, c: run_cmaes_pca(o, s, c),
    "cmaes_pca32_refit": lambda o, s, c: run_cmaes_pca(
        o, s, c, refit_every=1_000, label="cmaes_pca32_refit"
    ),
    "cmaes_pca32_transfer": run_cmaes_pca_transfer,
    "cmaes_pca32_family": run_cmaes_pca_family,
    # 32 instances > 16 family-manifold dimensions; deeper per-instance
    # bootstrap gives cleaner elites. Pretraining cost 32 x 2,000 evals,
    # amortized across every future instance of the family.
    "cmaes_pca32_family32": lambda o, s, c: run_cmaes_pca_family(
        o, s, c,
        per_instance_evaluations=2_000,
        elites_per_instance=10,
        instance_seeds=tuple(range(100, 132)),
        label="cmaes_pca32_family32",
    ),
    "cmaes_pca32_family128": lambda o, s, c: run_cmaes_pca_family(
        o, s, c,
        per_instance_evaluations=2_000,
        elites_per_instance=10,
        instance_seeds=tuple(range(100, 228)),
        label="cmaes_pca32_family128",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES)
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES)
    )
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
        for strategy_name in args.strategies:
            for seed in args.seeds:
                objective = OBJECTIVES[objective_name]()
                print(
                    f"run objective={objective_name:<14} strategy={strategy_name:<21} "
                    f"seed={seed} budget={config.evaluation_budget}",
                    flush=True,
                )
                result = STRATEGIES[strategy_name](objective, seed, config)
                print(
                    f"  {result.metric}={result.metric_at_budget:.6g} "
                    f"evals_run={result.evaluations_run}",
                    flush=True,
                )
                results.append(result)
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "pretrain_instance_seed": PRETRAIN_INSTANCE_SEED,
            "transfer_pretrain_evaluations": 2_000,
            "torch_version": torch.__version__,
            "runs": [asdict(result) for result in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
