"""Round 4: is the latent seam valuable to a strong optimizer?

Round 3 confirmed the bias-matching effect (DCT decoder beats MLP 10/0 on the
smooth target, ties on the rough control) — but direct CMA-ES still beat every
latent-GA variant by ~8x, because MSE targets are spheres and covariance/step
adaptation crushes fixed-sigma latent mutation. Two hypotheses remain:

  1. The decoder seam is real but the outer GA is the weak link: CMA-ES run
     *inside* a structure-matched latent space should inherit both advantages.
  2. Direct CMA-ES only wins because the round-3 landscapes are unimodal; a
     rugged landscape around a structured optimum should punish 256-dim direct
     search harder than 32-dim latent search.

Objectives:

  * smooth1d_256    — round 3's structured sphere (representation floor: the
                      sigmoid pre-image of the target is not exactly in the
                      DCT-32 span, so latent methods have a nonzero asymptote);
  * rugged_smooth_256 — Rastrigin ruggedness centred on the same smooth
                      target: the optimum lies on the manifold, the landscape
                      is multimodal everywhere.

Strategies: direct CMA-ES, CMA-ES over the 32-dim latent of the DCT decoder
(matched bias) and of a random frozen MLP (compression without bias), plus the
direct GA and the round-3 DCT+GA for continuity. Run at more than one budget:
compressed search should win early and may lose late to the representation
floor.
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

from latentspace import MLPDecoder

from benchmarks.compare import (
    BenchmarkConfig,
    Objective,
    TrackedFitness,
    _finish_result,
    _require_mps,
    _seed_everything,
    _warm_mps,
    print_summary,
    run_direct_ga,
    run_random_search,
    summarize,
)
from benchmarks.round1_deceptive import run_cmaes
from benchmarks.round3_structure import DCTDecoder, SmoothTarget, run_custom_decoder


class RuggedSmooth(SmoothTarget):
    """Rastrigin ruggedness centred on the smooth low-frequency target."""

    name = "rugged_smooth_256"
    metric_name = "rastrigin"

    def _values(self, phenotypes, target):
        return (phenotypes - target) * 10.24

    def loss_numpy(self, phenotypes):
        values = (np.asarray(phenotypes) - self.target) * 10.24
        return 10 * self.dimension + np.sum(
            values**2 - 10 * np.cos(2 * np.pi * values), axis=1
        )

    def loss_tensor(self, phenotypes):
        target = torch.as_tensor(
            self.target, device=phenotypes.device, dtype=phenotypes.dtype
        )
        values = (phenotypes - target) * 10.24
        return 10 * self.dimension + torch.sum(
            values**2 - 10 * torch.cos(2 * torch.pi * values), dim=1
        )


OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "rugged_smooth_256": RuggedSmooth,
}


def _cma_minimize(evaluate_batch, dim, budget_evaluations, evaluations_done,
                  rng, mean0, sigma0, lam=None):
    """Generic unbounded (mu/mu_w, lambda)-CMA-ES; returns generations run.

    `evaluate_batch(X) -> losses` must record evaluations itself; the loop
    stops exactly at the budget, discarding the final partial generation for
    distribution updates.
    """
    lam = lam if lam is not None else 4 + int(3 * np.log(dim))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights**2)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

    mean = np.array(mean0, dtype=np.float64)
    sigma = float(sigma0)
    covariance = np.eye(dim)
    ps = np.zeros(dim)
    pc = np.zeros(dim)
    generations = 0
    spent = evaluations_done

    while spent < budget_evaluations:
        covariance = (covariance + covariance.T) / 2
        eigenvalues, basis = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        scales = np.sqrt(eigenvalues)
        inv_sqrt_c = (basis / scales) @ basis.T

        n_sample = min(lam, budget_evaluations - spent)
        z = rng.standard_normal((n_sample, dim))
        y = z @ (basis * scales).T
        x = mean + sigma * y
        losses = evaluate_batch(x.astype(np.float32))
        spent += n_sample
        if n_sample < lam:
            break

        order = np.argsort(losses)
        selected = y[order[:mu]]
        y_weighted = weights @ selected
        mean = mean + sigma * y_weighted
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_c @ y_weighted)
        generations += 1
        hsig = (
            np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * generations)) / chi_n
            < 1.4 + 2 / (dim + 1)
        )
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_weighted
        covariance = (
            (1 - c1 - cmu) * covariance
            + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * covariance)
            + cmu * (selected.T * weights) @ selected
        )
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))

    return generations


def run_latent_cma(objective, seed, config, decoder_factory, label):
    """CMA-ES over the latent space of a frozen decoder (mean 0, sigma 1)."""
    _require_mps()
    _seed_everything(seed)
    tracker = TrackedFitness(objective)
    decoder = decoder_factory(config.latent, (objective.dimension,), "mps")
    rng = np.random.default_rng(seed)

    def evaluate_batch(latents):
        phenotypes = decoder.decode(latents)
        return -tracker(phenotypes).detach().cpu().numpy()

    started = time.perf_counter()
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


def _dct_factory(latent, output_shape, device):
    return DCTDecoder(latent, output_shape, device=device)


def _mlp_factory(latent, output_shape, device):
    return MLPDecoder(
        latent, output_shape, hidden_size=128, num_layers=2, lr=1e-3, device=device
    )


STRATEGIES: dict[str, Callable] = {
    "random_search": run_random_search,
    "direct_ga": run_direct_ga,
    "cmaes": run_cmaes,
    "cmaes_dct32": lambda o, s, c: run_latent_cma(
        o, s, c, _dct_factory, "cmaes_dct32"
    ),
    "cmaes_mlp32": lambda o, s, c: run_latent_cma(
        o, s, c, _mlp_factory, "cmaes_mlp32"
    ),
    "latent_dct_ga": lambda o, s, c: run_custom_decoder(
        o, s, c, _dct_factory, "latent_dct_ga", refine=False
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
    has_neural = any("latent" in name or name.endswith("32") for name in args.strategies)
    if has_neural:
        _require_mps()
    results = []
    for objective_name in args.objectives:
        if has_neural:
            _warm_mps(OBJECTIVES[objective_name](), config)
        for strategy_name in args.strategies:
            for seed in args.seeds:
                objective = OBJECTIVES[objective_name]()
                print(
                    f"run objective={objective_name:<18} strategy={strategy_name:<15} "
                    f"seed={seed} budget={config.evaluation_budget}",
                    flush=True,
                )
                result = STRATEGIES[strategy_name](objective, seed, config)
                print(
                    f"  {result.metric}={result.metric_at_budget:.6g} "
                    f"evals_run={result.evaluations_run} "
                    f"device={result.neural_device or 'numpy/cpu'}",
                    flush=True,
                )
                results.append(result)
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "torch_version": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "runs": [asdict(result) for result in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
