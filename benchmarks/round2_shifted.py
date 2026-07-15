"""Round 2: shifted continuous objectives — testing the init-centering confound.

Round 1 found every latent variant decisively beating direct search on
Rastrigin-64. But a freshly initialized sigmoid MLP decoder emits phenotypes
clustered at 0.500 +/- 0.026, and unshifted Rastrigin's global optimum sits at
exactly 0.5 in phenotype coordinates: the latent population starts with loss
~456 versus ~1188 for a uniform initialization. The apparent win may be an
architectural head start, not better search.

Following BBOB practice, each objective here hides its optimum at a random
interior point (fixed per instance seed, shared across algorithm seeds):

  * sphere64_shifted    — unimodal control;
  * rastrigin64_shifted — the round-1 winner, off-center;
  * ackley64_shifted    — a second rugged family.

`cmaes_pop64` adds a population-size-64 CMA-ES, the standard remedy for
multimodal landscapes, as a fairer strong baseline than default lambda.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from benchmarks.compare import (
    BenchmarkConfig,
    Objective,
    _require_mps,
    _warm_mps,
    print_summary,
    run_direct_ga,
    run_latent,
    run_random_search,
    summarize,
)
from benchmarks.round1_deceptive import run_cmaes as _run_cmaes_default
from benchmarks.round1_deceptive import run_latent_guarded


def _instance_shift(dimension: int, instance_seed: int) -> np.ndarray:
    """Optimum location in phenotype space, uniform in [0.25, 0.75]^d."""
    rng = np.random.default_rng(instance_seed)
    return rng.uniform(0.25, 0.75, dimension).astype(np.float32)


class ShiftedSphere(Objective):
    name = "sphere64_shifted"
    metric_name = "sphere"

    def __init__(self, dimension: int = 64, instance_seed: int = 2026):
        self.dimension = dimension
        self.center = _instance_shift(dimension, instance_seed)

    def loss_numpy(self, phenotypes):
        return np.sum((np.asarray(phenotypes) - self.center) ** 2, axis=1)

    def loss_tensor(self, phenotypes):
        center = torch.as_tensor(
            self.center, device=phenotypes.device, dtype=phenotypes.dtype
        )
        return torch.sum((phenotypes - center) ** 2, dim=1)


class ShiftedRastrigin(Objective):
    name = "rastrigin64_shifted"
    metric_name = "rastrigin"

    def __init__(self, dimension: int = 64, instance_seed: int = 2026):
        self.dimension = dimension
        self.center = _instance_shift(dimension, instance_seed)

    def _values_numpy(self, phenotypes):
        return (np.asarray(phenotypes) - self.center) * 10.24

    def loss_numpy(self, phenotypes):
        values = self._values_numpy(phenotypes)
        return 10 * self.dimension + np.sum(
            values**2 - 10 * np.cos(2 * np.pi * values), axis=1
        )

    def loss_tensor(self, phenotypes):
        center = torch.as_tensor(
            self.center, device=phenotypes.device, dtype=phenotypes.dtype
        )
        values = (phenotypes - center) * 10.24
        return 10 * self.dimension + torch.sum(
            values**2 - 10 * torch.cos(2 * torch.pi * values), dim=1
        )


class ShiftedAckley(Objective):
    name = "ackley64_shifted"
    metric_name = "ackley"

    def __init__(self, dimension: int = 64, instance_seed: int = 2026):
        self.dimension = dimension
        self.center = _instance_shift(dimension, instance_seed)

    def loss_numpy(self, phenotypes):
        values = (np.asarray(phenotypes) - self.center) * 10.0
        rms = np.sqrt(np.mean(values**2, axis=1))
        cos_mean = np.mean(np.cos(2 * np.pi * values), axis=1)
        return (
            -20 * np.exp(-0.2 * rms) - np.exp(cos_mean) + 20 + np.e
        ).astype(np.float32)

    def loss_tensor(self, phenotypes):
        center = torch.as_tensor(
            self.center, device=phenotypes.device, dtype=phenotypes.dtype
        )
        values = (phenotypes - center) * 10.0
        rms = torch.sqrt(torch.mean(values**2, dim=1))
        cos_mean = torch.mean(torch.cos(2 * torch.pi * values), dim=1)
        return -20 * torch.exp(-0.2 * rms) - torch.exp(cos_mean) + 20 + torch.e


OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "sphere64_shifted": ShiftedSphere,
    "rastrigin64_shifted": ShiftedRastrigin,
    "ackley64_shifted": ShiftedAckley,
}


def run_cmaes_pop64(objective, seed, config):
    """CMA-ES with lambda=64: the standard population fix for multimodality."""
    return _run_cmaes_default(objective, seed, config, lam=64, label="cmaes_pop64")


STRATEGIES: dict[str, Callable] = {
    "random_search": run_random_search,
    "direct_ga": run_direct_ga,
    "cmaes": _run_cmaes_default,
    "cmaes_pop64": run_cmaes_pop64,
    "latent_fixed": lambda o, s, c: run_latent(o, s, c, "latent_fixed"),
    "latent_gradient": lambda o, s, c: run_latent(o, s, c, "latent_gradient"),
    "latent_guarded": run_latent_guarded,
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
    has_neural = any(name.startswith("latent_") for name in args.strategies)
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
                    f"run objective={objective_name:<20} strategy={strategy_name:<16} "
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
