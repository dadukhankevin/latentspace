"""Round 7: the family-pretraining scaling law, across problem types.

Round 6 found that a PCA-32 decoder pretrained on K instances of a problem
family transfers to fresh instances, with quality scaling in K (8: fails,
32: beats direct GA, 128: 3.3x better). This round measures the law properly:
K in {8, 16, 32, 64, 128}, ten seeds each, across four families:

  * smooth1d_256      — 16-dim linear manifold, unimodal MSE landscape;
  * rugged_smooth_256 — same manifold style, multimodal Rastrigin landscape
                        (a different fitness topology);
  * image2d_1024      — 32x32 images from 25 random low-frequency 2D-DCT
                        components (a different domain, 4x the dimension);
  * rough1d_256       — full-rank family, the no-structure control: the law
                        must NOT appear here.

Pretraining cost is K x 2,000 direct-GA evaluations per family, amortizable
across every future instance; each fresh test instance gets exactly 5,000
evaluations. Direct GA on the test instance is the common reference.
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
    print_summary,
    run_direct_ga,
    summarize,
)
from benchmarks.round3_structure import RoughTarget, SmoothTarget, _dct_basis
from benchmarks.round4_latent_cma import RuggedSmooth
from benchmarks.round6_learned_structure import run_cmaes_pca_family


class SmoothImage2D(Objective):
    """32x32 image from 25 random low-frequency 2D-DCT atoms (flat 1024)."""

    name = "image2d_1024"
    metric_name = "mse"

    def __init__(self, size: int = 32, grid: int = 5, instance_seed: int = 2026):
        self.dimension = size * size
        rng = np.random.default_rng(instance_seed)
        basis_y = _dct_basis(size, grid)
        basis_x = _dct_basis(size, grid)
        atoms = np.einsum("hu,wv->uvhw", basis_y, basis_x).reshape(grid * grid, -1)
        atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
        frequencies = (
            np.add.outer(np.arange(grid), np.arange(grid)).reshape(-1) + 1.0
        )
        amplitudes = rng.normal(size=grid * grid) / frequencies
        signal = amplitudes @ atoms
        low, high = signal.min(), signal.max()
        self.target = (0.05 + 0.9 * (signal - low) / (high - low)).astype(np.float32)

    def loss_numpy(self, phenotypes):
        return np.mean((np.asarray(phenotypes) - self.target) ** 2, axis=1)

    def loss_tensor(self, phenotypes):
        target = torch.as_tensor(
            self.target, device=phenotypes.device, dtype=phenotypes.dtype
        )
        return torch.mean((phenotypes - target) ** 2, dim=1)


OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "rugged_smooth_256": RuggedSmooth,
    "image2d_1024": SmoothImage2D,
    "rough1d_256": RoughTarget,
}

FAMILY_SIZES = (8, 16, 32, 64, 128)


def make_family_strategy(k: int):
    return lambda o, s, c: run_cmaes_pca_family(
        o, s, c,
        per_instance_evaluations=2_000,
        elites_per_instance=10,
        instance_seeds=tuple(range(100, 100 + k)),
        label=f"family{k}",
    )


STRATEGIES: dict[str, Callable] = {
    "direct_ga": run_direct_ga,
    **{f"family{k}": make_family_strategy(k) for k in FAMILY_SIZES},
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
                    f"run objective={objective_name:<18} strategy={strategy_name:<10} "
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
            "family_sizes": list(FAMILY_SIZES),
            "per_instance_evaluations": 2_000,
            "elites_per_instance": 10,
            "torch_version": torch.__version__,
            "runs": [asdict(result) for result in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
