"""Round 5: the GeneSpace image demo, re-run with controls.

The original GeneSpace README evolves a 50x50 RGB image toward a target photo
(MSE fitness) and shows the result with no baseline and no evaluation budget.
Two facts change how that demo should be read:

  * A fresh sigmoid decoder emits every pixel at ~0.5: its starting MSE on the
    actual demo target (0.064) equals a flat gray canvas (0.0625) and is 2.3x
    better than uniform random pixels (0.147) — the round-2 init artifact.
  * Pixel-MSE image matching is a separable unimodal sphere — exactly the
    terrain where rounds 3-4 showed direct search and CMA-ES dominate latent
    GAs, and where a spatially-biased decoder (2D DCT) plus latent CMA-ES is
    the strongest latent configuration.

This suite runs the faithful scale-1 GeneSpace recipe (250 binary genes,
2000-wide sigmoid MLP, GOOD_TO_BEST distillation every 10 generations) against
direct search and the matched-bias latent at identical evaluation budgets.
Pass --image to point at the demo target (a .npy HxWx3 float array in [0,1]);
without it a synthetic smooth target is generated.
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

from latentspace import Decoder, Evolver
from latentspace.decoder import TrainMode

from benchmarks.compare import (
    BenchmarkConfig,
    Objective,
    TrackedFitness,
    _finish_result,
    _require_mps,
    _seed_everything,
    print_summary,
    run_direct_ga,
    run_latent,
    run_random_search,
    summarize,
)
from benchmarks.round3_structure import _dct_basis
from benchmarks.round4_latent_cma import _cma_minimize


class ImageTarget(Objective):
    """Flat MSE against a fixed HxWx3 image in [0,1]."""

    name = "image50"
    metric_name = "mse"

    image: np.ndarray | None = None  # set once by main()

    def __init__(self):
        if ImageTarget.image is None:
            raise RuntimeError("ImageTarget.image must be assigned before use")
        self.shape = ImageTarget.image.shape
        self.target = ImageTarget.image.reshape(-1).astype(np.float32)
        self.dimension = self.target.size

    def loss_numpy(self, phenotypes):
        flat = np.asarray(phenotypes).reshape(len(phenotypes), -1)
        return np.mean((flat - self.target) ** 2, axis=1)

    def loss_tensor(self, phenotypes):
        target = torch.as_tensor(
            self.target, device=phenotypes.device, dtype=phenotypes.dtype
        )
        flat = phenotypes.reshape(len(phenotypes), -1)
        return torch.mean((flat - target) ** 2, dim=1)


def synthetic_target(size: int = 50, instance_seed: int = 2026) -> np.ndarray:
    """Smooth Gaussian-blob image used when no --image is supplied."""
    rng = np.random.default_rng(instance_seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    image = np.zeros((size, size, 3), dtype=np.float32)
    for _ in range(6):
        cx, cy, radius = rng.random(), rng.random(), 0.08 + 0.2 * rng.random()
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2))
        image += blob[..., None] * rng.random(3)
    image -= image.min()
    image /= max(image.max(), 1e-8)
    return (0.05 + 0.9 * image).astype(np.float32)


class DCT2DDecoder(Decoder):
    """Per-channel low-frequency 2D DCT expansion — spatial bias for images."""

    def __init__(self, input_length: int, output_shape, image_shape,
                 device: str = "cpu"):
        super().__init__(input_length, output_shape, device)
        height, width, channels = image_shape
        per_channel = input_length // channels
        grid = int(np.sqrt(per_channel))
        if grid * grid * channels != input_length:
            raise ValueError("input_length must be channels * square")
        basis_y = _dct_basis(height, grid)          # (H, grid)
        basis_x = _dct_basis(width, grid)           # (W, grid)
        atoms = np.einsum("hu,wv->uvhw", basis_y, basis_x).reshape(grid * grid, -1)
        atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
        pixels = height * width
        basis = np.zeros((pixels * channels, input_length), dtype=np.float32)
        image_index = np.arange(pixels)
        for channel in range(channels):
            rows = image_index * channels + channel     # HxWxC flat layout
            cols = slice(channel * per_channel, (channel + 1) * per_channel)
            basis[rows, cols] = atoms.T
        row_norms = np.sqrt((basis**2).sum(axis=1, keepdims=True))
        self.basis = torch.as_tensor(
            2.0 * basis / np.maximum(row_norms, 1e-8), device=device
        )

    def decode(self, genes_batch) -> torch.Tensor:
        genes = torch.as_tensor(
            np.asarray(genes_batch, dtype=np.float32), device=self.device
        )
        with torch.no_grad():
            out = torch.sigmoid(genes @ self.basis.T)
        return out.view(-1, *self.output_shape)


def run_genespace_recipe(objective, seed, config):
    """Faithful scale-1 GeneralEvolution, budget-matched.

    250 binary genes, one 2000-wide hidden layer, sigmoid output, lr 1e-5,
    population 200, exponential rank pressure 20, 8-point crossover, 10% bit
    flips, GOOD_TO_BEST distillation every 10 generations.
    """
    _require_mps()
    _seed_everything(seed)
    tracker = TrackedFitness(objective)
    evolver = Evolver(
        tracker,
        output_shape=(objective.dimension,),
        device="mps",
        latent=250,
        population=200,
        hidden_size=2000,
        num_layers=1,
        lr=1e-5,
        binary=True,
        mutation_rate=0.1,
        mutation_sigma=0.1,
        refine_every=10,
        refine_percent=0.4,
        mode=TrainMode.GOOD_TO_BEST,
        pressure=20.0,
        scheme="exp",
        children=4,
        n_points=8,
    )
    started = time.perf_counter()
    while tracker.evaluations < config.evaluation_budget:
        evolver.solve(1, verbose_every=0)
    torch.mps.synchronize()
    result = _finish_result(
        objective, "genespace_recipe", seed, config, tracker, started,
        generations=evolver.env.generation, neural_device="mps",
    )
    torch.mps.empty_cache()
    return result


def run_cmaes_dct2d(objective, seed, config):
    """CMA-ES over the latent of a frozen per-channel 2D DCT decoder."""
    _require_mps()
    _seed_everything(seed)
    channels = objective.shape[2]
    latent = 64 * channels
    decoder = DCT2DDecoder(
        latent, (objective.dimension,), objective.shape, device="mps"
    )
    tracker = TrackedFitness(objective)
    rng = np.random.default_rng(seed)

    def evaluate_batch(latents):
        return -tracker(decoder.decode(latents)).detach().cpu().numpy()

    started = time.perf_counter()
    generations = _cma_minimize(
        evaluate_batch,
        dim=latent,
        budget_evaluations=config.evaluation_budget,
        evaluations_done=0,
        rng=rng,
        mean0=np.zeros(latent),
        sigma0=1.0,
    )
    torch.mps.synchronize()
    result = _finish_result(
        objective, "cmaes_dct2d", seed, config, tracker, started,
        generations=generations, neural_device="mps",
    )
    torch.mps.empty_cache()
    return result


STRATEGIES: dict[str, Callable] = {
    "random_search": run_random_search,
    "direct_ga": run_direct_ga,
    "genespace_recipe": run_genespace_recipe,
    "latent_mlp_gradient": lambda o, s, c: run_latent(o, s, c, "latent_gradient"),
    "cmaes_dct2d": run_cmaes_dct2d,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--image", type=Path, help=".npy HxWx3 float target")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    ImageTarget.image = (
        np.load(args.image) if args.image else synthetic_target()
    )
    config = BenchmarkConfig(evaluation_budget=args.budget)
    _require_mps()

    reference = ImageTarget()
    gray = np.full((1, reference.dimension), 0.5, dtype=np.float32)
    print(f"reference: constant-gray MSE {reference.loss_numpy(gray)[0]:.4f}")

    results = []
    for strategy_name in args.strategies:
        for seed in args.seeds:
            objective = ImageTarget()
            print(
                f"run strategy={strategy_name:<19} seed={seed} "
                f"budget={config.evaluation_budget}",
                flush=True,
            )
            result = STRATEGIES[strategy_name](objective, seed, config)
            print(
                f"  mse={result.metric_at_budget:.6g} "
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
            "torch_version": torch.__version__,
            "reference_gray_mse": float(reference.loss_numpy(gray)[0]),
            "runs": [asdict(result) for result in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
