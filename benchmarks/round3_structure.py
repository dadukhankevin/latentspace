"""Round 3: does decoder inductive bias matched to solution structure pay?

Rounds 1-2 eliminated every claimed habitat: the latent MLP loses to direct
search on deceptive, hierarchical, rugged-binary, and (once the init-centering
artifact is removed by shifting) all continuous objectives. Every benchmark so
far, however, has full-rank solutions — nothing for a 32-dimensional latent to
compress. This round builds the first objectives whose optima genuinely lie on
a low-dimensional manifold, and matches or mismatches the decoder to it:

  * smooth1d_256 — match a random low-frequency signal (16 DCT components,
                   fixed instance). The optimum IS a low-dimensional object.
  * rough1d_256  — match 256 iid uniform values: same dimension, zero
                   structure. Latent-32 methods cannot represent the optimum;
                   this is the falsification control.

Decoders, from oracle bias to no bias:

  * DCT      — fixed linear expansion over the 32 lowest DCT basis functions
               (oracle structure match, zero learning);
  * Conv1D   — learned upsampling stack (generic smoothness prior, the Deep
               Image Prior effect), frozen and gradient-refined;
  * MLP      — the standard unstructured decoder, frozen and gradient-refined.

Prediction if the structure-matching thesis is right: on the smooth target
DCT >> Conv > MLP with direct methods struggling in 256 dimensions; on the
rough target every latent variant loses to direct search.
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
import torch.optim as optim

from latentspace import Decoder, Evolver, MLPDecoder

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
    run_latent,
    run_random_search,
    summarize,
)
from benchmarks.round1_deceptive import run_cmaes


def _dct_basis(dimension: int, components: int) -> np.ndarray:
    """(dimension, components) orthonormal DCT-II basis, lowest frequencies."""
    positions = np.arange(dimension)[:, None] + 0.5
    frequencies = np.arange(components)[None, :]
    basis = np.cos(np.pi * frequencies * positions / dimension)
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    return basis.astype(np.float32)


class SmoothTarget(Objective):
    """Random low-frequency signal: the optimum lies on a 16-dim manifold."""

    name = "smooth1d_256"
    metric_name = "mse"

    def __init__(self, dimension: int = 256, components: int = 16,
                 instance_seed: int = 2026):
        self.dimension = dimension
        rng = np.random.default_rng(instance_seed)
        basis = _dct_basis(dimension, components + 1)[:, 1:]  # skip DC
        amplitudes = rng.normal(size=components) / np.arange(1, components + 1)
        signal = basis @ amplitudes
        low, high = signal.min(), signal.max()
        self.target = (0.05 + 0.9 * (signal - low) / (high - low)).astype(np.float32)

    def loss_numpy(self, phenotypes):
        return np.mean((np.asarray(phenotypes) - self.target) ** 2, axis=1)

    def loss_tensor(self, phenotypes):
        target = torch.as_tensor(
            self.target, device=phenotypes.device, dtype=phenotypes.dtype
        )
        return torch.mean((phenotypes - target) ** 2, dim=1)


class RoughTarget(SmoothTarget):
    """256 iid uniform values: same task, no structure to compress."""

    name = "rough1d_256"

    def __init__(self, dimension: int = 256, instance_seed: int = 2026):
        self.dimension = dimension
        rng = np.random.default_rng(instance_seed)
        self.target = (0.05 + 0.9 * rng.random(dimension)).astype(np.float32)


OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "rough1d_256": RoughTarget,
}


class DCTDecoder(Decoder):
    """Fixed linear low-frequency expansion — oracle structure match."""

    def __init__(self, input_length: int, output_shape, device: str = "cpu"):
        super().__init__(input_length, output_shape, device)
        basis = _dct_basis(int(np.prod(self.output_shape)), input_length)
        row_norms = np.sqrt((basis**2).sum(axis=1, keepdims=True))
        scaled = 2.0 * basis / np.maximum(row_norms, 1e-8)
        self.basis = torch.as_tensor(scaled, device=device)

    def decode(self, genes_batch) -> torch.Tensor:
        genes = torch.as_tensor(
            np.asarray(genes_batch, dtype=np.float32), device=self.device
        )
        with torch.no_grad():
            out = torch.sigmoid(genes @ self.basis.T)
        return out.view(-1, *self.output_shape)


class Conv1DDecoder(MLPDecoder):
    """Learned upsampling stack — a generic smoothness prior.

    Inherits MLPDecoder's refine/evolve_step, which only use forward, the
    optimizer, and the loss; the network is replaced wholesale.
    """

    def __init__(self, input_length: int, output_shape, base_length: int = 16,
                 channels: int = 32, lr: float = 1e-3, device: str = "cpu"):
        super().__init__(
            input_length, output_shape, hidden_size=1, num_layers=1,
            lr=lr, device=device,
        )
        length = int(np.prod(self.output_shape))
        if length % base_length or (length // base_length) & (length // base_length - 1):
            raise ValueError("output length must be base_length times a power of two")
        stages = int(np.log2(length // base_length))
        blocks: list[nn.Module] = [
            nn.Linear(input_length, channels * base_length),
            nn.Unflatten(1, (channels, base_length)),
            nn.LeakyReLU(),
        ]
        width = channels
        for _ in range(stages):
            narrower = max(4, width // 2)
            blocks += [
                nn.ConvTranspose1d(width, narrower, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(),
            ]
            width = narrower
        blocks += [nn.Conv1d(width, 1, kernel_size=5, padding=2), nn.Flatten(1)]
        self.net = nn.Sequential(*blocks).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.opt = self.optimizer


def run_custom_decoder(objective, seed, config, decoder_factory, label, refine):
    _require_mps()
    _seed_everything(seed)
    tracker = TrackedFitness(objective)
    decoder = decoder_factory(config.latent, (objective.dimension,), "mps")
    evolver = Evolver(
        tracker,
        output_shape=(objective.dimension,),
        device="mps",
        latent=config.latent,
        population=config.population,
        mutation_rate=config.mutation_rate,
        mutation_sigma=config.mutation_sigma,
        refine_every=(config.refine_every if refine else None),
        refine_percent=config.refine_percent,
        pressure=1.8,
        scheme="linear",
        families=max(1, config.offspring // 4),
        children=4,
        n_points=4,
        decoder=decoder,
    )
    parameter_devices = {p.device.type for p in evolver.decoder.parameters()}
    if parameter_devices and parameter_devices != {"mps"}:
        raise RuntimeError(f"decoder parameters are not exclusively on MPS: {parameter_devices}")

    started = time.perf_counter()
    while tracker.evaluations < config.evaluation_budget:
        evolver.solve(1, verbose_every=0)
    torch.mps.synchronize()
    result = _finish_result(
        objective, label, seed, config, tracker, started,
        generations=evolver.env.generation, neural_device="mps",
    )
    torch.mps.empty_cache()
    return result


def _dct_factory(latent, output_shape, device):
    return DCTDecoder(latent, output_shape, device=device)


def _conv_factory(latent, output_shape, device):
    return Conv1DDecoder(latent, output_shape, lr=1e-3, device=device)


STRATEGIES: dict[str, Callable] = {
    "random_search": run_random_search,
    "direct_ga": run_direct_ga,
    "cmaes": run_cmaes,
    "latent_mlp_fixed": lambda o, s, c: run_latent(o, s, c, "latent_fixed"),
    "latent_mlp_gradient": lambda o, s, c: run_latent(o, s, c, "latent_gradient"),
    "latent_dct_fixed": lambda o, s, c: run_custom_decoder(
        o, s, c, _dct_factory, "latent_dct_fixed", refine=False
    ),
    "latent_conv_fixed": lambda o, s, c: run_custom_decoder(
        o, s, c, _conv_factory, "latent_conv_fixed", refine=False
    ),
    "latent_conv_gradient": lambda o, s, c: run_custom_decoder(
        o, s, c, _conv_factory, "latent_conv_gradient", refine=True
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
                    f"run objective={objective_name:<14} strategy={strategy_name:<21} "
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
