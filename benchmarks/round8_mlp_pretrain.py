"""Round 8: a neural decoder on the SAME pretraining corpus as PCA.

Round 7 established the family-pretraining scaling law with a purely linear
decoder (PCA-32 of pooled practice-run elites). PCA can only learn flat
structure; this round asks whether a neural decoder trained on the identical
corpus (a) matches PCA where the family manifold is flat, and (b) beats it
where the manifold is curved.

The corpus, per run: for each of K = 128 practice instances (seeds 100..227),
run a decoder-free direct GA for 2,000 evaluations and keep the 10 best
phenotypes — 1,280 vectors of "decent solutions to sibling problems." The
test instance (seed 2026) is never in the corpus. Fitness only selects the
corpus; nothing backpropagates from the objective.

Decoders fit on that corpus, then frozen, searched by CMA-ES (5,000 fresh
evaluations):

  * pca128 — round 7's linear fit (flat map);
  * mlp128 — a 32-bottleneck autoencoder trained on elite logits; CMA-ES
             searches its code space, standardized by the training codes.

Families:

  * smooth1d_256 — flat manifold: the MLP should at best tie PCA;
  * blob2d_1024  — curved manifold (three Gaussian blobs, 12 nonlinear
                   parameters): PCA should floor, the MLP may pass it;
  * rough1d_256  — no structure: both must fail, or the method is a trick.
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

from latentspace import Decoder

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
from benchmarks.round3_structure import RoughTarget, SmoothTarget
from benchmarks.round4_latent_cma import _cma_minimize
from benchmarks.round6_learned_structure import (
    _bootstrap_direct_ga,
    fit_pca_decoder,
)

PER_INSTANCE_EVALUATIONS = 2_000
ELITES_PER_INSTANCE = 10
FAMILY_SIZES = (8, 16, 32, 64, 128)


class BlobImage2D(Objective):
    """32x32 image of three Gaussian blobs — a curved 12-parameter family."""

    name = "blob2d_1024"
    metric_name = "mse"

    def __init__(self, size: int = 32, blobs: int = 3, instance_seed: int = 2026):
        self.dimension = size * size
        rng = np.random.default_rng(instance_seed)
        grid = (np.arange(size) + 0.5) / size
        yy, xx = np.meshgrid(grid, grid, indexing="ij")
        image = np.zeros((size, size), dtype=np.float64)
        for _ in range(blobs):
            cx, cy = rng.uniform(0.2, 0.8, 2)
            radius = rng.uniform(0.06, 0.18)
            amplitude = rng.uniform(0.4, 1.0)
            image += amplitude * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2)
            )
        low, high = image.min(), image.max()
        image = 0.05 + 0.9 * (image - low) / max(high - low, 1e-8)
        self.target = image.reshape(-1).astype(np.float32)

    def loss_numpy(self, phenotypes):
        return np.mean((np.asarray(phenotypes) - self.target) ** 2, axis=1)

    def loss_tensor(self, phenotypes):
        target = torch.as_tensor(
            self.target, device=phenotypes.device, dtype=phenotypes.dtype
        )
        return torch.mean((phenotypes - target) ** 2, dim=1)


OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "blob2d_1024": BlobImage2D,
    "rough1d_256": RoughTarget,
}


def harvest_corpus(objective, rng, config, k: int = 128):
    """Elite phenotypes from cheap direct-GA runs on K practice instances."""
    pooled_x, pooled_loss = [], []
    for instance_seed in range(100, 100 + k):
        practice = type(objective)(instance_seed=instance_seed)
        tracker = TrackedFitness(practice)
        archive_x, archive_loss = _bootstrap_direct_ga(
            practice, rng, tracker, config, PER_INSTANCE_EVALUATIONS
        )
        order = np.argsort(archive_loss)[:ELITES_PER_INSTANCE]
        pooled_x.extend(np.asarray(archive_x)[order])
        pooled_loss.extend(np.asarray(archive_loss)[order])
    return np.asarray(pooled_x, dtype=np.float32), np.asarray(pooled_loss)


class EliteAutoencoderDecoder(Decoder):
    """32-bottleneck autoencoder fit on elite logits; decode from code space.

    CMA-ES proposes standardized codes z; the decoder maps them through the
    training-code statistics so z ~ N(0, I) covers the elite code cloud.
    """

    def __init__(self, elites: np.ndarray, latent: int, device: str,
                 hidden: int = 128, steps: int = 2_000, batch: int = 128,
                 lr: float = 1e-3, seed: int = 0):
        dim = elites.shape[1]
        super().__init__(latent, (dim,), device)
        generator = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)

        clipped = np.clip(elites, 1e-3, 1 - 1e-3)
        logits = torch.as_tensor(
            np.log(clipped / (1 - clipped)), dtype=torch.float32, device=device
        )
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.LeakyReLU(), nn.Linear(hidden, latent)
        ).to(device)
        self.decoder_net = nn.Sequential(
            nn.Linear(latent, hidden), nn.LeakyReLU(), nn.Linear(hidden, dim)
        ).to(device)
        optimizer = torch.optim.Adam(
            [*self.encoder.parameters(), *self.decoder_net.parameters()], lr=lr
        )
        loss_fn = nn.MSELoss()
        n = len(logits)
        for _ in range(steps):
            index = torch.randint(0, n, (min(batch, n),), generator=generator)
            batch_logits = logits[index.to(device)]
            optimizer.zero_grad()
            reconstruction = self.decoder_net(self.encoder(batch_logits))
            loss = loss_fn(reconstruction, batch_logits)
            loss.backward()
            optimizer.step()
        self.final_loss = float(loss.item())

        with torch.no_grad():
            codes = self.encoder(logits)
            self.code_mean = codes.mean(dim=0)
            self.code_std = codes.std(dim=0).clamp_min(1e-6)
        self.encoder.eval()
        self.decoder_net.eval()

    def decode(self, genes_batch) -> torch.Tensor:
        genes = torch.as_tensor(
            np.asarray(genes_batch, dtype=np.float32), device=self.device
        )
        with torch.no_grad():
            codes = self.code_mean + genes * self.code_std
            out = torch.sigmoid(self.decoder_net(codes))
        return out.view(-1, *self.output_shape)


def run_pretrained(objective, seed, config, kind, k: int = 128):
    _require_mps()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    corpus_x, corpus_loss = harvest_corpus(objective, rng, config, k=k)

    if kind == "pca":
        decoder = fit_pca_decoder(
            corpus_x, corpus_loss, config.latent, "mps", top=len(corpus_x)
        )
    elif kind == "mlp":
        decoder = EliteAutoencoderDecoder(
            corpus_x, config.latent, "mps", seed=seed
        )
    else:
        raise ValueError(kind)

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
        objective, f"{kind}{k}", seed, config, tracker, started,
        generations=generations, neural_device="mps",
    )
    torch.mps.empty_cache()
    return result


STRATEGIES: dict[str, Callable] = {
    "direct_ga": run_direct_ga,
    **{
        f"{kind}{k}": (
            lambda o, s, c, kind=kind, k=k: run_pretrained(o, s, c, kind, k=k)
        )
        for kind in ("pca", "mlp")
        for k in FAMILY_SIZES
    },
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
                    f"run objective={objective_name:<14} strategy={strategy_name:<10} "
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
            "practice_seed_base": 100,
            "family_sizes": list(FAMILY_SIZES),
            "per_instance_evaluations": PER_INSTANCE_EVALUATIONS,
            "elites_per_instance": ELITES_PER_INSTANCE,
            "torch_version": torch.__version__,
            "runs": [asdict(result) for result in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
