"""Round 10: warm-start from the PCA scaffold, then refine on real candidates.

The proposal: kickstart the decoder so candidates are good immediately (the
round-9 PCA-augmented neural fit), then during evolution keep training the
decoder on the better real candidates that CMA-ES discovers. This is NOT the
round-6 self-locking trap, for two reasons:

  * the decoder is nonlinear, so training on points near-but-off the current
    manifold can bend it toward them (a linear PCA refit provably cannot);
  * CMA-ES finds better candidates than the weak direct-GA bootstrap that
    built the original corpus, so the refit data is genuinely new, not
    recycled search output.

Refinement uses supervised regression on full phenotypes (the EDA/GLO view),
not policy-gradient RL: the whole good phenotype is richer than one reward
scalar. Refit data is candidates the search already evaluated, so online
refinement costs zero extra objective evaluations. Because retraining changes
the latent->phenotype map (the repo's core non-stationarity), CMA is
re-anchored after each refit at the encoding of the current best phenotype.

Arms at K = 128 bootstrap, 5,000 fresh evaluations:

  * frozen_aug — round-9 winner: PCA-augmented MLP, frozen, one CMA run;
  * online     — same warm start, then 5 epochs of refine-and-reanchor;
  * pca        — linear reference.

Families: smooth1d_256 (flat, expect neutral) and blob2d_1024 (curved,
expect online to help if anything does).
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
    summarize,
)
from benchmarks.round3_structure import SmoothTarget
from benchmarks.round4_latent_cma import _cma_minimize
from benchmarks.round6_learned_structure import fit_pca_decoder
from benchmarks.round8_mlp_pretrain import BlobImage2D, harvest_corpus

OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "blob2d_1024": BlobImage2D,
}

EPOCHS = 5
REFIT_STEPS = 2_000
SYNTHETIC = 5_000
HARVEST_PER_EPOCH = 60


def _to_logits(phenotypes: np.ndarray) -> np.ndarray:
    clipped = np.clip(phenotypes, 1e-3, 1 - 1e-3)
    return np.log(clipped / (1 - clipped)).astype(np.float32)


class RefinableDecoder(Decoder):
    """Autoencoder with a persistent optimizer that can be trained repeatedly."""

    def __init__(self, dim: int, latent: int, device: str, hidden: int = 128,
                 lr: float = 1e-3, seed: int = 0):
        super().__init__(latent, (dim,), device)
        self.generator = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.LeakyReLU(), nn.Linear(hidden, latent)
        ).to(device)
        self.decoder_net = nn.Sequential(
            nn.Linear(latent, hidden), nn.LeakyReLU(), nn.Linear(hidden, dim)
        ).to(device)
        self.optimizer = torch.optim.Adam(
            [*self.encoder.parameters(), *self.decoder_net.parameters()], lr=lr
        )
        self.loss_fn = nn.MSELoss()
        self.code_mean = torch.zeros(latent, device=device)
        self.code_std = torch.ones(latent, device=device)

    def fit(self, logits_np: np.ndarray, steps: int, batch: int = 128):
        logits = torch.as_tensor(logits_np, dtype=torch.float32, device=self.device)
        n = len(logits)
        self.encoder.train()
        self.decoder_net.train()
        for _ in range(steps):
            index = torch.randint(0, n, (min(batch, n),), generator=self.generator)
            batch_logits = logits[index.to(self.device)]
            self.optimizer.zero_grad()
            reconstruction = self.decoder_net(self.encoder(batch_logits))
            loss = self.loss_fn(reconstruction, batch_logits)
            loss.backward()
            self.optimizer.step()
        self.encoder.eval()
        self.decoder_net.eval()
        with torch.no_grad():
            codes = self.encoder(logits)
            self.code_mean = codes.mean(dim=0)
            self.code_std = codes.std(dim=0).clamp_min(1e-6)

    def decode(self, genes_batch) -> torch.Tensor:
        genes = torch.as_tensor(
            np.asarray(genes_batch, dtype=np.float32), device=self.device
        )
        with torch.no_grad():
            out = torch.sigmoid(self.decoder_net(self.code_mean + genes * self.code_std))
        return out.view(-1, *self.output_shape)

    def anchor_z(self, phenotype: np.ndarray) -> np.ndarray:
        """Standardized latent whose decode best matches a target phenotype."""
        logits = torch.as_tensor(
            _to_logits(phenotype[None]), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            code = self.encoder(logits)[0]
            z = (code - self.code_mean) / self.code_std
        return z.detach().cpu().numpy()


def _synthetic_from_pca(pca_decoder, rng, latent, count):
    z = rng.standard_normal((count, latent)).astype(np.float32)
    samples = pca_decoder.decode(z).detach().cpu().numpy().reshape(count, -1)
    return _to_logits(samples)


def run_arm(objective, seed, config, arm):
    _require_mps()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    corpus_x, corpus_loss = harvest_corpus(objective, rng, config, k=128)
    pca = fit_pca_decoder(corpus_x, corpus_loss, config.latent, "mps", top=len(corpus_x))

    if arm == "pca":
        tracker = TrackedFitness(objective)
        started = time.perf_counter()
        _cma_minimize(
            lambda z: -tracker(pca.decode(z)).detach().cpu().numpy(),
            dim=config.latent, budget_evaluations=config.evaluation_budget,
            evaluations_done=0, rng=rng, mean0=np.zeros(config.latent), sigma0=1.0,
        )
        torch.mps.synchronize()
        result = _finish_result(objective, "pca", seed, config, tracker,
                                started, neural_device="mps")
        torch.mps.empty_cache()
        return result

    synthetic = _synthetic_from_pca(pca, rng, config.latent, SYNTHETIC)
    real_logits = _to_logits(corpus_x)
    decoder = RefinableDecoder(objective.dimension, config.latent, "mps", seed=seed)
    decoder.fit(np.concatenate([real_logits, synthetic]), steps=8_000)

    tracker = TrackedFitness(objective)
    started = time.perf_counter()

    if arm == "frozen_aug":
        _cma_minimize(
            lambda z: -tracker(decoder.decode(z)).detach().cpu().numpy(),
            dim=config.latent, budget_evaluations=config.evaluation_budget,
            evaluations_done=0, rng=rng, mean0=np.zeros(config.latent), sigma0=1.0,
        )
        torch.mps.synchronize()
        result = _finish_result(objective, "frozen_aug", seed, config, tracker,
                                started, neural_device="mps")
        torch.mps.empty_cache()
        return result

    if arm != "online":
        raise ValueError(arm)

    accumulated = [real_logits]
    mean_z = np.zeros(config.latent)
    per_epoch = config.evaluation_budget // EPOCHS
    for epoch in range(EPOCHS):
        epoch_x: list[np.ndarray] = []
        epoch_loss: list[float] = []

        def evaluate_batch(z):
            phenotypes = decoder.decode(z)
            fitness = tracker(phenotypes).detach().cpu().numpy()
            epoch_x.extend(phenotypes.detach().cpu().numpy().reshape(len(z), -1))
            epoch_loss.extend((-fitness).tolist())
            return -fitness

        chunk_end = min(config.evaluation_budget, (epoch + 1) * per_epoch)
        if epoch == EPOCHS - 1:
            chunk_end = config.evaluation_budget
        _cma_minimize(
            evaluate_batch, dim=config.latent, budget_evaluations=chunk_end,
            evaluations_done=tracker.evaluations, rng=rng,
            mean0=mean_z, sigma0=(1.0 if epoch == 0 else 0.5),
        )
        if epoch == EPOCHS - 1:
            break
        # Harvest this epoch's best real candidates, refit, re-anchor.
        order = np.argsort(epoch_loss)[:HARVEST_PER_EPOCH]
        accumulated.append(_to_logits(np.asarray(epoch_x)[order]))
        training = np.concatenate([*accumulated, synthetic])
        decoder.fit(training, steps=REFIT_STEPS)
        mean_z = decoder.anchor_z(tracker.best_phenotype)

    torch.mps.synchronize()
    result = _finish_result(objective, "online", seed, config, tracker, started,
                            neural_device="mps")
    torch.mps.empty_cache()
    return result


STRATEGIES = ("pca", "frozen_aug", "online")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES)
    )
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
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
        for arm in args.strategies:
            for seed in args.seeds:
                objective = OBJECTIVES[objective_name]()
                print(
                    f"run objective={objective_name:<14} arm={arm:<11} "
                    f"seed={seed} budget={config.evaluation_budget}",
                    flush=True,
                )
                result = run_arm(objective, seed, config, arm)
                print(f"  {result.metric}={result.metric_at_budget:.6g}", flush=True)
                results.append(result)
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "epochs": EPOCHS, "refit_steps": REFIT_STEPS,
            "synthetic": SYNTHETIC, "harvest_per_epoch": HARVEST_PER_EPOCH,
            "torch_version": torch.__version__,
            "runs": [asdict(r) for r in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
