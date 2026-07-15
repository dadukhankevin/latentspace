"""Round 9: is the neural decoder's deficit a training-budget artifact?

Round 8b held autoencoder training fixed at 2,000 Adam steps while the corpus
grew — guaranteeing growing underfit and a shallower scaling slope than
closed-form PCA. This round scales training compute at K = 128 and tests the
PCA-as-scaffolding idea:

  * mlp8k / mlp32k — same architecture and corpus, 4x / 16x the training;
  * mlp8k_aug      — 8k steps on the real 1,280 elites PLUS 5,000 synthetic
                     samples drawn from the fitted PCA decoder. Synthetic
                     data lies exactly on PCA's plane, so it cannot add
                     expressiveness beyond PCA (a student cannot out-learn
                     its teacher's outputs) — but as augmentation around the
                     real elites it may regularize the fit.

References from round 8/8b at K = 128: PCA 0.0042 (smooth) / 0.0214 (blob);
MLP at 2k steps 0.0164 / 0.0544; direct GA 0.0140 / 0.0582.
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
from benchmarks.round8_mlp_pretrain import (
    BlobImage2D,
    EliteAutoencoderDecoder,
    harvest_corpus,
)

OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "blob2d_1024": BlobImage2D,
}

VARIANTS = {
    "mlp8k": {"steps": 8_000, "augment": 0},
    "mlp32k": {"steps": 32_000, "augment": 0},
    "mlp8k_aug": {"steps": 8_000, "augment": 5_000},
}


def run_variant(objective, seed, config, name):
    spec = VARIANTS[name]
    _require_mps()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    corpus_x, corpus_loss = harvest_corpus(objective, rng, config, k=128)

    if spec["augment"]:
        pca = fit_pca_decoder(
            corpus_x, corpus_loss, config.latent, "mps", top=len(corpus_x)
        )
        z = rng.standard_normal((spec["augment"], config.latent)).astype(np.float32)
        synthetic = pca.decode(z).detach().cpu().numpy().reshape(len(z), -1)
        corpus_x = np.concatenate([corpus_x, synthetic.astype(np.float32)])

    decoder = EliteAutoencoderDecoder(
        corpus_x, config.latent, "mps", steps=spec["steps"], seed=seed
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
        objective, name, seed, config, tracker, started,
        generations=generations, neural_device="mps",
    )
    torch.mps.empty_cache()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES)
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
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
        for variant in args.variants:
            for seed in args.seeds:
                objective = OBJECTIVES[objective_name]()
                print(
                    f"run objective={objective_name:<14} variant={variant:<10} "
                    f"seed={seed} budget={config.evaluation_budget}",
                    flush=True,
                )
                result = run_variant(objective, seed, config, variant)
                print(
                    f"  {result.metric}={result.metric_at_budget:.6g}",
                    flush=True,
                )
                results.append(result)
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "variants": VARIANTS,
            "k": 128,
            "torch_version": torch.__version__,
            "runs": [asdict(result) for result in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
