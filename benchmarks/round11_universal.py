"""Round 11: one universal decoder across families, versus one per family.

The vision under test: a single expressive MLP decoder serving every problem
family at once — a genuinely shared genetic code. Expressiveness is not the
question (capacity is cheap); the measured questions are latent geometry
(one code space must host all families' manifolds; search must find or be
told its region) and training interference (shared weights fit all corpora).

Both families are 256-dimensional so one output head can serve both
(blob images at 16x16 for this round; a production universal decoder needs
conditioning or masking to span output shapes — downstream engineering).

Arms (each solves fresh instances of BOTH families, 5,000 evaluations,
K = 128 practice instances per family, round-9 scaffolding recipe):

  * perfam     — one decoder per family (the incumbent recipe), latent 32;
  * uni32      — ONE decoder trained on both corpora + both PCA scaffolds,
                 latent 32; CMA-ES anchored at the mean code of the solving
                 family's real elites (you always know which problem you're
                 solving);
  * uni32_blind— same decoder, CMA-ES from the origin: the price of not
                 knowing where on the shared map you are;
  * uni64      — the anchored universal decoder with the combined width.
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
from benchmarks.round8_mlp_pretrain import BlobImage2D, harvest_corpus
from benchmarks.round10_online_refine import RefinableDecoder, _to_logits


class Blob16(BlobImage2D):
    name = "blob2d_256"

    def __init__(self, instance_seed: int = 2026):
        super().__init__(size=16, blobs=3, instance_seed=instance_seed)


FAMILIES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "blob2d_256": Blob16,
}

SYNTHETIC_PER_FAMILY = 5_000
FIT_STEPS = 8_000


def family_data(cls, rng, config, latent):
    """Real elite logits and PCA-scaffold synthetic logits for one family."""
    objective = cls()
    corpus_x, corpus_loss = harvest_corpus(objective, rng, config, k=128)
    pca = fit_pca_decoder(corpus_x, corpus_loss, latent, "mps", top=len(corpus_x))
    z = rng.standard_normal((SYNTHETIC_PER_FAMILY, latent)).astype(np.float32)
    synthetic = pca.decode(z).detach().cpu().numpy().reshape(len(z), -1)
    return _to_logits(corpus_x), _to_logits(synthetic)


def solve(objective, decoder, tracker, rng, config, mean0):
    started = time.perf_counter()
    _cma_minimize(
        lambda z: -tracker(decoder.decode(z)).detach().cpu().numpy(),
        dim=decoder.input_length,
        budget_evaluations=config.evaluation_budget,
        evaluations_done=0,
        rng=rng,
        mean0=mean0,
        sigma0=1.0,
    )
    torch.mps.synchronize()
    return started


def elite_code_mean(decoder, real_logits):
    logits = torch.as_tensor(real_logits, dtype=torch.float32, device=decoder.device)
    with torch.no_grad():
        codes = decoder.encoder(logits)
        z = (codes - decoder.code_mean) / decoder.code_std
    return z.mean(dim=0).detach().cpu().numpy()


def run_arm(family_name, seed, config, arm):
    _require_mps()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    latent = 64 if arm in ("uni64", "perfam64") else 32

    if arm.startswith("perfam"):
        real, synthetic = family_data(FAMILIES[family_name], rng, config, latent)
        decoder = RefinableDecoder(real.shape[1], latent, "mps", seed=seed)
        decoder.fit(np.concatenate([real, synthetic]), steps=FIT_STEPS)
        anchor = np.zeros(latent)
    else:
        # One decoder over the union of every family's corpus and scaffold.
        # Practice corpora are harvested with the same rng regardless of which
        # family is being solved, so the shared decoder is identical across
        # solve targets at a given seed.
        per_family = {
            name: family_data(cls, rng, config, 32)
            for name, cls in FAMILIES.items()
        }
        union = np.concatenate(
            [array for pair in per_family.values() for array in pair]
        )
        dim = union.shape[1]
        decoder = RefinableDecoder(dim, latent, "mps", seed=seed)
        decoder.fit(union, steps=FIT_STEPS)
        anchor = (
            np.zeros(latent)
            if arm == "uni32_blind"
            else elite_code_mean(decoder, per_family[family_name][0])
        )

    objective = FAMILIES[family_name]()
    tracker = TrackedFitness(objective)
    started = solve(objective, decoder, tracker, rng, config, anchor)
    result = _finish_result(
        objective, arm, seed, config, tracker, started, neural_device="mps"
    )
    torch.mps.empty_cache()
    return result


ARMS = ("perfam", "perfam64", "uni32", "uni32_blind", "uni64")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = BenchmarkConfig(evaluation_budget=args.budget)
    _require_mps()
    results = []
    for family_name in args.families:
        for arm in args.arms:
            for seed in args.seeds:
                print(
                    f"run family={family_name:<14} arm={arm:<12} seed={seed} "
                    f"budget={config.evaluation_budget}",
                    flush=True,
                )
                result = run_arm(family_name, seed, config, arm)
                print(f"  {result.metric}={result.metric_at_budget:.6g}", flush=True)
                results.append(result)
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "families": list(FAMILIES),
            "synthetic_per_family": SYNTHETIC_PER_FAMILY,
            "fit_steps": FIT_STEPS,
            "torch_version": torch.__version__,
            "runs": [asdict(r) for r in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
