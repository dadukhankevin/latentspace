"""Round 13: two competing decoders — nobody trains on their own outputs.

Round 12 established that the fix for self-distillation is a teacher that is
not the student, but its teacher (a weight-mutant champion) was only ever
epsilon away from the student, so the win was ~1.4%. The proposed
generalization (Daniel's): TWO decoders, A and B, where A is only ever
trained on elites found by searching through B and vice versa. Cross-data is
off the student's manifold by construction and verified-good (search paid
real evaluations for it), and the teachers can be arbitrarily far apart. A
subtle weight decay AWAY from each other fights the failure mode where the
two decoders converge until cross-data becomes self-data again.

Arms (all dual arms: two round-9 warm-started decoders differing only in
init seed; CMA-ES search alternates decoders per epoch, the global best
phenotype re-anchored into whichever decoder searches next):

  * dual_frozen     — no training; isolates the two-manifold ensemble effect;
  * dual_self       — each decoder refits on elites IT produced (self-distill
                      control inside the dual structure);
  * dual_cross      — each decoder refits only on the OTHER's elites;
  * dual_cross_repel— cross + per-epoch weight step away from each other.

Single-decoder baselines (frozen_aug, online) come from round 12's JSON:
identical warm start, seeds, and refit machinery. A `frozen` arm is kept
here only for spot-check reruns.

Diagnostic: representation floors of both decoders, before and after.
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
from benchmarks.round10_online_refine import (
    RefinableDecoder,
    _synthetic_from_pca,
    _to_logits,
)
from benchmarks.round12_weight_mutation import _floor

OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "blob2d_1024": BlobImage2D,
}

EPOCHS = 5
REFIT_STEPS = 2_000
SYNTHETIC = 5_000
HARVEST_PER_EPOCH = 60
REPEL = 0.02                    # per-epoch fractional step apart, "subtle"
INIT_SEED_OFFSET = 7_919        # decoder B's init seed = seed + this


def _repel(a: RefinableDecoder, b: RefinableDecoder, coef: float) -> None:
    """Push the two decoder nets a small step apart along their current
    weight difference (mutual decay AWAY from each other)."""
    with torch.no_grad():
        for pa, pb in zip(a.decoder_net.parameters(), b.decoder_net.parameters()):
            delta = coef * (pa - pb)
            pa.add_(delta)
            pb.sub_(delta)


def run_arm(objective, seed, config, arm):
    _require_mps()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    corpus_x, corpus_loss = harvest_corpus(objective, rng, config, k=128)
    pca = fit_pca_decoder(corpus_x, corpus_loss, config.latent, "mps", top=len(corpus_x))
    synthetic = _synthetic_from_pca(pca, rng, config.latent, SYNTHETIC)
    real_logits = _to_logits(corpus_x)

    decoders = [RefinableDecoder(objective.dimension, config.latent, "mps", seed=seed)]
    if arm != "frozen":
        decoders.append(RefinableDecoder(
            objective.dimension, config.latent, "mps",
            seed=seed + INIT_SEED_OFFSET))
    warm = np.concatenate([real_logits, synthetic])
    for d in decoders:
        d.fit(warm, steps=8_000)
    floors_start = [_floor(d, objective) for d in decoders]

    tracker = TrackedFitness(objective)
    started = time.perf_counter()

    if arm == "frozen":
        decoder = decoders[0]
        _cma_minimize(
            lambda z: -tracker(decoder.decode(z)).detach().cpu().numpy(),
            dim=config.latent, budget_evaluations=config.evaluation_budget,
            evaluations_done=0, rng=rng, mean0=np.zeros(config.latent), sigma0=1.0,
        )
    else:
        accumulated = [[real_logits], [real_logits]]
        mean_z = np.zeros(config.latent)
        per_epoch = config.evaluation_budget // EPOCHS
        for epoch in range(EPOCHS):
            active = epoch % 2
            decoder = decoders[active]
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

            if arm != "dual_frozen":
                order = np.argsort(epoch_loss)[:HARVEST_PER_EPOCH]
                elites = _to_logits(np.asarray(epoch_x)[order])
                # The one rule under test: who learns from these elites?
                student = active if arm.startswith("dual_self") else 1 - active
                accumulated[student].append(elites)
                decoders[student].fit(
                    np.concatenate([*accumulated[student], synthetic]),
                    steps=REFIT_STEPS)
                if arm.endswith("_repel"):
                    _repel(decoders[0], decoders[1], REPEL)

            # Carry the global best across manifolds by re-encoding it into
            # whichever decoder searches next.
            mean_z = decoders[(epoch + 1) % 2].anchor_z(tracker.best_phenotype)

    torch.mps.synchronize()
    result = _finish_result(objective, arm, seed, config, tracker, started,
                            neural_device="mps")
    floors_end = [_floor(d, objective) for d in decoders]
    torch.mps.empty_cache()
    return result, {
        "objective": objective.name, "strategy": arm, "seed": seed,
        "floor_start": floors_start, "floor_end": floors_end,
    }


STRATEGIES = ("frozen", "dual_frozen", "dual_self", "dual_cross",
              "dual_cross_repel", "dual_self_repel")
DEFAULT_STRATEGIES = ("dual_frozen", "dual_self", "dual_cross",
                      "dual_cross_repel")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES)
    )
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                        default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = BenchmarkConfig(evaluation_budget=args.budget)
    _require_mps()
    results, floors = [], []
    for objective_name in args.objectives:
        for arm in args.strategies:
            for seed in args.seeds:
                objective = OBJECTIVES[objective_name]()
                print(
                    f"run objective={objective_name:<14} arm={arm:<16} "
                    f"seed={seed} budget={config.evaluation_budget}",
                    flush=True,
                )
                result, floor = run_arm(objective, seed, config, arm)
                floor_txt = " ".join(
                    f"{s:.5f}->{e:.5f}"
                    for s, e in zip(floor["floor_start"], floor["floor_end"]))
                print(
                    f"  {result.metric}={result.metric_at_budget:.6g}  "
                    f"floors {floor_txt}",
                    flush=True,
                )
                results.append(result)
                floors.append(floor)
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "epochs": EPOCHS, "refit_steps": REFIT_STEPS,
            "synthetic": SYNTHETIC, "harvest_per_epoch": HARVEST_PER_EPOCH,
            "repel": REPEL, "init_seed_offset": INIT_SEED_OFFSET,
            "torch_version": torch.__version__,
            "runs": [asdict(r) for r in results],
            "floors": floors,
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
