"""Round 12: weight-mutation as the off-manifold data channel.

Round 10 showed online refinement is inert: candidates found by searching
through the decoder lie ON its output manifold, so refitting teaches density,
never geometry. The proposed fix (Daniel's): temporarily mutate the decoder's
WEIGHTS. A weight-perturbed decoder has a different output manifold, so its
outputs are things the base decoder cannot currently express — yet they are
still decoder-shaped, structured variations rather than raw noise. Push the
current best latent (plus jitter) through a few mutants, evaluate the
results, and distill any that compete with the search's own best candidates
back into the base weights. If the self-referentiality principle is the true
blocker, this channel should move the representation floor where round-10's
on-manifold refits could not.

Arms (identical warm start — round-9 PCA-scaffolded autoencoder — and
identical refit machinery; the ONLY difference is the training data):

  * frozen_aug — no refits (round-9/10 control);
  * online     — refit on the top CMA candidates each epoch (round 10's
                 inert arm, rerun here so floors are logged);
  * wmut       — same refits, but each epoch also spends ~2% of the
                 evaluation budget on weight-mutant outputs and adds the
                 competitive ones to the refit corpus.

Diagnostic: the representation floor — how well the decoder can express the
true target via encode->decode — measured after the warm start and again at
the end. Self-referentiality predicts online's floor does not move; the
channel works only if wmut's does.
"""

from __future__ import annotations

import argparse
import copy
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

OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "blob2d_1024": BlobImage2D,
}

EPOCHS = 5
REFIT_STEPS = 2_000
SYNTHETIC = 5_000
HARVEST_PER_EPOCH = 60

MUTANTS_PER_EPOCH = 12
LATENTS_PER_MUTANT = 8          # ~96 evals/epoch, ~7.7% of a 5,000 budget
# Diagnostic (seed 0, blob): frac(mutant beats parent) = 43% at sigma 0.003,
# 24% at 0.01, ~0 at >=0.03 — sample only the viable regime.
SIGMA_W_LOW, SIGMA_W_HIGH = 0.003, 0.02
JITTER = 0.15
MUTANT_KEEP = 40
MUTANT_REPLICATE = 10           # accepted samples are few; upweight in refit


def _perturb(net: torch.nn.Module, sigma_w: float, rng) -> torch.nn.Module:
    """A copy of a decoder net with Gaussian weight noise."""
    mutant = copy.deepcopy(net)
    with torch.no_grad():
        for p in mutant.parameters():
            scale = float(p.detach().std())
            if not np.isfinite(scale) or scale < 1e-3:
                scale = 1e-3
            noise = torch.as_tensor(
                rng.standard_normal(tuple(p.shape)).astype(np.float32),
                device=p.device,
            )
            p.add_(sigma_w * scale * noise)
    return mutant


def _make_mutant(decoder: RefinableDecoder, sigma_w: float, rng) -> torch.nn.Module:
    return _perturb(decoder.decoder_net, sigma_w, rng)


def _mutant_decode(decoder: RefinableDecoder, mutant, z: np.ndarray) -> torch.Tensor:
    genes = torch.as_tensor(np.asarray(z, dtype=np.float32), device=decoder.device)
    with torch.no_grad():
        out = torch.sigmoid(mutant(decoder.code_mean + genes * decoder.code_std))
    return out.view(-1, *decoder.output_shape)


def _floor(decoder: RefinableDecoder, objective: Objective) -> float:
    """Loss of the decoder's best expressible approximation of the true target
    (encode->decode reconstruction); costs no budget."""
    z = decoder.anchor_z(objective.target)
    phenotype = decoder.decode(z[None]).detach().cpu().numpy().reshape(1, -1)
    return float(objective.loss_numpy(phenotype)[0])


def run_arm(objective, seed, config, arm):
    _require_mps()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    corpus_x, corpus_loss = harvest_corpus(objective, rng, config, k=128)
    pca = fit_pca_decoder(corpus_x, corpus_loss, config.latent, "mps", top=len(corpus_x))
    synthetic = _synthetic_from_pca(pca, rng, config.latent, SYNTHETIC)
    real_logits = _to_logits(corpus_x)
    decoder = RefinableDecoder(objective.dimension, config.latent, "mps", seed=seed)
    decoder.fit(np.concatenate([real_logits, synthetic]), steps=8_000)
    floor_start = _floor(decoder, objective)

    tracker = TrackedFitness(objective)
    started = time.perf_counter()
    mutant_phase = arm if arm in ("wmut", "wmut_es") else None
    kept_total = 0

    if arm == "frozen_aug":
        _cma_minimize(
            lambda z: -tracker(decoder.decode(z)).detach().cpu().numpy(),
            dim=config.latent, budget_evaluations=config.evaluation_budget,
            evaluations_done=0, rng=rng, mean0=np.zeros(config.latent), sigma0=1.0,
        )
    elif arm in ("online", "wmut", "wmut_es"):
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

            order = np.argsort(epoch_loss)[:HARVEST_PER_EPOCH]
            accumulated.append(_to_logits(np.asarray(epoch_x)[order]))

            # Off-manifold channel: outputs of weight-mutated decoder copies.
            # A mutant output is kept iff it beats the BASE decoder's output
            # at the same latent — the weight mutation demonstrably improved
            # the phenotype, so its discovery is distilled into the base.
            phase_evals = (MUTANTS_PER_EPOCH + 1) * LATENTS_PER_MUTANT
            budget_left = config.evaluation_budget - tracker.evaluations
            if mutant_phase and budget_left > 2 * phase_evals:
                z_best = decoder.anchor_z(tracker.best_phenotype)
                z = np.tile(z_best, (LATENTS_PER_MUTANT, 1))
                z[1:] += JITTER * rng.standard_normal(z[1:].shape)
                parent_loss = (-tracker(decoder.decode(z))).detach().cpu().numpy()

                if mutant_phase == "wmut":
                    # One-step mutants: keep any output that beats the BASE
                    # decoder's output at the same latent.
                    mutant_x, mutant_gain = [], []
                    for _ in range(MUTANTS_PER_EPOCH):
                        sigma_w = float(np.exp(rng.uniform(
                            np.log(SIGMA_W_LOW), np.log(SIGMA_W_HIGH))))
                        mutant = _make_mutant(decoder, sigma_w, rng)
                        phenotypes = _mutant_decode(decoder, mutant, z)
                        losses = (-tracker(phenotypes)).detach().cpu().numpy()
                        mutant_x.extend(
                            phenotypes.detach().cpu().numpy().reshape(len(z), -1))
                        mutant_gain.extend((losses - parent_loss).tolist())
                    mutant_gain = np.asarray(mutant_gain)
                    good = np.flatnonzero(mutant_gain < 0)
                    good = good[np.argsort(mutant_gain[good])][:MUTANT_KEEP]
                    if len(good):
                        kept = _to_logits(np.asarray(mutant_x)[good])
                        accumulated.append(
                            np.repeat(kept, MUTANT_REPLICATE, axis=0))
                    kept_total += len(good)
                else:
                    # wmut_es: a (1+1)-ES walk on the decoder weights.
                    # Single mutations gain only ~0.5% (round-12a); compound
                    # accepted steps into one champion, then distill the
                    # champion's outputs around z_best into the base — true
                    # decoder-to-decoder distillation from a verified-better
                    # teacher, not self-imitation.
                    champion = None
                    champion_score = float(parent_loss.mean())
                    for _ in range(MUTANTS_PER_EPOCH):
                        sigma_w = float(np.exp(rng.uniform(
                            np.log(SIGMA_W_LOW), np.log(SIGMA_W_HIGH))))
                        source = (champion if champion is not None
                                  else decoder.decoder_net)
                        mutant = _perturb(source, sigma_w, rng)
                        losses = (-tracker(
                            _mutant_decode(decoder, mutant, z)
                        )).detach().cpu().numpy()
                        if float(losses.mean()) < champion_score:
                            champion, champion_score = mutant, float(losses.mean())
                            kept_total += 1
                    if champion is not None:
                        zc = np.tile(z_best, (MUTANT_KEEP, 1))
                        zc[1:] += JITTER * rng.standard_normal(zc[1:].shape)
                        outputs = _mutant_decode(decoder, champion, zc)
                        kept = _to_logits(
                            outputs.detach().cpu().numpy().reshape(len(zc), -1))
                        accumulated.append(
                            np.repeat(kept, MUTANT_REPLICATE, axis=0))

            decoder.fit(np.concatenate([*accumulated, synthetic]), steps=REFIT_STEPS)
            mean_z = decoder.anchor_z(tracker.best_phenotype)
    else:
        raise ValueError(arm)

    torch.mps.synchronize()
    result = _finish_result(objective, arm, seed, config, tracker, started,
                            neural_device="mps")
    floor_end = _floor(decoder, objective)
    torch.mps.empty_cache()
    return result, {
        "objective": objective.name, "strategy": arm, "seed": seed,
        "floor_start": floor_start, "floor_end": floor_end,
        "mutants_kept": kept_total,
    }


STRATEGIES = ("frozen_aug", "online", "wmut", "wmut_es")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES)
    )
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                        default=list(STRATEGIES))
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
                    f"run objective={objective_name:<14} arm={arm:<11} "
                    f"seed={seed} budget={config.evaluation_budget}",
                    flush=True,
                )
                result, floor = run_arm(objective, seed, config, arm)
                print(
                    f"  {result.metric}={result.metric_at_budget:.6g}  "
                    f"floor {floor['floor_start']:.5f}->{floor['floor_end']:.5f}  "
                    f"kept={floor['mutants_kept']}",
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
            "epochs": EPOCHS, "refit_steps": REFIT_STEPS, "synthetic": SYNTHETIC,
            "harvest_per_epoch": HARVEST_PER_EPOCH,
            "mutants_per_epoch": MUTANTS_PER_EPOCH,
            "latents_per_mutant": LATENTS_PER_MUTANT,
            "sigma_w": [SIGMA_W_LOW, SIGMA_W_HIGH], "jitter": JITTER,
            "mutant_keep": MUTANT_KEEP, "mutant_replicate": MUTANT_REPLICATE,
            "torch_version": torch.__version__,
            "runs": [asdict(r) for r in results],
            "floors": floors,
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
