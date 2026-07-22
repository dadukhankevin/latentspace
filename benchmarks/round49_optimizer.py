"""Round 49: give the decoder mutations a memory — fitness-only "Adam".

Daniel's question: do the decoders have an optimizer? They do not. Weight
changes are memoryless Gaussian noise under one global win-rate gain;
nothing watches how the network has been changing. Round 48 measured what
that costs: Adam training the same architecture with real gradients lands
at 0.000011 — evolution's 0.003248 uses ~0.3% of the decoder's expressive
ceiling, so the SEARCH is the bottleneck, not the architecture.

Gradients are off-limits (black-box rule), but selection produces a
learning signal the current code discards: an accepted child's weight-delta
is a measured descent direction bought entirely with fitness evaluations.
Two fitness-only analogues of Adam's machinery:

  * momentum   — each individual carries an EMA of its lineage's ACCEPTED
                 weight-steps; mutations become drift-along-the-path plus
                 fresh noise. CMA-ES's evolution path, transplanted into the
                 per-individual GA. Survivors inherit their path; crossover
                 children average their parents' paths (like the weights).
  * nes_adam   — per parent per generation, the fitness-weighted average of
                 its children's noise is a gradient ESTIMATE (OpenAI-ES);
                 the surviving parent takes an Adam step along it. Children
                 are still made by mutation+crossover as usual, so this adds
                 a second, Lamarckian-but-legal channel that moves parents
                 between generations.

Baseline = the shipped library loop (crossover "average", 8 survivors,
win-rate gain). References at 5k, 10 seeds (round 45): image 0.00336,
curve 0.00109, TSP-100 16.27.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round45_decoder_inheritance import make_problem
from latentspace.universal.architectures import resolve
from benchmarks.legacy_engines.explorer import ExplorerConfig, _Template

LATENT = 64
MOMENTUM_BETA = 0.7      # EMA horizon for the accepted-step path
DRIFT = 0.5              # fraction of one mutation step contributed by drift
ADAM_LR = 0.01           # nes_adam: relative step (scaled by weight std)
ADAM_B1, ADAM_B2 = 0.9, 0.999


def run(problem: str, arm: str, budget: int, seed: int) -> dict:
    loss_fn, output_shape, architecture = make_problem(problem, seed)
    c = ExplorerConfig()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = _Template(resolve(architecture, LATENT, output_shape), "mps")
    n_params = template.n_params

    def decode_all(zs, thetas):
        return template.decode_batch(np.asarray(thetas), np.asarray(zs))

    zs = rng.standard_normal((c.population, LATENT)).astype(np.float32)
    thetas = np.stack([template.init_theta(int(rng.integers(0, 2**31)))
                       for _ in range(c.population)])
    loss = loss_fn(decode_all(zs, thetas))
    spent, gain = len(zs), 1.0
    paths = np.zeros_like(thetas)                     # momentum arm
    adam_m = np.zeros_like(thetas)                    # nes_adam arm
    adam_v = np.zeros_like(thetas)
    adam_t = 0

    def mutate_z(z):
        m = rng.random(z.shape) < c.genome_mutation_rate
        if not m.any():
            m[rng.integers(0, len(z))] = True
        return (z + m * rng.normal(0, c.genome_mutation_sigma * gain, z.shape)
                ).astype(np.float32)

    def cross_z(base, donor):
        cut = int(rng.integers(1, LATENT))
        child = base.copy()
        child[cut:] = donor[cut:]
        return child.astype(np.float32)

    while spent < budget:
        order = np.argsort(loss)[:c.elite]
        zs, thetas, loss = zs[order], thetas[order], loss[order]
        paths = paths[order]
        adam_m, adam_v = adam_m[order], adam_v[order]
        n = min(c.population, budget - spent)

        par = rng.integers(0, len(zs), n)
        mate = rng.integers(0, len(zs), n)
        winner, loser = np.minimum(par, mate), np.maximum(par, mate)
        cz = np.stack([mutate_z(cross_z(zs[w], zs[l]))
                       for w, l in zip(winner, loser)])

        base_theta = (thetas[winner] + thetas[loser]) / 2.0
        sigmas = np.exp(rng.uniform(np.log(c.weight_sigma_low),
                                    np.log(c.weight_sigma_high), n)) * gain
        scales = np.maximum(base_theta.std(axis=1), 1e-3)
        noise = rng.standard_normal((n, n_params)).astype(np.float32)
        step = (sigmas * scales)[:, None] * noise
        if arm == "momentum":
            # drift along the lineage's accepted path, plus fresh noise
            drift = DRIFT * (paths[winner] + paths[loser]) / 2.0
            cth = (base_theta + drift + step).astype(np.float32)
        else:
            cth = (base_theta + step).astype(np.float32)

        cl = loss_fn(decode_all(cz, cth))
        spent += n
        wins_mask = cl <= loss[winner] + 1e-12
        wins = float(wins_mask.mean())
        gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, *c.gain_limits))

        child_paths = np.zeros_like(cth)
        for i in range(n):
            parent_path = (paths[winner[i]] + paths[loser[i]]) / 2.0
            accepted = cth[i] - base_theta[i] if wins_mask[i] else 0.0
            child_paths[i] = MOMENTUM_BETA * parent_path + (1 - MOMENTUM_BETA) * accepted

        if arm == "nes_adam":
            # per-parent NES gradient estimate from its children this
            # generation, then an Adam step on the surviving parent
            adam_t += 1
            for p in np.unique(winner):
                idx = np.where(winner == p)[0]
                if len(idx) < 2:
                    continue
                f = -cl[idx]
                if f.std() < 1e-12:
                    continue
                w = (f - f.mean()) / f.std()
                grad = (w[:, None] * noise[idx]).mean(axis=0)  # ascent dir
                adam_m[p] = ADAM_B1 * adam_m[p] + (1 - ADAM_B1) * grad
                adam_v[p] = ADAM_B2 * adam_v[p] + (1 - ADAM_B2) * grad ** 2
                mhat = adam_m[p] / (1 - ADAM_B1 ** adam_t)
                vhat = adam_v[p] / (1 - ADAM_B2 ** adam_t)
                scale = max(float(thetas[p].std()), 1e-3)
                thetas[p] = (thetas[p] + ADAM_LR * scale * mhat /
                             (np.sqrt(vhat) + 1e-8)).astype(np.float32)
            # parents moved: rescore them so selection stays honest
            moved = np.unique(winner)
            if len(moved) and spent + len(moved) <= budget:
                ml = loss_fn(decode_all(zs[moved], thetas[moved]))
                spent += len(moved)
                loss[moved] = ml

        zs = np.concatenate([zs, cz])
        thetas = np.concatenate([thetas, cth])
        loss = np.concatenate([loss, cl])
        paths = np.concatenate([paths, child_paths])
        adam_m = np.concatenate([adam_m, np.zeros_like(cth)])
        adam_v = np.concatenate([adam_v, np.zeros_like(cth)])

    return {"score": float(loss.min()), "final_gain": gain}


ARMS = ("baseline", "momentum", "nes_adam")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+",
                        default=["blob2d", "smooth1d", "tsp100"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for problem in args.problems:
        print(f"\n########## {problem} (budget {args.budget}) ##########",
              flush=True)
        for arm in args.arms:
            for seed in args.seeds:
                out = run(problem, arm, args.budget, seed)
                rows.append({"problem": problem, "arm": arm, "seed": seed,
                             **out})
                print(f"  {arm:<9} seed {seed} score {out['score']:.6g}",
                      flush=True)
            vals = [r["score"] for r in rows
                    if r["problem"] == problem and r["arm"] == arm]
            print(f"  {arm:<9} MEAN {np.mean(vals):.6g} +- "
                  f"{np.std(vals, ddof=1):.3g}", flush=True)

    print("\n--- summary (score; lower is better) ---")
    for problem in args.problems:
        base = np.mean([r["score"] for r in rows
                        if r["problem"] == problem and r["arm"] == "baseline"])
        for arm in args.arms:
            v = np.mean([r["score"] for r in rows
                         if r["problem"] == problem and r["arm"] == arm])
            print(f"  {problem:<10} {arm:<9} {v:.6g} ({base/v:.3f}x vs baseline)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "momentum_beta": MOMENTUM_BETA,
             "drift": DRIFT, "adam_lr": ADAM_LR,
             "torch_version": torch.__version__, "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
