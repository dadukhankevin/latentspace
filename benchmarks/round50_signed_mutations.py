"""Round 50: mutations ARE the gradient samples — signed by fitness, pooled over time.

Daniel's refinement of round 49, which fixed three real deficiencies in that
round's arms:

  1. FAILURES ARE DATA. Round 49's momentum remembered only ACCEPTED steps —
     but the win rate is pinned near 20%, so 80% of the evaluations (the
     failures) were discarded. A mutation that made fitness worse is a
     measured BAD direction; its negation is signal, same as a win.
  2. ACCUMULATE OVER TIME. Four children per parent per generation is a
     hopeless gradient estimate taken alone; an exponential moving average
     with a ~10-generation horizon pools ~40 mutations per lineage (or
     ~300+ in the shared variant). Adam's first moment exists precisely to
     average noisy estimates over time.
  3. ROUND 49's BUG: children were born with ZEROED optimizer state, and
     since survivors are replaced by their children within a few
     generations, the accumulator kept resetting — it never actually
     accumulated. Here the state is INHERITED down the lineage like the
     weights (crossover children average their parents' state, exactly as
     they average the weights).

The estimator, per child i with birth mutation delta_i (the weight noise) and
fitness change df_i = parent_loss - child_loss (positive = improvement):

    g_i = df_i * delta_i / (|df| scale)        # signed, magnitude-weighted

Arms (all on top of the shipped loop: crossover "average", 8 survivors,
win-rate gain):

  * baseline       — the shipped library loop.
  * signed_lineage — per-individual Adam state (m, v), inherited through
                     selection AND crossover, updated by the child's own
                     birth mutation; a drift term proportional to the current
                     mutation scale is added when breeding from it.
  * signed_global  — ONE shared Adam state pooling all 32 signed mutations
                     each generation. Legal because within a run the
                     population is lineage-collapsed near-clones (round 40:
                     ~3 mutation steps apart), so their weight coordinates
                     are mutually meaningful; round 37's co-adaptation
                     objection applies across runs, not within.

References at 5k, 3 seeds (round 49): image 0.00323, curve 0.00081,
TSP-100 15.82.
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
B1, B2 = 0.9, 0.999      # Adam moments
DRIFT = 0.5              # drift, as a fraction of the current mutation step


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
    spent, gain, step_count = len(zs), 1.0, 0
    df_scale = None                                   # running |df| normalizer

    per = arm == "signed_lineage"
    m = np.zeros((c.population, n_params), np.float32) if per else np.zeros(n_params, np.float32)
    v = np.zeros_like(m)

    def mutate_z(z):
        mask = rng.random(z.shape) < c.genome_mutation_rate
        if not mask.any():
            mask[rng.integers(0, len(z))] = True
        return (z + mask * rng.normal(0, c.genome_mutation_sigma * gain, z.shape)
                ).astype(np.float32)

    def cross_z(base, donor):
        cut = int(rng.integers(1, LATENT))
        child = base.copy()
        child[cut:] = donor[cut:]
        return child.astype(np.float32)

    while spent < budget:
        order = np.argsort(loss)[:c.elite]
        zs, thetas, loss = zs[order], thetas[order], loss[order]
        if per:
            m, v = m[order], v[order]
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
        step_size = (sigmas * scales)[:, None]
        noise = rng.standard_normal((n, n_params)).astype(np.float32)
        delta = step_size * noise

        if arm != "baseline" and step_count > 0:
            mh_denom = 1 - B1 ** step_count
            vh_denom = 1 - B2 ** step_count
            if per:
                m_in = (m[winner] + m[loser]) / 2.0
                v_in = (v[winner] + v[loser]) / 2.0
            else:
                m_in, v_in = m, v
            direction = (m_in / mh_denom) / (np.sqrt(v_in / vh_denom) + 1e-8)
            cth = (base_theta + DRIFT * step_size * direction + delta
                   ).astype(np.float32)
        else:
            cth = (base_theta + delta).astype(np.float32)

        cl = loss_fn(decode_all(cz, cth))
        spent += n
        wins = float((cl <= loss[winner] + 1e-12).mean())
        gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, *c.gain_limits))

        # signed gradient samples from EVERY child, winners and losers alike
        df = loss[winner] - cl                        # >0 = improvement
        mag = float(np.abs(df).mean())
        df_scale = mag if df_scale is None else 0.9 * df_scale + 0.1 * mag
        w_signed = (df / max(df_scale, 1e-12))[:, None] * noise
        step_count += 1

        if per:
            child_m = np.empty_like(cth)
            child_v = np.empty_like(cth)
            for i in range(n):
                pm = (m[winner[i]] + m[loser[i]]) / 2.0
                pv = (v[winner[i]] + v[loser[i]]) / 2.0
                child_m[i] = B1 * pm + (1 - B1) * w_signed[i]
                child_v[i] = B2 * pv + (1 - B2) * w_signed[i] ** 2
            m = np.concatenate([m, child_m])
            v = np.concatenate([v, child_v])
        else:
            g = w_signed.mean(axis=0)
            m = B1 * m + (1 - B1) * g
            v = B2 * v + (1 - B2) * g ** 2

        zs = np.concatenate([zs, cz])
        thetas = np.concatenate([thetas, cth])
        loss = np.concatenate([loss, cl])

    return {"score": float(loss.min()), "final_gain": gain}


ARMS = ("baseline", "signed_lineage", "signed_global")


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
                print(f"  {arm:<15} seed {seed} score {out['score']:.6g}",
                      flush=True)
            vals = [r["score"] for r in rows
                    if r["problem"] == problem and r["arm"] == arm]
            print(f"  {arm:<15} MEAN {np.mean(vals):.6g} +- "
                  f"{np.std(vals, ddof=1):.3g}", flush=True)

    print("\n--- summary (score; lower is better) ---")
    for problem in args.problems:
        base = np.mean([r["score"] for r in rows
                        if r["problem"] == problem and r["arm"] == "baseline"])
        for arm in args.arms:
            vals = np.mean([r["score"] for r in rows
                            if r["problem"] == problem and r["arm"] == arm])
            print(f"  {problem:<10} {arm:<15} {vals:.6g} "
                  f"({base/vals:.3f}x vs baseline)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "b1": B1, "b2": B2, "drift": DRIFT,
             "torch_version": torch.__version__, "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
