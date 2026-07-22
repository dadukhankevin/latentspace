"""Round 44: a classical, stochastic reproduction scheme vs the rigid conveyor.

Everything this campaign has run is an evolution-strategy-shaped loop: exactly
32 children every generation, every child produced the same way (since round
42: crossover + mutation), the whole parent set refreshed each cycle. Daniel's
GAs are built differently, and this round implements that shape:

  * a STANDING POPULATION of 32 that persists across epochs;
  * a random number of parent PAIRS each epoch, selected with RANK BIAS
    (linear rank weights — fitter individuals breed more often, but nothing
    is deterministic);
  * each pair has a random FAMILY SIZE (1-3 children);
  * some individuals ONLY MUTATE this epoch (no partner);
  * the rest sit the epoch out (our elitism already lets survivors persist,
    so idling is the default fate of anyone not selected);
  * new arrivals are evaluated, then the population is trimmed back to 32 by
    dropping the worst — parents are never killed for having bred.

Two things this buys beyond faithfulness to classical practice. First,
mutation-only children mean crossover's contribution finally becomes
attributable: round 42/43 gave EVERY child both operators, so the
smooth-problem wins could belong to either. Second, family-size randomness
plus rank bias is a soft, threshold-free version of the allocation idea that
round 39's controllers failed to build — good individuals get more children
IN EXPECTATION, without any survivor-count knob at all.

Honest caveat: this arm changes several ingredients at once (pair selection,
family sizes, operator mix, steady-state trimming). If it wins, round 45
decomposes it; if it loses cleanly, none of the ingredients was the missing
piece and one probe settled it.

Arms:

  * rigid  — the incumbent: (survivors + 32) with uniform-mate crossover on
             every child (round 43's uniform_free at 8 survivors).
  * loose  — the scheme above.

References at 5k: image 0.00342 (tournament crossover, 8 survivors) and
0.00416 (uniform crossover, 8 survivors); curve 0.00086 (uniform crossover,
8 survivors); TSP-100 15.23 (no crossover, 16 survivors, 5 seeds).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round39_survivors import make_problem
from latentspace.universal.architectures import resolve
from benchmarks.legacy_engines.explorer import ExplorerConfig, _Template

LATENT = 64
POP = 32          # standing population
PAIRS = (4, 12)   # parent pairs per epoch, drawn uniformly from this range
FAMILY = (1, 3)   # children per pair
SOLO = (4, 12)    # mutation-only individuals per epoch


def rank_weights(n: int) -> np.ndarray:
    """Linear rank selection: the best of n gets weight n, the worst gets 1."""
    w = np.arange(n, 0, -1, dtype=np.float64)
    return w / w.sum()


def run(problem: str, arm: str, budget: int, seed: int,
        survivors: int = 8) -> dict:
    loss_fn, output_shape, architecture = make_problem(problem, seed)
    c = ExplorerConfig()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = _Template(resolve(architecture, LATENT, output_shape), "mps")

    def decode_all(zs, thetas):
        return torch.stack([template.decode(t, z)
                            for z, t in zip(zs, thetas)])

    zs = rng.standard_normal((POP, LATENT)).astype(np.float32)
    thetas = np.stack([template.init_theta(int(rng.integers(0, 2**31)))
                       for _ in range(POP)])
    loss = loss_fn(decode_all(zs, thetas))
    spent, gain = len(zs), 1.0

    def mutate_z(z):
        m = rng.random(z.shape) < c.genome_mutation_rate
        if not m.any():
            m[rng.integers(0, len(z))] = True
        return (z + m * rng.normal(0, c.genome_mutation_sigma * gain, z.shape)
                ).astype(np.float32)

    def mutate_theta(theta):
        sigma = float(np.exp(rng.uniform(np.log(c.weight_sigma_low),
                                         np.log(c.weight_sigma_high)))) * gain
        scale = max(float(theta.std()), 1e-3)
        return (theta + rng.normal(0, sigma * scale, theta.shape)
                ).astype(np.float32)

    def cross_z(base, donor):
        cut = int(rng.integers(1, LATENT))
        child = base.copy()
        child[cut:] = donor[cut:]
        return child.astype(np.float32)

    while spent < budget:
        order = np.argsort(loss)
        zs, thetas, loss = zs[order], thetas[order], loss[order]

        child_z, child_th, child_parent = [], [], []
        if arm == "rigid":
            zs, thetas, loss = zs[:survivors], thetas[:survivors], loss[:survivors]
            n = min(POP, budget - spent)
            par = rng.integers(0, len(zs), n)
            mate = rng.integers(0, len(zs), n)
            winner, loser = np.minimum(par, mate), np.maximum(par, mate)
            for w, l in zip(winner, loser):
                child_z.append(mutate_z(cross_z(zs[w], zs[l])))
                child_th.append(mutate_theta(thetas[w]))
                child_parent.append(w)
        else:
            zs, thetas, loss = zs[:POP], thetas[:POP], loss[:POP]
            w = rank_weights(len(zs))
            n_pairs = int(rng.integers(PAIRS[0], PAIRS[1] + 1))
            for _ in range(n_pairs):
                a, b = rng.choice(len(zs), size=2, replace=False, p=w)
                fitter, other = min(a, b), max(a, b)
                for _ in range(int(rng.integers(FAMILY[0], FAMILY[1] + 1))):
                    child_z.append(mutate_z(cross_z(zs[fitter], zs[other])))
                    child_th.append(mutate_theta(thetas[fitter]))
                    child_parent.append(fitter)
            n_solo = int(rng.integers(SOLO[0], SOLO[1] + 1))
            for p in rng.choice(len(zs), size=n_solo, replace=False, p=w):
                child_z.append(mutate_z(zs[p]))
                child_th.append(mutate_theta(thetas[p]))
                child_parent.append(p)
            overshoot = spent + len(child_z) - budget
            if overshoot > 0:
                child_z = child_z[:-overshoot]
                child_th = child_th[:-overshoot]
                child_parent = child_parent[:-overshoot]
        if not child_z:
            break

        cz, cth = np.stack(child_z), np.stack(child_th)
        par_idx = np.asarray(child_parent)
        cl = loss_fn(decode_all(cz, cth))
        spent += len(cz)
        wins = float((cl <= loss[par_idx] + 1e-12).mean())
        gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, *c.gain_limits))
        zs = np.concatenate([zs, cz])
        thetas = np.concatenate([thetas, cth])
        loss = np.concatenate([loss, cl])

    return {"score": float(loss.min()), "final_gain": gain}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+",
                        default=["blob2d", "smooth1d", "tsp100"])
    parser.add_argument("--arms", nargs="+", choices=("rigid", "loose"),
                        default=["rigid", "loose"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--survivors", type=int, default=8,
                        help="truncation for the rigid arm only")
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
                out = run(problem, arm, args.budget, seed, args.survivors)
                rows.append({"problem": problem, "arm": arm, "seed": seed,
                             **out})
                print(f"  {arm:<6} seed {seed} score {out['score']:.6g}",
                      flush=True)
            vals = [r["score"] for r in rows
                    if r["problem"] == problem and r["arm"] == arm]
            print(f"  {arm:<6} MEAN {np.mean(vals):.6g} +- "
                  f"{np.std(vals, ddof=1):.3g}", flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "pop": POP, "pairs": PAIRS,
             "family": FAMILY, "solo": SOLO, "survivors": args.survivors,
             "torch_version": torch.__version__, "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
