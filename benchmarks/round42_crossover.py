"""Round 42: give the decoder GA crossover. It has never had any.

The universal explorer has been mutation-only since it was written — children
are noisy copies of ONE parent. Nothing was ever removed: `universal/explorer.py`
had no crossover call in its first commit, and its docstring's claim that
"mutation and crossover exist only for tensors" describes an intention the code
never implemented. The `Crossover` layer in `latentspace/layers.py` is real and
works, but it is wired into `evolver.py` — the TRADITIONAL GA baseline, the arm
this project has been beating. So every result in this campaign comes from what
is really a (mu + lambda) evolution strategy, not a genetic algorithm.

Round 25e is the only crossover evidence on record and it does not transfer: it
tested order crossover on the traditional TOUR GA (8.07 -> 8.88 at 50 cities,
21.01 -> 22.05 at 100 — worse). That is a different algorithm operating on a
different representation.

THE DESIGN, deliberately minimal:

  * ONE CUT on the genome. The 64 genes are a plain float vector, so a single
    crossover point is modality-independent — it works identically for pixels,
    curves and tours, and touches nothing but a tensor evolution already owns.
  * THE DECODER IS NOT MIXED. One parent wins it whole. Round 37 measured why:
    decoder/genome pairs are totally co-adapted, and swapping them across runs
    scores 18-28x worse than flat gray. Blending two weight vectors would very
    likely produce garbage, so the decoder is inherited, never averaged.
  * THE FITTER PARENT DONATES. It contributes both its decoder and the genome
    it is co-adapted to; the other parent grafts in one contiguous segment.
    This keeps the child near a working genome/decoder pair and makes the graft
    the perturbation, rather than asking a decoder to read a genome half of
    which it has never seen.

Crossover needs distinct parents, so every arm here breeds from the shipped
elite=16 survivor set. That also makes this a live test of a round-38 puzzle:
16 survivors are 1.9x WORSE than a single champion on the image, and the
suspicion is that keeping many lineages only pays if something can COMBINE
them. Mutation alone cannot; that is what crossover is for.

Arms:

  * no_crossover — the incumbent: one parent, mutate genome and weights.
  * crossover    — two parents, one-point genome graft, fitter parent's
                   decoder, then the same mutation as above.

Reference points at 5k budget, elite=16 (round 38/39): image 0.00771,
curve 0.00229, TSP-100 14.91. Best single champion (elite=1): image 0.00398,
curve 0.00098, TSP 18.55.
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


def run(problem: str, arm: str, budget: int, seed: int, elite: int,
        n_points: int) -> dict:
    loss_fn, output_shape, architecture = make_problem(problem, seed)
    c = ExplorerConfig(elite=elite)
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = _Template(resolve(architecture, LATENT, output_shape), "mps")

    def decode_all(zs, thetas):
        return torch.stack([template.decode(t, z)
                            for z, t in zip(zs, thetas)])

    zs = rng.standard_normal((c.population, LATENT)).astype(np.float32)
    thetas = np.stack([template.init_theta(int(rng.integers(0, 2**31)))
                       for _ in range(c.population)])
    loss = loss_fn(decode_all(zs, thetas))
    spent, gain, trace = len(zs), 1.0, []

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

    def cross_z(base: np.ndarray, donor: np.ndarray) -> np.ndarray:
        """Graft contiguous segments of `donor` into `base`. `base` belongs to
        the parent whose decoder the child inherits, so the child stays on a
        co-adapted genome/decoder pair and the graft is the perturbation."""
        cuts = np.sort(rng.choice(np.arange(1, LATENT),
                                  size=min(n_points, LATENT - 1),
                                  replace=False))
        child, take, prev = base.copy(), False, 0
        for cut in cuts:
            if take:
                child[prev:cut] = donor[prev:cut]
            take, prev = not take, cut
        if take:
            child[prev:] = donor[prev:]
        return child.astype(np.float32)

    while spent < budget:
        order = np.argsort(loss)[:c.elite]
        zs, thetas, loss = zs[order], thetas[order], loss[order]
        n = min(c.population, budget - spent)
        par = rng.integers(0, len(zs), n)

        if arm == "crossover" and len(zs) > 1:
            mate = rng.integers(0, len(zs), n)
            # survivors are sorted, so the lower index IS the fitter parent
            winner = np.minimum(par, mate)
            loser = np.maximum(par, mate)
            cz = np.stack([mutate_z(cross_z(zs[w], zs[l]))
                           for w, l in zip(winner, loser)])
            cth = np.stack([mutate_theta(thetas[w]) for w in winner])
        else:
            cz = np.stack([mutate_z(zs[p]) for p in par])
            cth = np.stack([mutate_theta(thetas[p]) for p in par])

        cl = loss_fn(decode_all(cz, cth))
        spent += n
        wins = float((cl <= loss[par] + 1e-12).mean())
        gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, *c.gain_limits))
        zs = np.concatenate([zs, cz])
        thetas = np.concatenate([thetas, cth])
        loss = np.concatenate([loss, cl])
        trace.append({"e": spent, "m": float(loss.min()), "gain": gain,
                      "win": wins})

    return {"score": float(loss.min()), "final_gain": gain,
            "trace": trace[::10]}


ARMS = ("no_crossover", "crossover")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+",
                        default=["blob2d", "smooth1d", "tsp100"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--elite", type=int, default=16)
    parser.add_argument("--n-points", type=int, default=1,
                        help="genome cut points (1 = single-point crossover)")
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for problem in args.problems:
        print(f"\n########## {problem} (budget {args.budget}, elite "
              f"{args.elite}, {args.n_points}-point) ##########", flush=True)
        for arm in args.arms:
            for seed in args.seeds:
                out = run(problem, arm, args.budget, seed, args.elite,
                          args.n_points)
                out.pop("trace")
                rows.append({"problem": problem, "arm": arm, "seed": seed,
                             "elite": args.elite, "n_points": args.n_points,
                             **out})
                print(f"  {arm:<13} seed {seed} score {out['score']:.6g} "
                      f"(gain {out['final_gain']:.2f})", flush=True)
            vals = [r["score"] for r in rows
                    if r["problem"] == problem and r["arm"] == arm]
            print(f"  {arm:<13} MEAN {np.mean(vals):.6g}", flush=True)

    print("\n--- summary (score; lower is better) ---")
    for problem in args.problems:
        base = np.mean([r["score"] for r in rows
                        if r["problem"] == problem
                        and r["arm"] == "no_crossover"])
        for arm in args.arms:
            v = np.mean([r["score"] for r in rows
                         if r["problem"] == problem and r["arm"] == arm])
            print(f"  {problem:<10} {arm:<13} {v:.6g} "
                  f"({base/v:.3f}x vs no crossover)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "elite": args.elite,
             "n_points": args.n_points, "torch_version": torch.__version__,
             "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
