"""Round 41: steer the survivor count from FITNESS SPREAD.

Round 40 measured what the survivors look like and falsified the idea it was
built to test. Genotype diversity does NOT discriminate: on image, curve and
TSP alike the survivors sit ~2-3 mutation steps apart and dedupe to mu ~10-11,
so a diversity-driven controller would have answered "about ten" everywhere —
wrong on the image (round 38: 1 is 1.9x better) and wrong on TSP (round 38: 16
is 1.25x better). Including the decoder changed nothing; it tracked the genome.

But one column in that probe separated the problems by 10-100x — the one
predicted to be useless:

    relative fitness spread among survivors
      image 1.0e-2 | curve 1.7e-2 | TSP-100 1.5e-4

The prediction was that fitness spread would be BLIND on a plateau, because
many different tours share one length. It does read near-zero on TSP for
exactly that reason — but that near-zero IS the ruggedness signature. Survivors
that are genotypically far apart (2.3 steps) and yet fitness-identical is the
definition of a plateau, and a plateau is precisely where independent bets pay.
On the smooth problems genotypic distance converts into real fitness
differences, so there is a gradient to climb and one champion suffices.

The appeal: this is the strictest tier-0 signal available. Nothing but scores.
No output-space metric, no genotype geometry, no information the fitness
function did not already hand over.

TWO honest caveats, both load-bearing.

1. THE DETECTOR MUST NOT READ ITS OWN OUTPUT. Spread measured across the mu
   survivors is a feedback loop: at mu=1 the spread is exactly 0, which reads
   as "plateau", which grows mu, forever. So spread is measured across the 32
   CHILDREN of each generation — always the same count however many parents
   bred them, so the signal is mu-independent by construction.

2. THE THRESHOLD IS FITTED, AND THAT IS THE THING THIS PROJECT EXISTS TO
   AVOID. `SPREAD_TARGET` below sits in the gap between the measured 1.5e-4 and
   1.0e-2 — chosen by looking at the answer. This round is a MECHANISM TEST,
   not a shippable controller: if the signal cannot steer mu even with the
   constant hand-picked, then no principled version of it is worth hunting for.
   If it does work, the threshold becomes the next problem, not a solved one.

Arms:

  * fixed_1          — round 38's champion on image and curve.
  * fixed_16         — round 38's champion on TSP, and the shipped default.
  * adaptive_spread  — mu grows while children are fitness-tied (plateau) and
                       shrinks while they are spread out (gradient).

The bar is unchanged and it is strict: match the better fixed arm on EVERY
problem without being told which problem it is. Round 39's three controllers
all failed it (best: 1.33x on image, 0.62x on curve, 0.94x on TSP).
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
from latentspace.universal.explorer import ExplorerConfig, _Template

LATENT = 64
MU_START = 4
MU_STEP = 1.15
# Fitted. See caveat 2 in the module docstring: the measured relative child
# spread is ~1e-2 on smooth problems and ~1.5e-4 on TSP, so this sits in the
# gap. It is a knob, and naming it one is the point.
SPREAD_TARGET = 1e-3
REFERENCE_WINDOW = 16   # fixed observation window; NOT the breeding count


def run(problem: str, arm: str, budget: int, seed: int) -> dict:
    loss_fn, output_shape, architecture = make_problem(problem, seed)
    c = ExplorerConfig()
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
    mu_real = float(MU_START if arm.startswith("adaptive")
                    else int(arm.split("_")[1]))

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

    while spent < budget:
        mu = int(np.clip(round(mu_real), 1, c.population // 2))
        order = np.argsort(loss)[:mu]
        zs, thetas, loss = zs[order], thetas[order], loss[order]
        n = min(c.population, budget - spent)
        par = rng.integers(0, len(zs), n)
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

        # Ruggedness detector: fitness spread across a FIXED-SIZE window of the
        # best of the pool. Two failed designs bracket this one. Measured over
        # the mu survivors, the detector reads its own output back: at mu=1 the
        # spread is exactly 0, which reads as plateau, which grows mu, forever.
        # Measured over all 32 children, the signal drowns — children include
        # every failed mutation, and failure is equally spread out on every
        # problem, so TSP read 1.9e-2 instead of the 1.5e-4 round 40 saw among
        # elites, and mu collapsed to 1 everywhere. The tied-ness that marks a
        # plateau is visible only among SURVIVORS. So the window is fixed at 16
        # regardless of how many actually breed: measurement size and breeding
        # size are decoupled, and 16 is an observation window, not a knob.
        window = np.sort(loss)[:REFERENCE_WINDOW]
        spread = float(window.std() / max(abs(window.mean()), 1e-12))
        if arm == "adaptive_spread":
            # tied survivors => plateau => keep more independent bets
            mu_real *= MU_STEP if spread < SPREAD_TARGET else 1 / MU_STEP
            mu_real = float(np.clip(mu_real, 1.0, c.population // 2))
        trace.append({"e": spent, "m": float(loss.min()), "gain": gain,
                      "mu": mu, "spread": spread})

    mus = [t["mu"] for t in trace]
    return {"score": float(loss.min()), "final_mu": mus[-1],
            "mean_mu": float(np.mean(mus)),
            "mean_spread": float(np.mean([t["spread"] for t in trace])),
            "trace": trace[::10]}


ARMS = ("fixed_1", "fixed_16", "adaptive_spread")


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
                out.pop("trace")
                rows.append({"problem": problem, "arm": arm, "seed": seed,
                             **out})
                print(f"  {arm:<16} seed {seed} score {out['score']:.6g} "
                      f"(mu {out['mean_mu']:.1f} mean -> {out['final_mu']} "
                      f"final, spread {out['mean_spread']:.2e})", flush=True)
            vals = [r["score"] for r in rows
                    if r["problem"] == problem and r["arm"] == arm]
            print(f"  {arm:<16} MEAN {np.mean(vals):.6g}", flush=True)

    print("\n--- summary (score; lower is better) ---")
    for problem in args.problems:
        fixed = [np.mean([r["score"] for r in rows
                          if r["problem"] == problem and r["arm"] == a])
                 for a in ("fixed_1", "fixed_16") if a in args.arms]
        best_fixed = min(fixed) if fixed else float("nan")
        for arm in args.arms:
            v = np.mean([r["score"] for r in rows
                         if r["problem"] == problem and r["arm"] == arm])
            print(f"  {problem:<10} {arm:<16} {v:.6g} "
                  f"({best_fixed/v:.3f}x vs best fixed arm)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "mu_start": MU_START, "mu_step": MU_STEP,
             "spread_target": SPREAD_TARGET,
             "torch_version": torch.__version__, "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
