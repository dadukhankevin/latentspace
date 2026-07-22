"""Round 44: how children get handed out, and how hard we truncate.

Two things the decoder GA has been doing without ever testing the alternative.

1. CHILDREN ARE ALLOCATED BY DICE ROLL. Parents are drawn 32 times WITH
   REPLACEMENT from the survivor set, so a survivor gets 4 children on average
   but may get zero — a lineage can die from sampling luck alone, despite having
   survived selection. Every earlier round in this campaign inherits that noise.
   The classical alternative is deterministic: each survivor gets exactly
   population/survivors children, everyone breeds, nothing is lost to variance.

2. TRUNCATION IS HARSH AND NEVER JUSTIFIED. 8 survivors out of a 40-strong pool
   (8 parents + 32 children) is a 20% cut. Nothing tested whether that is right;
   the shipped 16 was never compared against a gentler regime with a bigger
   population behind it. Note "population" here means CHILDREN PER GENERATION —
   the population has always been 32, and the survivor count is a separate
   number, which the survivor/population naming has been obscuring.

Both arms keep crossover on (uniform mates, free cuts — round 43's best on the
curve and TSP; tournament won only on the image and is a separate axis).

Reference at 5k: image 0.00342 (8 survivors + tournament crossover) / 0.00398
(1 survivor, no crossover); curve 0.00086 (8 survivors + uniform crossover);
TSP-100 14.91 (16 survivors, no crossover, 3 seeds) or 15.23 (5 seeds).

With equal allocation, survivors must divide the children evenly, so the
survivor counts tested are divisors of the population.
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


def run(problem: str, allocation: str, survivors: int, population: int,
        budget: int, seed: int) -> dict:
    loss_fn, output_shape, architecture = make_problem(problem, seed)
    c = ExplorerConfig()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = _Template(resolve(architecture, LATENT, output_shape), "mps")

    def decode_all(zs, thetas):
        return torch.stack([template.decode(t, z)
                            for z, t in zip(zs, thetas)])

    zs = rng.standard_normal((population, LATENT)).astype(np.float32)
    thetas = np.stack([template.init_theta(int(rng.integers(0, 2**31)))
                       for _ in range(population)])
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

    def cross_z(base, donor):
        cut = int(rng.integers(1, LATENT))
        child = base.copy()
        child[cut:] = donor[cut:]
        return child.astype(np.float32)

    while spent < budget:
        keep = min(survivors, len(loss))
        order = np.argsort(loss)[:keep]
        zs, thetas, loss = zs[order], thetas[order], loss[order]
        n = min(population, budget - spent)

        if allocation == "equal":
            # every survivor breeds exactly the same number of children; no
            # lineage is lost to a dice roll. Remainder (when n is not a
            # multiple of keep, i.e. the final truncated generation) goes to
            # the fittest.
            par = np.repeat(np.arange(keep), int(np.ceil(n / keep)))[:n]
        else:
            par = rng.integers(0, keep, n)
        mate = rng.integers(0, keep, n)
        winner, loser = np.minimum(par, mate), np.maximum(par, mate)

        cz = np.stack([mutate_z(cross_z(zs[w], zs[l]))
                       for w, l in zip(winner, loser)])
        cth = np.stack([mutate_theta(thetas[w]) for w in winner])
        cl = loss_fn(decode_all(cz, cth))
        spent += n
        wins = float((cl <= loss[winner] + 1e-12).mean())
        gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, *c.gain_limits))
        zs = np.concatenate([zs, cz])
        thetas = np.concatenate([thetas, cth])
        loss = np.concatenate([loss, cl])
        trace.append({"e": spent, "m": float(loss.min()), "gain": gain})

    return {"score": float(loss.min()), "final_gain": gain}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+",
                        default=["blob2d", "smooth1d", "tsp100"])
    parser.add_argument("--allocations", nargs="+", default=["random", "equal"],
                        choices=["random", "equal"])
    parser.add_argument("--survivors", nargs="+", type=int,
                        default=[8, 16, 32])
    parser.add_argument("--population", type=int, default=32,
                        help="children per generation (NOT the survivor count)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for problem in args.problems:
        print(f"\n########## {problem} (budget {args.budget}, population "
              f"{args.population} children/gen) ##########", flush=True)
        for allocation in args.allocations:
            for survivors in args.survivors:
                for seed in args.seeds:
                    out = run(problem, allocation, survivors, args.population,
                              args.budget, seed)
                    rows.append({"problem": problem, "allocation": allocation,
                                 "survivors": survivors,
                                 "population": args.population,
                                 "seed": seed, **out})
                vals = [r["score"] for r in rows
                        if r["problem"] == problem
                        and r["allocation"] == allocation
                        and r["survivors"] == survivors]
                print(f"  {allocation:<7} survivors {survivors:<3} MEAN "
                      f"{np.mean(vals):.6g} +- {np.std(vals, ddof=1):.3g}",
                      flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "population": args.population,
             "torch_version": torch.__version__, "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
