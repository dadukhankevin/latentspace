"""Round 43: fix the two things round 42's crossover got wrong.

Round 42 added crossover to the decoder GA for the first time and the survivor
sweep changed shape: with crossover, mu=8 became the best setting on image,
curve AND TSP at once (0.00403 / 0.00115 / 15.50), where without it mu=8 was
near-worst on the smooth problems (0.00526 / 0.00299). Round 38's
problem-dependent survivor count looks largely like an artifact of having no
recombination operator. But mu=8 is near-best and never actually best — it ties
elite=1 on the image, loses on the curve, loses on TSP — and the operator is
still crude in two independent ways.

1. MATE SELECTION IS UNIFORM. `par` and `mate` are both flat random over the
   survivors, so rank 0 mates with rank 15 as often as with rank 1, and the
   average mate is rank ~7.5. Under the round-42 winner rule this is actively
   destructive: the fitter parent donates its decoder AND the genome it is
   co-adapted to, then a middling genome's segment is grafted into that pair.
   "Elite" only means top-16-of-48; inside the survivor set there is no bias at
   all. Tournament selection biases both parents toward the top without a hard
   percentile cutoff, so good-with-good is common and good-with-poor still
   happens sometimes.

2. THE CUTS IGNORE THE GENE GRAMMAR. TSP's anchor field reads the 64 genes as
   8 anchors x 8 genes (2 position + 6 features). A cut at a random index slices
   THROUGH an anchor, so the child gets a chimeric anchor holding one parent's
   position and the other parent's features — an anchor pointing at a location
   whose message came from somewhere else. Conv decoders have no block
   structure, so any cut is harmless. That matches round 42 exactly: crossover
   helped both conv problems (1.17x, 1.51x) and hurt the one anchor problem
   (0.89x). Cutting on anchor boundaries is tier-1 legal — the decoder already
   declares its own architecture, and `gene_block` is just that architecture
   describing the units its genome comes in.

The two causes are independent, so they form a 2x2. On conv problems block ==
free by construction (block size 1), so only the selection axis varies there.

Arms: <selection>_<cuts>, selection in {uniform, tourney}, cuts in {free, block}.
Plus `no_crossover` for reference.

Round 42 reference at mu=8: image 0.00403, curve 0.00115, TSP 15.50.
Best known overall: image 0.00398 (elite=1), curve 0.00098 (elite=1),
TSP 14.91 (elite=16, no crossover).
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
TOURNEY_K = 3

# How many genes the decoder's grammar treats as one unit. The anchor field
# reads 64 genes as 8 anchors x (2 position + 6 features), so its genome comes
# in blocks of 8. Conv decoders impose no structure on the genome: block 1.
GENE_BLOCK = {"blob2d": 1, "smooth1d": 1, "tsp100": 8}


def run(problem: str, arm: str, budget: int, seed: int, elite: int,
        n_points: int) -> dict:
    loss_fn, output_shape, architecture = make_problem(problem, seed)
    c = ExplorerConfig(elite=elite)
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = _Template(resolve(architecture, LATENT, output_shape), "mps")
    block = GENE_BLOCK[problem] if arm.endswith("_block") else 1

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

    def select(n: int, mu: int) -> np.ndarray:
        """Survivors are sorted, so a LOWER index is a fitter individual and a
        tournament is just the min of k uniform draws."""
        if arm.startswith("tourney") and mu > 1:
            return np.min(rng.integers(0, mu, (n, TOURNEY_K)), axis=1)
        return rng.integers(0, mu, n)

    def cross_z(base: np.ndarray, donor: np.ndarray) -> np.ndarray:
        """Graft contiguous segments of `donor` into `base`, cutting only on
        `block` boundaries so a gene grammar's units are never split."""
        sites = np.arange(block, LATENT, block)
        if len(sites) == 0:
            return base.copy()
        cuts = np.sort(rng.choice(sites, size=min(n_points, len(sites)),
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
        par = select(n, len(zs))

        if arm != "no_crossover" and len(zs) > 1:
            mate = select(n, len(zs))
            winner = np.minimum(par, mate)
            loser = np.maximum(par, mate)
            cz = np.stack([mutate_z(cross_z(zs[w], zs[l]))
                           for w, l in zip(winner, loser)])
            cth = np.stack([mutate_theta(thetas[w]) for w in winner])
            bar = winner
        else:
            cz = np.stack([mutate_z(zs[p]) for p in par])
            cth = np.stack([mutate_theta(thetas[p]) for p in par])
            bar = par

        cl = loss_fn(decode_all(cz, cth))
        spent += n
        wins = float((cl <= loss[bar] + 1e-12).mean())
        gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, *c.gain_limits))
        zs = np.concatenate([zs, cz])
        thetas = np.concatenate([thetas, cth])
        loss = np.concatenate([loss, cl])
        trace.append({"e": spent, "m": float(loss.min()), "gain": gain,
                      "win": wins})

    return {"score": float(loss.min()), "final_gain": gain,
            "trace": trace[::10]}


ARMS = ("no_crossover", "uniform_free", "uniform_block", "tourney_free",
        "tourney_block")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+",
                        default=["blob2d", "smooth1d", "tsp100"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--elite", type=int, default=8)
    parser.add_argument("--n-points", type=int, default=1)
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for problem in args.problems:
        arms = list(args.arms)
        if GENE_BLOCK[problem] == 1:   # block == free here; do not re-run it
            arms = [a for a in arms if not a.endswith("_block")]
        print(f"\n########## {problem} (budget {args.budget}, elite "
              f"{args.elite}, gene block {GENE_BLOCK[problem]}) ##########",
              flush=True)
        for arm in arms:
            for seed in args.seeds:
                out = run(problem, arm, args.budget, seed, args.elite,
                          args.n_points)
                out.pop("trace")
                rows.append({"problem": problem, "arm": arm, "seed": seed,
                             "elite": args.elite, **out})
                print(f"  {arm:<14} seed {seed} score {out['score']:.6g}",
                      flush=True)
            vals = [r["score"] for r in rows
                    if r["problem"] == problem and r["arm"] == arm]
            print(f"  {arm:<14} MEAN {np.mean(vals):.6g} +- "
                  f"{np.std(vals, ddof=1):.3g}", flush=True)

    print("\n--- summary (score; lower is better) ---")
    for problem in args.problems:
        sel = [r for r in rows if r["problem"] == problem]
        base = np.mean([r["score"] for r in sel if r["arm"] == "uniform_free"])
        for arm in sorted({r["arm"] for r in sel}):
            v = np.mean([r["score"] for r in sel if r["arm"] == arm])
            print(f"  {problem:<10} {arm:<14} {v:.6g} "
                  f"({base/v:.3f}x vs uniform_free)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "elite": args.elite,
             "n_points": args.n_points, "tourney_k": TOURNEY_K,
             "gene_block": GENE_BLOCK, "torch_version": torch.__version__,
             "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
