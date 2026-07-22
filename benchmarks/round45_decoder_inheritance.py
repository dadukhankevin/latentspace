"""Round 45: what should a crossover child inherit, and does crossover itself act?

TSP-100 only (fast), 10 seeds, so paired same-seed stats finally have teeth
(t threshold 2.262). All arms: 8 survivors of a 40-pool, uniform mates, one
genome cut, every child mutated after birth — the configuration that scored
14.80 in round 44.

Question 1 — ATTRIBUTION. Since round 42 every child gets crossover AND
mutation, so "crossover helped" has strictly meant "the pipeline with
crossover in it helped." The mutation_only arm isolates the operator.

Question 2 — DECODER INHERITANCE. The genome is recombined; the decoder
cannot be cut in half meaningfully, so the child must get it some other way:

  * fitter_decoder — the incumbent since round 42: the fitter parent donates
    its whole decoder (and its genome is the base; the other parent grafts a
    segment in).
  * averaged_decoder — the child's weights are the elementwise mean of both
    parents' weight vectors.

Round 37 pulls both ways on averaging. Co-adaptation says disaster: decoder/
genome pairs are so entangled that cross-run swaps score 18-28x worse than
flat gray, and blending two networks usually produces neither. Lineage
collapse says harmless-or-better: survivors are near-clones a few mutation
steps apart, and averaging near-clones is noise-cancellation, not surgery.
Which force wins is exactly what the arm measures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import re

from benchmarks.batched_decode import BatchedTemplate
from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round21_tsp import make_instance
from benchmarks.round25_anchor_field import AnchorFieldTransformer
from benchmarks.round39_survivors import make_problem as _make_problem_39
from benchmarks.round40_diversity_probe import _tsp_loss
from latentspace.universal.architectures import resolve
from latentspace.universal.explorer import ExplorerConfig


def make_problem(name: str, seed: int):
    """round 39's problems, plus tsp<N> for any city count, plus the apple."""
    m = re.fullmatch(r"tsp(\d+)", name)
    if m:
        cities = make_instance(seed, int(m.group(1)))
        return (_tsp_loss(cities), (len(cities),),
                lambda latent, shape: AnchorFieldTransformer(latent, shape,
                                                             cities))
    if name == "apple":
        from benchmarks.round27_apple_no_cma import load_apple
        from benchmarks.round28_anchor_conv import ConvRGB
        target = load_apple()
        cache: dict[str, torch.Tensor] = {}

        def loss(phenos: torch.Tensor) -> np.ndarray:
            key = str(phenos.device)
            if key not in cache:
                cache[key] = torch.as_tensor(target, device=phenos.device)
            return ((phenos.flatten(1) - cache[key]) ** 2).mean(dim=1).cpu().numpy()
        return loss, (3, 96, 96), lambda latent, shape: ConvRGB(latent, shape)
    return _make_problem_39(name, seed)

LATENT = 64
POP = 32
SURVIVORS = 8

ARMS = ("mutation_only", "fitter_decoder", "averaged_decoder")


def run(problem: str, arm: str, budget: int, seed: int) -> dict:
    loss_fn, output_shape, architecture = make_problem(problem, seed)
    c = ExplorerConfig()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = BatchedTemplate(resolve(architecture, LATENT, output_shape),
                               "mps")

    def decode_all(zs, thetas):
        return template.decode_batch(np.asarray(thetas), np.asarray(zs))

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
        order = np.argsort(loss)[:SURVIVORS]
        zs, thetas, loss = zs[order], thetas[order], loss[order]
        n = min(POP, budget - spent)

        if arm == "mutation_only":
            par = rng.integers(0, len(zs), n)
            cz = np.stack([mutate_z(zs[p]) for p in par])
            cth = np.stack([mutate_theta(thetas[p]) for p in par])
            bar = par
        else:
            par = rng.integers(0, len(zs), n)
            mate = rng.integers(0, len(zs), n)
            winner, loser = np.minimum(par, mate), np.maximum(par, mate)
            cz = np.stack([mutate_z(cross_z(zs[w], zs[l]))
                           for w, l in zip(winner, loser)])
            if arm == "fitter_decoder":
                cth = np.stack([mutate_theta(thetas[w]) for w in winner])
            else:   # averaged_decoder
                cth = np.stack([
                    mutate_theta((thetas[w] + thetas[l]) / 2.0)
                    for w, l in zip(winner, loser)])
            bar = winner

        cl = loss_fn(decode_all(cz, cth))
        spent += n
        wins = float((cl <= loss[bar] + 1e-12).mean())
        gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, *c.gain_limits))
        zs = np.concatenate([zs, cz])
        thetas = np.concatenate([thetas, cth])
        loss = np.concatenate([loss, cl])

    return {"score": float(loss.min()), "final_gain": gain}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=["tsp100"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(10)))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for problem in args.problems:
        print(f"\n########## {problem} (budget {args.budget}, "
              f"{len(args.seeds)} seeds) ##########", flush=True)
        for arm in args.arms:
            for seed in args.seeds:
                out = run(problem, arm, args.budget, seed)
                rows.append({"problem": problem, "arm": arm, "seed": seed,
                             **out})
                print(f"  {arm:<17} seed {seed} score {out['score']:.6g}",
                      flush=True)
            vals = [r["score"] for r in rows
                    if r["problem"] == problem and r["arm"] == arm]
            print(f"  {arm:<17} MEAN {np.mean(vals):.6g} +- "
                  f"{np.std(vals, ddof=1):.3g}", flush=True)

        # paired same-seed t-tests against mutation_only
        base = {r["seed"]: r["score"] for r in rows
                if r["problem"] == problem and r["arm"] == "mutation_only"}
        if base:
            print("\n  paired vs mutation_only (t threshold 2.262 at n=10):")
            for arm in args.arms:
                if arm == "mutation_only":
                    continue
                pairs = [(r["score"], base[r["seed"]]) for r in rows
                         if r["problem"] == problem and r["arm"] == arm
                         and r["seed"] in base]
                d = np.array([a - b for a, b in pairs])
                t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                verdict = "SIGNIFICANT" if abs(t) > 2.262 else "not significant"
                print(f"    {arm:<17} mean diff {d.mean():+8.4f}  t={t:+5.2f}"
                      f"  {verdict}", flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "pop": POP, "survivors": SURVIVORS,
             "torch_version": torch.__version__, "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
