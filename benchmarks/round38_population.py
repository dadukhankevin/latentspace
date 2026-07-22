"""Round 38: if the population collapses to one lineage, how many survivors do we need?

Round 37's follow-up probe showed the per-individual-decoder population
collapses to a SINGLE ancestral lineage by generation 4 (of ~156) and
stays there — the 16 elites are near-clones, 1.3% apart in weight space.
So the "32 independent decoders" framing is fiction; what actually runs
is ONE decoder being refined, with a cloud of mutations tried each
generation. It is a (mu + lambda) evolution strategy on one decoder.

Daniel's reading: embrace it. If the survivors are becoming the same
decoder anyway, keeping many is waste — collapse to one. This round tests
exactly how far that goes by sweeping the number of survivors kept per
generation (elite = mu), everything else fixed (pure decoder GA,
win-rate step control, population/lambda = 32, 5k budget):

  * elite=1   — a clean (1 + 32) ES: ONE champion decoder carried forward,
                32 mutants tried, keep the single best. Daniel's "one
                decoder" in its strictest form.
  * elite=4   — a small survivor set.
  * elite=16  — the incumbent.

If elite=1 ties the others, the 1.3% clone spread does no work and the
method simplifies to a single evolving decoder with no loss — lighter,
clearer, and the same object as the one-backbone picture at scale. If
elite=1 loses, that small survivor diversity is a real (mu > 1)
exploration benefit and "they may as well be the same" is slightly too
strong.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round3_structure import SmoothTarget
from benchmarks.round8_mlp_pretrain import BlobImage2D
from benchmarks.round21_tsp import make_instance, nearest_neighbor_length
from benchmarks.round25_anchor_field import AnchorFieldTransformer
from benchmarks.legacy_engines.solver import solve_single as solve
from benchmarks.legacy_engines.explorer import ExplorerConfig

TSP_CITIES = 100


def image_fitness(objective):
    def f(phenotypes):
        return -objective.loss_tensor(phenotypes.flatten(1))
    return f


def tsp_fitness(cities):
    cache: dict[str, torch.Tensor] = {}

    def f(phenotypes):
        pr = phenotypes.reshape(len(phenotypes), -1)
        key = str(pr.device)
        if key not in cache:
            cache[key] = torch.as_tensor(cities, device=pr.device)
        pts = cache[key][torch.argsort(pr, dim=1)]
        return -(pts - pts.roll(-1, dims=1)).norm(dim=2).sum(dim=1)
    return f


def make_problem(name, seed):
    if name == "blob2d":
        o = BlobImage2D()
        return image_fitness(o), (32, 32), "conv2d", o.loss_numpy
    if name == "smooth1d":
        o = SmoothTarget()
        return image_fitness(o), (256,), "conv1d", o.loss_numpy
    if name == "tsp100":
        cities = make_instance(seed, TSP_CITIES)
        return (tsp_fitness(cities), (TSP_CITIES,),
                lambda l, s: AnchorFieldTransformer(l, s, cities), None)
    raise ValueError(name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+",
                        default=["blob2d", "smooth1d", "tsp100"])
    parser.add_argument("--elites", nargs="+", type=int, default=[1, 4, 16])
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
        for elite in args.elites:
            for seed in args.seeds:
                fitness, shape, arch, loss_np = make_problem(problem, seed)
                _seed_everything(seed)
                cfg = ExplorerConfig(elite=elite)
                result = solve(fitness, output_shape=shape, budget=args.budget,
                               architecture=arch, explore_fraction=1.0,
                               explorer_config=cfg, seed=seed)
                score = float(-result.best_fitness)
                rows.append({"problem": problem, "elite": elite,
                             "seed": seed, "score": score})
                print(f"  elite={elite:<2} seed {seed} score {score:.6g}",
                      flush=True)
            vals = [r["score"] for r in rows
                    if r["problem"] == problem and r["elite"] == elite]
            print(f"  elite={elite:<2} MEAN {np.mean(vals):.6g} +- "
                  f"{np.std(vals, ddof=1) if len(vals) > 1 else 0:.4g}",
                  flush=True)

    print("\n--- summary (score; lower is better) ---")
    for problem in args.problems:
        base = np.mean([r["score"] for r in rows
                        if r["problem"] == problem and r["elite"] == 16])
        for elite in args.elites:
            v = [r["score"] for r in rows
                 if r["problem"] == problem and r["elite"] == elite]
            print(f"  {problem:<10} elite={elite:<2} {np.mean(v):.6g} "
                  f"({base/np.mean(v):.3f}x vs elite=16)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "torch_version": torch.__version__,
             "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
