"""Round 47: equal WALL-CLOCK, not equal evaluations. The honest weak spot.

Every comparison in this campaign holds EVALUATIONS equal, which assumes
evaluations are the scarce resource. That is the regime the method is aimed
at (simulations, physical experiments, expensive scoring) — but on the apple
the fitness is a cheap MSE, and per evaluation the decoder GA pays a large
overhead: every child is decoded through a neural network on the GPU, while
a pixel GA's mutation is nearly free numpy noise. Under a TIME budget the
pixel GA gets orders of magnitude more evaluations. Daniel's question: does
ours still win then?

Protocol:

  1. Run the shipped decoder GA for its standard 150k evaluations and time
     it precisely, recording (wall seconds, best MSE) throughout.
  2. Give the pixel GA that exact wall-clock budget, same seed, and let it
     take as many evaluations as it can.
  3. Strongest-baseline rule: also run the pixel GA with OUR win-rate step
     control (grow sigma when >1/5 of children beat their parents, shrink
     otherwise). Its confetti floor is partly a fixed-sigma artifact and we
     know that handicap is worth 10-95x on decoders; the claim "ours is
     better" is only worth making against the pixel GA at its best.

Whatever wins, the result goes in FINDINGS: either the decoder GA survives
losing its sample-efficiency crutch, or the honest claim narrows to
"sample-efficient, not time-efficient, on cheap fitness functions."
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round27_apple_no_cma import load_apple
from benchmarks.round28_anchor_conv import ConvRGB
from benchmarks.legacy_engines.solver import solve_single as solve

SHAPE = (3, 96, 96)


def run_decoder(target, budget, seed):
    target_t = torch.as_tensor(target, device="mps")
    trace, state = [], {"spent": 0, "best": np.inf}
    t0 = time.perf_counter()

    def fitness(phenotypes):
        mse = ((phenotypes.flatten(1) - target_t) ** 2).mean(dim=1)
        state["spent"] += len(mse)
        lo = float(mse.min())
        if lo < state["best"]:
            state["best"] = lo
        trace.append({"t": time.perf_counter() - t0, "e": state["spent"],
                      "m": state["best"]})
        return -mse

    _seed_everything(seed)
    result = solve(fitness, output_shape=SHAPE, budget=budget,
                   architecture=lambda latent, shape: ConvRGB(latent, shape),
                   explore_fraction=1.0, seed=seed)
    seconds = time.perf_counter() - t0
    return {"mse": float(-result.best_fitness), "seconds": seconds,
            "evaluations": result.evaluations, "trace": trace[::20]}


def run_pixel_ga(target, seconds_budget, seed, adaptive: bool):
    """Rank selection, uniform crossover, per-pixel Gaussian mutation —
    round 31's baseline, stopped on WALL CLOCK. With `adaptive`, the mutation
    sigma is under the same win-rate rule the decoder GA uses."""
    rng = np.random.default_rng(seed)
    dim = target.size
    population, offspring = 32, 32
    pop = rng.random((population, dim)).astype(np.float32)
    loss = ((pop - target) ** 2).mean(axis=1)
    spent, sigma, gain = population, 0.1, 1.0
    trace = []
    t0 = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= seconds_budget:
            break
        ranked = pop[np.argsort(loss)]
        ranked_loss = np.sort(loss)
        weights = np.arange(population, 0, -1, dtype=np.float64)
        weights /= weights.sum()
        pick = rng.choice(population, size=(offspring, 2), p=weights)
        first, second = ranked[pick[:, 0]], ranked[pick[:, 1]]
        children = np.where(rng.random((offspring, dim)) < 0.5, first, second)
        mask = rng.random((offspring, dim)) < 0.01
        children = np.clip(
            children + rng.normal(0, sigma * gain, (offspring, dim)) * mask,
            0, 1).astype(np.float32)
        child_loss = ((children - target) ** 2).mean(axis=1)
        spent += offspring
        if adaptive:
            parent_best = np.minimum(ranked_loss[pick[:, 0]],
                                     ranked_loss[pick[:, 1]])
            wins = float((child_loss <= parent_best + 1e-12).mean())
            gain *= 1.15 if wins > 0.2 else 1 / 1.15
            gain = float(np.clip(gain, 1e-4, 1e4))
        pop = np.concatenate([pop, children])
        loss = np.concatenate([loss, child_loss])
        keep = np.argsort(loss)[:population]
        pop, loss = pop[keep], loss[keep]
        if len(trace) == 0 or elapsed - trace[-1]["t"] > 1.0:
            trace.append({"t": elapsed, "e": spent, "m": float(loss.min()),
                          "gain": gain})

    return {"mse": float(loss.min()), "seconds": time.perf_counter() - t0,
            "evaluations": spent, "final_gain": gain, "trace": trace}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=150_000,
                        help="decoder GA evaluation budget; sets the clock")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    target = load_apple()

    print("timing the decoder GA...", flush=True)
    dec = run_decoder(target, args.budget, args.seed)
    clock = dec["seconds"]
    print(f"  decoder GA: {dec['mse']:.6f} in {clock:.1f}s "
          f"({dec['evaluations']:,} evals, "
          f"{dec['evaluations']/clock:,.0f} evals/s)", flush=True)

    rows = {"decoder_ga": dec}
    for name, adaptive in (("pixel_ga_fixed", False),
                           ("pixel_ga_adaptive", True)):
        print(f"running {name} for {clock:.1f}s...", flush=True)
        out = run_pixel_ga(target, clock, args.seed, adaptive)
        rows[name] = out
        print(f"  {name}: {out['mse']:.6f} in {out['seconds']:.1f}s "
              f"({out['evaluations']:,} evals, "
              f"{out['evaluations']/out['seconds']:,.0f} evals/s, "
              f"final gain {out['final_gain']:.3g})", flush=True)

    print(f"\nequal wall-clock ({clock:.1f}s), all seed {args.seed}:")
    for name, out in rows.items():
        ratio = out["mse"] / dec["mse"]
        print(f"  {name:<18} {out['mse']:.6f}  ({out['evaluations']:>11,} "
              f"evals)  {ratio:.2f}x vs decoder GA")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "seed": args.seed, "clock_seconds": clock,
             "torch_version": torch.__version__, "runs": rows}) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
