"""Round 48: the gradient ceiling. What does Adam do to the apple?

Not a competition — a boundary measurement. Adam sees the full 27,648-dim
gradient of the loss at every step; the decoder GA sees scalar fitness
scores and nothing else. This run prices that information gap, and answers
two questions the campaign has never measured:

  1. How fast does a gradient method reach our black-box record (0.003248)?
  2. What is the EXPRESSIVENESS CEILING of the decoder we evolve? Adam
     training the same ConvRGB (weights + genome jointly, deep-image-prior
     style) converges to roughly the best image this architecture can emit —
     the floor evolution could approach but never pass. How close is
     0.003248 to it?

Arms, each capped at the decoder GA's own wall-clock (from round 47):

  * adam_pixels  — Adam on the raw pixel tensor. Gradient descent straight
                   at the target; the trivial upper bound.
  * adam_decoder — Adam on ConvRGB weights + the 64-float genome, same
                   architecture the GA evolves, sigmoid output, MSE loss.
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

SHAPE = (3, 96, 96)
RECORD = 0.003248


def run_adam(target: np.ndarray, arm: str, seconds_budget: float, seed: int,
             lr: float) -> dict:
    _seed_everything(seed)
    device = "mps"
    target_t = torch.as_tensor(target.reshape(1, -1), device=device)

    if arm == "adam_pixels":
        params = torch.rand(1, target.size, device=device, requires_grad=True)
        trainables = [params]
        def output(): return torch.sigmoid(params)
    else:
        net = ConvRGB(64, SHAPE).to(device)
        z = torch.randn(1, 64, device=device, requires_grad=True)
        trainables = list(net.parameters()) + [z]
        def output(): return torch.sigmoid(net(z)).flatten(1)

    optimizer = torch.optim.Adam(trainables, lr=lr)
    trace, best, steps, t_record = [], np.inf, 0, None
    t0 = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= seconds_budget:
            break
        optimizer.zero_grad()
        loss = ((output() - target_t) ** 2).mean()
        loss.backward()
        optimizer.step()
        steps += 1
        v = float(loss)
        if v < best:
            best = v
            if t_record is None and best < RECORD:
                t_record = elapsed
        if len(trace) == 0 or elapsed - trace[-1]["t"] > 0.5:
            trace.append({"t": elapsed, "step": steps, "m": best})

    return {"mse": best, "steps": steps,
            "seconds": time.perf_counter() - t0,
            "seconds_to_record": t_record, "trace": trace}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock", type=float, default=None,
                        help="wall-clock cap; default reads round 47's")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr-pixels", type=float, default=0.05)
    parser.add_argument("--lr-decoder", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    target = load_apple()
    clock = args.clock
    if clock is None:
        clock = json.load(open("benchmark_results/mps_round47_time_budget.json"))["clock_seconds"]
    print(f"clock {clock:.1f}s (decoder GA: {RECORD} in that time, 150k "
          f"fitness evals)", flush=True)

    rows = {}
    for arm, lr in (("adam_pixels", args.lr_pixels),
                    ("adam_decoder", args.lr_decoder)):
        out = run_adam(target, arm, clock, args.seed, lr)
        rows[arm] = out
        rec = ("never" if out["seconds_to_record"] is None
               else f"{out['seconds_to_record']:.2f}s")
        print(f"  {arm:<13} final {out['mse']:.8f}  ({out['steps']:,} steps, "
              f"reached our record in {rec})", flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"clock_seconds": clock, "seed": args.seed, "record": RECORD,
             "lr_pixels": args.lr_pixels, "lr_decoder": args.lr_decoder,
             "torch_version": torch.__version__, "runs": rows}) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
