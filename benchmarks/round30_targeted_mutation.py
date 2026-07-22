"""Round 30: set mutation size from measured phenotype movement.

Daniel's proposal: mutate exactly the same things, but scale the mutation
RATE up or down from how much the phenotype actually moves, against a
target.

Round 29 found the premise is worse than suspected. The explorer's weight
sigma range [0.003, 0.02] carries a comment claiming "~0% of mutant
outputs beat their parent at 0.03+". That is architecture-specific and
false for the decoders we now use. Measured success-vs-sigma on the blob
image at initialization:

    mlp      sigma 0.02: 38% win   0.1: 28%   0.5:  0%   2.0:  0%
    conv2d   sigma 0.02: 70% win   0.1: 64%   0.5: 52%   2.0:  4%
    anchor   sigma 0.02: 50% win   0.1: 38%   0.5: 50%   2.0: 40%

The viable ceiling differs by more than an order of magnitude between
architectures, and the ONE constant we ship is far below all of them. By
the classic 1/5th rule (steps are right-sized when ~20% of children beat
their parent), 40-70% success means the step is far too small: the anchor
decoder still wins 40% of the time at sigma 2.0, which moves the
phenotype 0.22 RMS versus 0.0004 at the shipped sigma.

Three arms, identical mutation directions, differing only in magnitude:

  * fixed        — the shipped explorer (baseline).
  * displacement — Daniel's rule: one gain multiplies both channels'
    sigmas, adapted each generation so that the mean phenotype RMS
    displacement between child and parent tracks `--target`. The
    measurement is FREE: parent and child phenotypes are already decoded
    for evaluation, so no fitness budget is spent on control.
  * success      — the classic Rechenberg alternative: adapt the same
    gain from the fraction of children that beat their parent, targeting
    1/5. Included to test whether targeting DISPLACEMENT buys anything
    over targeting outcomes, since displacement needs a target constant
    and this does not.
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
from benchmarks.round26_anchor_universal import build_anchor
from latentspace.universal.architectures import resolve
from latentspace.universal.explorer import ExplorerConfig, _Template

LATENT = 64
GAIN_LIMITS = (1e-2, 1e4)
DAMPING = 0.3
STEP_LIMITS = (0.7, 1.4)

SETUPS = {
    "blob2d_anchor": (BlobImage2D, (32, 32), build_anchor),
    "blob2d_conv": (BlobImage2D, (32, 32), "conv2d"),
    "smooth1d_anchor": (SmoothTarget, (256,), build_anchor),
}


def _mutate_genome(z, rng, c, gain):
    mask = rng.random(z.shape) < c.genome_mutation_rate
    if not mask.any():
        mask[rng.integers(0, len(z))] = True
    sigma = c.genome_mutation_sigma * gain
    return (z + mask * rng.normal(0, sigma, z.shape)).astype(np.float32)


def _mutate_weights(theta, rng, c, gain):
    sigma = float(np.exp(rng.uniform(np.log(c.weight_sigma_low),
                                     np.log(c.weight_sigma_high)))) * gain
    scale = max(float(theta.std()), 1e-3)
    return (theta + rng.normal(0, sigma * scale, theta.shape)).astype(np.float32)


def run(setup: str, arm: str, budget: int, seed: int, target: float) -> dict:
    factory, output_shape, architecture = SETUPS[setup]
    objective = factory()
    c = ExplorerConfig()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = _Template(resolve(architecture, LATENT, output_shape), "mps")

    def decode_all(zs, thetas) -> torch.Tensor:
        return torch.stack([template.decode(t, z)
                            for z, t in zip(zs, thetas)])

    zs = rng.standard_normal((c.population, LATENT)).astype(np.float32)
    thetas = np.stack([template.init_theta(int(rng.integers(0, 2**31)))
                       for _ in range(c.population)])
    phenos = decode_all(zs, thetas)
    loss = objective.loss_tensor(phenos.flatten(1)).cpu().numpy()
    spent = len(zs)
    gain = 1.0
    trace = []

    while spent < budget:
        order = np.argsort(loss)[:c.elite]
        zs, thetas, loss = zs[order], thetas[order], loss[order]
        phenos = phenos[order]
        n = min(c.population, budget - spent)
        parents = rng.integers(0, len(zs), n)
        child_z = np.stack([_mutate_genome(zs[p], rng, c, gain) for p in parents])
        child_theta = np.stack([_mutate_weights(thetas[p], rng, c, gain)
                                for p in parents])
        child_ph = decode_all(child_z, child_theta)
        child_loss = objective.loss_tensor(child_ph.flatten(1)).cpu().numpy()
        spent += n

        # Free control signals: both phenotypes were decoded for evaluation.
        moved = torch.sqrt(((child_ph - phenos[parents]) ** 2)
                           .flatten(1).mean(dim=1)).cpu().numpy()
        realized = float(moved.mean())
        wins = float((child_loss < loss[parents]).mean())

        if arm == "displacement" and realized > 0:
            gain *= float(np.clip((target / realized) ** DAMPING, *STEP_LIMITS))
        elif arm == "success":
            gain *= 1.15 if wins > 0.2 else 1 / 1.15
        gain = float(np.clip(gain, *GAIN_LIMITS))

        trace.append({"evaluations": spent, "best_loss": float(child_loss.min()),
                      "gain": gain, "realized": realized, "win_rate": wins})
        zs = np.concatenate([zs, child_z])
        thetas = np.concatenate([thetas, child_theta])
        phenos = torch.cat([phenos, child_ph])
        loss = np.concatenate([loss, child_loss])

    keep = trace[::10] + [trace[-1]]
    return {"mse": float(loss.min()), "final_gain": trace[-1]["gain"],
            "mean_win_rate": float(np.mean([t["win_rate"] for t in trace])),
            "trace": keep}


ARMS = ("fixed", "displacement", "success")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setups", nargs="+", choices=SETUPS,
                        default=list(SETUPS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--target", type=float, default=0.05,
                        help="target phenotype RMS displacement per mutation")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for setup in args.setups:
        print(f"\n=== {setup} (budget {args.budget}, target {args.target}) ===",
              flush=True)
        for arm in args.arms:
            for seed in args.seeds:
                out = run(setup, arm, args.budget, seed, args.target)
                trace = out.pop("trace")
                rows.append({"setup": setup, "arm": arm, "seed": seed,
                             **out, "trace": trace})
                print(f"  {arm:<13} seed {seed} mse {out['mse']:.6g} "
                      f"(final gain {out['final_gain']:.3g}, mean win rate "
                      f"{out['mean_win_rate']:.0%})", flush=True)
            vals = [r["mse"] for r in rows
                    if r["setup"] == setup and r["arm"] == arm]
            print(f"  {arm:<13} MEAN {np.mean(vals):.6g}", flush=True)

    print("\n--- summary ---")
    for setup in args.setups:
        base = np.mean([r["mse"] for r in rows
                        if r["setup"] == setup and r["arm"] == "fixed"])
        for arm in args.arms:
            vals = [r["mse"] for r in rows
                    if r["setup"] == setup and r["arm"] == arm]
            print(f"  {setup:<16} {arm:<13} {np.mean(vals):.6g} "
                  f"({base / np.mean(vals):.2f}x vs fixed)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "target": args.target, "latent": LATENT,
             "torch_version": torch.__version__, "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
