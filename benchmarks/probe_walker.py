"""Probe: can the universal GA design a creature that walks? (2026-07-25)

Why this exists. The Design The Past demo has a structural flaw as a demo:
the user types the answer. You ask for HELLO and you get HELLO, so the
output carries zero surprise, and the only way an audience learns that the
problem was hard is if someone tells them.

The fix is a demo where the user specifies a GOAL and the machine returns
an ARTIFACT nobody could have predicted. Locomotion is the classic: you
give it a body budget and a ground to cross, and it hands back a creature
with a gait. Everyone can score the result by eye in one second, and
nobody can guess it in advance.

It also happens to sit exactly on the framing's three conditions:
  BLACK BOX      the simulator has hard ground contact and Coulomb-ish
                 friction — velocity clamps and positional corrections,
                 no gradient path from distance travelled back to the body
  CONTINUOUS     node coordinates and per-spring gait amplitude/phase
  A FAMILY       every terrain / gravity / body budget is a fresh instance
                 of the same problem, which is where the warm decoder's
                 measured transfer advantage should show up live

This file answers only the feasibility question: is the simulator stable,
do random bodies produce a spread of outcomes (so there is something to
search), and does solve() actually climb it?

Phenotype (a flat vector the decoder emits, all values in [0, 1]):
    8 node positions            -> the body
    28 spring amplitudes        -> how hard each muscle pulses
    28 spring phases            -> the gait's coordination
"""
from __future__ import annotations

import argparse
import math
import time

import torch

from latentspace.universal import solve

N_NODES = 8
PAIRS = [(i, j) for i in range(N_NODES) for j in range(i + 1, N_NODES)]
N_SPRINGS = len(PAIRS)
DIM = N_NODES * 2 + N_SPRINGS * 2

DT = 0.008
STEPS = 400
GRAVITY = 4.0
K_SPRING = 30.0
SPRING_DAMP = 0.6
MASS = 0.2
FRICTION = 0.35          # fraction of tangential speed kept on ground contact
FREQ = 1.6
MAX_SPEED = 8.0
AMP_MAX = 0.35


def unpack(pheno):
    """(B, DIM) in [0, 1] -> node positions, muscle amplitudes, phases."""
    batch = pheno.shape[0]
    pos = pheno[:, :N_NODES * 2].reshape(batch, N_NODES, 2).clone()
    pos[..., 0] = (pos[..., 0] - 0.5) * 1.0          # body ~1 unit wide
    pos[..., 1] = pos[..., 1] * 0.7                  # ~0.7 units tall
    rest = pheno[:, N_NODES * 2:]
    amp = rest[:, :N_SPRINGS] * AMP_MAX
    phase = rest[:, N_SPRINGS:] * (2 * math.pi)
    return pos, amp, phase


def simulate(pheno, steps=STEPS, record=False):
    """Mass-spring bodies dropped on flat ground. Returns (start, end, frames)."""
    device = pheno.device
    pos, amp, phase = unpack(pheno)
    pos[..., 1] = pos[..., 1] - pos[..., 1].min(dim=1, keepdim=True).values
    start = pos.clone()

    ia = torch.tensor([p[0] for p in PAIRS], device=device)
    ib = torch.tensor([p[1] for p in PAIRS], device=device)
    rest_len = (pos[:, ib] - pos[:, ia]).norm(dim=-1).clamp(min=0.05)

    vel = torch.zeros_like(pos)
    frames = []
    for step in range(steps):
        t = step * DT
        target = rest_len * (1.0 + amp * torch.sin(2 * math.pi * FREQ * t + phase))

        delta = pos[:, ib] - pos[:, ia]
        dist = delta.norm(dim=-1).clamp(min=1e-4)
        unit = delta / dist.unsqueeze(-1)
        closing = ((vel[:, ib] - vel[:, ia]) * unit).sum(-1)
        magnitude = K_SPRING * (dist - target) + SPRING_DAMP * closing
        spring_force = magnitude.unsqueeze(-1) * unit

        force = torch.zeros_like(pos)
        force.index_add_(1, ia, spring_force)
        force.index_add_(1, ib, -spring_force)
        force[..., 1] -= GRAVITY * MASS

        vel = (vel + force / MASS * DT).clamp(-MAX_SPEED, MAX_SPEED)
        pos = pos + vel * DT

        grounded = pos[..., 1] < 0.0                 # hard threshold: no gradient
        pos = torch.stack([pos[..., 0],
                           torch.where(grounded, torch.zeros_like(pos[..., 1]),
                                       pos[..., 1])], dim=-1)
        vel = torch.stack([torch.where(grounded, vel[..., 0] * FRICTION,
                                       vel[..., 0]),
                           torch.where(grounded, torch.zeros_like(vel[..., 1]),
                                       vel[..., 1])], dim=-1)
        if record:
            frames.append(pos.clone())

    return start, pos, frames


def distance(pheno):
    """Forward travel of the centre of mass — the whole objective."""
    start, end = simulate(pheno)[:2]
    moved = end[..., 0].mean(dim=1) - start[..., 0].mean(dim=1)
    return torch.nan_to_num(moved, nan=-100.0, posinf=-100.0, neginf=-100.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--random-only", action="store_true")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="scale on the decoder's final layer. The stock "
                             "decoder emits a near-constant phenotype, which "
                             "for a geometric output means every body is a "
                             "degenerate dot.")
    args = parser.parse_args()

    device = "cpu"        # tiny tensors, 400 sequential steps: launch-bound on MPS
    torch.manual_seed(args.seed)

    random_bodies = torch.rand(512, DIM, device=device)
    began = time.time()
    scores = distance(random_bodies)
    elapsed = time.time() - began
    print(f"random bodies (512): mean {scores.mean():+.3f}  "
          f"best {scores.max():+.3f}  worst {scores.min():+.3f}  "
          f"sd {scores.std():.3f}")
    print(f"  simulator: {elapsed:.2f}s for 512 bodies x {STEPS} steps "
          f"({elapsed / 512 * 1000:.2f} ms/body)")
    if not torch.isfinite(scores).all():
        print("  UNSTABLE: non-finite scores present")
    if args.random_only:
        return

    architecture = "auto"
    if args.gain != 1.0:
        from latentspace.universal.architectures import build_mlp, register_architecture

        def build_wide(latent, output_shape, gain=args.gain):
            net = build_mlp(latent, output_shape)
            with torch.no_grad():
                net[-1].weight.mul_(gain)
                net[-1].bias.mul_(gain)
            return net

        register_architecture("wide-mlp", build_wide)
        architecture = "wide-mlp"

    def fitness(phenotypes):
        return distance(phenotypes.reshape(len(phenotypes), -1).to(device))

    def progress(epoch, total, spent, phenos, values):
        print(f"  epoch {epoch:>5}/{total}  {spent:>7} evals  "
              f"best travel {values[0]:+.3f}", flush=True)

    began = time.time()
    result = solve(fitness, output_shape=(DIM,), epochs=args.epochs,
                   architecture=architecture,
                   seed=args.seed, device=device, progress=progress,
                   progress_every=max(1, args.epochs // 10))
    print(f"\nevolved: {result.best_fitness:+.3f} body-lengths in "
          f"{result.evaluations} evaluations ({time.time() - began:.0f}s)")
    print(f"random best of 512: {scores.max():+.3f}")


if __name__ == "__main__":
    main()
