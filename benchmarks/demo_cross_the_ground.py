"""CROSS THE GROUND — you give it a landscape, it designs a creature.

The demo. You choose a terrain. The system evolves a body plan AND a gait
from nothing, live, and you watch flailing turn into locomotion.

It was DESIGNED to end on a transfer finale — a held-out terrain solved
faster by the decoder that learned the earlier landscapes. That finale is
not real and the demo does not claim it. Measured 2026-07-25, paired by
landscape and evolution seed: training on flat/hills/stairs/ramp and
holding out rubble, warm won 3/6 (mean +3.8%, a tie); training on rubble
and holding out DIFFERENT rubble — the related quadrant, where images and
the CA family both transferred — warm won 2/6 (mean -11.6%). The obvious
mechanism was ruled out too: warm founders are exactly as diverse as cold
ones (spread 0.269 vs 0.272) and score slightly worse. On locomotion the
shared decoder learns nothing a new terrain wants. `--transfer` is kept
because the negative is worth being able to re-run, not because it works.

Why this problem and not a picture. A demo where the user types the answer
cannot surprise anyone with the answer. Here you specify a GOAL — "get
across this" — and what comes back is an artifact nobody can predict, which
you can score by eye in one second.

It also sits on all three conditions the framing claims as home ground:

  BLACK BOX    ground contact is a hard test (is this node below the
               terrain?) and friction is a velocity clamp. There is no
               gradient path from distance-travelled back to the body, so
               this is not "gradients are beaten", it is "gradients are
               not applicable".
  CONTINUOUS   node coordinates, muscle amplitudes and phases are a smooth
               manifold, and a spring-gate threshold makes the body PLAN
               discrete on top of it.
  A FAMILY     every terrain is a fresh instance of one problem. This is
               the condition the problem turned out NOT to satisfy in the
               way that matters — see the transfer note above.

Each terrain is a species: `solve()` takes one fitness function per
landscape, they share a population and one decoder, and fitness shares stop
the easy terrain from swamping the hard one.

    python3 -m benchmarks.demo_cross_the_ground --live --terrain hills
    python3 -m benchmarks.demo_cross_the_ground --train
    python3 -m benchmarks.demo_cross_the_ground --transfer

Note the decoder is built through `architecture=`, which needs the output
scale widened: a decoder born emitting a near-constant phenotype is a body
whose nodes all sit in the same place, and a dot cannot walk (FINDINGS
round fourteen).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from latentspace.universal import solve, register_architecture
from latentspace.universal.architectures import build_mlp

# ----------------------------------------------------------------- body
N_NODES = 8
PAIRS = [(i, j) for i in range(N_NODES) for j in range(i + 1, N_NODES)]
N_SPRINGS = len(PAIRS)
KEEP = 12                      # springs the body actually gets to use
DIM = N_NODES * 2 + N_SPRINGS * 3       # positions | gates | amps | phases

# --------------------------------------------------------------- physics
DT = 0.008
STEPS = 600
GRAVITY = 4.0
K_SPRING = 30.0
SPRING_DAMP = 0.6
MASS = 0.2
FRICTION = 0.35                # tangential speed kept on contact
FREQ = 1.6
MAX_SPEED = 8.0
AMP_MAX = 0.35

STORE = Path(__file__).resolve().parent.parent / "demo" / "cross_the_ground.pt"

# --------------------------------------------------------------- terrain
X0, X1, DX = -3.0, 16.0, 0.04


def terrain(name, seed=0):
    """Ground height sampled on a fixed grid. Returns (xs, heights)."""
    xs = np.arange(X0, X1, DX)
    if name == "flat":
        h = np.zeros_like(xs)
    elif name == "hills":
        h = 0.16 * np.sin(1.5 * xs) + 0.09 * np.sin(3.3 * xs + 1.0)
    elif name == "stairs":
        h = 0.10 * np.floor(np.clip(xs, 0, None) / 1.2)
    elif name == "rubble":
        rng = np.random.default_rng(seed)
        bumps = rng.uniform(0.04, 0.16, 26)
        centres = rng.uniform(0, X1, 26)
        widths = rng.uniform(0.18, 0.5, 26)
        h = sum(b * np.exp(-((xs - c) ** 2) / (2 * w * w))
                for b, c, w in zip(bumps, centres, widths))
    elif name == "ramp":
        h = 0.055 * np.clip(xs, 0, None)
    else:
        raise ValueError(f"unknown terrain {name!r}")
    h = np.where(xs < 0, 0.0, h)                 # flat launch pad
    return xs.astype(np.float32), h.astype(np.float32)


TRAIN_TERRAINS = ("flat", "hills", "stairs", "ramp")
HELD_OUT = "rubble"


# ------------------------------------------------------------ simulation
def unpack(pheno):
    """(B, DIM) in [0,1] -> positions, active-spring mask, amplitudes, phases."""
    batch = pheno.shape[0]
    pos = pheno[:, :N_NODES * 2].reshape(batch, N_NODES, 2).clone()
    pos[..., 0] = (pos[..., 0] - 0.5) * 1.0
    pos[..., 1] = pos[..., 1] * 0.7
    rest = pheno[:, N_NODES * 2:]
    gates, amp, phase = rest[:, :N_SPRINGS], rest[:, N_SPRINGS:2 * N_SPRINGS], \
        rest[:, 2 * N_SPRINGS:]
    # the body PLAN: keep the K highest-gated springs, discard the rest.
    # A hard top-K — no gradient survives it, which is the point.
    keep = gates.topk(KEEP, dim=1).indices
    mask = torch.zeros_like(gates).scatter_(1, keep, 1.0)
    return pos, mask, amp * AMP_MAX, phase * (2 * math.pi)


def simulate(pheno, heights, record=False):
    """Drop bodies on the terrain and run the muscles. Returns start, end, frames."""
    device = pheno.device
    pos, mask, amp, phase = unpack(pheno)
    ground = torch.as_tensor(heights, device=device)
    n_grid = len(ground)

    def height_at(x):
        idx = ((x - X0) / DX).long().clamp(0, n_grid - 1)
        return ground[idx]

    # settle each body onto the surface
    pos[..., 1] = pos[..., 1] - (pos[..., 1] - height_at(pos[..., 0])).min(
        dim=1, keepdim=True).values
    start = pos.clone()

    ia = torch.tensor([p[0] for p in PAIRS], device=device)
    ib = torch.tensor([p[1] for p in PAIRS], device=device)
    rest_len = (pos[:, ib] - pos[:, ia]).norm(dim=-1).clamp(min=0.05)

    vel = torch.zeros_like(pos)
    frames = []
    for step in range(STEPS):
        t = step * DT
        target = rest_len * (1.0 + amp * torch.sin(2 * math.pi * FREQ * t + phase))

        delta = pos[:, ib] - pos[:, ia]
        dist = delta.norm(dim=-1).clamp(min=1e-4)
        unit = delta / dist.unsqueeze(-1)
        closing = ((vel[:, ib] - vel[:, ia]) * unit).sum(-1)
        force_mag = (K_SPRING * (dist - target) + SPRING_DAMP * closing) * mask
        spring = force_mag.unsqueeze(-1) * unit

        force = torch.zeros_like(pos)
        force.index_add_(1, ia, spring)
        force.index_add_(1, ib, -spring)
        force[..., 1] -= GRAVITY * MASS

        vel = (vel + force / MASS * DT).clamp(-MAX_SPEED, MAX_SPEED)
        pos = pos + vel * DT

        floor = height_at(pos[..., 0])
        grounded = pos[..., 1] < floor              # hard test: no gradient
        pos = torch.stack([pos[..., 0],
                           torch.where(grounded, floor, pos[..., 1])], dim=-1)
        vel = torch.stack([torch.where(grounded, vel[..., 0] * FRICTION,
                                       vel[..., 0]),
                           torch.where(grounded, torch.zeros_like(vel[..., 1]),
                                       vel[..., 1])], dim=-1)
        if record:
            frames.append(pos.clone())
    return start, pos, frames


def make_fitness(heights):
    """How far the centre of mass gets. That is the entire objective."""
    def fitness(phenotypes):
        flat = phenotypes.reshape(len(phenotypes), -1)
        start, end, _ = simulate(flat, heights)
        moved = end[..., 0].mean(dim=1) - start[..., 0].mean(dim=1)
        return torch.nan_to_num(moved, nan=-100.0, posinf=-100.0, neginf=-100.0)
    return fitness


# ------------------------------------------------------------- decoder
def register_walker_decoder(gain=10.0):
    """A wide-output MLP. See the module docstring: the stock near-constant
    output collapses every body to a point, and a point cannot walk."""
    def build(latent, output_shape):
        net = build_mlp(latent, output_shape)
        with torch.no_grad():
            net[-1].weight.mul_(gain)
            net[-1].bias.mul_(gain)
        return net
    register_architecture("walker", build)
    return "walker"


def run(terrains, epochs, seed=3, init_decoder=None, progress=None,
        progress_every=None):
    arch = register_walker_decoder()
    fits = [make_fitness(h) for _, h in terrains]
    # The stock population of 32 is too small a gene pool here: the search
    # froze within ~100 epochs and whatever it happened to find first was
    # the answer, so runs scattered (rubble: 4.35 / 4.32 / 5.67 / 2.77
    # across seeds). At 96 it neither freezes nor scatters — two seeds
    # landed on +6.108 and +6.114, about 40% further.
    return solve(fits, output_shape=(DIM,), epochs=epochs, architecture=arch,
                 seed=seed, device="cpu", init_decoder=init_decoder,
                 population_cap=96, children=48,
                 progress=progress, progress_every=progress_every)


# --------------------------------------------------------------- drawing
def draw_creature(ax, points, mask_pairs, alpha=1.0):
    artists = []
    for i, j in mask_pairs:
        artists += ax.plot([points[i, 0], points[j, 0]],
                           [points[i, 1], points[j, 1]],
                           color="#4d7fa3", lw=1.6, alpha=0.8 * alpha, zorder=3)
    artists.append(ax.scatter(points[:, 0], points[:, 1], s=42,
                              color="#12314a", alpha=alpha, zorder=4))
    return artists


def active_pairs(pheno):
    mask = unpack(pheno.reshape(1, -1))[1][0]
    return [PAIRS[k] for k in range(N_SPRINGS) if mask[k] > 0.5]


def live(terrain_name, epochs, seed):
    """Watch one landscape get solved."""
    import matplotlib.pyplot as plt

    xs, heights = terrain(terrain_name, seed)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                  gridspec_kw={"height_ratios": [3, 1]})
    fig.canvas.manager.set_window_title(f"cross the ground — {terrain_name}")
    curve = []

    def redraw(epoch, total, spent, phenos, scores):
        pheno = torch.as_tensor(np.asarray(phenos[0])).reshape(1, -1).float()
        frames = simulate(pheno, heights, record=True)[2]
        pairs = active_pairs(pheno[0])
        curve.append(float(scores[0]))
        for idx in range(0, len(frames), 12):
            ax.clear()
            ax.fill_between(xs, heights - 2.0, heights, color="#d8c4a8",
                            zorder=1)
            ax.plot(xs, heights, color="#8a6a45", lw=1.6, zorder=2)
            draw_creature(ax, frames[idx][0].numpy(), pairs)
            ax.set_xlim(-1.5, 12.0)
            ax.set_ylim(float(heights.min()) - 0.3, float(heights.max()) + 1.6)
            ax.set_aspect("equal")
            ax.set_yticks([])
            ax.set_title(f"{terrain_name} — epoch {epoch}/{total}, "
                         f"{spent} evaluations, best {scores[0]:+.2f} "
                         f"body-lengths", fontsize=11)
            ax2.clear()
            ax2.plot(curve, color="#c2703a", lw=1.4)
            ax2.set_ylabel("best distance")
            ax2.set_xlabel("progress reports")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            plt.pause(0.001)

    plt.ion()
    plt.show()
    result = run([(terrain_name, heights)], epochs, seed=seed,
                 progress=redraw, progress_every=max(1, epochs // 40))
    print(f"final: {result.best_fitness:+.3f} body-lengths in "
          f"{result.evaluations} evaluations")
    plt.ioff()
    plt.show(block=True)


def train(epochs, seed):
    """Learn the family: one decoder, one population, a species per terrain."""
    terrains = [(n, terrain(n, seed)[1]) for n in TRAIN_TERRAINS]
    began = time.time()

    def log(epoch, total, spent, phenos, scores):
        best = "  ".join(f"{n} {s:+.2f}" for (n, _), s in zip(terrains, scores))
        print(f"  epoch {epoch:>5}/{total}  {spent:>7} evals   {best}",
              flush=True)

    result = run(terrains, epochs, seed=seed, progress=log,
                 progress_every=max(1, epochs // 12))
    STORE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"decoder": result.decoder,
                "terrains": list(TRAIN_TERRAINS),
                "scores": [p.best_fitness for p in result.problems]}, STORE)
    print(f"\ntrained on {len(terrains)} landscapes in "
          f"{time.time() - began:.0f}s -> {STORE}")
    for (name, _), p in zip(terrains, result.problems):
        print(f"  {name:<8} {p.best_fitness:+.3f} body-lengths")


def transfer(epochs, seed):
    """The finale: a landscape neither arm has seen, cold vs warm."""
    if not STORE.exists():
        raise SystemExit(f"no trained decoder at {STORE}; run --train first")
    warm = torch.load(STORE, weights_only=False)["decoder"]
    xs, heights = terrain(HELD_OUT, seed + 17)
    print(f"held-out landscape: {HELD_OUT} (never trained on)\n")
    print(f"  {'epoch':>6} {'cold':>10} {'warm':>10}")
    traces = {}
    for arm, init in (("cold", None), ("warm", warm)):
        marks = {}

        def log(epoch, total, spent, phenos, scores, marks=marks):
            marks[epoch] = float(scores[0])

        run([(HELD_OUT, heights)], epochs, seed=seed + 1, init_decoder=init,
            progress=log, progress_every=max(1, epochs // 10))
        traces[arm] = marks
    for e in sorted(traces["cold"]):
        c, w = traces["cold"][e], traces["warm"][e]
        flag = "  <-- warm ahead" if w > c else ""
        print(f"  {e:>6} {c:>10.3f} {w:>10.3f}{flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="watch it evolve")
    parser.add_argument("--train", action="store_true",
                        help="learn the family of landscapes, save the decoder")
    parser.add_argument("--transfer", action="store_true",
                        help="held-out landscape, cold vs warm")
    parser.add_argument("--terrain", default="hills",
                        choices=("flat", "hills", "stairs", "rubble", "ramp"))
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    if args.train:
        train(args.epochs, args.seed)
    elif args.transfer:
        transfer(args.epochs, args.seed)
    elif args.live:
        live(args.terrain, args.epochs, args.seed)
    else:
        xs, heights = terrain(args.terrain, args.seed)
        result = run([(args.terrain, heights)], args.epochs, seed=args.seed)
        print(f"{args.terrain}: {result.best_fitness:+.3f} body-lengths "
              f"in {result.evaluations} evaluations")


if __name__ == "__main__":
    main()
