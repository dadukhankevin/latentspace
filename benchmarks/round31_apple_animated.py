"""Round 31: the apple, rerun with adaptive step size, with animation frames.

The published note (loseylabs.ai/notes/watching-evolution-find-an-apple)
recorded, at 150k evaluations on a 96x96 RGB apple: pure conv decoder
evolution 0.004929, the distill->CMA hybrid 0.011163, a pixel GA 0.120026.

Every one of those decoder runs used the shipped mutation constant, which
rounds 29-30 showed is roughly 100x too small (measured success rates of
40-70% against the 1/5th rule's 20%, and a justifying comment that was
measured on an MLP and is false for conv and anchor decoders). This round
reruns the photo with the round-30 controller and captures frames so the
note can show the animation.

Arms, all PURE evolution — no distill, no CMA anywhere:

  * pixel_ga         — traditional GA mutating the 27,648 pixels directly.
  * conv_fixed       — the published champion's decoder and shipped sigma:
                       reproduces 0.0049 as the number to beat.
  * conv_targeted    — same decoder, adaptive step size.
  * anchor_targeted  — the anchor genome grammar, adaptive step size.

The controller (round 30, Daniel's rule): one gain multiplies both
channels' sigmas, adapted each generation so mean child-parent phenotype
RMS displacement tracks --target. The measurement is free — both
phenotypes are already decoded for scoring — so it costs zero fitness
evaluations.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round27_apple_no_cma import DEMO, AnchorRGB, load_apple
from benchmarks.round28_anchor_conv import ConvRGB
from latentspace.universal.architectures import resolve
from benchmarks.legacy_engines.explorer import ExplorerConfig, _Template

LATENT = 64
SHAPE = (3, 96, 96)
GAIN_LIMITS = (1e-2, 1e4)
DAMPING = 0.3
STEP_LIMITS = (0.7, 1.4)


def _png(flat: np.ndarray) -> str:
    """A (3,96,96) phenotype in [0,1] as a base64 data URI, matching the
    format the published demo already stores."""
    from PIL import Image

    array = (np.clip(flat.reshape(SHAPE).transpose(1, 2, 0), 0, 1) * 255)
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), "RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _mutate_genome(z, rng, c, gain):
    mask = rng.random(z.shape) < c.genome_mutation_rate
    if not mask.any():
        mask[rng.integers(0, len(z))] = True
    return (z + mask * rng.normal(0, c.genome_mutation_sigma * gain, z.shape)
            ).astype(np.float32)


def _mutate_weights(theta, rng, c, gain):
    sigma = float(np.exp(rng.uniform(np.log(c.weight_sigma_low),
                                     np.log(c.weight_sigma_high)))) * gain
    scale = max(float(theta.std()), 1e-3)
    return (theta + rng.normal(0, sigma * scale, theta.shape)).astype(np.float32)


ANNEAL_FRACTION = 0.3


def run_decoder(builder, target, budget, seed, adaptive, target_move,
                frames_wanted, stop_below: float | None = None,
                anneal: bool = False) -> dict:
    c = ExplorerConfig()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = _Template(resolve(builder, LATENT, SHAPE), "mps")
    target_t = torch.as_tensor(target, device="mps")

    def decode_all(zs, thetas):
        return torch.stack([template.decode(t, z)
                            for z, t in zip(zs, thetas)])

    def losses_of(ph):
        return ((ph.flatten(1) - target_t) ** 2).mean(dim=1).cpu().numpy()

    zs = rng.standard_normal((c.population, LATENT)).astype(np.float32)
    thetas = np.stack([template.init_theta(int(rng.integers(0, 2**31)))
                       for _ in range(c.population)])
    phenos = decode_all(zs, thetas)
    loss = losses_of(phenos)
    spent = len(zs)
    gain, frames, trace = 1.0, [], []
    best_flat = phenos[int(np.argmin(loss))].cpu().numpy()
    frames.append({"e": spent, "m": float(loss.min()), "p": _png(best_flat)})
    every = max(1, budget // max(frames_wanted, 1))
    next_frame = every

    while spent < budget:
        order = np.argsort(loss)[:c.elite]
        zs, thetas, loss, phenos = zs[order], thetas[order], loss[order], phenos[order]
        n = min(c.population, budget - spent)
        parents = rng.integers(0, len(zs), n)
        child_z = np.stack([_mutate_genome(zs[p], rng, c, gain) for p in parents])
        child_theta = np.stack([_mutate_weights(thetas[p], rng, c, gain)
                                for p in parents])
        child_ph = decode_all(child_z, child_theta)
        child_loss = losses_of(child_ph)
        spent += n

        wins = float((child_loss < loss[parents]).mean())
        if adaptive == "success":
            # Rechenberg's rule: right-sized steps win ~1/5 of the time.
            # Self-tuning at every error scale — no target constant to
            # deadlock on.
            gain *= 1.15 if wins > 0.2 else 1 / 1.15
            gain = float(np.clip(gain, *GAIN_LIMITS))
        elif adaptive:
            moved = torch.sqrt(((child_ph - phenos[parents]) ** 2)
                               .flatten(1).mean(dim=1)).cpu().numpy()
            realized = float(moved.mean())
            # The remaining error sets the useful step size: while error is
            # large, coarse steps; as the answer sharpens, so must the steps.
            effective = (min(target_move,
                             ANNEAL_FRACTION * float(np.sqrt(loss.min())))
                         if anneal else target_move)
            if realized > 0:
                gain *= float(np.clip((effective / realized) ** DAMPING,
                                      *STEP_LIMITS))
                gain = float(np.clip(gain, *GAIN_LIMITS))

        zs = np.concatenate([zs, child_z])
        thetas = np.concatenate([thetas, child_theta])
        phenos = torch.cat([phenos, child_ph])
        loss = np.concatenate([loss, child_loss])
        trace.append({"e": spent, "m": float(loss.min()), "gain": gain,
                      "win_rate": wins})

        if spent >= next_frame:
            best_flat = phenos[int(np.argmin(loss))].cpu().numpy()
            frames.append({"e": spent, "m": float(loss.min()),
                           "p": _png(best_flat)})
            next_frame += every

        if stop_below is not None and float(loss.min()) < stop_below:
            print(f"    early stop: {loss.min():.6f} < {stop_below} "
                  f"at {spent} evaluations", flush=True)
            break

    best_flat = phenos[int(np.argmin(loss))].cpu().numpy()
    frames.append({"e": spent, "m": float(loss.min()), "p": _png(best_flat)})
    return {"mse": float(loss.min()), "final_gain": gain, "frames": frames,
            "trace": trace[::20], "final_png": _png(best_flat)}


def run_pixel_ga(target, budget, seed, frames_wanted) -> dict:
    """Traditional GA: rank selection, uniform crossover, per-pixel
    Gaussian mutation — the operator engineering a universal GA avoids."""
    rng = np.random.default_rng(seed)
    dim = target.size
    population, offspring = 32, 32
    pop = rng.random((population, dim)).astype(np.float32)
    loss = ((pop - target) ** 2).mean(axis=1)
    spent = population
    frames, trace = [], []
    frames.append({"e": spent, "m": float(loss.min()),
                   "p": _png(pop[int(np.argmin(loss))])})
    every = max(1, budget // max(frames_wanted, 1))
    next_frame = every

    while spent < budget:
        n = min(offspring, budget - spent)
        ranked = pop[np.argsort(loss)]
        weights = np.arange(len(ranked), 0, -1, dtype=np.float64)
        weights /= weights.sum()
        pick = rng.choice(len(ranked), size=(n, 2), p=weights)
        first, second = ranked[pick[:, 0]], ranked[pick[:, 1]]
        children = np.where(rng.random((n, dim)) < 0.5, first, second)
        mask = rng.random((n, dim)) < 0.01
        children = np.clip(children + rng.normal(0, 0.1, (n, dim)) * mask,
                           0, 1).astype(np.float32)
        child_loss = ((children - target) ** 2).mean(axis=1)
        spent += n
        pop = np.concatenate([pop, children])
        loss = np.concatenate([loss, child_loss])
        keep = np.argsort(loss)[:population]
        pop, loss = pop[keep], loss[keep]
        trace.append({"e": spent, "m": float(loss.min())})
        if spent >= next_frame:
            frames.append({"e": spent, "m": float(loss.min()),
                           "p": _png(pop[int(np.argmin(loss))])})
            next_frame += every

    best = pop[int(np.argmin(loss))]
    frames.append({"e": spent, "m": float(loss.min()), "p": _png(best)})
    return {"mse": float(loss.min()), "final_gain": None, "frames": frames,
            "trace": trace[::20], "final_png": _png(best)}


ARMS = ("pixel_ga", "conv_fixed", "conv_targeted", "anchor_targeted")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--budget", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=float, default=0.05)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--stop-below", type=float, default=None,
                        help="stop a decoder arm once best MSE drops below "
                             "this (e.g. the best previously recorded run)")
    parser.add_argument("--anneal", action="store_true",
                        help="shrink the displacement target with the "
                             "remaining error (0.3 x RMS error, capped at "
                             "--target)")
    parser.add_argument("--controller", choices=("displacement", "success"),
                        default="displacement",
                        help="step-size feedback signal for the adaptive "
                             "arms")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    target = load_apple()
    recorded = json.loads(DEMO.read_text())["D"]["finalMse"]
    print(f"apple {target.size} values; published refs: conv evolution "
          f"{recorded['cf']}, hybrid {recorded['stack']}, GA {recorded['ga']}",
          flush=True)

    rows = {}
    for arm in args.arms:
        _seed_everything(args.seed)
        if arm == "pixel_ga":
            out = run_pixel_ga(target, args.budget, args.seed, args.frames)
        elif arm == "conv_fixed":
            out = run_decoder(ConvRGB, target, args.budget, args.seed,
                              False, args.target, args.frames)
        elif arm == "conv_targeted":
            mode = "success" if args.controller == "success" else True
            out = run_decoder(ConvRGB, target, args.budget, args.seed,
                              mode, args.target, args.frames,
                              args.stop_below, args.anneal)
        else:
            mode = "success" if args.controller == "success" else True
            out = run_decoder(lambda l, s: AnchorRGB(l, s), target,
                              args.budget, args.seed, mode, args.target,
                              args.frames, args.stop_below, args.anneal)
        rows[arm] = out
        gain = "n/a" if out["final_gain"] is None else f"{out['final_gain']:.2f}"
        print(f"  {arm:<16} FINAL {out['mse']:.6f}  (final gain {gain}, "
              f"{len(out['frames'])} frames)", flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "seed": args.seed,
             "target_displacement": args.target, "latent": LATENT,
             "published_references": recorded,
             "torch_version": torch.__version__, "arms": rows}) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
