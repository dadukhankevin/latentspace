"""Rotating-prompt CLIP evolution with HIBERNATION — Daniel's curriculum v2.

Changes from demo_clip_niches (v1), all from Daniel's design:

  * SPECIALISTS HIBERNATE INSTEAD OF BEING REASSIGNED. In v1 a retiring
    prompt handed its individuals to the next prompt. Here each prompt OWNS
    its population: when the prompt rotates out, its individuals are stored;
    when it rotates back in, they wake up and continue where they left off.
    A prompt appearing for the first time seeds its population by breeding
    from the currently active seats (the newcomer-transfer channel). Records
    (hall of fame) also persist across hibernation.
  * TWO ROTATION TRIGGERS, both on STRICT improvement (v1's tie-reset bug
    let a stuck niche sit for 900+ generations):
      - stuck:    no new niche record for `retire_after` generations;
      - champion: the seat has been the TOP-SCORING seat for `retire_after`
        straight generations — the star pupil graduates and studies
        something new (this was Daniel's original spec; v1 only retired
        the stuck).
  * CLEARER LIVE VIEW: one line PER PROMPT with a stable color, drawn only
    while that prompt is active (hibernation shows as a gap); panel titles
    carry the prompt and its record; the legend lists prompts, correctly.

Everything else keeps v1's working recipe: comma selection per seat
(parents die each generation, so negative mutations drift the population
through score valleys), a hall of fame so nothing is lost, per-seat
truncation, cross-seat mating, gain floor 0.3 with a stagnation kick.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.demo_clip_evolve import load_clip, make_fitness
from benchmarks.round28_anchor_conv import ConvRGB
from benchmarks.round31_apple_animated import _png
from latentspace.universal.architectures import resolve
from benchmarks.legacy_engines.explorer import ExplorerConfig, _Template

LATENT = 64
SHAPE = (3, 96, 96)


class CurriculumView:
    """One panel per seat; one score line per PROMPT (stable color), drawn
    only while that prompt is active."""

    def __init__(self, pool, seats, budget):
        import matplotlib
        try:
            matplotlib.use("MacOSX")
        except Exception:
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        self.plt = plt
        plt.ion()
        self.fig = plt.figure(figsize=(3.0 * seats, 6.4))
        self.ims, self.axes_img = [], []
        for i in range(seats):
            ax = self.fig.add_subplot(2, seats, i + 1)
            ax.axis("off")
            self.axes_img.append(ax)
            self.ims.append(ax.imshow(np.zeros((96, 96, 3))))
        self.ax = self.fig.add_subplot(2, 1, 2)
        cmap = colormaps["tab20"]
        self.colors = {p: cmap(i % 20) for i, p in enumerate(pool)}
        self.series: dict[str, tuple[list, list]] = {}
        self.lines: dict[str, object] = {}
        self.ax.set_xlim(0, budget)
        self.ax.set_xlabel("evaluations", fontsize=9)
        self.ax.set_ylabel("prompt record (CLIP score)", fontsize=9)
        self.ax.grid(alpha=0.25)
        self.fig.tight_layout()
        plt.show(block=False)

    def mark_swap(self, prompt):
        """Insert a gap so hibernation shows as a break in the line."""
        if prompt in self.series:
            xs, ys = self.series[prompt]
            xs.append(np.nan)
            ys.append(np.nan)

    def update(self, spent, seat_prompts, best_imgs, best_scores):
        for i, (prompt, img, score) in enumerate(
                zip(seat_prompts, best_imgs, best_scores)):
            self.axes_img[i].set_title(
                f"{prompt}\n{score:.3f}" if np.isfinite(score) else prompt,
                fontsize=9)
            if img is not None:
                self.ims[i].set_data(np.clip(img.transpose(1, 2, 0), 0, 1))
            if prompt not in self.series:
                self.series[prompt] = ([], [])
                (line,) = self.ax.plot([], [], lw=1.6,
                                       color=self.colors[prompt],
                                       label=prompt)
                self.lines[prompt] = line
                self.ax.legend(fontsize=7, loc="lower right", ncols=2)
            xs, ys = self.series[prompt]
            xs.append(spent)
            ys.append(score)
            self.lines[prompt].set_data(xs, ys)
        flat = [v for xs, ys in self.series.values()
                for v in ys if np.isfinite(v)]
        if flat:
            lo, hi = min(flat), max(flat)
            pad = 0.05 * max(hi - lo, 1e-3)
            self.ax.set_ylim(lo - pad, hi + pad)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", nargs="*", default=[
        "a mango", "a sports car", "a tree", "a sunflower",
        "a house", "a cat", "a sailboat", "a coffee cup",
        "a mountain", "a butterfly", "a mushroom", "a lighthouse",
        "a snowman", "a rubber duck", "a planet", "a human face"])
    parser.add_argument("--neg", nargs="*",
                        default=["noise", "static", "an abstract pattern",
                                 "text"])
    parser.add_argument("--model",
                        default="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
    parser.add_argument("--clip-res", type=int, default=128)
    parser.add_argument("--seats", type=int, default=6)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--retire-after", type=int, default=200)
    parser.add_argument("--cross-seat", type=float, default=0.25)
    parser.add_argument("--budget", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    device = "mps"
    pool = list(args.pool)
    seats = min(args.seats, len(pool))
    quota = args.population // seats
    c = ExplorerConfig(population=args.population)

    print(f"loading CLIP: {args.model}", flush=True)
    model, processor = load_clip(args.model, device)

    def make_scorer(prompt):
        return make_fitness(model, processor, [prompt], args.neg, device,
                            res=args.clip_res)

    _seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    template = _Template(resolve(lambda l, s: ConvRGB(l, s), LATENT, SHAPE),
                         device)

    seat_prompt = pool[:seats]
    queue = pool[seats:]
    scorers = [make_scorer(p) for p in seat_prompt]
    # bank[prompt] = hibernated specialists + persistent record
    bank: dict[str, dict] = {}
    hall = [{"score": -np.inf, "img": None} for _ in range(seats)]
    stall = [0] * seats          # gens without a STRICT niche record
    reign = [0] * seats          # gens as the top-scoring seat
    episode_start = [0] * seats
    episodes: list[dict] = []

    zs = rng.standard_normal((args.population, LATENT)).astype(np.float32)
    thetas = np.stack([template.init_theta(int(rng.integers(0, 2**31)))
                       for _ in range(args.population)])
    seat = np.arange(args.population) % seats

    def score_all(z_batch, theta_batch, seat_batch):
        phenos = template.decode_batch(theta_batch, z_batch)
        out = np.empty(len(z_batch))
        for k in range(seats):
            idx = np.where(seat_batch == k)[0]
            if len(idx):
                out[idx] = scorers[k](phenos[idx]).detach().cpu().numpy()
        return out, phenos

    fitness, phenos = score_all(zs, thetas, seat)
    spent, gain, gen = len(zs), float(c.initial_gain), 0
    global_stall = 0
    frames: list[dict] = []
    every = max(1, args.budget // max(args.frames, 1))
    next_frame = 0
    view = (CurriculumView(pool, seats, args.budget) if args.live else None)

    def mutate_z(z):
        m = rng.random(z.shape) < c.genome_mutation_rate
        if not m.any():
            m[rng.integers(0, len(z))] = True
        return (z + m * rng.normal(0, c.genome_mutation_sigma * gain, z.shape)
                ).astype(np.float32)

    def cross_z(base, donor):
        cut = int(rng.integers(1, LATENT))
        child = base.copy()
        child[cut:] = donor[cut:]
        return child.astype(np.float32)

    def breed(base_z, base_theta, mate_z, mate_theta, n):
        child_z = np.stack([mutate_z(cross_z(bz, mz))
                            for bz, mz in zip(base_z, mate_z)])
        sigmas = np.exp(rng.uniform(np.log(c.weight_sigma_low),
                                    np.log(c.weight_sigma_high), n)) * gain
        base = (base_theta + mate_theta) / 2.0
        scales = np.maximum(base.std(axis=1), 1e-3)
        child_theta = (base + (sigmas * scales)[:, None]
                       * rng.standard_normal((n, template.n_params))
                       ).astype(np.float32)
        return child_z, child_theta

    def rotate(k, reason):
        """Hibernate seat k's specialists with their prompt; wake the next."""
        nonlocal zs, thetas, fitness, seat
        old = seat_prompt[k]
        members = np.where(seat == k)[0]
        episodes.append({"prompt": old, "seat": k, "reason": reason,
                         "start_gen": episode_start[k], "end_gen": gen,
                         "best": hall[k]["score"],
                         "resumed": old in bank})
        bank[old] = {"zs": zs[members].copy(),
                     "thetas": thetas[members].copy(),
                     "score": hall[k]["score"], "img": hall[k]["img"]}
        queue.append(old)
        new = queue.pop(0)
        print(f"  == seat {k}: '{old}' retires ({reason}, "
              f"{gen - episode_start[k]} gens, best {hall[k]['score']:.3f})"
              f" -> '{new}'" + (" [waking]" if new in bank else " [fresh]"),
              flush=True)
        if view is not None:
            view.mark_swap(old)

        keepers = np.setdiff1d(np.arange(len(seat)), members)
        if new in bank:
            stored = bank.pop(new)
            nz, nth = stored["zs"], stored["thetas"]
            if len(nz) > quota:
                nz, nth = nz[:quota], nth[:quota]
            while len(nz) < quota:      # top up with mutated clones
                i = rng.integers(0, len(nz))
                nz = np.concatenate([nz, [mutate_z(nz[i])]])
                nth = np.concatenate([nth, [nth[i]]])
            hall[k] = {"score": stored["score"], "img": stored["img"]}
        else:
            # first appearance: breed the seed from the active population
            donors = rng.integers(0, len(keepers), 2 * quota)
            d = keepers[donors]
            nz, nth = breed(zs[d[:quota]], thetas[d[:quota]],
                            zs[d[quota:]], thetas[d[quota:]], quota)
            hall[k] = {"score": -np.inf, "img": None}
        seat_prompt[k] = new
        scorers[k] = make_scorer(new)
        stall[k] = reign[k] = 0
        episode_start[k] = gen
        new_fit, _ = score_all(nz, nth, np.full(len(nz), k))
        zs = np.concatenate([zs[keepers], nz])
        thetas = np.concatenate([thetas[keepers], nth])
        fitness = np.concatenate([fitness[keepers], new_fit])
        seat = np.concatenate([seat[keepers], np.full(len(nz), k)])

    while spent < args.budget:
        # comma selection per seat: survivors are the best CHILDREN only
        keep: list[int] = []
        for k in range(seats):
            members = np.where(seat == k)[0]
            order = members[np.argsort(-fitness[members])][:max(1, quota // 4)]
            keep.extend(int(i) for i in order)
        keep = np.asarray(keep)
        zs, thetas, fitness, seat = (zs[keep], thetas[keep], fitness[keep],
                                     seat[keep])

        n = min(args.population, args.budget - spent)
        base_idx = np.empty(n, dtype=int)
        per = n // seats
        pos = 0
        for k in range(seats):     # children allocated evenly across seats
            members = np.where(seat == k)[0]
            count = per if k < seats - 1 else n - pos
            base_idx[pos:pos + count] = members[
                rng.integers(0, len(members), count)]
            pos += count
        cross = rng.random(n) < args.cross_seat
        mate_idx = np.empty(n, dtype=int)
        for i, b in enumerate(base_idx):
            if cross[i]:
                mate_idx[i] = rng.integers(0, len(zs))
            else:
                same = np.where(seat == seat[b])[0]
                mate_idx[i] = same[rng.integers(0, len(same))]

        child_z, child_theta = breed(zs[base_idx], thetas[base_idx],
                                     zs[mate_idx], thetas[mate_idx], n)
        child_seat = seat[base_idx]
        child_fit, child_phenos = score_all(child_z, child_theta, child_seat)
        spent += n
        gen += 1
        wins = float((child_fit >= fitness[base_idx] - 1e-12).mean())
        gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, 0.3, c.gain_limits[1]))

        zs, thetas, fitness, seat = child_z, child_theta, child_fit, child_seat

        any_improved = False
        for k in range(seats):
            members = np.where(seat == k)[0]
            if not len(members):
                stall[k] += 1
                continue
            b = members[int(np.argmax(fitness[members]))]
            if fitness[b] > hall[k]["score"] + 1e-9:      # STRICT
                hall[k] = {"score": float(fitness[b]),
                           "img": (child_phenos[b].detach().cpu().numpy()
                                   .reshape(*SHAPE))}
                stall[k] = 0
                any_improved = True
            else:
                stall[k] += 1
        global_stall = 0 if any_improved else global_stall + 1
        if global_stall >= 25:
            gain = min(gain * 3.0, c.gain_limits[1])
            global_stall = 0

        top = int(np.argmax([hall[k]["score"] for k in range(seats)]))
        for k in range(seats):
            reign[k] = reign[k] + 1 if k == top else 0

        rotated = False
        for k in range(seats):
            if rotated:
                break               # at most one rotation per generation
            if stall[k] >= args.retire_after:
                rotate(k, "stuck")
                rotated = True
            elif reign[k] >= args.retire_after:
                rotate(k, "champion")
                rotated = True

        best_imgs = [hall[k]["img"] for k in range(seats)]
        best_scores = [hall[k]["score"] for k in range(seats)]
        if spent >= next_frame:
            row = {"e": spent, "prompts": list(seat_prompt),
                   "scores": best_scores}
            for k, img in enumerate(best_imgs):
                if img is not None:
                    row[f"p{k}"] = _png(img.reshape(-1))
            frames.append(row)
            next_frame += every
            print("  " + f"{spent:>7} evals  " + "  ".join(
                f"{seat_prompt[k].split()[-1][:6]} {best_scores[k]:.3f}"
                for k in range(seats)) + f"  (gain {gain:.2f})", flush=True)
        if view is not None and gen % 5 == 0:
            view.update(spent, seat_prompt, best_imgs, best_scores)

    print("\nFINAL seat records:")
    for k in range(seats):
        print(f"  {seat_prompt[k]:<22} {hall[k]['score']:.4f}")
    print("\nhibernating records:")
    for prompt, stored in bank.items():
        print(f"  {prompt:<22} {stored['score']:.4f}")
    if episodes:
        print("\nepisodes (later fresh episodes climbing faster = the "
              "population is learning to learn):")
        for e in episodes:
            kind = "resumed" if e["resumed"] else "fresh"
            print(f"  {e['prompt']:<22} {kind:<8} {e['reason']:<9} gens "
                  f"{e['end_gen'] - e['start_gen']:>5}  best {e['best']:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"pool": pool, "neg": args.neg, "budget": args.budget,
             "seed": args.seed, "seats": seats,
             "population": args.population,
             "retire_after": args.retire_after,
             "cross_seat": args.cross_seat, "episodes": episodes,
             "bank_records": {p: b["score"] for p, b in bank.items()},
             "frames": frames, "torch_version": torch.__version__}) + "\n")
        print(f"wrote {args.output}")

    if args.gif:
        from PIL import ImageDraw
        imgs = []
        for f in frames:
            tiles = []
            for k in range(seats):
                if f"p{k}" in f:
                    tile = (Image.open(io.BytesIO(base64.b64decode(
                        f[f"p{k}"].split(",", 1)[1]))).convert("RGB")
                        .resize((192, 192), Image.NEAREST))
                    d = ImageDraw.Draw(tile)
                    d.rectangle([0, 176, 192, 192], fill=(0, 0, 0))
                    d.text((4, 178), f["prompts"][k], fill=(255, 255, 255))
                    tiles.append(tile)
            if tiles:
                strip = Image.new("RGB", (192 * len(tiles), 192))
                for i, tile in enumerate(tiles):
                    strip.paste(tile, (192 * i, 0))
                imgs.append(strip)
        if imgs:
            imgs.append(imgs[-1])
            args.gif.parent.mkdir(parents=True, exist_ok=True)
            imgs[0].save(args.gif, save_all=True, append_images=imgs[1:],
                         duration=[80] * (len(imgs) - 1) + [2500], loop=0,
                         optimize=True)
            print(f"wrote {args.gif}")

    if view is not None:
        view.plt.ioff()
        view.plt.show()


if __name__ == "__main__":
    main()
