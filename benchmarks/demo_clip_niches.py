"""CLIP evolution with NICHES: sub-populations chasing different prompts.

Daniel's diagnosis of the wallpaper failures: the population collapses into
the prompt's nearest color-field basin within a few generations, and from
then on crossover has nothing diverse to combine — lineage collapse on a
smooth semantic landscape, the same disease as the campaign's rounds 37-41.
The fix is the second law applied to prompts: give sub-populations DIFFERENT
targets. Car lineages cannot converge onto mango-yellow, so the gene pool
stays structurally diverse, and cross-niche mating can transport what the
mango niche cannot find alone (e.g. "an object with a boundary").

Mechanics: each individual carries a niche (a prompt index) inherited from
its within-niche parent. Selection is per-niche truncation — every niche
keeps its own best, so no prompt's score scale can starve another. Mating:
the base parent is within-niche; the mate is drawn from ALL survivors with
probability `cross_niche` (the transfer channel), else within-niche.

Per Daniel's read that the FIRST CLIP run produced the best images, the
generation settings revert to it: gray canvas (no logit bias), initial gain
1.0, NO cutouts. Only the speed fixes stay (fp16, 128px scoring).

The live matplotlib view shows every niche's best image side by side, with
per-niche score curves.
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


class NicheLiveView:
    def __init__(self, prompts, budget):
        import matplotlib
        try:
            matplotlib.use("MacOSX")
        except Exception:
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        n = len(prompts)
        self.fig = plt.figure(figsize=(3.1 * n, 6.2))
        self.axes_img, self.ims = [], []
        for i, prompt in enumerate(prompts):
            ax = self.fig.add_subplot(2, n, i + 1)
            ax.set_title(prompt, fontsize=9)
            ax.axis("off")
            self.ims.append(ax.imshow(np.zeros((96, 96, 3))))
        self.ax_fit = self.fig.add_subplot(2, 1, 2)
        self.lines = []
        for i, prompt in enumerate(prompts):
            (line,) = self.ax_fit.plot([], [], lw=1.6, label=prompt)
            self.lines.append(line)
        self.ax_fit.set_xlim(0, budget)
        self.ax_fit.set_xlabel("evaluations", fontsize=9)
        self.ax_fit.set_ylabel("niche best score", fontsize=9)
        self.ax_fit.legend(fontsize=8, loc="lower right")
        self.ax_fit.grid(alpha=0.25)
        self.es = []
        self.hist = [[] for _ in prompts]
        self.fig.tight_layout()
        plt.show(block=False)

    def update(self, spent, best_imgs, best_scores):
        self.es.append(spent)
        for i, (img, score) in enumerate(zip(best_imgs, best_scores)):
            if img is not None:
                self.ims[i].set_data(np.clip(img.transpose(1, 2, 0), 0, 1))
            self.hist[i].append(score)
            self.lines[i].set_data(self.es, self.hist[i])
        flat = [v for h in self.hist for v in h if np.isfinite(v)]
        if flat:
            lo, hi = min(flat), max(flat)
            pad = 0.05 * max(hi - lo, 1e-3)
            self.ax_fit.set_ylim(lo - pad, hi + pad)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", nargs="+",
                        default=["a mango fruit", "a red sports car",
                                 "a green tree", "a sunflower"])
    parser.add_argument("--pool", nargs="*", default=[
        "a mango", "a sports car", "a tree", "a sunflower",
        "a house", "a cat", "a sailboat", "a coffee cup",
        "a mountain", "a butterfly", "a mushroom", "a lighthouse",
        "a snowman", "a rubber duck", "a planet", "a human face"],
        help="rotating prompt pool; overrides --prompts when set")
    parser.add_argument("--niches", type=int, default=6,
                        help="concurrent prompts when using --pool")
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--retire-after", type=int, default=200,
                        help="generations without a new niche best before "
                             "the prompt retires and a fresh one arrives")
    parser.add_argument("--neg", nargs="*",
                        default=["noise", "static", "an abstract pattern",
                                 "text"])
    parser.add_argument("--model",
                        default="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
    parser.add_argument("--clip-res", type=int, default=128)
    parser.add_argument("--budget", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cross-niche", type=float, default=0.25,
                        help="probability the mate comes from another niche")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    device = "mps"
    pool = list(args.pool) if args.pool else list(args.prompts)
    n_niches = min(args.niches, len(pool))
    prompts = pool[:n_niches]         # active; mutated by retirement
    queue = pool[n_niches:]           # waiting; retired prompts requeue
    c = ExplorerConfig(population=args.population)
    per_niche = max(1, args.population // (4 * n_niches))

    print(f"loading CLIP: {args.model}", flush=True)
    model, processor = load_clip(args.model, device)

    def make_scorer(prompt):
        return make_fitness(model, processor, [prompt], args.neg, device,
                            res=args.clip_res)

    scorers = [make_scorer(p) for p in prompts]

    _seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    template = _Template(resolve(lambda l, s: ConvRGB(l, s), LATENT, SHAPE),
                         device)

    zs = rng.standard_normal((c.population, LATENT)).astype(np.float32)
    thetas = np.stack([template.init_theta(int(rng.integers(0, 2**31)))
                       for _ in range(c.population)])
    niche = np.arange(c.population) % n_niches

    def score_all(z_batch, theta_batch, niche_batch):
        phenos = template.decode_batch(theta_batch, z_batch)
        out = np.empty(len(z_batch))
        for k in range(n_niches):
            idx = np.where(niche_batch == k)[0]
            if len(idx):
                s = scorers[k](phenos[idx])
                out[idx] = s.detach().cpu().numpy()
        return out, phenos

    fitness, phenos = score_all(zs, thetas, niche)
    spent, gain = len(zs), float(c.initial_gain)
    frames: list[dict] = []
    every = max(1, args.budget // max(args.frames, 1))
    next_frame = 0
    view = NicheLiveView(prompts, args.budget) if args.live else None
    gen = 0

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

    hall = {k: {"score": -np.inf, "img": None} for k in range(n_niches)}
    stall = 0
    niche_stall = [0] * n_niches      # generations since that niche improved
    episode_start = [0] * n_niches
    episodes: list[dict] = []
    while spent < args.budget:
        # COMMA selection, per niche: survivors are the best CHILDREN of the
        # last generation — parents die, so a child that scored WORSE than
        # its parent can still inherit the line. Negative mutations are the
        # drift that escapes deceptive basins; the hall of fame below keeps
        # the best-ever so nothing is lost.
        keep: list[int] = []
        for k in range(n_niches):
            members = np.where(niche == k)[0]
            order = members[np.argsort(-fitness[members])][:per_niche]
            keep.extend(int(i) for i in order)
        keep = np.asarray(keep)
        zs, thetas, fitness, niche = (zs[keep], thetas[keep], fitness[keep],
                                      niche[keep])

        n = min(c.population, args.budget - spent)
        base_idx = rng.integers(0, len(zs), n)
        cross = rng.random(n) < args.cross_niche
        mate_idx = np.empty(n, dtype=int)
        for i, b in enumerate(base_idx):
            if cross[i]:
                mate_idx[i] = rng.integers(0, len(zs))
            else:
                same = np.where(niche == niche[b])[0]
                mate_idx[i] = same[rng.integers(0, len(same))]

        child_z = np.stack([mutate_z(cross_z(zs[b], zs[m]))
                            for b, m in zip(base_idx, mate_idx)])
        sigmas = np.exp(rng.uniform(np.log(c.weight_sigma_low),
                                    np.log(c.weight_sigma_high), n)) * gain
        base_theta = (thetas[base_idx] + thetas[mate_idx]) / 2.0
        scales = np.maximum(base_theta.std(axis=1), 1e-3)
        child_theta = (base_theta + (sigmas * scales)[:, None]
                       * rng.standard_normal((n, template.n_params))
                       ).astype(np.float32)
        child_niche = niche[base_idx]

        child_fit, child_phenos = score_all(child_z, child_theta, child_niche)
        spent += n
        gen += 1
        wins = float((child_fit >= fitness[base_idx] - 1e-12).mean())
        gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, 0.3, c.gain_limits[1]))

        # comma: ONLY the children carry forward; parents are gone
        zs, thetas, fitness, niche = child_z, child_theta, child_fit, child_niche
        improved = False
        for k in range(n_niches):
            members = np.where(child_niche == k)[0]
            if len(members):
                b = members[int(np.argmax(child_fit[members]))]
                if child_fit[b] > hall[k]["score"]:
                    hall[k]["score"] = float(child_fit[b])
                    hall[k]["img"] = (child_phenos[b].detach().cpu().numpy()
                                      .reshape(*SHAPE))
                    improved = True
        stall = 0 if improved else stall + 1
        if stall >= 25:
            gain = min(gain * 3.0, c.gain_limits[1])   # stagnation kick
            stall = 0

        # PROMPT RETIREMENT (Daniel's curriculum): a niche whose best-ever
        # has not moved in `retire_after` generations is done-or-stuck, so
        # its prompt retires to the back of the queue and a fresh prompt
        # arrives. The niche KEEPS its individuals — decoders carrying their
        # accumulated painting machinery into the next concept is the whole
        # point. If later episodes climb faster than early ones, the
        # population is learning to learn.
        for k in range(n_niches):
            members = np.where(child_niche == k)[0]
            niche_improved = (len(members) > 0 and improved and
                              float(child_fit[members].max()) >= hall[k]["score"])
            niche_stall[k] = 0 if niche_improved else niche_stall[k] + 1
            if niche_stall[k] >= args.retire_after and queue:
                episodes.append({"prompt": prompts[k], "niche": k,
                                 "start_gen": episode_start[k],
                                 "end_gen": gen, "best": hall[k]["score"]})
                print(f"  == retiring '{prompts[k]}' (niche {k}) after "
                      f"{gen - episode_start[k]} gens at {hall[k]['score']:.3f}"
                      f" -> new prompt '{queue[0]}'", flush=True)
                queue.append(prompts[k])
                prompts[k] = queue.pop(0)
                scorers[k] = make_scorer(prompts[k])
                hall[k] = {"score": -np.inf, "img": None}
                niche_stall[k] = 0
                episode_start[k] = gen
                if view is not None:
                    view.ims[k].axes.set_title(prompts[k], fontsize=9)
        phenos = torch.cat([phenos[keep] if gen == 1 else phenos, child_phenos]) \
            if False else child_phenos  # only children needed for display

        best_imgs = [hall[k]["img"] for k in range(n_niches)]
        best_scores = [hall[k]["score"] for k in range(n_niches)]

        if spent >= next_frame:
            row = {"e": spent, "scores": best_scores}
            for k, img in enumerate(best_imgs):
                if img is not None:
                    row[f"p{k}"] = _png(img.reshape(-1))
            frames.append(row)
            next_frame += every
            print("  " + f"{spent:>7} evals  " + "  ".join(
                f"{prompts[k].split()[-1][:6]} {best_scores[k]:.3f}"
                for k in range(n_niches)) + f"  (gain {gain:.2f})", flush=True)
            row["prompts"] = list(prompts)
        if view is not None and gen % 5 == 0:
            view.update(spent, best_imgs, best_scores)

    print("\nFINAL per-niche best-ever (current prompts):")
    for k in range(n_niches):
        print(f"  {prompts[k]:<22} {hall[k]['score']:.4f}")
    if episodes:
        print("\ncompleted episodes (learning-to-learn check: do later "
              "episodes reach their best in fewer generations?):")
        for e in episodes:
            print(f"  {e['prompt']:<22} gens {e['end_gen'] - e['start_gen']:>5}"
                  f"  best {e['best']:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"prompts": prompts, "neg": args.neg, "budget": args.budget,
             "seed": args.seed, "cross_niche": args.cross_niche,
             "population": args.population, "retire_after": args.retire_after,
             "pool": pool, "episodes": episodes,
             "frames": frames, "torch_version": torch.__version__}) + "\n")
        print(f"wrote {args.output}")

    if args.gif:
        imgs = []
        for f in frames:
            tiles = []
            for k in range(len(prompts)):
                if f"p{k}" in f:
                    tiles.append(Image.open(io.BytesIO(base64.b64decode(
                        f[f"p{k}"].split(",", 1)[1]))).convert("RGB")
                        .resize((192, 192), Image.NEAREST))
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
