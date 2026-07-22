"""CLIP evolution as an ISLAND MODEL: every prompt is a species, always alive.

Daniel's v3 design, which deletes machinery rather than adding it. v2 had six
"seats" that prompts rotated through, with retiring prompts hibernating their
specialists into a bank and waking them later. His observation: none of that
is needed if every species simply stays alive and we breed only a SUBSET each
generation. A species that is not sampled is resting — hibernation becomes the
default state of the world instead of an engineered feature, no banking, no
waking, no reassignment. Individuals are never handed to a different prompt.

This is the classical island model (demes under separate selection pressures,
with migration), which is what Daniel's "each fitness function guides a
species" framing names exactly. `--cross-species` IS the migration rate.

HONEST STATUS of migration: no controlled evidence yet. v1 showed a niche
rocket off the floor after 40k flat evaluations, which LOOKED like a migration
event, but the stagnation gain-kick is an equally good explanation. That is an
ablation (`--cross-species 0` vs 0.15 vs 0.4), not a thing to assert.

Kept from v2 (all measured to matter): comma selection per species (parents
die each generation, so negative mutations drift the population out of
deceptive basins — this is what unfroze the CLIP runs), a hall of fame so
nothing is lost, gain floor 0.3 with a stagnation kick, fp16 CLIP at 128px.

Speed: CLIP's image encoder does not depend on the prompt, so every child is
encoded ONCE per generation and scored against its own species' text vector —
one forward per generation instead of one per active species.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.demo_clip_evolve import CLIP_MEAN, CLIP_STD, load_clip
from benchmarks.round28_anchor_conv import ConvRGB
from benchmarks.round31_apple_animated import _png
from latentspace.universal.architectures import resolve
from latentspace.universal.explorer import ExplorerConfig, _Template

LATENT = 64
SHAPE = (3, 96, 96)

DEFAULT_POOL = [
    "a mango", "a sports car", "a tree", "a sunflower",
    "a house", "a cat", "a sailboat", "a coffee cup",
    "a mountain", "a butterfly", "a mushroom", "a lighthouse",
    "a snowman", "a rubber duck", "a planet", "a human face",
]


class SpeciesView:
    def __init__(self, prompts, budget):
        import matplotlib
        try:
            matplotlib.use("MacOSX")
        except Exception:
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        self.plt = plt
        plt.ion()
        n = len(prompts)
        cols = min(8, n)
        rows = int(np.ceil(n / cols))
        self.fig = plt.figure(figsize=(2.0 * cols, 2.3 * rows + 4.0))
        grid = (rows + 2, cols)
        self.ims, self.axes = [], []
        for i, prompt in enumerate(prompts):
            ax = self.plt.subplot2grid(grid, (i // cols, i % cols),
                                       fig=self.fig)
            ax.axis("off")
            self.axes.append(ax)
            self.ims.append(ax.imshow(np.zeros((96, 96, 3))))
        self.ax = self.plt.subplot2grid(grid, (rows, 0), colspan=cols,
                                        rowspan=2, fig=self.fig)
        cmap = colormaps["tab20"]
        self.lines = []
        for i, prompt in enumerate(prompts):
            (line,) = self.ax.plot([], [], lw=1.4, color=cmap(i % 20),
                                   label=prompt)
            self.lines.append(line)
        self.ax.set_xlim(0, budget)
        self.ax.set_xlabel("evaluations", fontsize=9)
        self.ax.set_ylabel("species record", fontsize=9)
        self.ax.legend(fontsize=6, loc="lower right", ncols=4)
        self.ax.grid(alpha=0.25)
        self.es: list[int] = []
        self.hist = [[] for _ in prompts]
        self.fig.tight_layout()
        plt.show(block=False)

    def update(self, spent, prompts, imgs, scores, active):
        self.es.append(spent)
        for i, (prompt, img, score) in enumerate(zip(prompts, imgs, scores)):
            mark = "*" if i in active else ""
            self.axes[i].set_title(
                f"{prompt}{mark}\n{score:.3f}" if np.isfinite(score)
                else f"{prompt}{mark}", fontsize=8)
            if img is not None:
                self.ims[i].set_data(np.clip(img.transpose(1, 2, 0), 0, 1))
            self.hist[i].append(score if np.isfinite(score) else np.nan)
            self.lines[i].set_data(self.es, self.hist[i])
        flat = [v for h in self.hist for v in h if np.isfinite(v)]
        if flat:
            lo, hi = min(flat), max(flat)
            pad = 0.05 * max(hi - lo, 1e-3)
            self.ax.set_ylim(lo - pad, hi + pad)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", nargs="*", default=DEFAULT_POOL)
    parser.add_argument("--neg", nargs="*",
                        default=["noise", "static", "an abstract pattern",
                                 "text"])
    parser.add_argument("--model",
                        default="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
    parser.add_argument("--clip-res", type=int, default=128)
    parser.add_argument("--survivors", type=int, default=8,
                        help="individuals kept per species")
    parser.add_argument("--children", type=int, default=32,
                        help="children bred per ACTIVE species per generation")
    parser.add_argument("--active", type=int, default=6,
                        help="species bred each generation (rest are resting)")
    parser.add_argument("--cross-species", type=float, default=0.15,
                        help="migration rate: probability a mate comes from "
                             "another species. 0 = isolated islands")
    parser.add_argument("--budget", type=int, default=600_000)
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
    prompts = list(args.pool)
    S = len(prompts)
    c = ExplorerConfig()

    print(f"loading CLIP: {args.model}", flush=True)
    model, processor = load_clip(args.model, device)
    dtype = next(model.parameters()).dtype
    res = args.clip_res or model.config.vision_config.image_size
    mean, std = CLIP_MEAN.to(device, dtype), CLIP_STD.to(device, dtype)

    with torch.no_grad():
        tok = processor(text=prompts + args.neg, return_tensors="pt",
                        padding=True)
        out = model.text_model(**{k: v.to(device) for k, v in tok.items()})
        text = F.normalize(model.text_projection(out.pooler_output).float(),
                           dim=-1)
    text_pos, text_neg = text[:S], text[S:]

    def embed(phenos: torch.Tensor) -> torch.Tensor:
        """One CLIP image forward for the whole generation — the image
        encoder does not depend on the prompt, so all species share it."""
        with torch.no_grad():
            x = phenos.reshape(len(phenos), *SHAPE)
            x = F.interpolate(x, size=(res, res), mode="bicubic",
                              align_corners=False).clamp(0, 1).to(dtype)
            v = model.vision_model(pixel_values=(x - mean) / std,
                                   interpolate_pos_encoding=True)
            return F.normalize(model.visual_projection(v.pooler_output).float(),
                               dim=-1)

    _seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    template = _Template(resolve(lambda l, s: ConvRGB(l, s), LATENT, SHAPE),
                         device)

    # every species owns its population, permanently. no seats, no banking.
    pop_z = [rng.standard_normal((args.survivors, LATENT)).astype(np.float32)
             for _ in range(S)]
    pop_th = [np.stack([template.init_theta(int(rng.integers(0, 2**31)))
                        for _ in range(args.survivors)]) for _ in range(S)]
    pop_fit = [np.full(args.survivors, -np.inf) for _ in range(S)]
    hall = [{"score": -np.inf, "img": None} for _ in range(S)]

    spent, gain, gen, global_stall = 0, float(c.initial_gain), 0, 0
    frames: list[dict] = []
    every = max(1, args.budget // max(args.frames, 1))
    next_frame = 0
    view = SpeciesView(prompts, args.budget) if args.live else None

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

    # score the founders so the halls start honest
    all_z = np.concatenate(pop_z)
    all_th = np.concatenate(pop_th)
    phen = template.decode_batch(all_th, all_z)
    emb = embed(phen)
    spent += len(all_z)
    for s in range(S):
        sl = slice(s * args.survivors, (s + 1) * args.survivors)
        sc = ((emb[sl] @ text_pos[s]) -
              (emb[sl] @ text_neg.T).mean(dim=1)).cpu().numpy()
        pop_fit[s] = sc
        b = int(sc.argmax())
        hall[s] = {"score": float(sc[b]),
                   "img": phen[sl][b].cpu().numpy().reshape(*SHAPE)}

    while spent < args.budget:
        active = rng.choice(S, size=min(args.active, S), replace=False)
        gen += 1

        batch_z, batch_th, batch_owner = [], [], []
        for s in active:
            n = args.children
            base = rng.integers(0, len(pop_z[s]), n)
            mate_s = np.where(rng.random(n) < args.cross_species,
                              rng.integers(0, S, n), s)
            mate = np.array([rng.integers(0, len(pop_z[m])) for m in mate_s])
            cz = np.stack([mutate_z(cross_z(pop_z[s][b], pop_z[m][j]))
                           for b, m, j in zip(base, mate_s, mate)])
            base_th = np.stack([(pop_th[s][b] + pop_th[m][j]) / 2.0
                                for b, m, j in zip(base, mate_s, mate)])
            sig = np.exp(rng.uniform(np.log(c.weight_sigma_low),
                                     np.log(c.weight_sigma_high), n)) * gain
            scales = np.maximum(base_th.std(axis=1), 1e-3)
            cth = (base_th + (sig * scales)[:, None]
                   * rng.standard_normal((n, template.n_params))
                   ).astype(np.float32)
            batch_z.append(cz)
            batch_th.append(cth)
            batch_owner.append(np.full(n, s))

        cz = np.concatenate(batch_z)
        cth = np.concatenate(batch_th)
        owner = np.concatenate(batch_owner)
        phen = template.decode_batch(cth, cz)
        emb = embed(phen)                      # ONE forward for everyone
        spent += len(cz)

        wins = []
        improved = False
        for s in active:
            idx = np.where(owner == s)[0]
            sc = ((emb[idx] @ text_pos[s]) -
                  (emb[idx] @ text_neg.T).mean(dim=1)).cpu().numpy()
            wins.append((sc >= pop_fit[s].mean()).mean())
            # comma selection: survivors are the best CHILDREN only
            order = np.argsort(-sc)[:args.survivors]
            pop_z[s], pop_th[s] = cz[idx][order], cth[idx][order]
            pop_fit[s] = sc[order]
            if sc[order[0]] > hall[s]["score"] + 1e-9:
                hall[s] = {"score": float(sc[order[0]]),
                           "img": phen[idx][order[0]].cpu().numpy()
                           .reshape(*SHAPE)}
                improved = True

        w = float(np.mean(wins))
        gain *= c.gain_step if w > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, 0.3, c.gain_limits[1]))
        global_stall = 0 if improved else global_stall + 1
        if global_stall >= 25:
            gain = min(gain * 3.0, c.gain_limits[1])
            global_stall = 0

        scores = [hall[s]["score"] for s in range(S)]
        if spent >= next_frame:
            row = {"e": spent, "scores": scores,
                   "active": [int(a) for a in active]}
            for s in range(S):
                if hall[s]["img"] is not None:
                    row[f"p{s}"] = _png(hall[s]["img"].reshape(-1))
            frames.append(row)
            next_frame += every
            top = int(np.argmax(scores))
            print(f"  {spent:>7} evals  gain {gain:.2f}  best: "
                  f"{prompts[top]} {scores[top]:.3f}  |  mean "
                  f"{np.mean(scores):.3f}", flush=True)
        if view is not None and gen % 5 == 0:
            view.update(spent, prompts, [hall[s]["img"] for s in range(S)],
                        scores, set(int(a) for a in active))

    print("\nFINAL species records:")
    for s in np.argsort([-hall[k]["score"] for k in range(S)]):
        print(f"  {prompts[s]:<20} {hall[s]['score']:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"pool": prompts, "neg": args.neg, "budget": args.budget,
             "seed": args.seed, "survivors": args.survivors,
             "children": args.children, "active": args.active,
             "cross_species": args.cross_species,
             "records": {prompts[s]: hall[s]["score"] for s in range(S)},
             "frames": frames, "torch_version": torch.__version__}) + "\n")
        print(f"wrote {args.output}")

    if args.gif:
        from PIL import ImageDraw
        cols = min(8, S)
        rows = int(np.ceil(S / cols))
        imgs = []
        for f in frames:
            sheet = Image.new("RGB", (128 * cols, 128 * rows), (20, 20, 20))
            for s in range(S):
                if f"p{s}" not in f:
                    continue
                tile = (Image.open(io.BytesIO(base64.b64decode(
                    f[f"p{s}"].split(",", 1)[1]))).convert("RGB")
                    .resize((128, 128), Image.NEAREST))
                d = ImageDraw.Draw(tile)
                d.rectangle([0, 116, 128, 128], fill=(0, 0, 0))
                d.text((3, 117), f"{prompts[s]} {f['scores'][s]:.2f}",
                       fill=(255, 255, 255))
                sheet.paste(tile, (128 * (s % cols), 128 * (s // cols)))
            imgs.append(sheet)
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
