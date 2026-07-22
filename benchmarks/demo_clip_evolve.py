"""Evolve an image against CLIP text prompts — no target image exists.

The fitness function is a tiny CLIP model: score = mean cosine similarity to
the positive prompts minus mean similarity to the negative prompts. The GA
never sees pixels of any target because there is none — it is climbing a
semantic gradient defined entirely by text. Fair warning from the
literature: population search against CLIP tends to find images CLIP loves
that humans find abstract; the negative prompts are the steering wheel.

With --live, a matplotlib window shows the run as it happens: the current
best image, the fitness curve, and the win-rate controller's step size.

    python3 -m benchmarks.demo_clip_evolve \
        --pos "a photo of a ripe mango" "a mango on a white background" \
        --neg "noise" "an apple" --budget 20000 --live --gif mango_clip.gif
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
from benchmarks.round28_anchor_conv import ConvRGB
from benchmarks.round31_apple_animated import _png
from latentspace.universal import solve
from latentspace.universal.explorer import ExplorerConfig


class WhiteCanvas(torch.nn.Module):
    """ConvRGB with a logit bias so the UNTRAINED decoder emits white
    (sigmoid(+2.94) ~ 0.95) instead of mid-gray. Round 29 measured that an
    untrained decoder's first job is inflating weights ~100x just to paint
    anything; for a product-photo prompt the background it must reach IS
    white, so start there and let evolution spend its budget on the fruit."""

    def __init__(self, latent, shape):
        super().__init__()
        self.net = ConvRGB(latent, shape)

    def forward(self, z):
        return self.net(z) + 2.944   # logit(0.95)

SHAPE = (3, 96, 96)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_clip(name: str, device: str):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(name).to(device).eval()
    if device == "mps":
        model = model.half()   # ~2x on Apple GPUs; scoring needs no fp32
    processor = CLIPProcessor.from_pretrained(name)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def make_fitness(model, processor, pos, neg, device, on_batch=None,
                 res=None):
    # transformers 5.x: the encoder submodules return output objects; project
    # the pooled state ourselves so this works across API versions.
    def text_features(**tokens):
        out = model.text_model(**tokens)
        return model.text_projection(out.pooler_output)

    def image_features(pixel_values):
        out = model.vision_model(pixel_values=pixel_values,
                                 interpolate_pos_encoding=True)
        return model.visual_projection(out.pooler_output)

    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        tokens = processor(text=pos + neg, return_tensors="pt", padding=True)
        text = text_features(**{k: v.to(device) for k, v in tokens.items()})
        text = F.normalize(text.float(), dim=-1)
    text_pos, text_neg = text[:len(pos)], text[len(pos):]
    size = res or model.config.vision_config.image_size
    mean = CLIP_MEAN.to(device, dtype)
    std = CLIP_STD.to(device, dtype)

    def fitness(phenotypes: torch.Tensor, cutouts: int = 1,
                rng: np.random.Generator | None = None) -> torch.Tensor:
        with torch.no_grad():
            imgs = phenotypes.reshape(len(phenotypes), *SHAPE)
            views = []
            for k in range(cutouts):
                if k == 0 or rng is None:
                    v = imgs                          # the full frame, always
                else:
                    # random crop between 60% and 100% of the frame — an
                    # adversarial texture only fools CLIP from one exact
                    # framing; a real object survives being looked at from
                    # many angles
                    side = int(96 * rng.uniform(0.6, 1.0))
                    y = int(rng.integers(0, 96 - side + 1))
                    x0 = int(rng.integers(0, 96 - side + 1))
                    v = imgs[:, :, y:y + side, x0:x0 + side]
                views.append(F.interpolate(v, size=(size, size),
                                           mode="bicubic",
                                           align_corners=False).clamp(0, 1))
            batch = torch.cat(views).to(dtype)        # (cutouts*B, 3, S, S)
            emb = F.normalize(image_features((batch - mean) / std).float(),
                              dim=-1)
            score = (emb @ text_pos.T).mean(dim=1)
            if len(text_neg):
                score = score - (emb @ text_neg.T).mean(dim=1)
            score = score.reshape(len(views), len(imgs)).mean(dim=0)
        if on_batch is not None:
            on_batch(phenotypes, score)
        return score

    return fitness


class LiveView:
    """A matplotlib window updated as the run progresses."""

    def __init__(self, pos, budget):
        import matplotlib
        try:
            matplotlib.use("MacOSX")
        except Exception:
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, (self.ax_img, self.ax_fit) = plt.subplots(
            1, 2, figsize=(10, 4.4),
            gridspec_kw={"width_ratios": [1, 1.6]})
        self.fig.suptitle(f'evolving: "{pos[0]}"', fontsize=11)
        self.im = self.ax_img.imshow(np.zeros((96, 96, 3)))
        self.ax_img.set_title("best individual", fontsize=10)
        self.ax_img.axis("off")
        (self.line_best,) = self.ax_fit.plot([], [], lw=2, color="#2a78d6",
                                             label="best CLIP score")
        (self.line_mean,) = self.ax_fit.plot([], [], lw=1, color="#eb6834",
                                             alpha=0.7, label="batch mean")
        self.ax_fit.set_xlim(0, budget)
        self.ax_fit.set_xlabel("evaluations", fontsize=9)
        self.ax_fit.set_ylabel("pos sim − neg sim", fontsize=9)
        self.ax_fit.legend(fontsize=9, loc="lower right")
        self.ax_fit.grid(alpha=0.25)
        self.es, self.bests, self.means = [], [], []
        self.fig.tight_layout()
        plt.show(block=False)

    def update(self, spent, best_img, best_score, mean_score):
        self.es.append(spent)
        self.bests.append(best_score)
        self.means.append(mean_score)
        self.im.set_data(np.clip(best_img.transpose(1, 2, 0), 0, 1))
        self.ax_img.set_title(f"best individual — {spent:,} evals",
                              fontsize=10)
        self.line_best.set_data(self.es, self.bests)
        self.line_mean.set_data(self.es, self.means)
        lo = min(min(self.means), min(self.bests))
        hi = max(self.bests)
        pad = 0.05 * max(hi - lo, 1e-3)
        self.ax_fit.set_ylim(lo - pad, hi + pad)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", nargs="+", required=True)
    parser.add_argument("--neg", nargs="*", default=[])
    parser.add_argument("--model",
                        default="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
    parser.add_argument("--budget", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--cutouts", type=int, default=8,
                        help="random crops averaged per score; 1 = off")
    parser.add_argument("--clip-res", type=int, default=None,
                        help="score at this resolution (interpolated pos embeddings); 128 is ~3x faster than 224 with near-identical judgment")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--start-gain", type=float, default=10.0,
                        help="initial mutation gain; the controller anneals from here")
    parser.add_argument("--canvas", choices=("white", "gray"), default="white")
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    device = "mps"
    print(f"loading CLIP: {args.model}", flush=True)
    try:
        model, processor = load_clip(args.model, device)
    except Exception as exc:
        print(f"  failed ({exc}); falling back to openai/clip-vit-base-patch32",
              flush=True)
        model, processor = load_clip("openai/clip-vit-base-patch32", device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M params, image size "
          f"{model.config.vision_config.image_size}", flush=True)

    view = LiveView(args.pos, args.budget) if args.live else None
    frames: list[dict] = []
    state = {"spent": 0, "best": -np.inf, "next": 0, "gen": 0}
    every = max(1, args.budget // max(args.frames, 1))
    import time as _time
    t0 = _time.perf_counter()

    def on_batch(phenotypes, scores):
        state["spent"] += len(scores)
        state["gen"] += 1
        hi = int(scores.argmax())
        if float(scores[hi]) > state["best"]:
            state["best"] = float(scores[hi])
            state["pheno"] = phenotypes[hi].detach().cpu().numpy()
        if state["spent"] >= state["next"]:
            frames.append({"e": state["spent"], "m": state["best"],
                           "p": _png(state["pheno"].reshape(-1))})
            state["next"] += every
            rate = state["spent"] / (_time.perf_counter() - t0)
            print(f"  {state['spent']:>7} evals  best {state['best']:.4f}  "
                  f"({rate:,.0f} evals/s)", flush=True)
        # redraw every 5th generation — matplotlib is not free
        if view is not None and state["gen"] % 5 == 0:
            view.update(state["spent"], state["pheno"].reshape(3, 96, 96),
                        state["best"], float(scores.mean()))

    base_fitness = make_fitness(model, processor, args.pos, args.neg,
                                device, on_batch, res=args.clip_res)
    cut_rng = np.random.default_rng(args.seed + 4243)
    fitness = lambda ph: base_fitness(ph, cutouts=args.cutouts, rng=cut_rng)
    _seed_everything(args.seed)
    arch = WhiteCanvas if args.canvas == "white" else ConvRGB
    result = solve(fitness, output_shape=SHAPE, budget=args.budget,
                   architecture=lambda latent, shape: arch(latent, shape),
                   explore_fraction=1.0, seed=args.seed,
                   explorer_config=ExplorerConfig(initial_gain=args.start_gain))
    final = float(result.best_fitness)
    frames.append({"e": result.evaluations, "m": final,
                   "p": _png(result.best_phenotype.reshape(-1))})
    print(f"\nFINAL CLIP score {final:.4f} at {result.evaluations} evaluations")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"pos": args.pos, "neg": args.neg, "model": args.model,
             "budget": args.budget, "seed": args.seed, "final_score": final,
             "history": result.history[::50], "frames": frames,
             "torch_version": torch.__version__}) + "\n")
        print(f"wrote {args.output}")

    if args.gif:
        imgs = [Image.open(io.BytesIO(base64.b64decode(f["p"].split(",", 1)[1])))
                .convert("RGB").resize((288, 288), Image.NEAREST)
                for f in frames]
        imgs.append(imgs[-1])
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:],
                     duration=[80] * (len(imgs) - 1) + [2500], loop=0,
                     optimize=True)
        print(f"wrote {args.gif}")

    if view is not None:
        print("close the matplotlib window to exit", flush=True)
        view.plt.ioff()
        view.plt.show()


if __name__ == "__main__":
    main()
