"""DESIGN THE PAST — the demo (2026-07-22).

Type a word. The system shows initial seed fields — scattered static — and
then runs a thresholded cellular automaton forward, live, and the static
blooms into your word. The dynamics have NO gradient (hard thresholds sever
the graph entirely), so gradient descent cannot attempt this problem;
evolution with a learned prior over good seeds solves it, and because
letters share stroke structure, the warmed decoder solves each new symbol
faster than the first — the universal-optimizer thesis performed live.

Usage:
  python -m benchmarks.demo_design_the_past --warm            # once: learn A-Z
  python -m benchmarks.demo_design_the_past --word HELLO      # the show
  python -m benchmarks.demo_design_the_past --live-solve '?'  # finale: a novel
                                                              # symbol, solved
                                                              # before your eyes

CA rule: a cell turns on iff >=4 of its 3x3 neighbourhood (incl. itself) are
on, K=4 steps, 64x64 grid. Fitness compares BLURRED CA output to the blurred
letter target (partial credit for search); the display is the raw binary CA.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from benchmarks.experimental_distill_loop import Config, run

G = 64                # grid
K = 4                 # CA steps
VOTE = 4              # rescues ring topologies (letter O: 0% at 5, 73%+ at 4)
STORE = Path(__file__).resolve().parent.parent / "demo/design_the_past.pt"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def ca_steps(fields: torch.Tensor, steps: int = K) -> list[torch.Tensor]:
    """All intermediate states, for the bloom animation. fields (n, G, G)."""
    kernel = torch.ones(1, 1, 3, 3, device=fields.device)
    s = (fields[:, None] > 0.5).float()
    out = [s[:, 0].clone()]
    for _ in range(steps):
        s = (torch.conv2d(s, kernel, padding=1) >= VOTE).float()
        out.append(s[:, 0].clone())
    return out


def make_transform(device):
    """Search objective: blur(CA(x)) — the CA is the black box, the blur just
    grants partial credit so evolution has signal through the thresholds."""
    kernel = torch.ones(1, 1, 3, 3, device=device)
    g = torch.tensor([1., 2., 1.], device=device)
    blur2d = (g[:, None] * g[None, :]).reshape(1, 1, 3, 3) / 16.0

    def transform(flat):
        s = (flat.reshape(-1, 1, G, G) > 0.5).float()
        for _ in range(K):
            s = (torch.conv2d(s, kernel, padding=1) >= VOTE).float()
        for _ in range(2):
            s = torch.conv2d(s, blur2d, padding=1)
        return s.reshape(len(flat), -1)
    return transform


def blur_np(img: np.ndarray, device) -> np.ndarray:
    g = torch.tensor([1., 2., 1.], device=device)
    blur2d = (g[:, None] * g[None, :]).reshape(1, 1, 3, 3) / 16.0
    s = torch.as_tensor(img, device=device).reshape(1, 1, G, G)
    for _ in range(2):
        s = torch.conv2d(s, blur2d, padding=1)
    return s.reshape(G, G).cpu().numpy()


def render_char(ch: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("L", (G, G), 0)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 58)
    box = d.textbbox((0, 0), ch, font=font, stroke_width=3)
    d.text(((G - box[2] - box[0]) / 2, (G - box[3] - box[1]) / 2 - box[1] * 0),
           ch, fill=255, font=font, stroke_width=3, stroke_fill=255,
           anchor=None)
    return (np.asarray(img, np.float32) / 255.0 > 0.5).astype(np.float32)


def solve_chars(chars, device, epochs, init_state=None, seed=7, log=None,
                progress=None):
    transform = make_transform(device)
    raw = [render_char(c) for c in chars]
    targets = [blur_np(r, device) for r in raw]     # blurred, to match transform
    cfg = Config(epochs=epochs, children=16, patch=1024, genes=32)
    mean, per, gen, _, seeds = run(
        targets, (G, G), cfg, seed=seed, device=device, transform=transform,
        init_state=init_state, return_full=True, log=log, progress=progress)
    return mean, per, gen, [s.reshape(G, G) for s in seeds], raw


def animate(word, seeds, raws, device, hold=12, step_frames=10):
    import matplotlib
    try:
        matplotlib.use("MacOSX")
    except Exception:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fields = torch.as_tensor(np.stack(seeds), device=device)
    states = ca_steps(fields)                       # K+1 frames per letter
    gutter = np.zeros((G, 6))

    def strip(frames):
        cols = []
        for i in range(len(seeds)):
            cols += [frames[i].cpu().numpy() if torch.is_tensor(frames[i])
                     else frames[i], gutter]
        return np.concatenate(cols[:-1], axis=1)

    plt.ion()
    fig, (ax_t, ax_s) = plt.subplots(2, 1, figsize=(1.6 * len(word), 4.2))
    fig.canvas.manager.set_window_title("DESIGN THE PAST")
    ax_t.imshow(strip(raws), cmap="gray"); ax_t.axis("off")
    ax_t.set_title(f"the future we want: “{word}”", fontsize=11)
    im = ax_s.imshow(strip([s for s in states[0]]), cmap="gray", vmin=0, vmax=1)
    ax_s.axis("off")
    ax_s.set_title("the past we designed (seeds)", fontsize=11)
    fig.tight_layout()
    plt.pause(2.0)
    for k in range(1, K + 1):
        for _ in range(step_frames):
            plt.pause(0.03)
        im.set_data(strip([s for s in states[k]]))
        ax_s.set_title(f"automaton step {k}/{K}", fontsize=11)
        fig.canvas.draw_idle()
    ax_s.set_title(f"step {K}/{K} — the static became the word", fontsize=11)
    for _ in range(hold):
        plt.pause(0.25)
    plt.ioff()
    plt.show(block=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", action="store_true",
                        help="learn the alphabet (run once; saves decoder)")
    parser.add_argument("--warm-epochs", type=int, default=4000)
    parser.add_argument("--word")
    parser.add_argument("--live-solve", metavar="CHAR",
                        help="solve a novel symbol live with the warm decoder")
    parser.add_argument("--solve-epochs", type=int, default=600)
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.warm:
        print(f"learning the alphabet ({args.warm_epochs} epochs, "
              f"{len(ALPHABET)} species)...", flush=True)
        mean, per, gen, seeds, raws = solve_chars(
            list(ALPHABET), device, args.warm_epochs,
            log=max(1, args.warm_epochs // 10))
        torch.save({"state": gen.net.state_dict(),
                    "seeds": {c: s for c, s in zip(ALPHABET, seeds)},
                    "match": {c: p for c, p in zip(ALPHABET, per)}},
                   STORE)
        print(f"alphabet learned: mean match {mean:.1f}%  -> {STORE}")
        worst = min(zip(ALPHABET, per), key=lambda t: t[1])
        print(f"worst letter: {worst[0]} at {worst[1]:.1f}%")
        return

    if not STORE.exists():
        raise SystemExit("run --warm once first")
    bank = torch.load(STORE, weights_only=False)

    if args.live_solve:
        ch = args.live_solve[0].upper()
        print(f"solving never-seen symbol '{ch}' with the warm decoder "
              f"({args.solve_epochs} epochs)...", flush=True)
        _, per, _, seeds, raws = solve_chars(
            [ch], device, args.solve_epochs, init_state=bank["state"],
            log=max(1, args.solve_epochs // 6))
        print(f"match {per[0]:.1f}%")
        animate(ch, seeds, raws, device)
        return

    word = (args.word or input("word: ")).upper()
    missing = [c for c in word if c not in bank["seeds"] and c != " "]
    if missing:
        raise SystemExit(f"not in the learned bank: {missing} "
                         f"(use --live-solve)")
    seeds = [bank["seeds"][c] for c in word if c != " "]
    raws = [render_char(c) for c in word if c != " "]
    animate(word, seeds, raws, device)


if __name__ == "__main__":
    main()
