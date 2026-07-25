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


def salt_seed(seed: np.ndarray, rng, noise: float, device,
              display_steps: int, tol: float = 0.02) -> np.ndarray:
    """Sprinkle real random static over an evolved seed — the drama knob.
    The majority rule is a denoiser (isolated flips die, small holes
    refill), so heavy salt vanishes in the first steps while the structure
    condenses. VERIFIED honest: we re-run the CA and require the salted
    seed's final frame to match the clean seed's within `tol`; if a draw
    breaks the word, we redraw with less noise."""
    clean = torch.as_tensor(seed[None], device=device)
    final = ca_steps(clean, display_steps)[-1]
    on = seed > 0.5
    # Physics note: under vote>=4 any 2x2 noise clump GROWS, so a dense
    # white-noise past of a clean word does not exist — the honest maximum
    # is sparse scattered static plus a FRAGMENTED letter body (strokes
    # broken into dots the rule fuses back). Both knobs below back off
    # automatically until the verified final frame matches.
    bg_level, fg_level = noise, min(0.45, noise * 2.2)
    for _ in range(12):
        flip_bg = (~on) & (rng.random(seed.shape) < bg_level)
        flip_fg = on & (rng.random(seed.shape) < fg_level)
        mask = flip_bg | flip_fg
        salted = np.where(mask, 1.0 - on, seed).astype(np.float32)
        got = ca_steps(torch.as_tensor(salted[None], device=device),
                       display_steps)[-1]
        if float((got - final).abs().mean()) <= tol:
            return salted
        bg_level *= 0.8
        fg_level *= 0.85
    return seed


def make_transform(device, noisy_weight: float = 0.0):
    """Search objective: blur(CA(x)) — the CA is the black box, the blur just
    grants partial credit so evolution has signal through the thresholds.

    With `noisy_weight` > 0 the objective is WIDENED (Daniel's "look like
    random pixels" ask): the comparison vector gains w*(local_density(x) -
    0.5), scored against zeros — so among the many pasts that produce the
    same word (the map destroys information), evolution prefers seeds whose
    local density is ~50% everywhere, i.e. genuinely static-looking. The
    target vectors must be padded with zeros to match (see solve_chars)."""
    kernel = torch.ones(1, 1, 3, 3, device=device)
    g = torch.tensor([1., 2., 1.], device=device)
    blur2d = (g[:, None] * g[None, :]).reshape(1, 1, 3, 3) / 16.0

    def transform(flat):
        x = flat.reshape(-1, 1, G, G)
        s = (x > 0.5).float()
        seed_bin = s
        for _ in range(K):
            s = (torch.conv2d(s, kernel, padding=1) >= VOTE).float()
        for _ in range(2):
            s = torch.conv2d(s, blur2d, padding=1)
        out = s.reshape(len(flat), -1)
        if noisy_weight > 0:
            density = torch.conv2d(seed_bin, blur2d, padding=1)
            noise_term = noisy_weight * (density - 0.5)
            out = torch.cat([out, noise_term.reshape(len(flat), -1)], dim=1)
        return out
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
                progress=None, noisy_weight: float = 0.0):
    transform = make_transform(device, noisy_weight)
    raw = [render_char(c) for c in chars]
    targets = [blur_np(r, device) for r in raw]     # blurred, to match transform
    if noisy_weight > 0:                # widened objective: noise term vs 0
        targets = [np.concatenate([t.reshape(-1),
                                   np.zeros(G * G, np.float32)])
                   for t in targets]
    cfg = Config(epochs=epochs, children=16, patch=1024, genes=32)
    mean, per, gen, _, seeds = run(
        targets, (G, G), cfg, seed=seed, device=device, transform=transform,
        init_state=init_state, return_full=True, log=log, progress=progress)
    return mean, per, gen, [s.reshape(G, G) for s in seeds], raw


def animate(word, seeds, raws, device, hold=12, step_frames=10,
            noise=0.18, display_steps=6, pace=0.7):
    import matplotlib
    try:
        matplotlib.use("MacOSX")
    except Exception:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    salted = [salt_seed(sd, rng, noise, device, display_steps)
              for sd in seeds]
    fields = torch.as_tensor(np.stack(salted), device=device)
    states = ca_steps(fields, display_steps)
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
    ax_s.set_title("the past we designed (random-looking static)",
                   fontsize=11)
    fig.tight_layout()
    plt.pause(2.5)
    for k in range(1, display_steps + 1):
        plt.pause(pace)
        im.set_data(strip([s for s in states[k]]))
        ax_s.set_title(f"automaton step {k}/{display_steps}", fontsize=11)
        fig.canvas.draw_idle()
    ax_s.set_title(f"step {display_steps}/{display_steps} — "
                   "the static became the word", fontsize=11)
    for _ in range(hold):
        plt.pause(0.25)
    plt.ioff()
    plt.show(block=True)


class PlayWindow:
    """The interactive show: a persistent window with a text box. Type a
    word, press enter, watch it bloom. Symbols not yet in the bank are
    solved LIVE with the warm decoder and added to it."""

    def __init__(self, bank, device, solve_epochs=600, noise=0.18,
                 display_steps=6, pace=0.8):
        import matplotlib
        try:
            matplotlib.use("MacOSX")
        except Exception:
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import TextBox

        self.plt = plt
        self.bank = bank
        self.device = device
        self.solve_epochs = solve_epochs
        self.noise = noise
        self.display_steps = display_steps
        self.pace = pace
        self.rng = np.random.default_rng(0)
        self.busy = False
        self.fig = plt.figure(figsize=(11, 5.2))
        self.fig.canvas.manager.set_window_title("DESIGN THE PAST")
        self.ax_t = self.fig.add_axes([0.03, 0.60, 0.94, 0.30])
        self.ax_s = self.fig.add_axes([0.03, 0.20, 0.94, 0.36])
        for ax in (self.ax_t, self.ax_s):
            ax.axis("off")
        self.ax_t.set_title("type a word below and press enter", fontsize=12)
        box_ax = self.fig.add_axes([0.25, 0.05, 0.50, 0.07])
        self.box = TextBox(box_ax, "word ", initial="")
        self.box.on_submit(self.on_word)
        plt.show(block=True)

    @staticmethod
    def _strip(panels):
        gutter = np.zeros((G, 6))
        cols = []
        for p in panels:
            cols += [p, gutter]
        return np.concatenate(cols[:-1], axis=1)

    def on_word(self, text):
        if self.busy or not text.strip():
            return
        self.busy = True
        try:
            word = "".join(c for c in text.upper() if c != " ")
            # solve any never-seen symbols live, off the warm decoder
            for ch in dict.fromkeys(word):
                if ch not in self.bank["seeds"]:
                    self.ax_t.set_title(
                        f"never seen '{ch}' — evolving its past live "
                        f"(~{self.solve_epochs} epochs)...", fontsize=12)
                    self.fig.canvas.draw_idle()
                    self.plt.pause(0.05)
                    try:
                        _, per, _, seeds, _ = solve_chars(
                            [ch], self.device, self.solve_epochs,
                            init_state=self.bank["state"], seed=11)
                    except Exception as err:
                        self.ax_t.set_title(f"cannot render '{ch}' ({err})",
                                            fontsize=12)
                        return
                    self.bank["seeds"][ch] = seeds[0]
                    self.bank["match"][ch] = per[0]
                    torch.save(self.bank, STORE)
            seeds = [salt_seed(self.bank["seeds"][c], self.rng, self.noise,
                               self.device, self.display_steps)
                     for c in word]
            raws = [render_char(c) for c in word]
            self.ax_t.clear(); self.ax_t.axis("off")
            self.ax_t.imshow(self._strip(raws), cmap="gray")
            self.ax_t.set_title(f"the future we want: “{word}”", fontsize=12)
            fields = torch.as_tensor(np.stack(seeds), device=self.device)
            states = ca_steps(fields, self.display_steps)
            self.ax_s.clear(); self.ax_s.axis("off")
            im = self.ax_s.imshow(
                self._strip([s.cpu().numpy() for s in states[0]]),
                cmap="gray", vmin=0, vmax=1)
            self.ax_s.set_title(
                "the past we designed (random-looking static)", fontsize=12)
            self.fig.canvas.draw_idle()
            self.plt.pause(2.2)
            for k in range(1, self.display_steps + 1):
                self.plt.pause(self.pace)
                im.set_data(self._strip([s.cpu().numpy() for s in states[k]]))
                self.ax_s.set_title(
                    f"automaton step {k}/{self.display_steps}", fontsize=12)
                self.fig.canvas.draw_idle()
            self.ax_s.set_title(
                f"the static became “{word}”. type another.", fontsize=12)
            self.fig.canvas.draw_idle()
        finally:
            self.busy = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", action="store_true",
                        help="learn the alphabet (run once; saves decoder)")
    parser.add_argument("--warm-epochs", type=int, default=4000)
    parser.add_argument("--word")
    parser.add_argument("--live-solve", metavar="CHAR",
                        help="solve a novel symbol live with the warm decoder")
    parser.add_argument("--noise", type=float, default=0.18,
                        help="fraction of seed pixels flipped to random "
                             "static (physics scrubs it; verified honest)")
    parser.add_argument("--steps", type=int, default=6,
                        help="display steps for the bloom")
    parser.add_argument("--pace", type=float, default=0.8,
                        help="seconds per automaton step in the animation")
    parser.add_argument("--play", action="store_true",
                        help="interactive window: type words, watch them "
                             "bloom; unknown symbols are solved live")
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

    if args.play:
        PlayWindow(bank, device, solve_epochs=args.solve_epochs,
                   noise=args.noise, display_steps=args.steps,
                   pace=args.pace)
        return

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
