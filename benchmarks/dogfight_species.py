"""DOGFIGHT AS SPECIES — every opponent a fitness function (2026-07-30).

The floor problem, solved by the library's own design instead of fitness
arithmetic. The mean aggregator abandoned matchups (the 1W/39L chaser
hole); the min aggregator destroyed selection variance. But fitness SHARES
already ARE the multi-objective mechanism: each species owns a permanent
1/K of the environment's fitness mass, so the population structurally
cannot abandon a niche — the guarantee both aggregators failed to provide.

Six species, one population, one decoder:
    chaser / dogfighter / ace / brawler   anti-script specialists
    field                                 vs the hall of fame (self-play)
    generalist                            vs everything; the SUBMISSION

And the reason to do it now: distillation (round thirty-one, 10/10,
t=+16.7) is a CROSS-PROBLEM consolidator. Specialists mine opponent-
specific tactics; the base absorbs what all of them keep discovering —
flying, aiming, not crashing; the generalist harvests a base that already
knows the shared skills. n_fns=6 turns `distill="auto"` on.

Substrate note: the pilot decoder is ~15M weights (64 genes -> 256 hidden
-> 58k outputs), too large for the sparse path's vmapped per-individual
weight copies, so this uses `directions="frozen"` — the VECTOR LoRA path.
That is not the substrate where distillation was falsified: the falsified
one was the conv decoder, whose MIXED conditioning feeds half of each
individual's coefficients in as decoder input the base cannot absorb. The
vector path's coefficients only gate directions; base(genes) is the whole
story, so the absorb target is coherent. This run is the test of that.

    PYTHONPATH=~/.local/gymlibs python3 -m benchmarks.dogfight_species --live
"""
from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from latentspace.universal import solve, register_architecture
from latentspace.universal.architectures import build_mlp
from benchmarks.dogfight_win import (OPPONENTS, WEIGHT_SCALE, build_template,
                                     make_writer, match_score, play)

HOF_DIR = Path("/tmp/dogfight_species_hof")
HOF_KEEP = 10
WORKERS = 4
SPECIES = list(OPPONENTS) + ["field", "generalist"]
GENERALIST = len(SPECIES) - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    model, inits, sizes, n_weights = build_template(args.hidden)
    write = make_writer(model, inits, sizes)
    print(f"pilot {n_weights} weights | {len(SPECIES)} species "
          f"(4 scripts + field + generalist) | distill auto-on | 180s",
          flush=True)

    if HOF_DIR.exists():
        shutil.rmtree(HOF_DIR)
    HOF_DIR.mkdir(parents=True)
    hof: list[Path] = []
    state = {"epoch": 0}
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    paths = [Path(f"/tmp/dogfight_spc_{k}.onnx") for k in range(WORKERS)]

    def matches_for(kind, base):
        """(opponent, seed) pairs for one evaluation of this species."""
        if kind in OPPONENTS:
            return [(kind, base + k) for k in range(6)]
        if kind == "field":
            rivals = hof[-2:] if hof else []
            if not rivals:                       # before the archive exists
                return [(o, base + k) for k, o in enumerate(OPPONENTS)]
            return [(str(r), base + k) for r in rivals for k in range(3)]
        # generalist: breadth over everything
        jobs = [(o, base + k) for k, o in enumerate(OPPONENTS)]
        if hof:
            jobs += [(str(hof[-1]), base + 7), (str(hof[-1]), base + 8)]
        return jobs

    def score_one(w, kind, base, slot):
        path = paths[slot % WORKERS]
        write(w, path)
        vals = []
        for opponent, seed in matches_for(kind, base):
            r = play(path, opponent, seed)
            if r is None:
                return -100.0
            vals.append(match_score(r))
        return float(np.mean(vals))

    def make_fn(kind):
        def fn(phenotypes):
            flat = phenotypes.reshape(len(phenotypes), -1).cpu().numpy()
            weights = (flat - 0.5) * (2.0 * WEIGHT_SCALE)
            base = 2000 + 61 * state["epoch"]
            jobs = [pool.submit(score_one, w, kind, base, i)
                    for i, w in enumerate(weights)]
            return torch.tensor([j.result() for j in jobs],
                                dtype=torch.float32)
        return fn

    fns = [make_fn(k) for k in SPECIES]

    def build(latent, output_shape, gain=10.0):
        net = build_mlp(latent, output_shape, hidden=args.hidden)
        with torch.no_grad():
            net[-1].weight.mul_(gain); net[-1].bias.mul_(gain)
        return net
    register_architecture("pilot-species", build)

    view = None
    if args.live:
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.canvas.manager.set_window_title("dogfight species")
        history = {name: [] for name in SPECIES}
        view = (plt, fig, ax, history)

    def progress(epoch, total, spent, phenos, scores):
        state["epoch"] = epoch
        if epoch % 8 == 0 and phenos[GENERALIST] is not None:
            entry = HOF_DIR / f"gen_{epoch:04d}.onnx"
            w = (np.asarray(phenos[GENERALIST]).reshape(-1) - 0.5) * 2.0
            write(w, entry)
            hof.append(entry)
            while len(hof) > HOF_KEEP:
                hof.pop(0).unlink(missing_ok=True)
        line = "  ".join(f"{n[:4]} {s:+.2f}" for n, s in zip(SPECIES, scores))
        print(f"  epoch {epoch:>4}/{total}  {spent:>6} matches-evals  {line}",
              flush=True)
        if view is None:
            return
        plt, fig, ax, history = view
        ax.clear()
        for name, s in zip(SPECIES, scores):
            history[name].append(s)
        for name in SPECIES:
            ax.plot(history[name], lw=2.2 if name == "generalist" else 1.0,
                    label=name)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_ylabel("best fitness per species (tournament-scored)")
        ax.set_title(f"six species, one decoder — distillation consolidates "
                     f"({len(hof)} in hall of fame)", fontsize=11)
        plt.pause(0.001)

    began = time.time()
    result = solve(fns, output_shape=(n_weights,), epochs=args.epochs,
                   architecture="pilot-species", seed=args.seed, device="cpu",
                   directions="frozen", children=48,
                   progress=progress, progress_every=4)
    print(f"\n{result.evaluations} evaluations, {time.time() - began:.0f}s")

    best = result.problems[GENERALIST].best_phenotype
    w = (np.asarray(best).reshape(-1) - 0.5) * 2.0
    out = Path(__file__).resolve().parent.parent / "demo" / \
        "dogfight_species.onnx"
    write(w, out)
    print("held-out: 40 unseen spawns per opponent (generalist champion)")
    tot, pts = [0, 0, 0], 0
    for opponent in OPPONENTS:
        rec = [0, 0, 0]
        for s in range(500, 540):
            r = play(out, opponent, s)
            if r is None:
                continue
            if r[3] == "Player0Win":
                rec[0] += 1; pts += 3 if r[4] == "Elimination" else 2
            elif r[3] == "Player1Win":
                rec[2] += 1
            else:
                rec[1] += 1; pts += 1
        tot = [t + x for t, x in zip(tot, rec)]
        print(f"  vs {opponent:<11} {rec[0]:>2}W /{rec[1]:>3}D /{rec[2]:>3}L",
              flush=True)
    print(f"  TOTAL         {tot[0]:>2}W /{tot[1]:>3}D /{tot[2]:>3}L  "
          f"{pts} tournament points", flush=True)
    rival = Path(__file__).resolve().parent.parent / "demo" / \
        "dogfight_selfplay.onnx"
    if rival.exists():
        w1 = d1 = l1 = 0
        for s in range(600, 640):
            r = play(out, str(rival), s)
            if r is None:
                continue
            w1 += r[3] == "Player0Win"; l1 += r[3] == "Player1Win"
            d1 += r[3] == "Draw"
        for s in range(600, 640):
            r = play(str(rival), out, s)
            if r is None:
                continue
            l1 += r[3] == "Player0Win"; w1 += r[3] == "Player1Win"
            d1 += r[3] == "Draw"
        print(f"  vs ELO-583 pilot (80 matches, sides swapped): "
              f"{w1}W/{d1}D/{l1}L", flush=True)
    print(f"saved {out}", flush=True)
    if view is not None:
        view[0].ioff(); view[0].show(block=True)


if __name__ == "__main__":
    main()
