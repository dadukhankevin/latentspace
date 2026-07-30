"""DOGFIGHT SELF-PLAY — competition within the population (2026-07-29).

The external verdict on the previous pilot was 0 wins in 60 leaderboard
matches, ELO -200, against a 28% win rate on the scripted built-ins. The
diagnosis: beating four fixed scripted opponents teaches a pilot to beat
those four, and nothing about a competent adversary. Daniel's directive:
competition within the population, scripted opponents kept as the floor,
a larger network, other fixes as needed.

What each individual now faces per evaluation:
  - the four SCRIPTED opponents (the floor: never regress below them)
  - a sample of the HALL OF FAME: past population champions written to
    disk as ONNX. An archive rather than only the current best, because
    pure current-best co-evolution famously cycles (A beats B beats C
    beats A); the archive forces progress against the whole history.

Spawn seeds VARY BY EPOCH (all children within an epoch share them, so
ranking stays fair) — the fixed-seed overfitting of rounds eighteen and
twenty-two, converted into rotating pressure. Fitness is the tournament's
own 3/2/1/0 with the asymmetric time shaping.

Also fixed: the local sim ran 90-second matches while the site's replays
run to 3:00, and `ticks_remaining` is in the observation — the pilot's
sense of time was compressed 2x relative to the deployed game. The sim is
rebuilt at 180s.

    PYTHONPATH=~/.local/gymlibs python3 -m benchmarks.dogfight_selfplay --live
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

HOF_DIR = Path("/tmp/dogfight_hof")
HOF_KEEP = 12               # archive size
HOF_SAMPLE = 3              # opponents drawn from it per evaluation
SCRIPT_SEEDS = 4            # spawn seeds per scripted opponent
HOF_SEEDS = 3               # spawn seeds per hall-of-fame opponent
WORKERS = 4


def evaluate(weights, path, write, hof_paths, seed_base, mode="mean"):
    """Scripted floor + hall-of-fame sample, tournament scoring.

    mode="min" (Daniel, 2026-07-30): fitness is the MINIMUM over opponent
    GROUPS (each group's mean over its spawns) — maximin. Round
    twenty-eight measured why the average fails as a floor: the population
    paid for a 1W/39L chaser hole with hall-of-fame wins, because an
    average lets strong matchups subsidise abandoned ones. Under min, your
    worst matchup IS your fitness, and once it improves a different
    matchup becomes binding, so pressure rotates until nothing is
    neglected. Groups, not individual matches: a per-match min would make
    every early pilot's fitness "my worst spawn" = a loss, flattening the
    landscape into the plateau that kills this method (FINDINGS sixteen).
    """
    write(weights, path)
    groups, record = [], [0, 0, 0]
    for opponent in OPPONENTS:
        g = []
        for k in range(SCRIPT_SEEDS):
            r = play(path, opponent, seed_base + k)
            if r is None:
                return -100.0, record
            g.append(match_score(r))
            record[0 if r[3] == "Player0Win"
                   else (2 if r[3] == "Player1Win" else 1)] += 1
        groups.append(float(np.mean(g)))
    for rival in hof_paths:
        g = []
        for k in range(HOF_SEEDS):
            r = play(path, str(rival), seed_base + 100 + k)
            if r is None:
                continue                    # a corrupt archive entry
            g.append(match_score(r))
            record[0 if r[3] == "Player0Win"
                   else (2 if r[3] == "Player1Win" else 1)] += 1
        if g:
            groups.append(float(np.mean(g)))
    if mode == "min":
        return float(np.min(groups)), record
    return float(np.mean(groups)), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--fitness", default="min", choices=("mean", "min"))
    args = parser.parse_args()

    model, inits, sizes, n_weights = build_template(args.hidden)
    write = make_writer(model, inits, sizes)
    print(f"pilot {n_weights} weights (cap 250000) | scripted floor + "
          f"{HOF_SAMPLE} of {HOF_KEEP} hall-of-fame rivals | 180s matches | "
          f"fitness={args.fitness}", flush=True)

    if HOF_DIR.exists():
        shutil.rmtree(HOF_DIR)
    HOF_DIR.mkdir(parents=True)
    hof: list[Path] = []
    rng = np.random.default_rng(args.seed + 999)
    state = {"epoch": 0, "record": [0, 0, 0], "champion": None}
    paths = [Path(f"/tmp/dogfight_sp_{k}.onnx") for k in range(WORKERS)]
    pool = ThreadPoolExecutor(max_workers=WORKERS)

    def fitness(phenotypes):
        flat = phenotypes.reshape(len(phenotypes), -1).cpu().numpy()
        weights = (flat - 0.5) * (2.0 * WEIGHT_SCALE)
        # everyone in this epoch faces the same rivals and the same spawns;
        # both rotate across epochs.
        rivals = ([hof[-1]] + list(rng.choice(hof[:-1],
                                              min(HOF_SAMPLE - 1,
                                                  max(0, len(hof) - 1)),
                                              replace=False))
                  if hof else [])
        base = 1000 + 37 * state["epoch"]
        jobs = [pool.submit(evaluate, w, paths[i % WORKERS], write, rivals,
                            base, args.fitness)
                for i, w in enumerate(weights)]
        out = [j.result() for j in jobs]
        best = int(np.argmax([s for s, _ in out]))
        state["record"] = out[best][1]
        state["champion"] = weights[best].copy()
        return torch.tensor([s for s, _ in out], dtype=torch.float32)

    view = None
    if args.live:
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.ion()
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.5))
        fig.canvas.manager.set_window_title("dogfight self-play")
        view = (plt, fig, ax, ax2, [], [])

    def progress(epoch, total, spent, phenos, scores):
        state["epoch"] = epoch
        # every 8 epochs the current champion joins the hall of fame
        if epoch % 8 == 0 and state["champion"] is not None:
            entry = HOF_DIR / f"champ_{epoch:04d}.onnx"
            write(state["champion"], entry)
            hof.append(entry)
            while len(hof) > HOF_KEEP:
                old = hof.pop(0)
                old.unlink(missing_ok=True)
        w, d, l = state["record"]
        print(f"  epoch {epoch:>4}/{total}  {spent:>6} evals  "
              f"best {scores[0]:+.3f}  record {w}W/{d}D/{l}L  "
              f"hof {len(hof)}", flush=True)
        if view is None:
            return
        plt, fig, ax, ax2, xs, ys = view
        xs.append(spent); ys.append(scores[0])
        ax.clear(); ax.plot(xs, ys, color="#c2703a", lw=1.6)
        ax.set_ylabel("fitness (tournament-scored)")
        ax.set_title(f"self-play — {n_weights} weights, "
                     f"{len(hof)} pilots in the hall of fame", fontsize=11)
        ax2.clear()
        ax2.bar(["win", "draw", "loss"], [w, d, l],
                color=["#4d7fa3", "#bbb", "#c2703a"])
        ax2.set_ylabel("best pilot, this epoch's matches")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
        plt.pause(0.001)

    began = time.time()

    def build(latent, output_shape, gain=10.0):
        net = build_mlp(latent, output_shape)
        with torch.no_grad():
            net[-1].weight.mul_(gain); net[-1].bias.mul_(gain)
        return net
    register_architecture("pilot", build)

    result = solve(fitness, output_shape=(n_weights,), epochs=args.epochs,
                   architecture="pilot", seed=args.seed, device="cpu",
                   directions="frozen", population_cap=96, children=48,
                   progress=progress, progress_every=4)
    print(f"\n{result.evaluations} evaluations, {time.time() - began:.0f}s")

    # The objective drifts (rivals and spawns rotate), so "best ever" is
    # polluted by whichever epoch was easiest. Pick the final pilot by a
    # FIXED judgment suite: best-ever vs the last champion, scripted
    # opponents x 20 unseen spawns, tournament points.
    candidates = {
        "best-ever": (np.asarray(result.best_phenotype).reshape(-1) - 0.5)
        * 2.0 * WEIGHT_SCALE,
        "last-champion": state["champion"],
    }
    out = Path(__file__).resolve().parent.parent / "demo" / \
        f"dogfight_selfplay_{args.fitness}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    best_name, best_pts = None, -1
    for name, w in candidates.items():
        if w is None:
            continue
        write(w, paths[0])
        pts, rec = 0, [0, 0, 0]
        for opponent in OPPONENTS:
            for s in range(700, 720):
                r = play(paths[0], opponent, s)
                if r is None:
                    continue
                if r[3] == "Player0Win":
                    rec[0] += 1; pts += 3 if r[4] == "Elimination" else 2
                elif r[3] == "Player1Win":
                    rec[2] += 1
                else:
                    rec[1] += 1; pts += 1
        print(f"  {name:<13} {rec[0]:>2}W/{rec[1]:>3}D/{rec[2]:>3}L  "
              f"{pts} pts on the judgment suite")
        if pts > best_pts:
            best_pts = pts
            best_name = name
            write(w, out)
    print(f"saved {best_name} -> {out}")
    if view is not None:
        view[0].ioff(); view[0].show(block=True)


if __name__ == "__main__":
    main()
