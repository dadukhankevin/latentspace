"""DOGFIGHT, aiming at WINS — and the thesis test (2026-07-27).

Two changes from benchmarks/dogfight_evolve.py, both from measured failures:

FITNESS, in two revisions, both from watching a run fail.
v1 scored HP differential: a draw paid 0 and a loss paid negative, so
"disengage and never get hit" was the best average policy and the pilot
drew 143 of 160 held-out matches. v2 weighted wins — and draws still
stalled at half of all matches, because a draw paying 0 against an
elimination death paying -3 left passivity a cheap refuge.
v3 (current): A DRAW SCORES AS A LOSS, and time enters ASYMMETRICALLY —
losing slowly beats losing fast, winning fast beats winning slowly. A flat
survival reward would pay a pilot to run out the clock, which is the exact
behaviour being bred out. Dense terms (HP differential, hits) stay on top:
a pure win/loss signal is sparse, and sparse feedback is measured to
degenerate this method into random sampling (FINDINGS sixteen).

THE COMPARISON. `--arm direct` evolves the pilot's 14,595 weights straight,
which is classic neuroevolution — no decoder, no genes, the individual IS
the network. `--arm decoder` is this library. Matched evaluations. This is
the thesis test the campaign has never run on an outside problem: the claim
for an indirect encoding is that its advantage GROWS with network size,
because it searches the same 64 genes whether the pilot has 14k weights or
250k (the challenge's cap). `--hidden` steps the pilot size.

    PYTHONPATH=~/.local/gymlibs python3 -m benchmarks.dogfight_win --live
    PYTHONPATH=~/.local/gymlibs python3 -m benchmarks.dogfight_win --arm direct
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from latentspace.universal import solve, register_architecture
from latentspace.universal.architectures import build_mlp

REPO = Path.home() / "Documents" / "dogfight-challenge"
BIN = REPO / "target" / "release" / "dogfight"
OPPONENTS = ("chaser", "dogfighter", "ace", "brawler")
WEIGHT_SCALE = 1.0
WORKERS = 4
_STATS = re.compile(r"HP=(\d+), Hits=(\d+), Shots=(\d+)")
_OUT = re.compile(r"Outcome:\s+(\w+)")
_WHY = re.compile(r"Reason:\s+(\w+)")
_TICK = re.compile(r"Final tick:\s+(\d+)")
MAX_TICKS = 10800          # 90s at 120Hz


class Pilot(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(224, hidden)
        self.fc2 = nn.Linear(hidden, 3)

    def forward(self, x):
        y = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        yaw, thr, shoot = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        return torch.cat([yaw, (thr + 1.0) * 0.5, shoot], dim=1)


def build_template(hidden):
    import onnx
    path = Path(f"/tmp/dogfight_tmpl_{hidden}.onnx")
    if not path.exists():
        torch.onnx.export(Pilot(hidden).eval(), torch.zeros(1, 224),
                          str(path), input_names=["obs"],
                          output_names=["act"], opset_version=17)
    model = onnx.load(str(path))
    inits = {i.name: i for i in model.graph.initializer}
    names = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
    sizes = [(n, int(np.prod(inits[n].dims))) for n in names]
    return model, inits, sizes, sum(n for _, n in sizes)


def make_writer(model, inits, sizes):
    def write(weights, path):
        off = 0
        for name, count in sizes:
            inits[name].raw_data = weights[off:off + count] \
                .astype(np.float32).tobytes()
            off += count
        with open(path, "wb") as fh:
            fh.write(model.SerializeToString())
    return write


def play(path, opponent, seed):
    out = subprocess.run(
        [str(BIN), "run", "--p0", str(path), "--p1", opponent,
         "--seed", str(seed), "--randomize"],
        capture_output=True, text=True, cwd=str(REPO)).stdout
    stats, who, why = _STATS.findall(out), _OUT.search(out), _WHY.search(out)
    tick = _TICK.search(out)
    if len(stats) != 2 or who is None:
        return None
    (hp, hits, _), (ohp, _, _) = stats
    return (int(hp), int(hits), int(ohp), who.group(1),
            why.group(1) if why else "",
            int(tick.group(1)) if tick else MAX_TICKS)


def match_score(result):
    """A DRAW SCORES AS A LOSS (Daniel, 2026-07-27): under the previous
    scheme a draw paid 0 while dying paid -3, so "never engage" stayed a
    cheap local optimum and draws stalled at half of all matches. A draw is
    a failure to win, and pricing it as one removes the passive refuge.

    TIME is used ASYMMETRICALLY, which is the only way it does not reward
    running away: when LOSING, surviving longer is better (dying at 80s
    beats dying at 10s — dense gradient out of the die-instantly basin);
    when WINNING, killing faster is better. A flat survival reward would
    pay a pilot to disengage for the full 90 seconds, which is precisely
    the behaviour being bred out."""
    hp, hits, ohp, who, why, ticks = result
    frac = min(1.0, ticks / MAX_TICKS)
    # THE TOURNAMENT'S OWN SCORING (RULES.md): elimination win 3, HP win 2,
    # DRAW 1, loss 0. Pricing a draw at zero (the previous revision) was
    # measured catastrophic against this metric — held-out 9W/34D/117L = 61
    # tournament points, against the passive HP-differential pilot's
    # 5W/143D/12L = 158. A draw is worth a third of an elimination win here,
    # so teaching the pilot to throw drawable matches costs more than the
    # extra wins earn. Align the objective with the scoreboard.
    if who == "Player0Win":
        base = 3.0 if why == "Elimination" else 2.0
        base += 0.3 * (1.0 - frac)            # finish it quickly
    elif who == "Player1Win":
        base = 0.0 + 0.3 * frac               # at least last longer
    else:
        base = 1.0                            # a draw scores as the rules say
    return base + 0.1 * (hp - ohp) + 0.02 * hits


def evaluate(weights, path, write, opponents, seeds):
    write(weights, path)
    total, record = 0.0, [0, 0, 0]
    for opponent in opponents:
        for seed in seeds:
            r = play(path, opponent, seed)
            if r is None:
                return -100.0, record
            total += match_score(r)
            record[0 if r[3] == "Player0Win"
                   else (2 if r[3] == "Player1Win" else 1)] += 1
    return total / (len(opponents) * len(seeds)), record


def make_fitness(write, n_weights, opponents, seeds, tracker):
    paths = [Path(f"/tmp/dogfight_win_{k}.onnx") for k in range(WORKERS)]
    pool = ThreadPoolExecutor(max_workers=WORKERS)

    def fitness(phenotypes):
        flat = phenotypes.reshape(len(phenotypes), -1).cpu().numpy()
        weights = (flat - 0.5) * (2.0 * WEIGHT_SCALE)
        jobs = [pool.submit(evaluate, w, paths[i % WORKERS], write,
                            opponents, seeds)
                for i, w in enumerate(weights)]
        out = [j.result() for j in jobs]
        best = int(np.argmax([s for s, _ in out]))
        tracker["record"] = out[best][1]
        return torch.tensor([s for s, _ in out], dtype=torch.float32)
    return fitness


def direct_search(fitness_raw, n_weights, budget, seed, report):
    """Classic neuroevolution: the individual IS the weight vector."""
    rng = np.random.default_rng(seed)
    mu, lam, sigma = 12, 48, 0.3
    pop = rng.normal(0, 0.3, (mu, n_weights))
    scores = fitness_raw(pop)
    spent = mu
    best_i = int(np.argmax(scores))
    best, best_s = pop[best_i].copy(), float(scores[best_i])
    while spent + lam <= budget:
        parents = rng.integers(0, mu, lam)
        kids = pop[parents] + sigma * rng.normal(0, 1, (lam, n_weights))
        ks = fitness_raw(kids)
        spent += lam
        allp = np.concatenate([pop, kids])
        alls = np.concatenate([scores, ks])
        keep = np.argsort(-alls)[:mu]
        pop, scores = allp[keep], alls[keep]
        if scores[0] > best_s:
            best_s, best = float(scores[0]), pop[0].copy()
        sigma *= 1.05 if ks.max() > best_s - 1e-9 else 1 / 1.05
        sigma = float(np.clip(sigma, 0.01, 1.5))
        report(spent, best_s)
    return best, best_s, spent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="decoder",
                        choices=("decoder", "direct"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    model, inits, sizes, n_weights = build_template(args.hidden)
    write = make_writer(model, inits, sizes)
    seeds = tuple(range(1, args.seeds + 1))
    tracker = {"record": [0, 0, 0]}
    fitness = make_fitness(write, n_weights, OPPONENTS, seeds, tracker)
    print(f"pilot {n_weights} weights (cap 250000) | arm {args.arm} | "
          f"{len(OPPONENTS)} opponents x {len(seeds)} spawns", flush=True)

    view = None
    if args.live:
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.ion()
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.5))
        fig.canvas.manager.set_window_title(f"dogfight — {args.arm}")
        view = (plt, fig, ax, ax2, [], [])

    def show(spent, best):
        w, d, l = tracker["record"]
        print(f"  {spent:>6} evals   best {best:+.3f}   "
              f"best-pilot record {w}W/{d}D/{l}L", flush=True)
        if view is None:
            return
        plt, fig, ax, ax2, xs, ys = view
        xs.append(spent); ys.append(best)
        ax.clear(); ax.plot(xs, ys, color="#c2703a", lw=1.6)
        ax.set_ylabel("fitness (win-weighted)")
        ax.axhline(0, color="#999", lw=0.8, ls=":")
        ax.set_title(f"{args.arm} — {n_weights} pilot weights", fontsize=11)
        ax2.clear()
        ax2.bar(["win", "draw", "loss"], [w, d, l],
                color=["#4d7fa3", "#bbb", "#c2703a"])
        ax2.set_ylabel(f"best pilot, {len(OPPONENTS) * len(seeds)} training matches")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
        plt.pause(0.001)

    began = time.time()
    if args.arm == "decoder":
        def build(latent, output_shape, gain=10.0):
            net = build_mlp(latent, output_shape)
            with torch.no_grad():
                net[-1].weight.mul_(gain); net[-1].bias.mul_(gain)
            return net
        register_architecture("pilot", build)

        def progress(epoch, total, spent, phenos, scores):
            show(spent, scores[0])

        result = solve(fitness, output_shape=(n_weights,), epochs=args.epochs,
                       architecture="pilot", seed=args.seed, device="cpu",
                       directions="frozen", population_cap=96, children=48,
                       progress=progress, progress_every=5)
        best = (np.asarray(result.best_phenotype).reshape(-1) - 0.5) * 2.0
        spent, score = result.evaluations, result.best_fitness
    else:
        def raw(batch):
            return fitness(torch.as_tensor(batch / (2.0 * WEIGHT_SCALE) + 0.5,
                                           dtype=torch.float32)).numpy()
        budget = args.epochs * 48
        best, score, spent = direct_search(raw, n_weights, budget, args.seed,
                                           lambda s, b: show(s, b)
                                           if s % 480 < 48 else None)

    print(f"\n{args.arm}: {score:+.3f} in {spent} evaluations "
          f"({time.time() - began:.0f}s)")
    out = Path(__file__).resolve().parent.parent / "demo" / \
        f"dogfight_win_{args.arm}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    write(best, out)
    print("HELD-OUT: 40 unseen spawn seeds per opponent")
    totals = [0, 0, 0]
    for opponent in OPPONENTS:
        rec = [0, 0, 0]
        for s in range(500, 540):
            r = play(out, opponent, s)
            if r is None:
                continue
            rec[0 if r[3] == "Player0Win"
                else (2 if r[3] == "Player1Win" else 1)] += 1
        totals = [t + x for t, x in zip(totals, rec)]
        print(f"  vs {opponent:<11} {rec[0]:>2}W /{rec[1]:>3}D /{rec[2]:>3}L")
    print(f"  TOTAL        {totals[0]:>2}W /{totals[1]:>3}D /{totals[2]:>3}L"
          f"   win rate {totals[0] / max(1, sum(totals)):.1%}")
    if view is not None:
        view[0].ioff(); view[0].show(block=True)


if __name__ == "__main__":
    main()
