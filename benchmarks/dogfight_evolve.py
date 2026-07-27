"""Dogfight — evolving a fighter pilot for someone else's benchmark (2026-07-26).

Feasibility probe. The Paradigm "Dogfight" challenge asks for a stateless
ONNX network: 224 floats of battlefield state in (four stacked frames), three
floats out (turn, throttle, shoot), 250,000 parameters maximum, judged by Elo
against everyone else's. The simulator is public and runs locally, so
evaluations are free.

Why this problem is worth the campaign's time:
  - It is black box in the strict sense. Bullets hit or they do not, the
    shoot output is thresholded, the match returns a result. No derivative
    of "did I win" with respect to my weights exists.
  - Feedback is DENSE, which is the condition MountainCarContinuous failed
    (FINDINGS round sixteen): HP differential and hits landed both vary
    across random pilots, so selection has a gradient of improvement to
    climb rather than a plateau with a cliff.
  - It tests the thesis's untested half. The decoder's whole claim is that
    it searches the same 64 genes however large the solution is — measured
    on TSP, where margins widened with city count, but never on a big
    weight vector. Here the pilot is ~14.6k weights and the cap allows
    250k, so this is a direct test of indirect encoding against dimension.
  - It is judged by someone else. Every benchmark in FINDINGS is one we
    chose ourselves.

Setup (the simulator lives outside this repo):
    git clone https://github.com/benedictbrady/dogfight-challenge
    cd dogfight-challenge && cargo build --release
    PYTHONPATH=~/.local/gymlibs python3 -m benchmarks.dogfight_evolve

ONNX is written by rewriting the weight tensors of a template graph
(0.11 ms) rather than re-exporting from torch (3.8 s) — a 34,000x
difference that decides whether this is runnable at all.
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
TEMPLATE = Path("/tmp/dogfight_template.onnx")
PARAMS = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
OPPONENTS = ("chaser", "dogfighter", "ace", "brawler")
WEIGHT_SCALE = 1.0
HIDDEN = 64
WORKERS = 4                      # parallel matches; kept low deliberately
_STATS = re.compile(r"HP=(\d+), Hits=(\d+), Shots=(\d+)")


class Pilot(nn.Module):
    """The sim's output ranges are baked into the graph so every output is
    smooth and in range. Saturated outputs would make most weight changes
    invisible to selection, which is the same failure as a decoder that
    emits a near-constant phenotype (FINDINGS round fourteen)."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(224, hidden)
        self.fc2 = nn.Linear(hidden, 3)

    def forward(self, x):
        y = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        yaw, throttle, shoot = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        return torch.cat([yaw, (throttle + 1.0) * 0.5, shoot], dim=1)


def build_template():
    """Export the graph once; afterwards only its weights are rewritten."""
    if not TEMPLATE.exists():
        torch.onnx.export(Pilot().eval(), torch.zeros(1, 224), str(TEMPLATE),
                          input_names=["obs"], output_names=["act"],
                          opset_version=17)
    import onnx
    model = onnx.load(str(TEMPLATE))
    inits = {i.name: i for i in model.graph.initializer}
    sizes = [(name, int(np.prod(inits[name].dims))) for name in PARAMS]
    return model, inits, sizes, sum(n for _, n in sizes)


MODEL, INITS, SIZES, N_WEIGHTS = build_template()


def write_onnx(weights, path):
    offset = 0
    for name, count in SIZES:
        INITS[name].raw_data = weights[offset:offset + count] \
            .astype(np.float32).tobytes()
        offset += count
    with open(path, "wb") as handle:
        handle.write(MODEL.SerializeToString())


def play(path, opponent, seed, randomize=True):
    """One match. Returns None if the sim rejected the model.

    `--randomize` is NOT optional. Without it the seed changes nothing: 40
    matches at 40 different seeds came back byte-identical (HP differential
    sd 0.00). A run trained that way optimizes ONE fixed starting position
    per opponent, and the first probe did exactly that — it went 160/160
    against the built-in opponents on fixed spawns and collapsed to 7W/30D/3L
    against the weakest of them once spawns varied. Fixed SEEDS with varied
    spawns keeps the objective deterministic per individual while still
    spanning real scenarios; unseen seeds then measure generalization.
    """
    out = subprocess.run(
        [str(BIN), "run", "--p0", str(path), "--p1", opponent,
         "--seed", str(seed)] + (["--randomize"] if randomize else []),
        capture_output=True, text=True, cwd=str(REPO)).stdout
    found = _STATS.findall(out)
    if len(found) != 2:
        return None
    (hp, hits, shots), (opp_hp, opp_hits, _) = found
    return (int(hp), int(hits), int(shots), int(opp_hp), int(opp_hits))


def pilot_fitness(weights, path, opponents, seeds):
    """HP differential is the objective; hits landed is a finer-grained
    tie-breaker so that pilots which never connect can still be ranked
    against each other while the population is still bad."""
    write_onnx(weights, path)
    total = 0.0
    for opponent in opponents:
        for seed in seeds:
            result = play(path, opponent, seed)
            if result is None:
                return -100.0
            hp, hits, shots, opp_hp, _ = result
            total += (hp - opp_hp) + 0.3 * hits - 0.02 * max(0, shots - 40)
    return total / (len(opponents) * len(seeds))


def make_fitness(opponents, seeds):
    paths = [Path(f"/tmp/dogfight_worker_{k}.onnx") for k in range(WORKERS)]
    pool = ThreadPoolExecutor(max_workers=WORKERS)

    def fitness(phenotypes):
        flat = phenotypes.reshape(len(phenotypes), -1).cpu().numpy()
        weights = (flat - 0.5) * (2.0 * WEIGHT_SCALE)
        jobs = [pool.submit(pilot_fitness, w, paths[i % WORKERS],
                            opponents, seeds) for i, w in enumerate(weights)]
        return torch.tensor([j.result() for j in jobs], dtype=torch.float32)
    return fitness


def register_pilot_decoder(gain=10.0):
    def build(latent, output_shape):
        net = build_mlp(latent, output_shape)
        with torch.no_grad():
            net[-1].weight.mul_(gain)
            net[-1].bias.mul_(gain)
        return net
    register_architecture("pilot", build)
    return "pilot"


def baseline(opponents, seeds, draws, seed):
    """Random draws through the same decoder, batched through the same
    parallel scorer so the budgets are honestly matched. This is the control
    that found the search adds nothing on sparse reward (round sixteen) and
    8.1x on images (round seventeen)."""
    from latentspace.universal.conditional import build_conditional_decoder
    rng = np.random.default_rng(seed)
    decoder = build_conditional_decoder("pilot", 64, (N_WEIGHTS,), 64, "cpu")
    fitness = make_fitness(opponents, seeds)
    best, done = -np.inf, 0
    while done < draws:
        n = min(48, draws - done)
        genes = rng.standard_normal((n, 64)).astype(np.float32)
        latents = rng.standard_normal((n, 64)).astype(np.float32)
        phenos = torch.as_tensor(np.asarray(decoder.decode(genes, latents)))
        best = max(best, float(fitness(phenos).max()))
        done += n
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--opponents", default="chaser")
    parser.add_argument("--seeds", type=int, default=8,
                        help="spawn seeds per opponent. The first probe used "
                             "3, giving 12 scenarios per individual, and the "
                             "pilot memorised them: training +1.350 against "
                             "held-out -0.09. This is the binding constraint "
                             "here, and it costs linearly.")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--baseline", type=int, default=0,
                        help="also spend this many random draws as a control")
    args = parser.parse_args()

    if not BIN.exists():
        raise SystemExit(f"simulator not built: {BIN}\n"
                         f"  cd {REPO} && cargo build --release")
    opponents = tuple(args.opponents.split(","))
    seeds = tuple(range(1, args.seeds + 1))
    arch = register_pilot_decoder()
    print(f"pilot: {N_WEIGHTS} weights (cap 250000) | opponents "
          f"{', '.join(opponents)} | {len(seeds)} seeds each")

    began = time.time()

    def progress(epoch, total, spent, phenos, scores):
        print(f"  epoch {epoch:>4}/{total}  {spent:>6} evals  "
              f"best {scores[0]:+.3f}  ({time.time() - began:.0f}s)",
              flush=True)

    result = solve(make_fitness(opponents, seeds), output_shape=(N_WEIGHTS,),
                   epochs=args.epochs, architecture=arch, seed=args.seed,
                   device="cpu", population_cap=96, children=48,
                   progress=progress,
                   progress_every=max(1, args.epochs // 12))
    print(f"\nevolved best {result.best_fitness:+.3f} in "
          f"{result.evaluations} evaluations, {time.time() - began:.0f}s")

    best = (np.asarray(result.best_phenotype).reshape(-1) - 0.5) * 2.0
    out = Path(__file__).resolve().parent.parent / "demo" / "dogfight_best.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_onnx(best, out)
    print(f"saved {out}")
    print("\nheld-out: 40 UNSEEN spawn seeds per opponent")
    for opponent in OPPONENTS:
        got = [g for g in (play(out, opponent, s) for s in range(500, 540)) if g]
        diffs = [hp - ohp for hp, _, _, ohp, _ in got]
        won = sum(1 for d in diffs if d > 0)
        lost = sum(1 for d in diffs if d < 0)
        print(f"  vs {opponent:<11} {won:>2}W /{len(got) - won - lost:>3}D /"
              f"{lost:>3}L   HP diff {np.mean(diffs):+.2f}")

    if args.baseline:
        control = baseline(opponents, seeds, args.baseline, args.seed)
        print(f"\ncontrol: best of {args.baseline} random draws through the "
              f"same decoder = {control:+.3f}")


if __name__ == "__main__":
    main()
