"""Round 39: let evolution choose how many survivors it keeps.

Round 38 measured the survivor count (elite = mu) and found the answer is
problem-dependent, which is exactly the kind of knob this project exists
to eliminate. Keeping ONE champion is 1.9x better on the image and 2.3x
better on the curve; keeping SIXTEEN is 1.25x better on 100-city TSP.
Smooth landscapes want a single lineage hill-climbing; the rugged
step-function tour landscape wants several bets exploring at once. A
universal solver cannot be told which of those it is looking at.

So mu becomes closed-loop, the same way the mutation gain already is. The
gain controller does not ask what step size a problem wants; it measures
whether mutations are landing (Rechenberg's 1/5th rule) and adjusts. This
round applies that one level up: measure whether the extra survivors are
EARNING their keep, and adjust mu.

The signal is RANK CREDIT. Every child has a parent with a known rank
inside the survivor set. Ask what fraction of the generation's genuine
improvements came from parents OTHER than the rank-0 champion, and
compare it to the fraction of children those parents were given in the
first place:

  * credit share > sampling share -> the non-champion lineages are
    out-performing their allocation. They are exploring somewhere the
    champion is not. Grow mu.
  * credit share < sampling share -> improvements are coming from the
    champion and the rest are spending evaluations on inferior copies.
    Shrink mu.

Comparing observed credit against the sampling share is what keeps this
constant-free: the null hypothesis ("all survivors are equally good") is
computed from the parent assignment itself, not from a fitted threshold.
The alternative signal — measure the tie/plateau rate as a ruggedness
proxy and map it to mu — needs exactly such a mapping constant, which is
the tuning knob wearing a disguise.

NOTE the tie convention is deliberately OPPOSITE to the gain
controller's. For step size, a tie must count as a SUCCESS: a tie means
the mutation changed nothing, which on a discrete phenotype is evidence
the step is too small, and counting ties as failures death-spirals (see
explorer.py). For survivor credit, a tie is precisely NOT evidence that a
lineage is contributing — a child that reproduces its parent's tour
exactly has told us nothing about whether its lineage deserves to live.
So this controller requires STRICT improvement. Same event, read as a
success by one controller and as non-evidence by the other; both readings
are correct.

Arms (pure decoder GA, win-rate step control, no distill/CMA):

  * fixed_1     — round 38's champion on image and curve: a (1 + 32) ES.
  * fixed_16    — round 38's champion on TSP, and the shipped default.
  * adaptive    — mu under rank-credit control, starting between the two.

The bar: `adaptive` must match the better fixed arm on EVERY problem
without being told which problem it is on. Matching the winner on one and
losing on the other is a failure — that is what a fixed knob already does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.round3_structure import SmoothTarget
from benchmarks.round8_mlp_pretrain import BlobImage2D
from benchmarks.round21_tsp import make_instance
from benchmarks.round25_anchor_field import AnchorFieldTransformer
from latentspace.universal.architectures import resolve
from benchmarks.legacy_engines.explorer import ExplorerConfig, _Template

LATENT = 64
TSP_CITIES = 100
MU_START = 4        # geometric middle of [1, 32]; the controller moves it
MU_STEP = 1.15      # same multiplicative rate as the gain controller
# Survivors are capped at half the population, the standard ES convention
# (Hansen's CMA-ES defaults to mu = lambda/2). Not a tuned value: at
# mu = population truncation keeps EVERYTHING and selection pressure
# vanishes, so an unbounded controller that wants more diversity just pins
# itself against the ceiling and drifts.


def _image_loss(objective):
    def loss(phenos: torch.Tensor) -> np.ndarray:
        return objective.loss_tensor(phenos.flatten(1)).cpu().numpy()
    return loss


def _tsp_loss(cities: np.ndarray):
    pts_cache: dict[str, torch.Tensor] = {}

    def loss(phenos: torch.Tensor) -> np.ndarray:
        pr = phenos.reshape(len(phenos), -1)
        key = str(pr.device)
        if key not in pts_cache:
            pts_cache[key] = torch.as_tensor(cities, device=pr.device,
                                             dtype=torch.float32)
        pts = pts_cache[key][torch.argsort(pr, dim=1)]
        return (pts - pts.roll(-1, dims=1)).norm(dim=2).sum(dim=1).cpu().numpy()
    return loss


def make_problem(name: str, seed: int):
    if name == "blob2d":
        return _image_loss(BlobImage2D()), (32, 32), "conv2d"
    if name == "smooth1d":
        return _image_loss(SmoothTarget()), (256,), "conv1d"
    if name == "tsp100":
        cities = make_instance(seed, TSP_CITIES)
        return (_tsp_loss(cities), (TSP_CITIES,),
                lambda latent, shape: AnchorFieldTransformer(latent, shape,
                                                             cities))
    raise ValueError(name)


def run(problem: str, arm: str, budget: int, seed: int) -> dict:
    loss_fn, output_shape, architecture = make_problem(problem, seed)
    c = ExplorerConfig()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = _Template(resolve(architecture, LATENT, output_shape), "mps")

    def decode_all(zs, thetas):
        return torch.stack([template.decode(t, z)
                            for z, t in zip(zs, thetas)])

    zs = rng.standard_normal((c.population, LATENT)).astype(np.float32)
    thetas = np.stack([template.init_theta(int(rng.integers(0, 2**31)))
                       for _ in range(c.population)])
    loss = loss_fn(decode_all(zs, thetas))
    spent, gain, trace = len(zs), 1.0, []
    mu_real = float(c.population // 2 if arm == "adaptive_anneal"
                    else MU_START if arm.startswith("adaptive")
                    else int(arm.split("_")[1]))

    def mutate_z(z):
        m = rng.random(z.shape) < c.genome_mutation_rate
        if not m.any():
            m[rng.integers(0, len(z))] = True
        return (z + m * rng.normal(0, c.genome_mutation_sigma * gain, z.shape)
                ).astype(np.float32)

    def mutate_theta(theta):
        sigma = float(np.exp(rng.uniform(np.log(c.weight_sigma_low),
                                         np.log(c.weight_sigma_high)))) * gain
        scale = max(float(theta.std()), 1e-3)
        return (theta + rng.normal(0, sigma * scale, theta.shape)
                ).astype(np.float32)

    while spent < budget:
        mu = int(np.clip(round(mu_real), 1, c.population // 2))
        order = np.argsort(loss)[:mu]
        zs, thetas, loss = zs[order], thetas[order], loss[order]
        n = min(c.population, budget - spent)
        # parent RANK is the index into the (already sorted) survivor set
        par = rng.integers(0, len(zs), n)
        cz = np.stack([mutate_z(zs[p]) for p in par])
        cth = np.stack([mutate_theta(thetas[p]) for p in par])
        cl = loss_fn(decode_all(cz, cth))
        spent += n

        # step-size control: ties count as SUCCESSES (see module docstring)
        wins = float((cl <= loss[par] + 1e-12).mean())
        gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, *c.gain_limits))

        # Survivor control. Both rules score a generation by whether any child
        # beat the GLOBAL CHAMPION — never by whether a child beat its own
        # parent. Per-parent wins are an easier bar the worse the parent is,
        # so they reward a lineage for being bad: measured, mu climbed to
        # 14-19 on the image, where 1 is 1.9x better.
        credit_share = sampling_share = float("nan")
        beat_champion = cl < loss[0] - 1e-12
        if arm == "adaptive_credit" and mu > 1 and beat_champion.any():
            # Rank credit: did non-champion survivors produce more than their
            # allocated share of the champion-beating children?
            non_champion = par > 0
            credit_share = float(non_champion[beat_champion].mean())
            sampling_share = float(non_champion.mean())
            mu_real *= (MU_STEP if credit_share > sampling_share
                        else 1 / MU_STEP)
        elif arm == "adaptive_anneal":
            # Monotone: mu starts wide and only ever contracts. Shrinking is
            # IRREVERSIBLE — culling from 16 survivors to 12 makes 4 lineages
            # extinct, and growing mu back later cannot resurrect them, it just
            # retains more copies of whoever is left. So a controller that
            # oscillates destroys the diversity it later asks for (measured:
            # the two-way rules reach mu=16 on TSP and still score 16.7 vs
            # 14.9 for a fixed 16, because each dip cost lineages the run
            # needed from generation one). Annealing one way avoids the trap:
            # every improvement over the champion is evidence the search is
            # still climbing and can afford to narrow.
            if beat_champion.any():
                mu_real /= MU_STEP
        elif arm == "adaptive_stall":
            # Stall response: climbing means intensify, stuck means diversify.
            # The rank-credit rule above has a structural downward bias — the
            # rank-0 parent IS the champion, so it starts from the best point
            # on the map and is the likeliest single parent to beat it. Other
            # survivors are handicapped by definition, credit share sits under
            # sampling share almost always, and mu collapses everywhere. This
            # rule asks only whether the search is still moving, which needs no
            # null hypothesis at all.
            mu_real *= (1 / MU_STEP if beat_champion.any() else MU_STEP)
        if arm.startswith("adaptive"):
            mu_real = float(np.clip(mu_real, 1.0, c.population // 2))

        zs = np.concatenate([zs, cz])
        thetas = np.concatenate([thetas, cth])
        loss = np.concatenate([loss, cl])
        trace.append({"e": spent, "m": float(loss.min()), "gain": gain,
                      "mu": mu, "win": wins, "credit": credit_share,
                      "sampling": sampling_share})

    mus = [t["mu"] for t in trace]
    return {"score": float(loss.min()), "final_gain": gain,
            "final_mu": mus[-1], "mean_mu": float(np.mean(mus)),
            "trace": trace[::10]}


ARMS = ("fixed_1", "fixed_16", "adaptive_credit", "adaptive_stall",
        "adaptive_anneal")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+",
                        default=["blob2d", "smooth1d", "tsp100"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for problem in args.problems:
        print(f"\n########## {problem} (budget {args.budget}) ##########",
              flush=True)
        for arm in args.arms:
            for seed in args.seeds:
                out = run(problem, arm, args.budget, seed)
                trace = out.pop("trace")
                rows.append({"problem": problem, "arm": arm, "seed": seed,
                             **out, "trace": trace})
                print(f"  {arm:<16} seed {seed} score {out['score']:.6g} "
                      f"(mu {out['mean_mu']:.1f} mean -> {out['final_mu']} "
                      f"final, gain {out['final_gain']:.2f})", flush=True)
            vals = [r["score"] for r in rows
                    if r["problem"] == problem and r["arm"] == arm]
            print(f"  {arm:<16} MEAN {np.mean(vals):.6g}", flush=True)

    print("\n--- summary (score; lower is better) ---")
    for problem in args.problems:
        fixed = [np.mean([r["score"] for r in rows
                          if r["problem"] == problem and r["arm"] == a])
                 for a in ("fixed_1", "fixed_16") if a in args.arms]
        best_fixed = min(fixed) if fixed else float("nan")
        for arm in args.arms:
            v = np.mean([r["score"] for r in rows
                         if r["problem"] == problem and r["arm"] == arm])
            print(f"  {problem:<10} {arm:<16} {v:.6g} "
                  f"({best_fixed/v:.3f}x vs best fixed arm)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "mu_start": MU_START, "mu_step": MU_STEP,
             "torch_version": torch.__version__, "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
