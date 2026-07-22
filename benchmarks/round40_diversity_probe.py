"""Round 40: are the survivors actually distinct, and where does the distinctness live?

Round 39 failed to control mu from performance signals. Every rule had to
INFER whether a lineage would pay off later, and that evidence only exists
in hindsight — by the time TSP's plateaus prove diversity was needed, the
lineages were already culled during the early climb.

Diversity inverts that. Redundancy is measurable NOW: if two survivors are
near-clones, one is provably wasted whatever happens later. So a controller
could cull only demonstrated duplicates and never touch a distinct lineage.
That makes mu an OUTPUT — however many genuinely distinct lineages exist —
rather than a knob.

This round measures, it does not control. It runs the shipped explorer
(elite=16) and asks how distinct those 16 survivors really are.

WHICH distance is legal matters. Phenotype spread is out: it needs a metric
on the output space, which is more than a fitness score and is not universal
(L2 on TSP priorities is close to meaningless, and a real tour distance would
be hand-built for routing). Fitness spread is legal but blind exactly where
it counts — on a plateau many DIFFERENT tours share one length, so the spread
reads zero when diversity matters most. Genotype distance uses only tensors
evolution already owns and already perturbs: no output-space metric, no new
information, and identical for every modality.

Two channels, measured separately because they may disagree:

  * genome  — 64 floats, the code.
  * decoder — the weight vector, thousands of floats, the map.

Distances are normalized by ONE MUTATION STEP in that channel, which is the
constant-free threshold the round-39 designs lacked: two lineages closer
together than a single mutation can carry them are, operationally, the same
lineage. The gain controller already tracks the step size, so the threshold
rescales itself as the run anneals.

Reported per channel: mean pairwise distance in step units, and the EFFECTIVE
LINEAGE COUNT (greedy dedupe at the one-step threshold) — the mu a
diversity-driven controller would have produced, had it been running.

The prediction under test. Round 37 found the image elites sit 1.3% apart in
weight space by generation 4, so they should dedupe hard toward 1 — which
matches round 38, where a single champion was 1.9x better. The open question
is TSP: round 38 wants 16 survivors there. If TSP's elites are genuinely
distinct, a diversity rule can produce that 16 honestly. If TSP's elites are
near-clones too, then whatever makes many survivors help on tours is NOT
genotype diversity, and this whole approach cannot work — one cheap probe
instead of a full controller sweep.
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


def _image_loss(objective):
    def loss(phenos: torch.Tensor) -> np.ndarray:
        return objective.loss_tensor(phenos.flatten(1)).cpu().numpy()
    return loss


def _tsp_loss(cities: np.ndarray):
    cache: dict[str, torch.Tensor] = {}

    def loss(phenos: torch.Tensor) -> np.ndarray:
        pr = phenos.reshape(len(phenos), -1)
        key = str(pr.device)
        if key not in cache:
            cache[key] = torch.as_tensor(cities, device=pr.device,
                                         dtype=torch.float32)
        pts = cache[key][torch.argsort(pr, dim=1)]
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


def _pairwise(x: np.ndarray) -> np.ndarray:
    """Euclidean distances between every pair of rows."""
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    return d[np.triu_indices(len(x), k=1)]


def _effective_lineages(x: np.ndarray, step: float) -> int:
    """Greedy dedupe: how many rows are further than one mutation step from
    every row already accepted. Rows arrive best-first, so a survivor is only
    dropped in favour of something already proven better."""
    if step <= 0:
        return len(x)
    kept: list[np.ndarray] = []
    for row in x:
        if all(np.linalg.norm(row - k) > step for k in kept):
            kept.append(row)
    return len(kept)


def run(problem: str, budget: int, seed: int) -> dict:
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

    # Expected length of ONE mutation step in each channel. Genome: `rate` of
    # `latent` genes move by sigma*gain each. Decoder: every weight moves by
    # sigma_w*gain*scale, with sigma_w the geometric mean of the log-uniform
    # draw. Both scale with `gain`, so the thresholds anneal with the run.
    geo_sigma_w = float(np.sqrt(c.weight_sigma_low * c.weight_sigma_high))
    genome_step_1 = float(np.sqrt(c.genome_mutation_rate * LATENT)
                          * c.genome_mutation_sigma)

    while spent < budget:
        order = np.argsort(loss)[:c.elite]
        zs, thetas, loss = zs[order], thetas[order], loss[order]
        n = min(c.population, budget - spent)
        par = rng.integers(0, len(zs), n)
        cz = np.stack([mutate_z(zs[p]) for p in par])
        cth = np.stack([mutate_theta(thetas[p]) for p in par])
        cl = loss_fn(decode_all(cz, cth))
        spent += n
        wins = float((cl <= loss[par] + 1e-12).mean())
        gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, *c.gain_limits))
        zs = np.concatenate([zs, cz])
        thetas = np.concatenate([thetas, cth])
        loss = np.concatenate([loss, cl])

        # measure the CURRENT survivor set (what a controller would see)
        elite_idx = np.argsort(loss)[:c.elite]
        ez, et, el = zs[elite_idx], thetas[elite_idx], loss[elite_idx]
        w_scale = max(float(et.std()), 1e-3)
        genome_step = genome_step_1 * gain
        decoder_step = float(np.sqrt(template.n_params)) * geo_sigma_w * gain * w_scale
        gd, wd = _pairwise(ez), _pairwise(et)
        # joint space: each channel in its own step units, so neither the
        # 64-float genome nor the 7.5k-float decoder dominates by size alone
        joint = np.sqrt((gd / max(genome_step, 1e-12)) ** 2
                        + (wd / max(decoder_step, 1e-12)) ** 2)
        trace.append({
            "e": spent, "m": float(loss.min()), "gain": gain,
            "genome_steps": float(gd.mean() / max(genome_step, 1e-12)),
            "decoder_steps": float(wd.mean() / max(decoder_step, 1e-12)),
            "joint_steps": float(joint.mean()),
            "fitness_spread": float(el.std() / max(abs(el.mean()), 1e-12)),
            "mu_genome": _effective_lineages(ez, genome_step),
            "mu_decoder": _effective_lineages(et, decoder_step),
        })

    tail = trace[len(trace) // 2:]   # second half of the run: the settled regime
    return {"score": float(loss.min()),
            "genome_steps": float(np.mean([t["genome_steps"] for t in tail])),
            "decoder_steps": float(np.mean([t["decoder_steps"] for t in tail])),
            "joint_steps": float(np.mean([t["joint_steps"] for t in tail])),
            "fitness_spread": float(np.mean([t["fitness_spread"] for t in tail])),
            "mu_genome": float(np.mean([t["mu_genome"] for t in tail])),
            "mu_decoder": float(np.mean([t["mu_decoder"] for t in tail])),
            "trace": trace[::10]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+",
                        default=["blob2d", "smooth1d", "tsp100"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    print("survivor distances in MUTATION-STEP units (>1 = further apart "
          "than one mutation can carry them); mu_* = lineages surviving "
          "a one-step dedupe, out of 16", flush=True)
    for problem in args.problems:
        print(f"\n########## {problem} (budget {args.budget}) ##########",
              flush=True)
        for seed in args.seeds:
            out = run(problem, args.budget, seed)
            out.pop("trace")
            rows.append({"problem": problem, "seed": seed, **out})
            print(f"  seed {seed} score {out['score']:.6g} | genome "
                  f"{out['genome_steps']:.2f} steps -> mu {out['mu_genome']:.1f}"
                  f" | decoder {out['decoder_steps']:.2f} steps -> mu "
                  f"{out['mu_decoder']:.1f} | joint {out['joint_steps']:.2f}"
                  f" | fitness spread {out['fitness_spread']:.2e}", flush=True)

    print("\n--- means over seeds ---")
    for problem in args.problems:
        sel = [r for r in rows if r["problem"] == problem]
        print(f"  {problem:<10} genome {np.mean([r['genome_steps'] for r in sel]):6.2f} "
              f"steps (mu {np.mean([r['mu_genome'] for r in sel]):4.1f}) | "
              f"decoder {np.mean([r['decoder_steps'] for r in sel]):6.2f} steps "
              f"(mu {np.mean([r['mu_decoder'] for r in sel]):4.1f}) | joint "
              f"{np.mean([r['joint_steps'] for r in sel]):6.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "elite": ExplorerConfig().elite,
             "torch_version": torch.__version__, "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
