"""Round 36: one shared decoder, per-individual low-rank modifications.

Daniel's proposal: instead of every individual carrying a full private
copy of the decoder weights, share ONE backbone and store only a small
per-individual modification — a LoRA-style adapter.

Reframed for what actually matters here. Memory is not the bottleneck
(7.5k-47k weights, ~1MB for a whole population); the SEARCH SPACE is.
Evolution currently perturbs ~7,500 numbers per individual and its sample
efficiency degrades with dimension — round 28 measured this directly: the
38k-weight anchor decoder was 2x WORSE than the 7.5k one, mutations
spread thinner climb slower. This round keeps the expressive backbone and
shrinks only what evolution must search.

Implementation is the flat-vector form rather than per-layer LoRA:

    weights = backbone + P @ adapter

with `backbone` one shared random initialization, `P` a fixed random
projection (n_params x rank) scaled so a unit adapter perturbs the
weights on the same scale as the backbone itself, and `adapter` the
per-individual vector of `rank` numbers. Individuals are then
(genome: 64, adapter: rank) instead of (genome: 64, weights: 7500).
This needs no per-layer surgery, so it works for ANY architecture — the
weights are already a flat vector — which keeps the universality
property. Same idea as Li et al.'s intrinsic-dimension subspaces.

Note this is NOT round 27's frozen decoder (CMA on frozen weights capped
at 0.0513 because capacity lives in the weights). The weights still
evolve here; they are merely compressed. That is distillation's
compression without distillation's ceiling.

Arms (pure decoder GA, win-rate step control, no distill/CMA):

  * full_weights   — the incumbent: private full weights per individual.
  * subspace_<r>   — shared backbone + rank-r adapter per individual.

Two risks the run is instrumented for: (1) error correlation — 32
individuals sharing one backbone inherit its blind spots together, which
is the campaign's second law (a teacher's value is the independence of
its errors) and the diagnosis behind the green-leaf failure; population
phenotype spread is logged to catch it. (2) An untrained network emits
near-flat output (round 29: std 0.002 vs target 0.246), so evolution's
first job is inflating the weights ~100x — a low-rank update has to be
able to express that.
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
from benchmarks.round26_anchor_universal import build_anchor
from latentspace.universal.architectures import resolve
from benchmarks.legacy_engines.explorer import ExplorerConfig, _Template

LATENT = 64

SETUPS = {
    "blob2d_anchor": (BlobImage2D, (32, 32), build_anchor),
    "blob2d_conv": (BlobImage2D, (32, 32), "conv2d"),
    "smooth1d_anchor": (SmoothTarget, (256,), build_anchor),
}


class Subspace:
    """weights = backbone + P @ adapter, with P a fixed random projection."""

    def __init__(self, template: _Template, rank: int, seed: int):
        self.backbone = template.init_theta(seed)
        scale = max(float(self.backbone.std()), 1e-3)
        rng = np.random.default_rng(seed + 7919)
        # Column scale sqrt(rank) keeps ||P @ a|| ~ backbone scale for a ~ N(0,1),
        # so a unit adapter perturbs the net meaningfully without destroying it.
        self.P = (rng.standard_normal((len(self.backbone), rank))
                  * (scale / np.sqrt(rank))).astype(np.float32)
        self.rank = rank

    def weights(self, adapter: np.ndarray) -> np.ndarray:
        return (self.backbone + self.P @ adapter).astype(np.float32)


def run(setup: str, arm: str, budget: int, seed: int) -> dict:
    factory, output_shape, architecture = SETUPS[setup]
    objective = factory()
    c = ExplorerConfig()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = _Template(resolve(architecture, LATENT, output_shape), "mps")
    subspace = None if arm == "full_weights" else Subspace(
        template, int(arm.split("_")[1]), seed)

    def decode_all(zs, params):
        thetas = (params if subspace is None
                  else [subspace.weights(a) for a in params])
        return torch.stack([template.decode(t, z)
                            for z, t in zip(zs, thetas)])

    zs = rng.standard_normal((c.population, LATENT)).astype(np.float32)
    if subspace is None:
        params = np.stack([template.init_theta(int(rng.integers(0, 2**31)))
                           for _ in range(c.population)])
    else:
        params = rng.standard_normal(
            (c.population, subspace.rank)).astype(np.float32)

    phenos = decode_all(zs, params)
    loss = objective.loss_tensor(phenos.flatten(1)).cpu().numpy()
    spent, gain, trace = len(zs), 1.0, []

    def mutate_z(z):
        m = rng.random(z.shape) < c.genome_mutation_rate
        if not m.any():
            m[rng.integers(0, len(z))] = True
        return (z + m * rng.normal(0, c.genome_mutation_sigma * gain, z.shape)
                ).astype(np.float32)

    def mutate_p(p):
        sigma = float(np.exp(rng.uniform(np.log(c.weight_sigma_low),
                                         np.log(c.weight_sigma_high)))) * gain
        # full weights scale by their own spread; adapters live at unit scale
        scale = max(float(p.std()), 1e-3) if subspace is None else 1.0
        return (p + rng.normal(0, sigma * scale, p.shape)).astype(np.float32)

    while spent < budget:
        order = np.argsort(loss)[:c.elite]
        zs, params, loss, phenos = zs[order], params[order], loss[order], phenos[order]
        n = min(c.population, budget - spent)
        par = rng.integers(0, len(zs), n)
        cz = np.stack([mutate_z(zs[p]) for p in par])
        cp = np.stack([mutate_p(params[p]) for p in par])
        cph = decode_all(cz, cp)
        cl = objective.loss_tensor(cph.flatten(1)).cpu().numpy()
        spent += n
        wins = float((cl <= loss[par] + 1e-12).mean())
        gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, *c.gain_limits))
        zs = np.concatenate([zs, cz])
        params = np.concatenate([params, cp])
        phenos = torch.cat([phenos, cph])
        loss = np.concatenate([loss, cl])
        # population phenotype spread: the error-independence canary
        spread = float(phenos[np.argsort(loss)[:c.elite]].flatten(1).std(dim=0).mean())
        trace.append({"e": spent, "m": float(loss.min()), "gain": gain,
                      "win": wins, "spread": spread})

    return {"mse": float(loss.min()), "final_gain": gain,
            "evolved_dims": LATENT + (len(template.init_theta(0))
                                      if subspace is None else subspace.rank),
            "mean_spread": float(np.mean([t["spread"] for t in trace])),
            "trace": trace[::10]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setups", nargs="+", choices=SETUPS,
                        default=["blob2d_anchor"])
    parser.add_argument("--arms", nargs="+",
                        default=["full_weights", "subspace_16",
                                 "subspace_64", "subspace_256"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for setup in args.setups:
        print(f"\n########## {setup} (budget {args.budget}) ##########",
              flush=True)
        for arm in args.arms:
            for seed in args.seeds:
                out = run(setup, arm, args.budget, seed)
                trace = out.pop("trace")
                rows.append({"setup": setup, "arm": arm, "seed": seed,
                             **out, "trace": trace})
                print(f"  {arm:<14} seed {seed} mse {out['mse']:.6g} "
                      f"(evolved dims {out['evolved_dims']}, spread "
                      f"{out['mean_spread']:.4f}, final gain "
                      f"{out['final_gain']:.2f})", flush=True)
            vals = [r["mse"] for r in rows
                    if r["arm"] == arm and r["setup"] == setup]
            print(f"  {arm:<14} MEAN {np.mean(vals):.6g}", flush=True)

    print("\n--- summary ---")
    for setup in args.setups:
        base = np.mean([r["mse"] for r in rows
                        if r["setup"] == setup and r["arm"] == "full_weights"])
        for arm in args.arms:
            v = [r["mse"] for r in rows
                 if r["setup"] == setup and r["arm"] == arm]
            d = [r["evolved_dims"] for r in rows
                 if r["setup"] == setup and r["arm"] == arm][0]
            print(f"  {setup:<16} {arm:<14} {np.mean(v):.6g} "
                  f"({base/np.mean(v):.2f}x vs full, {d} evolved dims)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "latent": LATENT,
             "torch_version": torch.__version__, "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
