"""Round 29: does a fixed mutation sigma hold the phenotype step steady?

Daniel's proposal: keep mutating exactly the same things (genome and
decoder weights, same operators), but set the mutation MAGNITUDE from
measured phenotype displacement against a target, instead of from
hand-set constants in parameter space.

Before building that controller, this round measures whether the premise
holds. The explorer currently mutates weights at a nominal sigma sampled
log-uniformly from [0.003, 0.02] and already rescales it by theta.std(),
and mutates genes at a fixed rate 0.1 / sigma 0.12. Those constants carry
a comment recording that they were measured on this campaign's own
benchmarks ("43% of mutant outputs beat their parent at 0.003; ~0% at
0.03+") — the last hand-tuned problem knowledge in the explorer.

The question: over a run, does a FIXED nominal sigma keep producing the
same phenotype displacement? If yes, fixed sigma already controls the
step and the controller buys little. If the sensitivity drifts, then the
real phenotype step size has been wandering uncontrolled the whole time,
and targeting displacement is the fix.

Method: run ordinary per-individual evolution (mutation code mirrored
from `benchmarks.legacy_engines.explorer` exactly, including the theta.std()
rescale). Every `--probe-every` generations, freeze the current best
individual and apply each channel's mutation ALONE at fixed nominal
sigmas, decoding K probes per setting and recording the RMS phenotype
displacement from the parent. Probes call the DECODER only, never the
fitness function, so they cost zero evaluations — which is the whole
point of Daniel's idea: displacement is free to measure in the currency
that is actually scarce.

Reported per probe point: RMS displacement, sensitivity (displacement per
unit nominal sigma), theta.std(), and the best loss so far.
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
GENOME_PROBE_SIGMAS = (0.12,)
WEIGHT_PROBE_SIGMAS = (0.003, 0.02)

SETUPS = {
    "blob2d_anchor": (BlobImage2D, (32, 32), build_anchor),
    "blob2d_conv": (BlobImage2D, (32, 32), "conv2d"),
    "smooth1d_anchor": (SmoothTarget, (256,), build_anchor),
    "smooth1d_conv": (SmoothTarget, (256,), "conv1d"),
}


def _mutate_genome(z, rng, c, sigma=None):
    """Mirrors explorer._mutate_genome, with an overridable sigma."""
    sigma = c.genome_mutation_sigma if sigma is None else sigma
    mask = rng.random(z.shape) < c.genome_mutation_rate
    if not mask.any():
        mask[rng.integers(0, len(z))] = True
    return (z + mask * rng.normal(0, sigma, z.shape)).astype(np.float32)


def _mutate_weights(theta, rng, c, sigma=None):
    """Mirrors explorer._mutate_weights, including the theta.std() rescale."""
    if sigma is None:
        sigma = float(np.exp(rng.uniform(np.log(c.weight_sigma_low),
                                         np.log(c.weight_sigma_high))))
    scale = max(float(theta.std()), 1e-3)
    return (theta + rng.normal(0, sigma * scale, theta.shape)).astype(np.float32)


def _rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(((a - b) ** 2).mean()))


def probe(template, z, theta, rng, config, samples: int) -> dict:
    """Phenotype displacement from mutating ONE channel at a fixed nominal
    sigma. Decoder forwards only — no fitness evaluations are spent."""
    parent = template.decode(theta, z)
    out = {}
    for sigma in GENOME_PROBE_SIGMAS:
        moves = [_rms(template.decode(theta, _mutate_genome(z, rng, config, sigma)),
                      parent) for _ in range(samples)]
        out[f"genome@{sigma}"] = float(np.mean(moves))
    for sigma in WEIGHT_PROBE_SIGMAS:
        moves = [_rms(template.decode(_mutate_weights(theta, rng, config, sigma), z),
                      parent) for _ in range(samples)]
        out[f"weights@{sigma}"] = float(np.mean(moves))
    return out


def run(setup: str, budget: int, seed: int, probe_every: int,
        samples: int) -> list[dict]:
    factory, output_shape, architecture = SETUPS[setup]
    objective = factory()
    config = ExplorerConfig()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    builder = resolve(architecture, LATENT, output_shape)
    template = _Template(builder, "mps")

    def losses_of(zs, thetas) -> np.ndarray:
        phenotypes = torch.stack([template.decode(t, z)
                                  for z, t in zip(zs, thetas)])
        return objective.loss_tensor(phenotypes.flatten(1)).cpu().numpy()

    zs = rng.standard_normal((config.population, LATENT)).astype(np.float32)
    thetas = np.stack([template.init_theta(int(rng.integers(0, 2**31)))
                       for _ in range(config.population)])
    loss = losses_of(zs, thetas)
    spent = len(zs)
    rows, generation = [], 0

    while spent < budget:
        if generation % probe_every == 0:
            best = int(np.argmin(loss))
            measured = probe(template, zs[best], thetas[best], rng, config,
                             samples)
            row = {"generation": generation, "evaluations": spent,
                   "best_loss": float(loss.min()),
                   "theta_std": float(thetas[best].std()), **measured}
            rows.append(row)
            print(f"  gen {generation:>4} evals {spent:>5} "
                  f"loss {row['best_loss']:.5f} theta_std {row['theta_std']:.4f} "
                  + " ".join(f"{k} {v:.4f}" for k, v in measured.items()),
                  flush=True)

        order = np.argsort(loss)[:config.elite]
        zs, thetas, loss = zs[order], thetas[order], loss[order]
        n = min(config.population, budget - spent)
        parents = rng.integers(0, len(zs), n)
        child_z = np.stack([_mutate_genome(zs[p], rng, config) for p in parents])
        child_theta = np.stack([_mutate_weights(thetas[p], rng, config)
                                for p in parents])
        child_loss = losses_of(child_z, child_theta)
        spent += n
        zs = np.concatenate([zs, child_z])
        thetas = np.concatenate([thetas, child_theta])
        loss = np.concatenate([loss, child_loss])
        generation += 1

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setups", nargs="+", choices=SETUPS,
                        default=list(SETUPS))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-every", type=int, default=10)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    payload = {}
    for setup in args.setups:
        print(f"\n=== {setup} (budget {args.budget}) ===", flush=True)
        rows = run(setup, args.budget, args.seed, args.probe_every,
                   args.samples)
        payload[setup] = rows
        for key in [f"genome@{s}" for s in GENOME_PROBE_SIGMAS] + \
                   [f"weights@{s}" for s in WEIGHT_PROBE_SIGMAS]:
            series = [r[key] for r in rows]
            first, last = series[0], series[-1]
            drift = last / first if first > 0 else float("nan")
            print(f"  {key:<16} first {first:.4f} -> last {last:.4f} "
                  f"({drift:.2f}x), min {min(series):.4f} max {max(series):.4f}",
                  flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "latent": LATENT, "seed": args.seed,
             "genome_probe_sigmas": list(GENOME_PROBE_SIGMAS),
             "weight_probe_sigmas": list(WEIGHT_PROBE_SIGMAS),
             "torch_version": torch.__version__, "probes": payload},
            indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
