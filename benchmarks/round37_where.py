"""Round 37: should evolution choose WHERE it mutates the decoder?

Daniel's follow-up to round 36: instead of a fixed random subspace, let
each individual modify a DIFFERENT part of the base decoder, and let that
choice itself mutate.

This is the anchor grammar transplanted into weight space. An anchor has
a LOCATION and a MESSAGE; here an adapter has a location (which weights)
and a value (how much). Round 28 supports the direction: concentrated
mutations climb faster than diffuse ones (38k-weight decoder was 2x worse
than 7.5k — mutations spread thinner climb slower). The caveat is that
weight space has no metric — weight 500 is not "near" weight 501 — so
"where" is a subset choice, not a position.

Note round 36's shared backbone is FROZEN random noise for the whole run;
it never learns anything. Sharing it saves search dimensions, nothing
more. So these arms keep full private weights (memory was never the
bottleneck) and vary only WHICH coordinates a mutation is allowed to
touch — which isolates concentration and locality from expressiveness.
Every arm can still reach all of weight space over enough generations, so
none of them inherits round 36's low-rank ceiling.

Arms (pure decoder GA, win-rate step control, no distill/CMA):

  * full_weights   — the incumbent: every mutation perturbs all ~7,500
                     weights at once.
  * sparse_random  — each mutation perturbs k random coordinates, freshly
                     drawn every time. Concentration without memory.
  * sparse_evolved — each individual inherits a set of k coordinates from
                     its parent and mutates it slowly (a few indices move
                     per generation); its mutations only ever touch those.
                     Daniel's proposal: WHERE is a gene under selection.

If sparse_evolved beats sparse_random, the location genuinely carries
information worth inheriting; if they tie, concentration was the whole
story and the "where" is incidental.
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
INDEX_CHURN = 0.05   # fraction of an individual's coordinate set that moves

SETUPS = {
    "blob2d_anchor": (BlobImage2D, (32, 32), build_anchor),
    "smooth1d_anchor": (SmoothTarget, (256,), build_anchor),
}


def run(setup: str, arm: str, k: int, budget: int, seed: int) -> dict:
    factory, output_shape, architecture = SETUPS[setup]
    objective = factory()
    c = ExplorerConfig()
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    template = _Template(resolve(architecture, LATENT, output_shape), "mps")
    n_params = template.n_params

    def decode_all(zs, thetas):
        return torch.stack([template.decode(t, z)
                            for z, t in zip(zs, thetas)])

    zs = rng.standard_normal((c.population, LATENT)).astype(np.float32)
    thetas = np.stack([template.init_theta(int(rng.integers(0, 2**31)))
                       for _ in range(c.population)])
    # sparse_evolved: each individual owns a coordinate set it inherits
    sites = (np.stack([rng.choice(n_params, k, replace=False)
                       for _ in range(c.population)])
             if arm == "sparse_evolved" else None)

    phenos = decode_all(zs, thetas)
    loss = objective.loss_tensor(phenos.flatten(1)).cpu().numpy()
    spent, gain, trace = len(zs), 1.0, []

    def mutate_z(z):
        m = rng.random(z.shape) < c.genome_mutation_rate
        if not m.any():
            m[rng.integers(0, len(z))] = True
        return (z + m * rng.normal(0, c.genome_mutation_sigma * gain, z.shape)
                ).astype(np.float32)

    def mutate_theta(theta, site):
        sigma = float(np.exp(rng.uniform(np.log(c.weight_sigma_low),
                                         np.log(c.weight_sigma_high)))) * gain
        scale = max(float(theta.std()), 1e-3)
        out = theta.copy()
        if arm == "full_weights":
            out += rng.normal(0, sigma * scale, theta.shape).astype(np.float32)
        else:
            idx = (rng.choice(n_params, k, replace=False)
                   if arm == "sparse_random" else site)
            out[idx] += rng.normal(0, sigma * scale, k).astype(np.float32)
        return out.astype(np.float32)

    def mutate_site(site):
        """Move a few of this individual's coordinates — the 'where' gene."""
        moved = site.copy()
        n_move = max(1, int(k * INDEX_CHURN))
        slots = rng.choice(k, n_move, replace=False)
        moved[slots] = rng.choice(n_params, n_move, replace=False)
        return moved

    while spent < budget:
        order = np.argsort(loss)[:c.elite]
        zs, thetas, loss, phenos = zs[order], thetas[order], loss[order], phenos[order]
        if sites is not None:
            sites = sites[order]
        n = min(c.population, budget - spent)
        par = rng.integers(0, len(zs), n)
        cz = np.stack([mutate_z(zs[p]) for p in par])
        csites = (np.stack([mutate_site(sites[p]) for p in par])
                  if sites is not None else None)
        cth = np.stack([mutate_theta(thetas[p],
                                     csites[i] if csites is not None else None)
                        for i, p in enumerate(par)])
        cph = decode_all(cz, cth)
        cl = objective.loss_tensor(cph.flatten(1)).cpu().numpy()
        spent += n
        wins = float((cl <= loss[par] + 1e-12).mean())
        gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
        gain = float(np.clip(gain, *c.gain_limits))
        zs = np.concatenate([zs, cz])
        thetas = np.concatenate([thetas, cth])
        phenos = torch.cat([phenos, cph])
        loss = np.concatenate([loss, cl])
        if sites is not None:
            sites = np.concatenate([sites, csites])
        spread = float(phenos[np.argsort(loss)[:c.elite]].flatten(1)
                       .std(dim=0).mean())
        trace.append({"e": spent, "m": float(loss.min()), "gain": gain,
                      "win": wins, "spread": spread})

    out = {"mse": float(loss.min()), "final_gain": gain,
           "mean_spread": float(np.mean([t["spread"] for t in trace])),
           "trace": trace[::10]}
    if sites is not None:
        # how much do surviving elites agree on WHERE to mutate?
        elite_sites = sites[np.argsort(loss)[:c.elite]]
        counts = np.bincount(elite_sites.reshape(-1), minlength=n_params)
        out["site_overlap"] = float((counts > 1).sum() / max((counts > 0).sum(), 1))
    return out


ARMS = ("full_weights", "sparse_random", "sparse_evolved")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setups", nargs="+", choices=SETUPS,
                        default=["blob2d_anchor"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--k", type=int, default=256,
                        help="coordinates a sparse mutation may touch")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_mps()
    rows = []
    for setup in args.setups:
        print(f"\n########## {setup} (k={args.k}, budget {args.budget}) "
              f"##########", flush=True)
        for arm in args.arms:
            for seed in args.seeds:
                out = run(setup, arm, args.k, args.budget, seed)
                trace = out.pop("trace")
                rows.append({"setup": setup, "arm": arm, "seed": seed,
                             "k": args.k, **out, "trace": trace})
                extra = (f", site overlap {out['site_overlap']:.2f}"
                         if "site_overlap" in out else "")
                print(f"  {arm:<15} seed {seed} mse {out['mse']:.6g} "
                      f"(spread {out['mean_spread']:.4f}, gain "
                      f"{out['final_gain']:.2f}{extra})", flush=True)
            vals = [r["mse"] for r in rows
                    if r["arm"] == arm and r["setup"] == setup]
            print(f"  {arm:<15} MEAN {np.mean(vals):.6g}", flush=True)

    print("\n--- summary ---")
    for setup in args.setups:
        base = np.mean([r["mse"] for r in rows
                        if r["setup"] == setup and r["arm"] == "full_weights"])
        for arm in args.arms:
            v = [r["mse"] for r in rows
                 if r["setup"] == setup and r["arm"] == arm]
            print(f"  {setup:<16} {arm:<15} {np.mean(v):.6g} "
                  f"({base/np.mean(v):.2f}x vs full)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"budget": args.budget, "k": args.k, "index_churn": INDEX_CHURN,
             "torch_version": torch.__version__, "runs": rows}, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
