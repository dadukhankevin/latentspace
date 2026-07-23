"""EXPERIMENTAL — the distillation loop (Daniel's synthesis, 2026-07-22).

The day's failures triangulated on this. Two honest observations:
  * The decoder is DIFFERENTIABLE even though the fitness is a black box.
    Evolution's job is the black-box search; once it has found good
    phenotypes, teaching the decoder to reproduce them is ordinary
    supervised learning with gradients fully available. We had banned
    ourselves from that on a technicality.
  * A universal decoder's whole reason to exist is TRANSFER across related
    problems (CIFAR images share grass, sky, edges). The rising shared
    baseline is the mechanism; a single apple or unrelated targets can't
    reward it.

The loop:
  1. Each individual reaches BEYOND the current decoder with a sparse weight
     patch (the modifier that doubled the apple).
  2. Evolution vets which patched phenotypes are actually good, against the
     black-box fitness (species + shares, imported unchanged).
  3. Gradient descent DISTILLS the vetted phenotypes into the shared
     decoder: train it so base(z_i) ~= patched_phenotype_i. Zero fitness
     evaluations — the fitness is never differentiated.
  4. The patch is decayed; individuals now paint from the higher floor and
     re-patch. Repeat.

Evolution does the black-box search it is uniquely good at; gradients do the
differentiable learning they are uniquely good at; the two stop fighting
over the same job. This file touches nothing in the library.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, vmap

from latentspace.universal import (fitness_shares, make_species_selection,
                                    resolve)


class DistillGenerator:
    """One shared conv decoder. Individuals reach past it with sparse weight
    patches; gradient distillation absorbs their vetted discoveries."""

    def __init__(self, architecture, genes, out_shape, patch_size, device,
                 value_scale=0.25, lr=1e-3):
        self.device = device
        self.out_shape = out_shape
        self.net = resolve(architecture, genes, out_shape)().to(device)
        self._names = [n for n, _ in self.net.named_parameters()]
        self._shapes = [tuple(p.shape) for _, p in self.net.named_parameters()]
        numels = [int(np.prod(s)) if s else 1 for s in self._shapes]
        self._offsets = np.concatenate([[0], np.cumsum(numels)]).astype(int)
        self.n_params = int(self._offsets[-1])
        self.patch_size = int(min(patch_size, self.n_params))
        self.value_scale = float(
            nn.utils.parameters_to_vector(self.net.parameters()).std().item()
        ) * value_scale
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self._sites: dict[int, np.ndarray] = {}
        self.replay_z: list[np.ndarray] = []
        self.replay_p: list[np.ndarray] = []

        def _forward(params, z):
            return torch.sigmoid(functional_call(self.net, params, (z[None],)))[0]
        self._vforward = vmap(_forward)

    def sites_for(self, seed):
        key = int(seed)
        if key not in self._sites:
            self._sites[key] = np.random.default_rng(key).integers(
                0, self.n_params, self.patch_size)
        return self._sites[key]

    def _flat_base(self):
        return nn.utils.parameters_to_vector(
            self.net.parameters()).detach()

    def decode(self, z, values, seeds):
        """Per-individual PATCHED phenotypes (evolution's forward)."""
        n = len(z)
        flat = self._flat_base()[None, :].repeat(n, 1)
        idx = torch.as_tensor(
            np.stack([self.sites_for(s) for s in seeds]), device=self.device)
        vals = torch.as_tensor(
            np.ascontiguousarray(values.astype(np.float32)), device=self.device)
        flat = flat.scatter_add(1, idx, vals * self.value_scale)
        params = {name: flat[:, self._offsets[i]:self._offsets[i + 1]]
                  .reshape(n, *self._shapes[i])
                  for i, name in enumerate(self._names)}
        genes = torch.as_tensor(
            np.ascontiguousarray(z.astype(np.float32)), device=self.device)
        with torch.no_grad():
            out = self._vforward(params, genes)
        return out.reshape(n, -1)

    def base_decode(self, z):
        """The shared decoder with NO patch — the rising baseline."""
        genes = torch.as_tensor(
            np.ascontiguousarray(z.astype(np.float32)), device=self.device)
        return torch.sigmoid(self.net(genes)).reshape(len(z), -1)

    def distill(self, z_new, pheno_new, steps, buffer_cap=256):
        """Train the shared decoder so base(z) reproduces the vetted patched
        phenotypes. A replay buffer over past champions resists forgetting.
        Zero fitness evaluations."""
        for z, p in zip(z_new, pheno_new):
            self.replay_z.append(z.astype(np.float32))
            self.replay_p.append(p.astype(np.float32))
        if len(self.replay_z) > buffer_cap:
            self.replay_z = self.replay_z[-buffer_cap:]
            self.replay_p = self.replay_p[-buffer_cap:]
        Z = torch.as_tensor(np.stack(self.replay_z), device=self.device)
        P = torch.as_tensor(np.stack(self.replay_p), device=self.device)
        for _ in range(steps):
            idx = torch.randint(0, len(Z), (min(64, len(Z)),),
                                device=self.device)
            self.opt.zero_grad()
            out = torch.sigmoid(self.net(Z[idx])).reshape(len(idx), -1)
            loss = ((out - P[idx]) ** 2).mean()
            loss.backward()
            self.opt.step()
        return float(loss.detach())


@dataclass
class Config:
    genes: int = 32
    patch: int = 1024
    children: int = 16
    epochs: int = 1500
    distill_every: int = 50
    distill_steps: int = 40
    patch_decay: float = 0.3       # shrink patches after their absorption
    outcross: float = 0.05
    mut_gene_sigma: float = 0.12
    mut_patch_sigma: float = 0.2
    win_target: float = 0.2
    dial_step: float = 1.15


def run(images, out_shape, cfg, seed=0, device="cpu", distill=True, log=None,
        progress=None, progress_every=None, init_state=None,
        return_full=False, transform=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(0, 2 ** 31)))
    n_fns = len(images)
    flats = [torch.as_tensor(img.reshape(-1), device=device) for img in images]
    gen = DistillGenerator("auto", cfg.genes, out_shape, cfg.patch, device)
    if init_state is not None:                     # warm-start (transfer)
        gen.net.load_state_dict(init_state)
    select = make_species_selection(cfg.outcross)
    best_pheno = [None] * n_fns
    prog_every = progress_every or max(1, cfg.epochs // 50)
    trace: list[tuple] = []

    def score(phenos, fn_of):
        # `transform` (optional) is a non-differentiable forward map applied
        # to the phenotype before comparison — e.g. a cellular automaton. The
        # objective is a black box; distillation still targets the phenotype
        # (the CA's INPUT field), never the transform.
        cmp = transform(phenos) if transform is not None else phenos
        v = np.empty(len(fn_of))
        for f in np.unique(fn_of):
            pk = np.flatnonzero(fn_of == f)
            v[pk] = (-(cmp[pk] - flats[int(f)]) ** 2).mean(1).cpu().numpy()
        return v

    cap = 2 * n_fns
    pop_z = rng.standard_normal((cap, cfg.genes)).astype(np.float32)
    pop_patch = (rng.standard_normal((cap, cfg.patch)) * 0.1).astype(np.float32)
    pop_seed = rng.integers(0, 2 ** 31, cap)
    pop_fn = np.repeat(np.arange(n_fns), 2)
    best = np.full(n_fns, -np.inf)
    founder = np.full(n_fns, np.nan)
    patch_dial = 1.0

    def evaluate(z, patch, seeds, fn_of, keep_pheno=False):
        nonlocal best, founder
        ph = gen.decode(z, patch, seeds)
        v = score(ph, fn_of)
        for f in np.unique(fn_of):
            picks = np.flatnonzero(fn_of == f)
            top = int(picks[np.argmax(v[picks])])
            if np.isnan(founder[f]):
                founder[f] = float(v[top])
            if v[top] > best[f]:
                best[f] = float(v[top])
                best_pheno[int(f)] = ph[top].detach().cpu().numpy().copy()
        return (v, ph) if keep_pheno else v

    pop_score = evaluate(pop_z, pop_patch, pop_seed, pop_fn)

    for epoch in range(cfg.epochs):
        w = fitness_shares(pop_score, pop_fn)
        a, b = select(w, pop_fn, rng, cfg.children)
        # genes: one-point crossover then mutate
        cz = pop_z[a].copy()
        cuts = rng.integers(1, cfg.genes, cfg.children)
        for i in range(cfg.children):
            cz[i, cuts[i]:] = pop_z[b[i], cuts[i]:]
        cz += (rng.standard_normal(cz.shape) * cfg.mut_gene_sigma
               ).astype(np.float32)
        # patch: inherited whole from a (with its sites), mutated
        cpatch = pop_patch[a].copy()
        cseed = pop_seed[a].copy()
        cpatch += (rng.standard_normal(cpatch.shape)
                   * cfg.mut_patch_sigma * patch_dial).astype(np.float32)
        cfn = pop_fn[a]
        cscore = evaluate(cz, cpatch, cseed, cfn)

        wins = float((cscore >= pop_score[a] - 1e-12).mean())
        patch_dial *= cfg.dial_step if wins > cfg.win_target else 1 / cfg.dial_step
        patch_dial = float(np.clip(patch_dial, 1e-3, 1e3))

        A = np.concatenate
        az, ap, asd = A([pop_z, cz]), A([pop_patch, cpatch]), A([pop_seed, cseed])
        af, asc = A([pop_fn, cfn]), A([pop_score, cscore])
        keep = np.argsort(-fitness_shares(asc, af))[:cap]
        pop_z, pop_patch, pop_seed = az[keep], ap[keep], asd[keep]
        pop_fn, pop_score = af[keep], asc[keep]

        if distill and (epoch + 1) % cfg.distill_every == 0:
            champs = [int(np.flatnonzero(pop_fn == f)[
                np.argmax(pop_score[pop_fn == f])]) for f in np.unique(pop_fn)]
            champs = np.array(champs)
            _, ph = evaluate(pop_z[champs], pop_patch[champs],
                             pop_seed[champs], pop_fn[champs], keep_pheno=True)
            gen.distill(pop_z[champs], ph.cpu().numpy(), cfg.distill_steps)
            pop_patch *= cfg.patch_decay          # discovery now in the base
            pop_score = evaluate(pop_z, pop_patch, pop_seed, pop_fn)

        if (epoch + 1) % prog_every == 0:
            # absolute mean best MSE (fitness = -MSE), the transfer metric
            trace.append((epoch + 1, float(np.mean(-best))))
            if progress:
                progress(epoch + 1, cfg.epochs,
                         [None if p is None else p.copy() for p in best_pheno],
                         best.copy())
        if log and (epoch + 1) % log == 0:
            removed = np.mean([100 * (1 - (-best[f]) / (-founder[f]))
                               for f in range(n_fns)])
            print(f"  epoch {epoch + 1:>5}  mean removed {removed:.1f}%  "
                  f"mse {np.mean(-best):.5f}  patch_dial {patch_dial:.2f}",
                  flush=True)

    removed = [100 * (1 - (-best[f]) / (-founder[f])) for f in range(n_fns)]
    if return_full:
        return float(np.mean(removed)), removed, gen, trace, best_pheno
    return float(np.mean(removed)), removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-distill", action="store_true",
                        help="ablation: sparse patch + arithmetic fold only")
    args = parser.parse_args()

    from pathlib import Path
    from PIL import Image
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    files = sorted(Path("/tmp/latentspace_cifar100_scaling_1024")
                   .glob("*.png"))[:args.count]
    images = [np.asarray(Image.open(f), dtype=np.float32) / 255.0
              for f in files]
    cfg = Config(epochs=args.epochs)
    mean, per = run(images, (32, 32, 3), cfg, seed=args.seed, device=device,
                    distill=not args.no_distill, log=max(1, args.epochs // 15))
    print(f"final mean error removed: {mean:.1f}%  "
          f"({'DISTILL' if not args.no_distill else 'no-distill'})")


if __name__ == "__main__":
    main()
