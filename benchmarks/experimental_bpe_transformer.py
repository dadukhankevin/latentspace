"""EXPERIMENTAL — go-all-out: a tokenized genome + full attention transformer.

Daniel's fusion (2026-07-22), a separate experiment that touches nothing in
the working library:

  1. The genome is a string of base symbols. Byte-pair encoding learns, from
     the population, which adjacent symbol pairs co-occur so often they should
     be ONE token — exactly the "co-adapted genes" idea, but realized as
     tokenization. Because a co-adapted block becomes a single atomic token,
     a plain cut between tokens CANNOT split it. "Crossover can't break a
     gene" stops being a rule and becomes a property of the representation.

  2. The decoder is a real transformer over that tokenized genome: a learned
     embedding per token, positional embeddings, multi-head SELF-attention
     over the genome, then a grid of learned pixel queries that CROSS-attend
     to the genome to paint the image. Every attention and feed-forward
     projection is a Linear, so the whole thing is LoRA-gateable.

The invariant is kept: ONE shared transformer decoder. An individual is its
token sequence (genes) plus a small latent that LoRA-gates the shared
decoder (the modifier). Fold absorbs a proven latent into the shared weights
by exact arithmetic (no training, no fitness gradients), same as the library.
Species + fitness-shares are imported from the library unchanged.

This is a research artifact, not a library feature. It exists to test whether
BPE tokenization finally makes building-block crossover pay — the record's
open question — in a setting with real population diversity (multi-image).
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from latentspace.universal import fitness_shares, make_species_selection


# ----------------------------------------------------------------- BPE

class BPE:
    """Learns merge rules from the population and tokenizes genomes.

    A genome is a length-`base_len` sequence of base symbols in [0, alphabet).
    `learn` merges the single most frequent adjacent token pair across the
    population into a new token id; repeated calls grow the vocabulary. A
    merged token spans >=2 base positions, so `encode` also returns, per
    genome, which base gaps are TOKEN BOUNDARIES — the only places crossover
    may cut without splitting a co-adapted block.
    """

    def __init__(self, alphabet: int, base_len: int, max_merges: int):
        self.alphabet = alphabet
        self.base_len = base_len
        self.max_merges = max_merges
        self.merges: list[tuple[int, int]] = []           # (a, b) -> new id
        self.vocab = alphabet + max_merges                # reserve id space

    def _tokenize_one(self, base: np.ndarray):
        """Greedily apply the current merges to one base sequence.
        Returns (token_ids, spans) where spans[i] = number of base symbols
        the i-th token covers."""
        toks = [int(x) for x in base]
        spans = [1] * len(toks)
        for k, (a, b) in enumerate(self.merges):
            new_id = self.alphabet + k
            i = 0
            while i < len(toks) - 1:
                if toks[i] == a and toks[i + 1] == b:
                    toks[i] = new_id
                    spans[i] += spans[i + 1]
                    del toks[i + 1]
                    del spans[i + 1]
                else:
                    i += 1
        return toks, spans

    def learn(self, population_base: np.ndarray) -> bool:
        """Merge the most frequent adjacent token pair. Returns False when the
        merge budget is spent or nothing repeats."""
        if len(self.merges) >= self.max_merges:
            return False
        counts: dict[tuple[int, int], int] = {}
        for base in population_base:
            toks, _ = self._tokenize_one(base)
            for i in range(len(toks) - 1):
                pair = (toks[i], toks[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            return False
        best, freq = max(counts.items(), key=lambda kv: kv[1])
        if freq < 2:
            return False
        self.merges.append(best)
        return True

    def encode(self, population_base: np.ndarray):
        """Tokenize the whole population. Returns:
        ids  (M, base_len) padded token ids (pad id = vocab, an extra slot),
        mask (M, base_len) True where padding,
        boundary (M, base_len - 1) True where a base gap is a token boundary.
        """
        M = len(population_base)
        pad = self.vocab
        ids = np.full((M, self.base_len), pad, dtype=np.int64)
        mask = np.ones((M, self.base_len), dtype=bool)
        boundary = np.zeros((M, self.base_len - 1), dtype=bool)
        for m, base in enumerate(population_base):
            toks, spans = self._tokenize_one(base)
            ids[m, :len(toks)] = toks
            mask[m, :len(toks)] = False
            pos = 0
            for s in spans[:-1]:
                pos += s
                boundary[m, pos - 1] = True          # gap after this token
        return ids, mask, boundary


# --------------------------------------------------- transformer decoder

class LoRALinear(nn.Module):
    """A Linear whose output can be bent per-individual by a shared latent:
    y = base(x) + scale * up(coeff * down(x)). `coeff` is set per batch."""

    def __init__(self, d_in: int, d_out: int, rank: int):
        super().__init__()
        self.base = nn.Linear(d_in, d_out)
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up = nn.Linear(rank, d_out, bias=False)
        self.scale = rank ** -0.5
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        self.coeff: torch.Tensor | None = None            # (B, rank)

    def forward(self, x):                                 # x (B, T, d_in)
        out = self.base(x)
        if self.coeff is not None:
            bent = self.up(self.down(x) * self.coeff[:, None, :])
            out = out + self.scale * bent
        return out

    def absorb(self, coeff: torch.Tensor) -> None:
        with torch.no_grad():
            self.base.weight += self.scale * (
                self.up.weight @ (coeff[:, None] * self.down.weight))


class Attention(nn.Module):
    """Multi-head attention with LoRA on every projection (so folding can
    reach the attention weights, not just the feed-forward ones)."""

    def __init__(self, d_model: int, heads: int, rank: int):
        super().__init__()
        self.h, self.dk = heads, d_model // heads
        self.q = LoRALinear(d_model, d_model, rank)
        self.k = LoRALinear(d_model, d_model, rank)
        self.v = LoRALinear(d_model, d_model, rank)
        self.o = LoRALinear(d_model, d_model, rank)

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dk).transpose(1, 2)   # (B,h,T,dk)

    def forward(self, q_in, kv_in, key_pad=None):
        B, Tq, _ = q_in.shape
        q, k, v = self._split(self.q(q_in)), self._split(self.k(kv_in)), \
            self._split(self.v(kv_in))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        if key_pad is not None:                            # (B, Tk) True=pad
            scores = scores.masked_fill(
                key_pad[:, None, None, :], float("-inf"))
        att = F.softmax(scores, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, Tq, -1)
        return self.o(out)


class Block(nn.Module):
    def __init__(self, d_model, heads, rank):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = Attention(d_model, heads, rank)
        self.ff1 = LoRALinear(d_model, 2 * d_model, rank)
        self.ff2 = LoRALinear(2 * d_model, d_model, rank)

    def forward(self, x, key_pad=None):
        x = x + self.attn(self.n1(x), self.n1(x), key_pad)
        h = self.n2(x)
        return x + self.ff2(F.gelu(self.ff1(h)))


class TransformerImageDecoder:
    """ONE shared transformer. decode(token_ids, coeff) paints images; a grid
    of learned pixel queries cross-attends to the self-attended genome."""

    def __init__(self, vocab, base_len, out_shape, device,
                 d_model=48, heads=3, layers=2, rank=32):
        self.device = device
        self.out_shape = out_shape
        h, w, c = out_shape
        self.rank = rank
        net = nn.Module()
        net.embed = nn.Embedding(vocab + 1, d_model)       # +1 pad row
        net.pos = nn.Parameter(torch.randn(base_len, d_model) * 0.02)
        net.blocks = nn.ModuleList(
            [Block(d_model, heads, rank) for _ in range(layers)])
        net.queries = nn.Parameter(torch.randn(h * w, d_model) * 0.02)
        net.cross = Attention(d_model, heads, rank)
        net.qnorm = nn.LayerNorm(d_model)
        net.head = LoRALinear(d_model, c, rank)
        self.net = net.to(device)
        self._lora = [m for m in self.net.modules()
                      if isinstance(m, LoRALinear)]
        self.n_params = sum(p.numel() for p in self.net.parameters())

    def _set_coeff(self, coeff):
        for m in self._lora:
            m.coeff = coeff

    def decode(self, token_ids: np.ndarray, coeff: np.ndarray,
               pad_mask: np.ndarray) -> torch.Tensor:
        ids = torch.as_tensor(token_ids, device=self.device)
        cf = torch.as_tensor(coeff.astype(np.float32), device=self.device)
        kp = torch.as_tensor(pad_mask, device=self.device)
        h, w, c = self.out_shape
        self._set_coeff(cf)
        with torch.no_grad():
            x = self.net.embed(ids) + self.net.pos[None]
            for blk in self.net.blocks:
                x = blk(x, kp)
            q = self.net.qnorm(self.net.queries)[None].expand(len(ids), -1, -1)
            painted = self.net.cross(q, x, kp)
            out = torch.sigmoid(self.net.head(painted))    # (B, h*w, c)
        self._set_coeff(None)
        return out.reshape(len(ids), h * w * c)

    def absorb(self, coeff: np.ndarray) -> None:
        cf = torch.as_tensor(coeff.astype(np.float32), device=self.device)
        for m in self._lora:
            m.absorb(cf)


# ------------------------------------------------------------ the GA

@dataclass
class Config:
    alphabet: int = 24
    base_len: int = 48
    max_merges: int = 64
    rank: int = 32
    children: int = 24
    epochs: int = 1200
    token_xover: bool = True          # cut only at BPE token boundaries
    bpe_every: int = 40               # learn one merge this often
    fold_every: int = 32
    outcross: float = 0.05
    mut_token: float = 0.15           # per-base symbol mutation rate
    mut_latent_sigma: float = 0.2


def run(images, out_shape, cfg: Config, seed=0, device="cpu", log=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(0, 2 ** 31)))
    n_fns = len(images)
    flats = [torch.as_tensor(img.reshape(-1), device=device) for img in images]

    def score_on(phenos, fn_of):
        vals = np.empty(len(fn_of))
        for f in np.unique(fn_of):
            pk = np.flatnonzero(fn_of == f)
            vals[pk] = (-(phenos[pk] - flats[int(f)]) ** 2).mean(1).cpu().numpy()
        return vals

    bpe = BPE(cfg.alphabet, cfg.base_len, cfg.max_merges)
    decoder = TransformerImageDecoder(bpe.vocab, cfg.base_len, out_shape,
                                      device, rank=cfg.rank)
    select = make_species_selection(cfg.outcross)

    # seed two individuals per image
    cap = 2 * n_fns
    pop_base = rng.integers(0, cfg.alphabet, (cap, cfg.base_len))
    pop_lat = rng.standard_normal((cap, cfg.rank)).astype(np.float32)
    pop_fn = np.repeat(np.arange(n_fns), 2)
    best = np.full(n_fns, -np.inf)
    founder = np.full(n_fns, np.nan)

    def evaluate(base, lat, fn_of):
        nonlocal best, founder
        ids, mask, _ = bpe.encode(base)
        ph = decoder.decode(ids, lat, mask)
        vals = score_on(ph, fn_of)
        for f in np.unique(fn_of):
            v = vals[fn_of == f]
            if np.isnan(founder[f]):
                founder[f] = float(v.max())
            best[f] = max(best[f], float(v.max()))
        return vals

    pop_score = evaluate(pop_base, pop_lat, pop_fn)

    for epoch in range(cfg.epochs):
        w = fitness_shares(pop_score, pop_fn)
        a, b = select(w, pop_fn, rng, cfg.children)

        # gene (token) crossover, boundary-respecting
        _, _, bound = bpe.encode(pop_base)
        child_base = pop_base[a].copy()
        if cfg.token_xover:
            for i in range(cfg.children):
                gaps = np.flatnonzero(bound[a[i]])
                if len(gaps):
                    cut = int(rng.choice(gaps)) + 1
                    child_base[i, cut:] = pop_base[b[i], cut:]
        else:
            cuts = rng.integers(1, cfg.base_len, cfg.children)
            for i in range(cfg.children):
                child_base[i, cuts[i]:] = pop_base[b[i], cuts[i]:]
        # token mutation
        mut = rng.random(child_base.shape) < cfg.mut_token
        child_base[mut] = rng.integers(0, cfg.alphabet, int(mut.sum()))

        # latent inherited whole from parent a, then mutated
        child_lat = pop_lat[a].copy()
        child_lat += (rng.standard_normal(child_lat.shape)
                      * cfg.mut_latent_sigma).astype(np.float32)

        child_fn = pop_fn[a]
        child_score = evaluate(child_base, child_lat, child_fn)

        # cap the population by fitness share
        all_base = np.concatenate([pop_base, child_base])
        all_lat = np.concatenate([pop_lat, child_lat])
        all_fn = np.concatenate([pop_fn, child_fn])
        all_sc = np.concatenate([pop_score, child_score])
        keep = np.argsort(-fitness_shares(all_sc, all_fn))[:cap]
        pop_base, pop_lat = all_base[keep], all_lat[keep]
        pop_fn, pop_score = all_fn[keep], all_sc[keep]

        if (epoch + 1) % cfg.bpe_every == 0:
            bpe.learn(pop_base)

        if (epoch + 1) % cfg.fold_every == 0:
            w2 = fitness_shares(pop_score, pop_fn)
            donor = int(np.argmax(w2))
            decoder.absorb(pop_lat[donor])
            pop_lat[donor] = 0.0
            pop_score = evaluate(pop_base, pop_lat, pop_fn)

        if log and (epoch + 1) % log == 0:
            removed = np.mean([100 * (1 - (-best[f]) / (-founder[f]))
                               for f in range(n_fns)])
            print(f"  epoch {epoch + 1:>5}  mean removed {removed:.1f}%  "
                  f"vocab {bpe.alphabet + len(bpe.merges)}", flush=True)

    removed = [100 * (1 - (-best[f]) / (-founder[f])) for f in range(n_fns)]
    return float(np.mean(removed)), removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-token-xover", action="store_true",
                        help="ablation: plain cuts that ignore BPE boundaries")
    args = parser.parse_args()

    from pathlib import Path
    from PIL import Image
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    files = sorted(Path("/tmp/latentspace_cifar100_scaling_1024")
                   .glob("*.png"))[:args.count]
    images = [np.asarray(Image.open(f), dtype=np.float32) / 255.0
              for f in files]
    cfg = Config(epochs=args.epochs, token_xover=not args.no_token_xover)
    mean, _ = run(images, (32, 32, 3), cfg, seed=args.seed, device=device,
                  log=max(1, args.epochs // 20))
    print(f"final mean error removed: {mean:.1f}%  "
          f"({'token' if cfg.token_xover else 'plain'} crossover)")


if __name__ == "__main__":
    main()
