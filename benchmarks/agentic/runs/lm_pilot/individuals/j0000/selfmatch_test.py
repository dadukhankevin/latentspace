"""Offline test: add a self-prefix (online, causal) match source on top
of the train-index blend, using cached dev lp. No training, no lock.
For row i (predicting dev[i+1]) the self index contains only grams
fully inside dev[:i+1], inserted incrementally — strictly causal."""
import json

import numpy as np

z = np.load("blend_cache.npz")
lp, best_k, rows, cmat, dev = (z["lp"].astype(np.float64), z["best_k"],
                               z["rows"], z["cmat"].astype(np.float64),
                               z["dev"])
n = len(dev) - 1
tgt = dev[1:].astype(np.int64)
ORDERS = (64, 48, 32, 24, 16, 12)
LAM = {64: .999, 48: .995, 32: .98, 24: .93, 16: .8, 12: .6}
TAU = 0.25

train_counts = {int(r): c for r, c in zip(rows, cmat)}
train_k = best_k


def bpb(p):
    return float(-np.log(p[np.arange(n), tgt]).mean() / np.log(2.0))


# --- causal self-index: dict per order, gram-bytes -> next-byte counts
def self_match(dev, orders):
    idx = {k: {} for k in orders}
    out_k = np.zeros(n, dtype=np.int64)
    out_c = {}
    b = dev.tobytes()
    for i in range(n):          # predicting dev[i+1]
        # 1) query longest order whose gram dev[i-k+1:i+1] was inserted
        for k in orders:
            if i - k + 1 < 0:
                continue
            g = b[i - k + 1: i + 1]
            d = idx[k].get(g)
            if d is not None:
                out_k[i] = k
                out_c[i] = d
                break
        # 2) insert grams ENDING at i-? : gram dev[i-k:i] predicts dev[i]
        #    (uses only bytes <= i, so future queries stay causal)
        for k in orders:
            if i - k < 0:
                continue
            g = b[i - k: i]
            idx[k].setdefault(g, []).append(int(dev[i]))
    return out_k, out_c


sk, sc = self_match(dev, ORDERS)
cov = (sk > 0).sum()
both = ((sk > 0) & (train_k > 0)).sum()
only_self = ((sk > 0) & (train_k == 0)).sum()
print(f"self-match coverage: {cov} rows ({100*cov/n:.1f}%), "
      f"self-only (no train match): {only_self} ({100*only_self/n:.1f}%)")

p_base = np.exp(lp)


def apply_train(p):
    for i, c in train_counts.items():
        tot = c.sum()
        lam = LAM[int(train_k[i])] * (tot / (tot + TAU))
        p[i] = (1 - lam) * p[i] + lam * (c / tot)
    return p


def apply_self(p, mode):
    for i, lst in sc.items():
        k = int(sk[i])
        if mode == "gaps" and train_k[i] > 0:
            continue
        if mode == "longer" and k <= train_k[i]:
            continue
        c = np.bincount(lst, minlength=256).astype(np.float64)
        tot = c.sum()
        lam = LAM[k] * (tot / (tot + TAU))
        p[i] = (1 - lam) * p[i] + lam * (c / tot)
    return p


print("train-only:",
      round(bpb(apply_train(p_base.copy())), 5))
print("train+self(gaps only):",
      round(bpb(apply_self(apply_train(p_base.copy()), "gaps"), ), 5))
print("train+self(when longer):",
      round(bpb(apply_self(apply_train(p_base.copy()), "longer")), 5))
print("self applied first, then train:",
      round(bpb(apply_train(apply_self(p_base.copy(), "all"))), 5))
