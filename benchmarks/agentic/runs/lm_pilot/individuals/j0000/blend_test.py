"""Offline experiment: does blending a longest-suffix-match retrieval
model (rolling-hash n-gram index over the train corpus) with the v1
transformer help on the dev slice? Trains ONCE (20s), then evaluates
many blend configs post-hoc. Takes the canonical GPU lock. Never
touches the scorer's val/holdout splits."""
import fcntl
import importlib.util
import json
import sys
import time

import numpy as np

TASK = "/Users/daniellosey/Documents/latentspace/benchmarks/agentic/tasks/lm"
spec = importlib.util.spec_from_file_location("canon", TASK + "/score.py")
canon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canon)

DEV = 32768
B = np.uint64(0x100000001B3)


def gram_hashes(arr, k):
    """Hash of every k-gram arr[i:i+k]: Horner over j, k shifted adds,
    h_k[i] = sum_j (arr[i+j]+1)*B^(k-1-j) mod 2^64 (numpy wraparound)."""
    x = arr.astype(np.uint64) + np.uint64(1)
    n = len(arr) - k + 1
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h * B + x[j:j + n]
    return h


def build_index(train, orders):
    idx = {}
    for k in orders:
        h = gram_hashes(train, k)
        # gram ending at position e = i+k-1 predicts train[e+1]
        # keep only grams with a next byte
        h = h[: len(train) - k]          # ends at i+k-1 <= len-2
        order_next = train[k:].astype(np.int64)  # next byte after each gram
        srt = np.argsort(h, kind="stable")
        idx[k] = (h[srt], order_next[srt])
    return idx


def match_stats(idx, dev, orders, cap=256):
    """For each dev row i (predict dev[i+1]), for the LONGEST order k
    with a train match of dev[i-k+1:i+1], return (k, counts256)."""
    n = len(dev) - 1
    best_k = np.zeros(n, dtype=np.int64)
    counts = {}
    remaining = np.ones(n, dtype=bool)
    for k in sorted(orders, reverse=True):
        hs, nxt = idx[k]
        # dev gram ending at row i needs i-k+1 >= 0
        gh = gram_hashes(dev, k)          # gram starting at s ends at s+k-1
        rows = np.arange(k - 1, n)        # row i = s+k-1
        q = gh[: len(rows)]
        lo = np.searchsorted(hs, q, side="left")
        hi = np.searchsorted(hs, q, side="right")
        hit = (hi > lo) & remaining[rows]
        for s in np.nonzero(hit)[0]:
            i = s + k - 1
            l, h2 = lo[s], hi[s]
            if h2 - l > cap:
                h2 = l + cap
            c = np.bincount(nxt[l:h2], minlength=256).astype(np.float64)
            counts[i] = c
            best_k[i] = k
            remaining[i] = False
    return best_k, counts


def blended_bpb(lp_model, dev, best_k, counts, lam_by_k, tau):
    p = np.exp(lp_model.astype(np.float64))
    n = len(dev) - 1
    tgt = dev[1:].astype(np.int64)
    for i, c in counts.items():
        tot = c.sum()
        lam = lam_by_k[best_k[i]] * (tot / (tot + tau))
        p[i] = (1 - lam) * p[i] + lam * (c / tot)
    nll = -np.log(p[np.arange(n), tgt])
    return float(nll.mean() / np.log(2.0))


ORDERS = [64, 48, 32, 24, 16, 12]


def main():
    train_full, _, _ = canon.splits()
    train_head = train_full[:-DEV]
    dev = train_full[-DEV:]
    with open(canon.LOCK, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        ns = {}
        with open("artifact.py") as f:
            exec(compile(f.read(), "artifact.py", "exec"), ns)
        model_fn = ns["train"](train_head, 20.0, 0)
        lp = np.asarray(model_fn(dev), dtype=np.float64)
    base = canon.bits_per_byte(lambda a: lp.astype(np.float32), dev)
    print("model-only dev bpb:", round(base, 5))
    t0 = time.time()
    idx = build_index(train_head, ORDERS)
    print("index build seconds:", round(time.time() - t0, 2))
    t0 = time.time()
    best_k, counts = match_stats(idx, dev, ORDERS)
    print("match seconds:", round(time.time() - t0, 2))
    for k in ORDERS:
        m = (best_k == k).sum()
        print(f"  longest-match order {k}: {m} rows "
              f"({100.0 * m / (len(dev) - 1):.1f}%)")
    np.savez_compressed(
        "blend_cache.npz", lp=lp.astype(np.float32), best_k=best_k,
        rows=np.array(sorted(counts)),
        cmat=np.stack([counts[i] for i in sorted(counts)]).astype(np.uint16),
        dev=dev)
    print("cached to blend_cache.npz")


if __name__ == "__main__":
    main()
