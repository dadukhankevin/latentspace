"""Grid-tune the retrieval blend on the cached dev-slice data.
No training, no GPU, no lock — pure post-hoc arithmetic."""
import json

import numpy as np

z = np.load("blend_cache.npz")
lp, best_k, rows, cmat, dev = (z["lp"].astype(np.float64), z["best_k"],
                               z["rows"], z["cmat"].astype(np.float64),
                               z["dev"])
n = len(dev) - 1
tgt = dev[1:].astype(np.int64)
p_base = np.exp(lp)


def bpb(p):
    return float(-np.log(p[np.arange(n), tgt]).mean() / np.log(2.0))


print("model-only:", round(bpb(p_base), 5))

schedules = {
    "aggr":   {64: .95, 48: .93, 32: .85, 24: .6, 16: .35, 12: .15},
    "aggr2":  {64: .98, 48: .95, 32: .9, 24: .7, 16: .45, 12: .25},
    "aggr3":  {64: .99, 48: .97, 32: .93, 24: .8, 16: .55, 12: .35},
    "max":    {64: .995, 48: .99, 32: .96, 24: .88, 16: .68, 12: .45},
    "insane": {64: .999, 48: .995, 32: .98, 24: .93, 16: .8, 12: .6},
}
best = (1e9, None)
for name, lam_by_k in schedules.items():
    for tau in (0.25, 0.5, 1.0):
        for g in (0.0, 0.1, 0.3):
            p = p_base.copy()
            for r, c in zip(rows, cmat):
                tot = c.sum()
                lam = lam_by_k[int(best_k[r])] * (tot / (tot + tau))
                png = c / tot
                if g > 0.0:
                    png = (1 - g) * png + g * p_base[r]
                p[r] = (1 - lam) * p[r] + lam * png
            b = bpb(p)
            if b < best[0]:
                best = (b, (name, tau, g))
            print(json.dumps({"sched": name, "tau": tau, "g": g,
                              "dev_bpb": round(b, 5)}))
print("BEST:", best)
