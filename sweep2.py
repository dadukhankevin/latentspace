"""Confirm the winning config, settle the training mode, and check that the
tuning generalizes to a structurally different problem (TSP)."""
import numpy as np
import torch

from latentspace import Evolver, TrainMode

# --- target-match (interpretable MSE) ------------------------------------- #
TARGET = torch.linspace(0, 1, 16)


def match_target(phenotypes):
    err = ((phenotypes - TARGET.to(phenotypes.device)) ** 2).mean(dim=1)
    return (1.0 / (err + 1e-6)).tolist()


TUNED = dict(latent=250, population=200, hidden_size=2000, num_layers=1, lr=1e-5,
             pressure=20, scheme="exp", children=4, n_points=8, refine_every=10)


def target_mse(mode, gens=200):
    np.random.seed(0); torch.manual_seed(0)
    ev = Evolver(match_target, output_shape=(16,), mode=mode, **TUNED)
    ev.solve(gens, verbose_every=0)
    decoded = ev.decode_best().cpu().numpy()
    return float(np.mean((decoded - TARGET.numpy()) ** 2)), decoded


print("=== training mode, full tuned config, target-match ===")
for m in (TrainMode.SELF_DISTILL, TrainMode.GOOD_TO_BEST, TrainMode.EACH_TO_NEXT):
    mse, dec = target_mse(m)
    print(f"{m.name:<14} MSE={mse:.5f}")

mse, dec = target_mse(TrainMode.SELF_DISTILL)
print("\nsanity: decoded best vs target (should track 0..1):")
print("decoded:", np.round(dec, 2))
print("target :", np.round(TARGET.numpy(), 2))

# --- TSP: does the SAME tuning help a different problem? ------------------- #
np.random.seed(0)
CITIES = np.random.rand(12, 2)
D = np.sqrt(((CITIES[:, None, :] - CITIES[None, :, :]) ** 2).sum(-1))


def tsp_fitness(phenotypes):
    out = []
    for row in phenotypes.cpu().numpy():
        route = np.argsort(row)
        out.append(1.0 / (sum(D[route[i], route[(i + 1) % 12]] for i in range(12)) + 1e-6))
    return out


def tsp_len(**kw):
    np.random.seed(0); torch.manual_seed(0)
    ev = Evolver(tsp_fitness, output_shape=(12,), **kw)
    ev.solve(200, verbose_every=0)
    r = np.argsort(ev.decode_best().cpu().numpy())
    return sum(D[r[i], r[(i + 1) % 12]] for i in range(12))


# brute-force optimal for reference (12 cities is small enough via held-karp-ish sampling)
def approx_optimal(trials=200000):
    best = np.inf
    for _ in range(trials):
        r = np.random.permutation(12)
        d = sum(D[r[i], r[(i + 1) % 12]] for i in range(12))
        best = min(best, d)
    return best


BASE = dict(latent=32, population=150, hidden_size=256, num_layers=2, lr=1e-4,
            pressure=1.8, scheme="linear", children=2, n_points=4,
            mode=TrainMode.SELF_DISTILL, refine_every=10)

print("\n=== TSP generalization (tour length, lower=better) ===")
print(f"random-search best (200k tries): {approx_optimal():.3f}")
print(f"latentspace baseline               : {tsp_len(**BASE):.3f}")
print(f"latentspace tuned                  : {tsp_len(mode=TrainMode.SELF_DISTILL, **TUNED):.3f}")
