"""Why does the tuned config hurt TSP? Hypothesis: a wide decoder + strong
self-distillation + high selection pressure is a *convergence* machine -- ideal
when there's one smooth target, fatal for combinatorial search that needs
sustained diversity. Test by dialing exploitation down."""
import numpy as np
import torch

from latentspace import Evolver, TrainMode

np.random.seed(0)
CITIES = np.random.rand(12, 2)
D = np.sqrt(((CITIES[:, None, :] - CITIES[None, :, :]) ** 2).sum(-1))


def tsp_fitness(phenotypes):
    out = []
    for row in phenotypes.cpu().numpy():
        route = np.argsort(row)
        out.append(1.0 / (sum(D[route[i], route[(i + 1) % 12]] for i in range(12)) + 1e-6))
    return out


def diversity(ev):
    # mean pairwise L2 between latents in the final population
    g = np.stack([ind.genes for ind in ev.env.population])
    return float(np.mean(np.linalg.norm(g[:, None, :] - g[None, :, :], axis=-1)))


def tsp(label, gens=300, **kw):
    np.random.seed(0); torch.manual_seed(0)
    ev = Evolver(tsp_fitness, output_shape=(12,), **kw)
    ev.solve(gens, verbose_every=0)
    r = np.argsort(ev.decode_best().cpu().numpy())
    length = sum(D[r[i], r[(i + 1) % 12]] for i in range(12))
    print(f"{label:<40} len={length:.3f}  diversity={diversity(ev):.2f}")
    return length


# reference
best_rand = min(
    sum(D[r[i], r[(i + 1) % 12]] for i in range(12))
    for r in (np.random.permutation(12) for _ in range(200000))
)
print(f"random-search best (200k): {best_rand:.3f}\n")

TUNED = dict(latent=250, population=200, hidden_size=2000, num_layers=1, lr=1e-5,
             pressure=20, scheme="exp", children=4, n_points=8,
             mode=TrainMode.SELF_DISTILL, refine_every=10)

tsp("tuned (exploitative, as-is)", **TUNED)
tsp("tuned + linear pressure 1.8", **{**TUNED, "pressure": 1.8, "scheme": "linear"})
tsp("tuned + gentle refine (every 50)", **{**TUNED, "refine_every": 50})
tsp("tuned + NO refine (fixed decoder)", **{**TUNED, "refine_every": 10**9})
tsp("tuned + linear + refine 50", **{**TUNED, "pressure": 1.8, "scheme": "linear", "refine_every": 50})
tsp("small decoder + linear + refine 50",
    latent=64, population=200, hidden_size=256, num_layers=2, lr=1e-4,
    pressure=1.8, scheme="linear", children=2, n_points=4,
    mode=TrainMode.SELF_DISTILL, refine_every=50)
