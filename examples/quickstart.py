"""One algorithm, two structurally different problems, ~2 lines each.

Problem A: match a target vector      (continuous, output_shape = (N,))
Problem B: traveling salesman         (permutation, output_shape = (N,) + argsort)

The ONLY things that change between them are the fitness function and the output
shape. Same latent, same operators, same decoder architecture.
"""
import numpy as np
import torch

from latentspace import Evolver

# --------------------------------------------------------------------------- #
# Problem A: evolve a latent whose decoded phenotype matches a target vector.
# --------------------------------------------------------------------------- #
TARGET = torch.linspace(0, 1, 16)

def match_target(phenotypes):
    # phenotypes: (B, 16) in [0, 1]. Higher fitness = closer to target.
    err = ((phenotypes - TARGET.to(phenotypes.device)) ** 2).mean(dim=1)
    return (1.0 / (err + 1e-6)).tolist()

print("=== Problem A: match target vector ===")
a = Evolver(match_target, output_shape=(16,), latent=32, population=150)
a.solve(120, verbose_every=30)
print("decoded best:", np.round(a.decode_best().cpu().numpy(), 2))
print("target      :", np.round(TARGET.numpy(), 2))

# --------------------------------------------------------------------------- #
# Problem B: TSP. The decoder emits continuous values; argsort turns them into a
# tour, so a continuous universal decoder can produce discrete permutations.
# --------------------------------------------------------------------------- #
np.random.seed(0)
CITIES = np.random.rand(12, 2)
D = np.sqrt(((CITIES[:, None, :] - CITIES[None, :, :]) ** 2).sum(-1))

def tsp_fitness(phenotypes):
    out = []
    for row in phenotypes.cpu().numpy():
        route = np.argsort(row)
        dist = sum(D[route[i], route[(i + 1) % len(route)]] for i in range(len(route)))
        out.append(1.0 / (dist + 1e-6))
    return out

print("\n=== Problem B: traveling salesman (same algorithm) ===")
b = Evolver(tsp_fitness, output_shape=(12,), latent=32, population=150)
b.solve(120, verbose_every=30)
best_route = np.argsort(b.decode_best().cpu().numpy())
best_dist = sum(D[best_route[i], best_route[(i + 1) % 12]] for i in range(12))
print("best route :", best_route.tolist())
print("tour length:", round(best_dist, 3))
