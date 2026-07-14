"""Selection strategies. Each is callable as `select(population, k) -> [k inds]`.
Kept deliberately small; add more only when a problem needs it."""
from __future__ import annotations

import random
from typing import List

import numpy as np

from .core import Individual


class Selection(Individual):
    def __call__(self, individuals: List[Individual], k: int) -> List[Individual]:
        raise NotImplementedError


class RandomSelection(Selection):
    def __call__(self, individuals, k):
        return random.choices(individuals, k=k)


class TournamentSelection(Selection):
    def __init__(self, size: int = 3):
        super().__init__()
        self.size = size

    def __call__(self, individuals, k):
        s = min(self.size, len(individuals))
        return [max(random.sample(individuals, s), key=lambda i: i.fitness) for _ in range(k)]


class RankSelection(Selection):
    """Probability from rank, not raw fitness -> robust to fitness scale.

    scheme='linear'  -> classic linear rank; `pressure` in [1, 2].
    scheme='exp'     -> exp(-pressure * (worst_rank / n)); `pressure` unbounded
                        (GeneSpace uses ~20, i.e. near-elitist exploitation).
    """

    def __init__(self, pressure: float = 1.8, scheme: str = "linear"):
        super().__init__()
        self.pressure = pressure
        self.scheme = scheme

    def __call__(self, individuals, k):
        n = len(individuals)
        if n == 1:
            return [individuals[0]] * k
        order = sorted(range(n), key=lambda i: individuals[i].fitness)  # worst -> best
        ranks = np.empty(n)
        for r, idx in enumerate(order):
            ranks[idx] = r  # 0 = worst, n-1 = best
        if self.scheme == "exp":
            # best rank (n-1) -> smallest exponent -> highest weight
            probs = np.exp(-self.pressure * (n - 1 - ranks) / n)
        else:
            probs = 2 - self.pressure + 2 * (self.pressure - 1) * ranks / (n - 1)
        probs = probs / probs.sum()
        chosen = np.random.choice(n, size=k, p=probs)
        return [individuals[i] for i in chosen]


class TruncationSelection(Selection):
    def __init__(self, keep: float = 0.5):
        super().__init__()
        self.keep = keep

    def __call__(self, individuals, k):
        ranked = sorted(individuals, key=lambda i: i.fitness, reverse=True)
        top = ranked[: max(1, int(len(ranked) * self.keep))]
        return [top[i % len(top)] for i in range(k)]
