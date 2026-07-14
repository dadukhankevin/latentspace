"""The layer pipeline. Each layer takes a population and returns one.

Genetic operators act ONLY on the latent vector, so there is exactly one
crossover and one mutation regardless of the problem. `DecodeAndEvaluate` is the
bridge to the decoder; `RefineDecoder` is the decoder's improvement step exposed
as a cadence-controlled pipeline stage.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .core import Layer, LatentIndividual, make_callable
from .decoder import TrainMode


class Populate(Layer):
    """Top the population up to `size` from the gene pool."""

    def __init__(self, size):
        super().__init__()
        self.size = make_callable(size)

    def __call__(self, pop):
        target = self.size()
        if len(pop) < target:
            pop = pop + self.env.genepool.generate(target - len(pop))
        return pop


class Crossover(Layer):
    """N-point crossover on the latent vector. Children are new individuals
    (evaluated_at = -1), so they are guaranteed to be scored next."""

    def __init__(self, selection, families, children=2, n_points=4):
        super().__init__()
        self.selection = selection
        self.families = make_callable(families)
        self.children = make_callable(children)
        self.n_points = n_points

    def _cross(self, p1: LatentIndividual, p2: LatentIndividual) -> LatentIndividual:
        length = len(p1.genes)
        pts = np.sort(np.random.choice(
            np.arange(1, length), size=min(self.n_points, length - 1), replace=False))
        child = p1.genes.copy()
        take_b, prev = False, 0
        for pt in pts:
            if take_b:
                child[prev:pt] = p2.genes[prev:pt]
            take_b, prev = not take_b, pt
        if take_b:
            child[prev:] = p2.genes[prev:]
        return LatentIndividual(child)

    def __call__(self, pop):
        offspring: List[LatentIndividual] = []
        for _ in range(self.families()):
            p1, p2 = self.selection(pop, 2)
            offspring.extend(self._cross(p1, p2) for _ in range(self.children()))
        return pop + offspring


class Mutate(Layer):
    """Perturb a random `percent` of the population in place. Gaussian for float
    latents (default), bit-flip for binary. Changed individuals have their cache
    invalidated so their fitness is recomputed."""

    def __init__(self, rate=0.1, sigma=0.1, percent=1.0, binary=False):
        super().__init__()
        self.rate = make_callable(rate)
        self.sigma = make_callable(sigma)
        self.percent = make_callable(percent)
        self.binary = binary

    def __call__(self, pop):
        k = max(1, int(len(pop) * self.percent()))
        for i in np.random.choice(len(pop), size=k, replace=False):
            genes = pop[i].genes
            mask = np.random.random(genes.shape) < self.rate()
            if not mask.any():
                continue
            if self.binary:
                genes[mask] = 1 - genes[mask]
            else:
                genes[mask] += (np.random.randn(int(mask.sum())) * self.sigma()).astype(genes.dtype)
            pop[i].evaluated_at = -1
        return pop


class DecodeAndEvaluate(Layer):
    """Batched latent -> phenotype -> fitness.

    Only (re)scores individuals whose cached fitness is stale with respect to the
    current decoder version. This fixes two things at once: individuals whose
    genes just changed (version -1), and EVERY individual after the decoder was
    refined (their version no longer matches the decoder's).

    `fitness_fn` receives a torch tensor of shape (B, *output_shape) and returns
    an iterable of B scalars.
    """

    def __init__(self, fitness_fn, batch_size=64):
        super().__init__()
        self.fitness_fn = fitness_fn
        self.batch_size = batch_size

    def __call__(self, pop):
        decoder = self.env.decoder
        stale = [ind for ind in pop if ind.evaluated_at != decoder.version]
        for i in range(0, len(stale), self.batch_size):
            batch = stale[i:i + self.batch_size]
            genes = np.stack([b.genes for b in batch]).astype(np.float32)
            phenotypes = decoder.decode(genes)
            for ind, fit in zip(batch, self.fitness_fn(phenotypes)):
                ind.fitness = float(fit)
                ind.evaluated_at = decoder.version
        return pop


class Sort(Layer):
    def __call__(self, pop):
        return sorted(pop, key=lambda ind: ind.fitness, reverse=True)


class Cap(Layer):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = make_callable(max_size)

    def __call__(self, pop):
        return pop[: self.max_size()]


class RefineDecoder(Layer):
    """Improve the decoder every `every` generations. Place AFTER Sort so the
    population is ranked. Bumps the decoder version, so the population is
    re-scored under the new mapping on the next generation."""

    def __init__(self, every=10, mode=TrainMode.SELF_DISTILL, percent=0.4,
                 epochs=1, batch_size=32, verbose=False):
        super().__init__()
        self.every = every
        self.mode = mode
        self.percent = percent
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self._gen = 0

    def __call__(self, pop):
        if len(pop) >= 2 and self._gen % self.every == 0:
            loss = self.env.decoder.refine(
                pop, mode=self.mode, percent=self.percent,
                batch_size=self.batch_size, epochs=self.epochs)
            self.env.decoder.fitness = pop[0].fitness  # decoder fitness = quality it supports
            if self.verbose:
                print(f"  [decoder] refined loss={loss:.5f} -> v{self.env.decoder.version}")
        self._gen += 1
        return pop
