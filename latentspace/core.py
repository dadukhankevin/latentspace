"""Core abstractions.

The design is a synthesis of three earlier projects:

  * Finch     -> the Keras-style *layer pipeline* and schedule-as-callable idea.
  * GeneSpace -> a single *universal latent* genotype that a co-evolving neural
                 *decoder* maps to a phenotype of ANY shape. This is what makes
                 one algorithm able to solve everything.
  * Aule      -> the ontology: *everything is an Individual*. Solutions, the
                 decoder, layers and the environment all share one root type,
                 so the decoder can be treated as an evolvable citizen too.

Only the latent vector is ever evolved by the genetic operators. All problem
specificity lives in exactly two places: the fitness function and the decoder.
That is why the operator set collapses to "cross and mutate a vector".
"""
from __future__ import annotations

import copy
from typing import Callable, List, Optional

import numpy as np


def make_callable(x) -> Callable:
    """Any scalar hyperparameter may be a schedule. (Finch.)"""
    return x if callable(x) else (lambda: x)


class Schedule:
    """A linear start->end schedule advanced once per call. (Finch's `Rate`.)"""

    def __init__(self, start: float, end: float, steps: int, integer: bool = False):
        self.start, self.end, self.integer = start, end, integer
        self.value = start
        self._delta = (end - start) / max(1, steps)

    def __call__(self):
        v = self.value
        past_end = (self._delta > 0 and v >= self.end) or (self._delta < 0 and v <= self.end)
        self.value = self.end if past_end else self.value + self._delta
        return int(v) if self.integer else v


class Individual:
    """Root type for every evolvable component.

    `evaluated_at` records the decoder *version* under which this individual's
    fitness was last computed. Because the decoder co-evolves, a stale version
    means the cached fitness is invalid and must be recomputed.
    """

    def __init__(self, fitness: float = float("-inf")):
        self.fitness = fitness
        self.age = 0
        self.evaluated_at = -1


class LatentIndividual(Individual):
    """A candidate solution: a fixed-length latent vector (float by default)."""

    def __init__(self, genes: np.ndarray):
        super().__init__()
        self.genes = genes

    def copy(self) -> "LatentIndividual":
        return copy.deepcopy(self)


class GenePool(Individual):
    """Generates fresh latent individuals."""

    def __init__(self, length: int, binary: bool = False):
        super().__init__()
        self.length = length
        self.binary = binary

    def _random(self) -> np.ndarray:
        if self.binary:
            return np.random.randint(0, 2, self.length).astype(np.float32)
        return np.random.randn(self.length).astype(np.float32)

    def generate(self, amount: int) -> List[LatentIndividual]:
        return [LatentIndividual(self._random()) for _ in range(amount)]


class Layer(Individual):
    """A pipeline stage. Also an Individual, so a layer could itself be evolved
    later without changing the type system (the Aule seed, left dormant)."""

    def __init__(self):
        super().__init__()
        self.env: Optional["Environment"] = None

    def bind(self, env: "Environment"):
        self.env = env

    def __call__(self, population: List[LatentIndividual]) -> List[LatentIndividual]:
        raise NotImplementedError


class Environment(Individual):
    """Runs a compiled list of layers over the population each generation."""

    def __init__(self, layers: List[Layer], genepool: GenePool, decoder):
        super().__init__()
        self.layers = layers
        self.genepool = genepool
        self.decoder = decoder
        self.population: List[LatentIndividual] = []
        self.best_ever: Optional[LatentIndividual] = None
        self.history = {"fitness": [], "population": [], "decoder_version": []}

    def compile(self) -> "Environment":
        for layer in self.layers:
            layer.bind(self)
        return self

    def evolve(self, generations: int, verbose_every: int = 1) -> LatentIndividual:
        for g in range(generations):
            for layer in self.layers:
                self.population = layer(self.population)
            for ind in self.population:
                ind.age += 1

            best = self.population[0]  # Sort keeps the population ordered best-first
            if self.best_ever is None or best.fitness > self.best_ever.fitness:
                self.best_ever = best.copy()

            self.history["fitness"].append(best.fitness)
            self.history["population"].append(len(self.population))
            self.history["decoder_version"].append(self.decoder.version)

            if verbose_every and g % verbose_every == 0:
                print(f"gen {g:4d} | best {best.fitness:.4f} | "
                      f"pop {len(self.population):4d} | decoder v{self.decoder.version}")
        return self.best_ever

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.history["fitness"])
        plt.xlabel("generation")
        plt.ylabel("best fitness")
        plt.show()
