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
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch


def make_callable(x) -> Callable:
    """Any scalar hyperparameter may be a schedule. (Finch.)"""
    return x if callable(x) else (lambda: x)


class Schedule:
    """A linear start->end schedule advanced once per call. (Finch's `Rate`.)"""

    def __init__(self, start: float, end: float, steps: int, integer: bool = False):
        if steps < 1:
            raise ValueError("steps must be at least 1")
        self.start, self.end, self.integer = start, end, integer
        self.value = start
        self._delta = (end - start) / steps

    def __call__(self):
        v = self.value
        if self._delta > 0:
            self.value = min(self.end, self.value + self._delta)
        elif self._delta < 0:
            self.value = max(self.end, self.value + self._delta)
        else:
            self.value = self.end
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


@dataclass(frozen=True, init=False)
class SolutionSnapshot:
    """An immutable record of a phenotype that was actually evaluated.

    A latent alone cannot reproduce a historical solution after the decoder has
    changed. The snapshot therefore keeps copies of both the genes and decoded
    phenotype alongside the exact fitness and decoder version.
    """

    _genes: np.ndarray = field(repr=False)
    _phenotype: torch.Tensor = field(repr=False)
    fitness: float
    decoder_version: int
    generation: int

    def __init__(self, genes, phenotype, fitness, decoder_version, generation):
        genes_copy = np.asarray(genes, dtype=np.float32).copy()
        genes_copy.setflags(write=False)
        phenotype_copy = torch.as_tensor(phenotype).detach().cpu().clone()
        object.__setattr__(self, "_genes", genes_copy)
        object.__setattr__(self, "_phenotype", phenotype_copy)
        object.__setattr__(self, "fitness", float(fitness))
        object.__setattr__(self, "decoder_version", int(decoder_version))
        object.__setattr__(self, "generation", int(generation))

    @property
    def genes(self) -> np.ndarray:
        return self._genes.copy()

    @property
    def phenotype(self) -> torch.Tensor:
        return self._phenotype.clone()


class GenePool(Individual):
    """Generates fresh latent individuals."""

    def __init__(self, length: int, binary: bool = False):
        super().__init__()
        if length < 1:
            raise ValueError("length must be at least 1")
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
        self.best_observed: Optional[SolutionSnapshot] = None
        self.generation = 0
        self.history = {
            "generation": [],
            "fitness": [],
            "population": [],
            "decoder_version": [],
        }

    def compile(self) -> "Environment":
        for layer in self.layers:
            layer.bind(self)
        return self

    @property
    def best_current(self) -> Optional[LatentIndividual]:
        return self.population[0] if self.population else None

    @property
    def best_ever(self) -> Optional[SolutionSnapshot]:
        """Compatibility alias for the reproducible historical best."""
        return self.best_observed

    def _snapshot(self, individual: LatentIndividual) -> SolutionSnapshot:
        genes = np.stack([individual.genes]).astype(np.float32)
        phenotype = self.decoder.decode(genes)[0]
        return SolutionSnapshot(
            genes=individual.genes,
            phenotype=phenotype,
            fitness=individual.fitness,
            decoder_version=self.decoder.version,
            generation=self.generation,
        )

    def evolve(self, generations: int, verbose_every: int = 1) -> Optional[SolutionSnapshot]:
        if generations < 0:
            raise ValueError("generations cannot be negative")

        for _ in range(generations):
            for layer in self.layers:
                self.population = layer(self.population)
            for ind in self.population:
                ind.age += 1

            if not self.population:
                raise RuntimeError("the evolution pipeline produced an empty population")
            stale = [ind for ind in self.population if ind.evaluated_at != self.decoder.version]
            if stale:
                raise RuntimeError(
                    "generation ended with stale fitness; place DecodeAndEvaluate "
                    "after every decoder update"
                )

            best = self.population[0]  # Sort keeps the population ordered best-first
            self.decoder.fitness = best.fitness
            if self.best_observed is None or best.fitness > self.best_observed.fitness:
                self.best_observed = self._snapshot(best)

            self.history["generation"].append(self.generation)
            self.history["fitness"].append(best.fitness)
            self.history["population"].append(len(self.population))
            self.history["decoder_version"].append(self.decoder.version)

            if verbose_every and self.generation % verbose_every == 0:
                print(f"gen {self.generation:4d} | best {best.fitness:.4f} | "
                      f"pop {len(self.population):4d} | decoder v{self.decoder.version}")
            self.generation += 1
        return self.best_observed

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.history["fitness"])
        plt.xlabel("generation")
        plt.ylabel("best fitness")
        plt.show()
