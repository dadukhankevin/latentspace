"""The layer pipeline. Each layer takes a population and returns one.

Genetic operators act ONLY on the latent vector, so there is exactly one
crossover and one mutation regardless of the problem. `DecodeAndEvaluate` is the
bridge to the decoder; `RefineDecoder` is the decoder's improvement step exposed
as a cadence-controlled pipeline stage.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch

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
        if n_points < 0:
            raise ValueError("n_points cannot be negative")
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
        families = int(self.families())
        children = int(self.children())
        if families < 0 or children < 0:
            raise ValueError("families and children cannot be negative")
        for _ in range(families):
            p1, p2 = self.selection(pop, 2)
            offspring.extend(self._cross(p1, p2) for _ in range(children))
        return pop + offspring


def _mutate_genes(genes, rate, sigma, binary, ensure_change):
    mask = np.random.random(genes.shape) < rate
    if ensure_change and rate > 0 and not mask.any():
        mask.flat[np.random.randint(mask.size)] = True
    if not mask.any():
        return False
    if binary:
        genes[mask] = 1 - genes[mask]
    else:
        genes[mask] += (
            np.random.randn(int(mask.sum())) * sigma
        ).astype(genes.dtype)
    return True


class Mutate(Layer):
    """Perturb a random ``percent`` of eligible individuals in place.

    Gaussian mutation is used for float latents and bit flips for binary ones.
    With ``offspring_only=True``, only individuals that have not yet been
    evaluated under the current decoder are eligible. Placed directly after
    :class:`Crossover`, this gives the usual elitist ``(mu + lambda)`` behavior:
    incumbent parents remain unchanged while newly bred children are mutated.
    """

    def __init__(self, rate=0.1, sigma=0.1, percent=1.0, binary=False,
                 offspring_only=False, ensure_change=True):
        super().__init__()
        self.rate = make_callable(rate)
        self.sigma = make_callable(sigma)
        self.percent = make_callable(percent)
        self.binary = binary
        self.offspring_only = bool(offspring_only)
        self.ensure_change = bool(ensure_change)

    def __call__(self, pop):
        if not pop:
            return pop
        percent = float(self.percent())
        rate = float(self.rate())
        sigma = float(self.sigma())
        if not 0 <= percent <= 1:
            raise ValueError("mutation percent must be in [0, 1]")
        if not 0 <= rate <= 1:
            raise ValueError("mutation rate must be in [0, 1]")
        if sigma < 0:
            raise ValueError("mutation sigma cannot be negative")
        eligible = [
            index for index, individual in enumerate(pop)
            if not self.offspring_only
            or individual.evaluated_at != self.env.decoder.version
        ]
        k = int(len(eligible) * percent)
        if k == 0:
            return pop
        selected = np.random.choice(eligible, size=k, replace=False)
        for i in np.atleast_1d(selected):
            genes = pop[i].genes
            if _mutate_genes(
                genes, rate, sigma, self.binary, self.ensure_change
            ):
                pop[i].evaluated_at = -1
        return pop


class MutationOffspring(Layer):
    """Append independently mutated copies of uniformly sampled parents.

    This is the clean counterpart to GeneSpace's second operator stage. Parents
    remain intact, every returned child is a distinct object, and sampling may
    be with or without replacement. Place it after an evaluate/sort/cap block
    to run crossover selection and mutation selection as separate stages.
    """

    def __init__(self, amount, rate=0.1, sigma=0.1, binary=False,
                 replace=True, ensure_change=True):
        super().__init__()
        self.amount = make_callable(amount)
        self.rate = make_callable(rate)
        self.sigma = make_callable(sigma)
        self.binary = bool(binary)
        self.replace = bool(replace)
        self.ensure_change = bool(ensure_change)

    def __call__(self, pop):
        if not pop:
            return pop
        amount = int(self.amount())
        rate = float(self.rate())
        sigma = float(self.sigma())
        if amount < 0:
            raise ValueError("mutation offspring amount cannot be negative")
        if not 0 <= rate <= 1:
            raise ValueError("mutation rate must be in [0, 1]")
        if sigma < 0:
            raise ValueError("mutation sigma cannot be negative")
        if not self.replace and amount > len(pop):
            raise ValueError(
                "cannot sample more mutation parents than the population "
                "without replacement"
            )
        if amount == 0:
            return pop

        parent_indices = np.random.choice(
            len(pop), size=amount, replace=self.replace
        )
        offspring = []
        for index in np.atleast_1d(parent_indices):
            genes = pop[index].genes.copy()
            _mutate_genes(
                genes, rate, sigma, self.binary, self.ensure_change
            )
            offspring.append(LatentIndividual(genes))
        return pop + offspring


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
            fitnesses = self.fitness_fn(phenotypes)
            if isinstance(fitnesses, torch.Tensor):
                fitnesses = fitnesses.detach().reshape(-1).cpu().tolist()
            else:
                fitnesses = list(fitnesses)
            if len(fitnesses) != len(batch):
                raise ValueError(
                    f"fitness_fn returned {len(fitnesses)} values for "
                    f"{len(batch)} phenotypes"
                )
            for ind, fit in zip(batch, fitnesses):
                ind.fitness = float(fit)
                if math.isnan(ind.fitness):
                    raise ValueError("fitness_fn returned NaN")
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
    """Improve the decoder after every completed ``every`` generations.

    ``None`` disables refinement. Place this after ``Sort`` and follow it with
    ``DecodeAndEvaluate`` so a generation never ends with stale fitness.
    """

    def __init__(self, every=10, mode=TrainMode.SELF_DISTILL, percent=0.4,
                 epochs=1, batch_size=32, verbose=False, trainer=None,
                 fitness_fn=None):
        super().__init__()
        if every is not None and every < 1:
            raise ValueError("every must be at least 1 or None")
        self.every = every
        self.mode = mode
        self.percent = percent
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.trainer = trainer
        self.fitness_fn = fitness_fn
        self._calls = 0
        self.last_loss = None

    def __call__(self, pop):
        self._calls += 1
        if self.every is None:
            return pop
        if len(pop) >= 2 and self._calls % self.every == 0:
            if self.trainer is not None:
                loss = self.trainer.step(
                    self.env.decoder, pop, fitness_fn=self.fitness_fn
                )
            elif not self.env.decoder.supports_refinement:
                raise TypeError(
                    "decoder does not support refinement; set refine_every=None "
                    "or implement Decoder.refine"
                )
            else:
                loss = self.env.decoder.refine(
                    pop, mode=self.mode, percent=self.percent,
                    batch_size=self.batch_size, epochs=self.epochs)
            self.last_loss = loss
            if self.verbose:
                print(f"  [decoder] refined loss={loss:.5f} -> v{self.env.decoder.version}")
        return pop


class EvolveDecoder(Layer):
    """Improve decoder weights with a simple perturb-and-select ES step."""

    def __init__(self, fitness_fn, every=10, n_candidates=8, percent=0.4,
                 sigma=None, verbose=False):
        super().__init__()
        if every < 1:
            raise ValueError("every must be at least 1")
        if n_candidates < 1:
            raise ValueError("n_candidates must be at least 1")
        self.fitness_fn = fitness_fn
        self.every = every
        self.n_candidates = n_candidates
        self.percent = percent
        self.sigma = sigma
        self.verbose = verbose
        self._calls = 0
        self.last_fitness = None

    def __call__(self, pop):
        self._calls += 1
        if len(pop) < 2 or self._calls % self.every:
            return pop
        decoder = self.env.decoder
        if not decoder.supports_evolution:
            raise TypeError("decoder does not implement Decoder.evolve_step")
        previous_version = decoder.version
        self.last_fitness = decoder.evolve_step(
            pop,
            fitness_fn=self.fitness_fn,
            n_candidates=self.n_candidates,
            percent=self.percent,
            sigma=self.sigma,
        )
        if self.verbose:
            accepted = decoder.version != previous_version
            print(
                f"  [decoder-es] fitness={self.last_fitness:.5f} "
                f"accepted={accepted} -> v{decoder.version}"
            )
        return pop
