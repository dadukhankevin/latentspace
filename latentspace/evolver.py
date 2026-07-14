"""The friendly front door.

    ev = Evolver(fitness_fn, output_shape=(H, W, 3), device="cuda")   # smooth default
    ev.solve(400)
    phenotype = ev.decode_best()

`fitness_fn(phenotypes)` receives a torch tensor (B, *output_shape) and returns
B scalars. Change the fitness function and the output shape; nothing else.

Two presets, because empirically there is no single best config -- the decoder's
refinement is a *convergence amplifier*:

  * Evolver(...)                         -> SMOOTH preset. Lean into the
      co-evolving decoder: wide-shallow net, low LR, frequent self-distillation,
      strong selection. Best for problems with a well-defined phenotype target
      (match an image/vector, function approximation, generative targets).

  * Evolver.combinatorial(...)           -> preserve diversity: mild selection,
      gentle/rare refinement. Best for permutation/deceptive/multimodal problems
      (TSP, scheduling) where premature convergence is fatal. For the hardest
      combinatorial cases, set refine_every very large to freeze the decoder into
      a fixed random projection and let the GA do all the searching.

The single most important dial is `refine_every`: small = exploit, large = explore.
"""
from __future__ import annotations

import numpy as np

from .core import Environment, GenePool
from .decoder import MLPDecoder, TrainMode
from .layers import (Cap, Crossover, DecodeAndEvaluate, Mutate, Populate,
                     RefineDecoder, Sort)
from .selection import RankSelection


class Evolver:
    def __init__(self, fitness_fn, output_shape, device="cpu",
                 latent=250, population=200, hidden_size=2000, num_layers=1,
                 lr=1e-5, binary=False, mutation_rate=0.1, mutation_sigma=0.1,
                 refine_every=10, refine_percent=0.4, mode=TrainMode.SELF_DISTILL,
                 pressure=20.0, scheme="exp", families=None, children=4, n_points=8):
        self.decoder = MLPDecoder(
            input_length=latent, output_shape=output_shape, hidden_size=hidden_size,
            num_layers=num_layers, lr=lr, device=device)
        self.genepool = GenePool(latent, binary=binary)
        families = families if families is not None else max(16, population // 12)
        self.env = Environment(
            layers=[
                Populate(population),
                Crossover(RankSelection(pressure=pressure, scheme=scheme),
                          families=families, children=children, n_points=n_points),
                Mutate(rate=mutation_rate, sigma=mutation_sigma, percent=1.0, binary=binary),
                DecodeAndEvaluate(fitness_fn),
                Sort(),
                RefineDecoder(every=refine_every, mode=mode, percent=refine_percent),
                Cap(population),
            ],
            genepool=self.genepool,
            decoder=self.decoder,
        ).compile()

    @classmethod
    def combinatorial(cls, fitness_fn, output_shape, **kw):
        """Diversity-preserving preset for permutation/deceptive problems."""
        defaults = dict(pressure=1.8, scheme="linear", refine_every=50,
                        mode=TrainMode.SELF_DISTILL)
        defaults.update(kw)
        return cls(fitness_fn, output_shape, **defaults)

    def solve(self, generations=300, verbose_every=10):
        return self.env.evolve(generations, verbose_every=verbose_every)

    @property
    def best(self):
        return self.env.best_ever

    def decode_best(self):
        genes = np.stack([self.env.best_ever.genes]).astype(np.float32)
        return self.decoder.decode(genes)[0]

    def plot(self):
        self.env.plot()
