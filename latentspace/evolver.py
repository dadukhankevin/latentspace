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

from .core import Environment, GenePool, Layer
from .decoder import Decoder, MLPDecoder, TrainMode
from .layers import (Cap, Crossover, DecodeAndEvaluate, Mutate,
                     MutationOffspring, Populate, RefineDecoder, Sort)
from .selection import RankSelection
from .training import DecoderTrainer


class Evolver:
    def __init__(self, fitness_fn, output_shape, device="cpu",
                 latent=250, population=200, hidden_size=2000, num_layers=1,
                 lr=1e-5, binary=False, mutation_rate=0.1, mutation_sigma=0.1,
                 refine_every=10, refine_percent=0.4, mode=TrainMode.SELF_DISTILL,
                 pressure=20.0, scheme="exp", families=None, children=4, n_points=8,
                 offspring_only_mutation=False,
                 operator_schedule="single_stage", mutation_children=None,
                 mutation_with_replacement=True, ensure_mutation=True,
                 decoder: Decoder | None = None, decoder_update: Layer | None = None,
                 trainer: DecoderTrainer | None = None):
        if latent < 1:
            raise ValueError("latent must be at least 1")
        if population < 2:
            raise ValueError("population must be at least 2")
        if operator_schedule not in {"single_stage", "two_stage"}:
            raise ValueError(
                "operator_schedule must be 'single_stage' or 'two_stage'"
            )

        if decoder is None:
            decoder = MLPDecoder(
                input_length=latent, output_shape=output_shape, hidden_size=hidden_size,
                num_layers=num_layers, lr=lr, device=device)
        elif not isinstance(decoder, Decoder):
            raise TypeError("custom decoders must inherit from latentspace.Decoder")
        elif decoder.input_length != latent:
            raise ValueError(
                f"decoder input_length {decoder.input_length} does not match latent {latent}"
            )
        elif decoder.output_shape != tuple(output_shape):
            raise ValueError(
                f"decoder output_shape {decoder.output_shape} does not match {tuple(output_shape)}"
            )
        if decoder_update is not None and trainer is not None:
            raise ValueError("decoder_update and trainer are mutually exclusive")
        if trainer is not None and not isinstance(trainer, DecoderTrainer):
            raise TypeError("trainer must implement DecoderTrainer")
        if decoder_update is not None and refine_every is not None:
            raise ValueError(
                "decoder_update replaces gradient refinement; set refine_every=None"
            )
        if (decoder_update is None and trainer is None and refine_every is not None
                and not decoder.supports_refinement):
            raise ValueError(
                "custom decoder does not support refinement; set refine_every=None "
                "or implement Decoder.refine"
            )

        self.decoder = decoder
        self.genepool = GenePool(latent, binary=binary)
        families = families if families is not None else max(16, population // 12)
        update_layer = decoder_update or RefineDecoder(
            every=refine_every, mode=mode, percent=refine_percent,
            trainer=trainer, fitness_fn=fitness_fn)
        operator_layers = [
            Crossover(
                RankSelection(pressure=pressure, scheme=scheme),
                families=families,
                children=children,
                n_points=n_points,
            )
        ]
        if operator_schedule == "single_stage":
            operator_layers += [
                Mutate(
                    rate=mutation_rate,
                    sigma=mutation_sigma,
                    percent=1.0,
                    binary=binary,
                    offspring_only=offspring_only_mutation,
                    ensure_change=ensure_mutation,
                ),
                DecodeAndEvaluate(fitness_fn),
                Sort(),
                Cap(population),
            ]
        else:
            mutation_children = (
                population if mutation_children is None else mutation_children
            )
            operator_layers += [
                DecodeAndEvaluate(fitness_fn),
                Sort(),
                Cap(population),
                MutationOffspring(
                    amount=mutation_children,
                    rate=mutation_rate,
                    sigma=mutation_sigma,
                    binary=binary,
                    replace=mutation_with_replacement,
                    ensure_change=ensure_mutation,
                ),
                DecodeAndEvaluate(fitness_fn),
                Sort(),
                Cap(population),
            ]
        self.env = Environment(
            layers=[
                Populate(population),
                # Parents must be current before selection, including generation 0
                # and the generation after any decoder update.
                DecodeAndEvaluate(fitness_fn),
                Sort(),
                *operator_layers,
                update_layer,
                # Refinement changes the genotype-to-phenotype map. Re-score now,
                # not after the next generation has already selected parents.
                DecodeAndEvaluate(fitness_fn),
                Sort(),
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
        return self.env.best_observed

    @property
    def best_current(self):
        return self.env.best_current

    def decode_best(self):
        """Return the phenotype snapshot that achieved ``best.fitness``."""
        if self.env.best_observed is None:
            raise RuntimeError("solve must be called before decode_best")
        return self.env.best_observed.phenotype

    def decode_current_best(self):
        """Decode the current population leader under the current decoder."""
        if self.env.best_current is None:
            raise RuntimeError("solve must be called before decode_current_best")
        genes = np.stack([self.env.best_current.genes]).astype(np.float32)
        return self.decoder.decode(genes)[0]

    def plot(self):
        self.env.plot()
