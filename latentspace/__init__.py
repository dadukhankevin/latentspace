"""latentspace -- one universal genetic algorithm for any problem.

The solver (`latentspace.universal.solve`) evolves individuals that are
genes plus latents read by ONE shared decoder network — never the
phenotype — one species per fitness function you provide, and folds discoveries into the decoder over time. Change the
fitness function(s), the output shape, and (optionally) the decoder
architecture; nothing else changes. See FINDINGS.md for the benchmark
campaign behind this design.

    from latentspace.universal import solve
    result = solve(fitness_fn, output_shape=(32, 32), epochs=1_000)

The original co-evolving `Evolver` API below is retained for research use.
"""
from .core import (Environment, GenePool, Individual, LatentIndividual, Layer,
                   Schedule, SolutionSnapshot, make_callable)
from .decoder import Decoder, MLPDecoder, TrainMode
from .evolver import Evolver
from .layers import (Cap, Crossover, DecodeAndEvaluate, EvolveDecoder, Mutate,
                     MutationOffspring, Populate, RefineDecoder, Sort)
from .selection import (RandomSelection, RankSelection, Selection,
                        TournamentSelection, TruncationSelection)
from .training import (AdaptiveMixtureTrainer, AdvantageWeightedTrainer,
                       BacktrackingTrainer, ContrastiveTrainer, DecoderTrainer,
                       DistillationTrainer, FrozenTrainer, GuardedTrainer,
                       MixtureTrainer, PermutationTrainer,
                       PolicyGradientTrainer)
from .universal import GAResult, solve

__all__ = [
    "solve", "GAResult",
    "Individual", "LatentIndividual", "GenePool", "Layer", "Environment",
    "Schedule", "SolutionSnapshot", "make_callable", "Decoder", "MLPDecoder",
    "TrainMode",
    "Populate", "Crossover", "Mutate", "MutationOffspring",
    "DecodeAndEvaluate", "Sort", "Cap",
    "RefineDecoder", "EvolveDecoder", "Selection", "RandomSelection", "TournamentSelection",
    "RankSelection", "TruncationSelection", "Evolver",
    "DecoderTrainer", "FrozenTrainer", "DistillationTrainer",
    "ContrastiveTrainer", "PolicyGradientTrainer", "AdvantageWeightedTrainer",
    "PermutationTrainer", "BacktrackingTrainer",
    "MixtureTrainer",
    "GuardedTrainer",
    "AdaptiveMixtureTrainer",
]
