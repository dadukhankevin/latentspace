"""latentspace -- one evolutionary algorithm for any problem.

Evolve a universal latent vector; a single co-evolving neural decoder maps it to
a phenotype of any shape. Change the fitness function and the output shape;
nothing else changes.

(Working name -- rename freely; it's just the package directory.)
"""
from .core import (Environment, GenePool, Individual, LatentIndividual, Layer,
                   Schedule, make_callable)
from .decoder import MLPDecoder, TrainMode
from .evolver import Evolver
from .layers import (Cap, Crossover, DecodeAndEvaluate, Mutate, Populate,
                     RefineDecoder, Sort)
from .selection import (RandomSelection, RankSelection, Selection,
                        TournamentSelection, TruncationSelection)

__all__ = [
    "Individual", "LatentIndividual", "GenePool", "Layer", "Environment",
    "Schedule", "make_callable", "MLPDecoder", "TrainMode",
    "Populate", "Crossover", "Mutate", "DecodeAndEvaluate", "Sort", "Cap",
    "RefineDecoder", "Selection", "RandomSelection", "TournamentSelection",
    "RankSelection", "TruncationSelection", "Evolver",
]
