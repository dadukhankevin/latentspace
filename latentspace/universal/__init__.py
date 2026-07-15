"""The universal solver: one fitness function, one algorithm, any modality.

    from latentspace.universal import solve

    result = solve(fitness_fn, output_shape=(32, 32), budget=5_000)
    result.best_phenotype   # (32, 32) array in [0, 1]

Modules (each phase is replaceable):
    architectures — decoder shapes ("mlp", "conv1d", "conv2d", register your own)
    explorer      — per-individual decoder evolution (genome + private weights)
    distill       — PCA compression of vetted solutions into a latent space
    cma           — self-contained CMA-ES for the exploit phase
    solver        — the explore -> distill -> exploit orchestration
"""
from .architectures import build_mlp, register_architecture, resolve
from .cma import cma_minimize
from .distill import LatentSpace, distill
from .explorer import Archive, ExplorerConfig, PerIndividualExplorer
from .solver import SolveResult, solve

__all__ = [
    "solve", "SolveResult",
    "PerIndividualExplorer", "ExplorerConfig", "Archive",
    "distill", "LatentSpace",
    "cma_minimize",
    "register_architecture", "resolve", "build_mlp",
]
