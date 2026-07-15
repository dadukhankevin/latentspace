"""The universal solver: explore -> distill -> exploit, one fitness function.

The method that first beat a traditional GA under the universality
constraint (image target, 10/0 at matched budget — see FINDINGS.md):

  1. EXPLORE  — per-individual decoder evolution. Every individual owns a
     genome and its own decoder weights; children mutate both. No operator
     ever touches the phenotype, so the same code runs for any modality.
  2. DISTILL  — compress the run's best fitness-vetted solutions (capped
     per lineage to keep their errors independent) into a small linear
     latent space.
  3. EXPLOIT  — CMA-ES over that latent space with the remaining budget.

The explore/exploit split is adaptive by default: exploration ends when
its best score stalls, and a reserve (10x latent) guarantees the exploit
phase enough evaluations to converge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .architectures import resolve
from .cma import cma_minimize
from .distill import LatentSpace, distill
from .explorer import Archive, ExplorerConfig, PerIndividualExplorer


def _auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class SolveResult:
    best_phenotype: np.ndarray      # shape = output_shape, values in [0, 1]
    best_fitness: float
    evaluations: int
    explore_evaluations: int
    history: list = field(repr=False, default_factory=list)  # best-so-far per evaluation
    latent_space: LatentSpace | None = field(repr=False, default=None)
    archive: Archive | None = field(repr=False, default=None)


def solve(fitness_fn, output_shape, budget=5_000, architecture="auto",
          latent=32, device="auto", explore_fraction="auto",
          distill_top=200, lineage_cap=None, explorer_config=None,
          seed=None) -> SolveResult:
    """Maximize `fitness_fn` over phenotypes of `output_shape` in [0, 1].

    fitness_fn: callable taking a torch tensor (B, *output_shape) and
        returning B fitness values, higher better. It is the ONLY
        problem-specific code. `architecture` may name a registered decoder
        shape ("mlp", "conv1d", "conv2d"), be "auto" (chosen from
        output_shape), or be a callable `(latent, output_shape) -> nn.Module`.
    explore_fraction: "auto" ends exploration when it stalls (recommended);
        a float in (0, 1] fixes the split.
    """
    output_shape = tuple(int(s) for s in output_shape)
    device = _auto_device() if device == "auto" else device
    rng = np.random.default_rng(seed)
    builder = resolve(architecture, latent, output_shape)

    spent = 0
    best_fitness = -math.inf
    best_phenotype: np.ndarray | None = None
    history: list[float] = []

    def evaluate_losses(phenotypes: torch.Tensor) -> np.ndarray:
        nonlocal spent, best_fitness, best_phenotype
        values = fitness_fn(phenotypes)
        if isinstance(values, torch.Tensor):
            values = values.detach().reshape(-1).cpu().numpy()
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) != len(phenotypes):
            raise ValueError(
                f"fitness_fn returned {len(values)} values for "
                f"{len(phenotypes)} phenotypes")
        flat = phenotypes.detach().cpu().numpy().reshape(len(values), -1)
        for value, phenotype in zip(values, flat):
            spent += 1
            if value > best_fitness:
                best_fitness = float(value)
                best_phenotype = phenotype.copy()
            history.append(best_fitness)
        return -values

    explorer = PerIndividualExplorer(builder, latent, device,
                                     explorer_config or ExplorerConfig())
    adaptive = explore_fraction == "auto"
    explore_budget = (budget if adaptive
                      else int(budget * float(explore_fraction)))
    reserve = 10 * latent
    archive = explorer.run(
        evaluate_losses, rng,
        stop_after=lambda _n: min(explore_budget, budget) - spent,
        adaptive=adaptive, reserve=reserve,
    )
    explore_evaluations = spent

    latent_space = None
    if spent < budget and len(archive.losses) >= 2:
        idx = archive.select(distill_top, lineage_cap=lineage_cap)
        latent_space = distill(
            archive.phenotypes[idx], latent, output_shape, device=device)
        cma_minimize(
            lambda z: evaluate_losses(latent_space.decode(z)),
            dim=latent, budget_evaluations=budget, evaluations_done=spent,
            rng=rng, mean0=np.zeros(latent), sigma0=1.0,
        )

    assert best_phenotype is not None, "budget too small to evaluate anything"
    return SolveResult(
        best_phenotype=best_phenotype.reshape(output_shape),
        best_fitness=best_fitness,
        evaluations=spent,
        explore_evaluations=explore_evaluations,
        history=history,
        latent_space=latent_space,
        archive=archive,
    )
