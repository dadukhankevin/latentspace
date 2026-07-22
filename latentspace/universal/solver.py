"""The universal solver: a pure decoder GA, one fitness function.

By default the whole budget goes to decoder evolution (EXPLORE below),
restarting fresh on stall — the exploit ablation (2026-07-21) measured
this as the robust configuration. The optional distill -> exploit tail
and the cycle mode reuse the same phases:

  1. EXPLORE  — per-individual decoder evolution. Every individual owns a
     genome and its own decoder weights; children mutate both. No operator
     ever touches the phenotype, so the same code runs for any modality.
  2. DISTILL  — (exploit="ga" only) compress the run's best fitness-vetted
     solutions (capped per lineage to keep their errors independent) into
     a small linear latent space.
  3. EXPLOIT  — (exploit="ga" only) a latent-space GA (selection + uniform
     crossover + win-rate mutation) evolves genotypes feeding that
     distilled decoder with the remaining budget. CMA-ES is NOT used
     anywhere in this pipeline — it is a baseline the method competes
     against (Daniel's ruling; cma.py exists only so benchmarks can field
     it as an opponent arm).

No phase ever mutates a phenotype: solutions are only computed from a
genotype through a decoder, then scored. Exploration searches
(genome, decoder-weight) pairs; exploitation searches genomes feeding the
distilled decoder. That invariant is the universality guarantee.

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
from .distill import LatentSpace, distill
from .explorer import Archive, ExplorerConfig, PerIndividualExplorer
from .exploit import ga_minimize


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

    @property
    def problems(self):
        """One-problem view matching MultiResult.problems, so code written
        against the unified solve() reads either result the same way."""
        from .multi import ProblemResult
        return [ProblemResult(
            best_phenotype=self.best_phenotype,
            best_fitness=self.best_fitness,
            initial_fitness=(self.history[0] if self.history
                             else self.best_fitness),
            evaluations=self.evaluations)]


def _weights_from_space(builder, space, latent, rng, device,
                        samples=2_048, steps=1_500):
    """Decompress a distilled latent space into decoder weights by
    supervised regression — costs no fitness evaluations. Used to warm-
    start re-entered exploration around the current knowledge."""
    net = builder().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    z = torch.as_tensor(
        rng.standard_normal((samples, latent)).astype(np.float32),
        device=device)
    with torch.no_grad():
        targets = torch.logit(
            space.decode(z.cpu().numpy()).clamp(1e-3, 1 - 1e-3)).flatten(1)
    generator = torch.Generator().manual_seed(int(rng.integers(0, 2**31)))
    for _ in range(steps):
        idx = torch.randint(0, samples, (128,), generator=generator).to(device)
        optimizer.zero_grad()
        loss = loss_fn(net(z[idx]), targets[idx])
        loss.backward()
        optimizer.step()
    return torch.nn.utils.parameters_to_vector(
        net.parameters()).detach().cpu().numpy().astype(np.float32)


def _merge(a: Archive, b: Archive) -> Archive:
    offset = int(a.lineages.max()) + 1 if len(a.lineages) else 0
    return Archive(
        np.concatenate([a.phenotypes, b.phenotypes]),
        np.concatenate([a.losses, b.losses]),
        np.concatenate([a.lineages, b.lineages + offset]),
    )


def solve_single(fitness_fn, output_shape, budget=5_000, architecture="auto",
                 latent=64, device="auto", explore_fraction="auto",
                 distill_top=200, lineage_cap=None, explorer_config=None,
                 phases="single", exploit=None, exploit_stall_window=20,
                 seed=None) -> SolveResult:
    """Maximize `fitness_fn` over phenotypes of `output_shape` in [0, 1].

    fitness_fn: callable taking a torch tensor (B, *output_shape) and
        returning B fitness values, higher better. It is the ONLY
        problem-specific code. `architecture` may name a registered decoder
        shape ("mlp", "conv1d", "conv2d"), be "auto" (chosen from
        output_shape), or be a callable `(latent, output_shape) -> nn.Module`.
    explore_fraction: "auto" ends exploration when it stalls (recommended);
        a float in (0, 1] fixes the split.
    exploit: None (default) resolves to "off" for phases="single" and "ga"
        for phases="cycle". "off" — pure decoder evolution for the whole
        budget, restarting exploration fresh whenever it stalls; the
        measured robust default (the exploit ablation found the distilled
        space itself detonates ~3/10 image seeds and any optimizer confined
        to it is trapped — GA and CMA exploit failures correlate 0.97 —
        while pure evolution sails through every seed at negligible cost on
        smooth signals). "ga" — after distillation, a latent-space GA
        (truncation selection + uniform crossover + win-rate mutation)
        spends the remaining budget. CMA-ES is deliberately not an option —
        it is a baseline the method competes against, never a component.
    phases: "single" runs explore -> distill -> exploit once; "cycle"
        alternates — when exploitation stalls too (exploit_stall_window
        generations without meaningful progress), exploration re-enters,
        half its population warm-started from the current distilled
        knowledge decompressed into decoder weights plus noise (the
        off-manifold escape channel) and half fresh; the archive is
        cumulative and each cycle re-distills a richer space.
    latent: size of the genome AND of the distilled search space. The
        benchmarked response is a cliff below ~32 and a broad plateau at
        32-128; 64 had the best means and the lowest variance. Scale it
        with the intrinsic variety of good solutions (not with raw output
        size), and scale `distill_top` alongside — the distilled space is
        fit from that many solutions, so keep distill_top >= ~3x latent.
    """
    if exploit is None:
        exploit = "ga" if phases == "cycle" else "off"
    if phases == "cycle" and exploit == "off":
        raise ValueError(
            "phases='cycle' re-enters exploration from the distilled "
            "exploit hand-off; it needs exploit='ga'")
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
    # With the exploit phase off there is nothing to save budget for:
    # exploration owns everything regardless of explore_fraction.
    explore_budget = (budget if adaptive or exploit == "off"
                      else int(budget * float(explore_fraction)))
    reserve = 0 if exploit == "off" else 10 * latent

    archive: Archive | None = None
    latent_space = None
    explore_evaluations = 0
    warm_theta = None
    while spent < budget:
        part = explorer.run(
            evaluate_losses, rng,
            stop_after=lambda _n: min(explore_budget, budget) - spent,
            adaptive=adaptive, reserve=reserve,
            warm_theta=warm_theta,
        )
        archive = part if archive is None else _merge(archive, part)
        explore_evaluations += len(part.losses)
        if spent >= budget or len(archive.losses) < 2:
            break
        if exploit == "off":
            continue    # stalled with budget left: restart exploration fresh
        idx = archive.select(distill_top, lineage_cap=lineage_cap)
        latent_space = distill(
            archive.phenotypes[idx], latent, output_shape, device=device)
        ga_minimize(
            lambda z: evaluate_losses(latent_space.decode(z)),
            dim=latent, budget_evaluations=budget, evaluations_done=spent,
            rng=rng, mean0=np.zeros(latent), sigma0=1.0,
            stall_window=exploit_stall_window if phases == "cycle" else None,
        )
        if phases != "cycle" or spent >= budget:
            break
        # exploitation stalled with budget left: decompress the current
        # knowledge into decoder weights and hand the budget back to
        # exploration.
        warm_theta = _weights_from_space(builder, latent_space, latent,
                                         rng, device)

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


# Keyword arguments that only the multi-problem engine understands; their
# presence routes even a one-element list of fitness functions there.
_MULTI_ONLY = {
    "coefficient_dim", "slots_per_problem", "children", "consolidate",
    "init_decoder", "crossover_rate", "crossover_mode", "crossover_cuts",
    "compat_distance", "latent_sigma_scale", "initial_gain",
    "progress", "progress_every",
}


def solve(fitness, output_shape, budget=5_000, **kwargs):
    """THE entry point: solve any problem(s) given only fitness and shape.

    `fitness` is a single callable or a list of them; every callable takes
    a torch tensor (B, *output_shape) in [0, 1] and returns B fitness
    values, higher better. One fitness function is simply the one-problem
    case of the same call.

    Two measured engines sit underneath, chosen by problem count:
    - one problem  -> per-individual decoder evolution (solve_single), the
      configuration holding every single-fitness record;
    - many problems -> one shared conditional-LoRA decoder population
      (solve_many): at equal per-problem compute it beats separate runs at
      every budget tested, most when evaluations are scarce.
    Unifying the two engines' machinery is open problem #5 in FINDINGS.md;
    unifying the API is this function. Passing any multi-engine keyword
    (children, consolidate, crossover_rate, ...) routes a one-element list
    to the multi engine so those knobs are honoured.
    """
    from .multi import solve_many
    if callable(fitness):
        return solve_single(fitness, output_shape, budget=budget, **kwargs)
    fns = list(fitness)
    if not fns:
        raise ValueError("at least one fitness function is required")
    if len(fns) == 1 and not (_MULTI_ONLY & set(kwargs)):
        return solve_single(fns[0], output_shape, budget=budget, **kwargs)
    return solve_many(fns, output_shape, budget=budget, **kwargs)
