"""Solve many problems in one population — one shared decoder, ever.

Architecture invariant (enforced here and in conditional.py): there is
exactly ONE decoder. No individual owns weights. Each individual is only
its GENES — the network input `z` (the universal genotype) plus a
coefficient vector that gates the decoder's shared LoRA directions. The
two halves are different kinds of object and must not be assumed to want
identical operators; where the code currently treats them alike (one
mutation sigma, one crossover cut, full-vector compatibility distance),
that is a flagged assumption, not a validated law. The shared backbone is
trained only at consolidation, at zero fitness-evaluation cost.

Every fitness function gets one champion lineage (slots_per_problem=1 by
default — per-problem mini-populations were falsified, see the doc); a
child inherits its problem from its base parent and is scored against
that problem ONLY — no problem pays for another's evaluations, sleeping
problems cost nothing, and cached fitness never goes stale.

Why one population instead of one run per problem: measured head-to-head
on ten image problems at exactly equal evaluations per problem, the shared
population won at every budget (+16 points at 1,500 evaluations each,
shrinking to +4 at 30,000; related problems roughly double the gap). The
advantage is a head start, not a ceiling change — arms converge at long
budgets.

Operator choices, each carrying its benchmark result:
- Parents are drawn uniformly from the whole living population. Fully-mixed
  one-at-a-time selection beat clutch/burst breeding (which also broke
  coverage) and beat improvement-biased weights (no measured gain).
- Crossover is RARE and compatibility-gated: it fires only when the drawn
  mate is among the closest `crossover_rate` fraction of gene distances
  that generation (default 0.05). Rarity is load-bearing (always-crossing
  removed 3% of error vs 81% at rate 0.05), the frequency bound is what
  prevents the runaway-crossover collapse (mate-selection round 5 measured
  the unbounded lottery collapsing ~1/4 of seeds), and the compatibility
  gate is what lets crossover help across objectives (+2.2 at the default
  vs blind-random rare crossover).
- Step size is win-rate controlled per problem, ties counted as successes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .conditional import build_conditional_decoder
from .explorer import ExplorerConfig


def _learn_genes(genes, n_bins=4, protect_frac=0.5):
    """Discover contiguous 'gene' blocks of the genome that coadapt in the
    current population (BPE-style: merge high-mutual-information neighbours);
    crossover then cuts only between weakly-coupled blocks. Returns (start,end).
    """
    M, L = genes.shape
    if M < 8 or L < 2:
        return [(0, L)]
    sym = np.empty((M, L), dtype=np.int64)
    for d in range(L):
        edges = np.quantile(genes[:, d], np.linspace(0, 1, n_bins + 1)[1:-1])
        sym[:, d] = np.digitize(genes[:, d], edges)
    mi = np.empty(L - 1)
    for i in range(L - 1):
        joint = np.histogram2d(sym[:, i], sym[:, i + 1],
                               bins=n_bins, range=[[0, n_bins]] * 2)[0]
        p = joint / joint.sum()
        px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        nz = p > 0
        mi[i] = float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())
    n_protect = int(round(protect_frac * (L - 1)))
    if n_protect <= 0:
        return [(d, d + 1) for d in range(L)]
    cut = mi < np.sort(mi)[::-1][min(n_protect, L - 2)]
    blocks, start = [], 0
    for i in range(L - 1):
        if cut[i]:
            blocks.append((start, i + 1))
            start = i + 1
    blocks.append((start, L))
    return blocks


@dataclass
class ProblemResult:
    best_phenotype: np.ndarray
    best_fitness: float
    initial_fitness: float
    evaluations: int


@dataclass
class MultiResult:
    problems: list[ProblemResult]
    evaluations: int
    history: list[dict] = field(default_factory=list)
    decoder: np.ndarray | None = None
    consolidations: int = 0

    @property
    def best_fitnesses(self) -> np.ndarray:
        return np.array([p.best_fitness for p in self.problems])

    # Single-problem ergonomics, so a one-element solve() call reads the
    # same regardless of which engine ran it.
    @property
    def best_phenotype(self) -> np.ndarray:
        if len(self.problems) != 1:
            raise ValueError("best_phenotype is ambiguous for "
                             f"{len(self.problems)} problems; use .problems")
        return self.problems[0].best_phenotype

    @property
    def best_fitness(self) -> float:
        if len(self.problems) != 1:
            raise ValueError("best_fitness is ambiguous for "
                             f"{len(self.problems)} problems; use .problems")
        return self.problems[0].best_fitness


def solve_many(fitness_fns, output_shape, budget=5_000, architecture="auto",
               latent=64, coefficient_dim=None, slots_per_problem=1, children=32,
               device="auto", explorer_config=None, consolidate="off",
               init_decoder=None, crossover_rate=0.05,
               crossover_mode="one_point", crossover_cuts="single",
               compat_distance="all", latent_sigma_scale="auto",
               initial_gain=64.0, progress=None, progress_every=None,
               seed=None) -> MultiResult:
    """Maximize every function in `fitness_fns` with one shared population.

    Architecture invariant: there is exactly ONE decoder. A single shared
    backbone carries shared LoRA directions; each individual's genome is its
    network input `z` plus a `coefficient_dim` vector that gates those shared
    directions (zero coefficients reproduce the backbone). Crossover and
    mutation act on the genes (z + coefficients); the shared decoder is trained
    only at consolidation. Never per-individual decoders.

    Each fitness function takes a torch tensor (B, *output_shape) in [0, 1] and
    returns B fitness values, higher better. `budget` counts phenotype
    evaluations; each child costs one, charged to the problem it inherits.

    consolidate: periodically FOLD the population's discoveries into the shared
        backbone (train it so the coefficient-zero output reproduces champion
        phenotypes — zero fitness evaluations), then reseed each problem's WORST
        member onto the improved backbone (coefficients zeroed; re-scored and
        charged to the budget). "champions" | "breeders" | "off" (default).
    init_decoder: a shared decoder from a previous run (`result.decoder`) —
        the backbone starts from it. (Caveat: at conv strength the inherited
        core HURT transfer in 3k; treat as experimental.)

    The genes/latents boundary (genes = z = the first `latent` dims; latents
    = the LoRA coefficients = the rest). Three knobs control whether the
    operators respect that boundary. Matrix verdict (2026-07-21, 5 paired
    seeds, image + curve): the MUTATION boundary matters — see
    latent_sigma_scale below, "auto" shipped as default; the crossover-cut
    and compatibility-distance boundaries measured as ties (gene-only cuts
    leaning harmful; the z-only gate positive alone but interfering with
    the auto ratio when combined), so their boundary-blind defaults stand.
    crossover_cuts: "single" (one cut across the concatenated genome),
        "separate" (independent cut in each half — the benchmark harness's
        operator), "genes_only" (latents inherited whole from the base
        parent), "latents_only" (genes inherited whole).
    compat_distance: which space the crossover-compatibility gate measures —
        "all" (full genome), "genes" (z only — the species-era harness
        rule), "latents" (coefficients only).
    slots_per_problem: living slots per problem. Default 1 — champion-only
        — after Daniel's challenge falsified per-problem mini-populations
        (2026-07-21): quality degrades MONOTONICALLY with added slots on
        both benchmarks (13/13 paired seeds, t = 4.8-13.7; image 58.2% at
        1 slot -> 45.1% at 8), because uniform parent selection breeds
        mostly from sub-champion members while the held diversity never
        pays. The irreducible per-problem state is a tag, a champion, and
        a gain — nothing more. (Formerly "family_size"; the family
        vocabulary is retired.) Note: consolidation's reseed-worst step
        needs >= 2 slots and no-ops at 1 (folding still runs).
    initial_gain: where each problem's win-rate step controller starts.
        Default 64 — hot — measured 2026-07-21: the cold start (1.0) taxes
        every scale because each problem's controller ramps on its own
        sparse child cadence — at 1,024 problems warmup ate ~9x of a 160k
        run (1.4% vs 12.5% hot), and hot also won at 16 problems (+2.7,
        t = 5.0) and on curves (+8.0). 512 is better still at extreme
        problem counts (14.8% at 1,024) but overshoots on curves (one
        seed detonated); 64 is best-or-near-best everywhere tested. The
        controller self-corrects downward quickly, so a hot start is
        cheap insurance. (This is solve_many's start; the single-problem
        explorer's measured best start remains 1.0 — per-problem child
        cadence, not generation size, is what sets the warmup bill.)
    progress: optional callable, called at most every `progress_every`
        evaluations (default budget/50) with (evaluations, budget,
        champion_phenotypes, champion_fitnesses) — champion_phenotypes is
        one decoded phenotype per problem, (n_problems, prod(output_shape)).
        Costs one extra batched decode per call, no fitness evaluations.
    latent_sigma_scale: mutation sigma for the latents half relative to the
        genes half. Measured (genes-vs-latents matrix, 2026-07-21): images
        want the latents mutated 2-8x HARDER than genes (+4.4 points at 2x,
        t = 4.41 at 5 seeds — coefficients must grow from exact zero), but
        the curve benchmark leans the OPPOSITE way — the right ratio is
        architecture-dependent, so no fixed value is universal. "auto"
        (default) self-tunes it per run: each child mutates only one
        channel (coin flip), and a shared ratio climbs toward whichever
        channel's children are winning (the same comparative win-rate logic
        as the step controller). A float pins the ratio; 1.0 is the
        historical boundary-blind behavior.
    """
    n_problems = len(fitness_fns)
    if n_problems < 1:
        raise ValueError("at least one fitness function is required")
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = explorer_config or ExplorerConfig()
    rng = np.random.default_rng(seed)
    coeff_dim = latent if coefficient_dim is None else coefficient_dim
    # The shared decoder's weights ARE part of the run; seed torch so the one
    # backbone + LoRA directions are reproducible for a given seed.
    torch.manual_seed(int(rng.integers(0, 2 ** 31)))
    decoder = build_conditional_decoder(
        architecture, latent, output_shape, coeff_dim, device)
    if init_decoder is not None:
        base = np.asarray(init_decoder, dtype=np.float32)
        if base.shape != (decoder.n_params,):
            raise ValueError("init_decoder does not match the architecture")
        decoder.set_params(base)

    gene_dim = latent + coeff_dim
    slots = slots_per_problem
    capacity = n_problems * slots
    # Genome = [ z (network input) | coefficients (shared-LoRA gates) ].
    pop_g = np.zeros((capacity, gene_dim), dtype=np.float32)
    pop_fit = np.full(capacity, -np.inf, dtype=np.float64)
    problem_of = np.repeat(np.arange(n_problems), slots)
    alive = np.zeros(capacity, dtype=bool)
    problem_gain = np.full(n_problems, float(initial_gain), dtype=np.float64)

    def decode(genes: np.ndarray) -> torch.Tensor:
        return decoder.decode(genes[:, :latent], genes[:, latent:])

    def score(phenotypes: torch.Tensor, problems_idx: np.ndarray) -> np.ndarray:
        values = np.empty(len(problems_idx), dtype=np.float64)
        for f in np.unique(problems_idx):
            picks = np.flatnonzero(problems_idx == f)
            out = fitness_fns[int(f)](phenotypes[picks])
            values[picks] = np.asarray(
                out.detach().cpu().numpy() if torch.is_tensor(out) else out,
                dtype=np.float64)
        return values

    # Latent-ratio controller state ("auto"): one shared ratio, moved toward
    # whichever channel's single-channel children are currently winning.
    auto_ratio = isinstance(latent_sigma_scale, str)
    ratio = 1.0 if auto_ratio else float(latent_sigma_scale)
    ratio_adapts = latent_sigma_scale == "auto"   # "alt" = alternation only

    def mutate(genes: np.ndarray, gain: np.ndarray,
               latents_child: np.ndarray | None = None) -> np.ndarray:
        mask = rng.random(genes.shape) < config.genome_mutation_rate
        noise = rng.normal(0, config.genome_mutation_sigma, genes.shape)
        if ratio != 1.0:
            noise[:, latent:] *= ratio
        if latents_child is not None:
            # Alternating single-channel mutation: clean credit assignment
            # for the ratio controller.
            noise[latents_child, :latent] = 0.0
            noise[~latents_child, latent:] = 0.0
        return (genes + mask * noise * gain[:, None]).astype(np.float32)

    founders = np.arange(n_problems) * slots
    pop_g[founders, :latent] = rng.standard_normal(
        (n_problems, latent)).astype(np.float32)     # coefficients start at 0
    alive[founders] = True
    pop_attempts = np.zeros(capacity, dtype=np.int64)
    pop_wins = np.zeros(capacity, dtype=np.int64)
    pop_fit[founders] = score(decode(pop_g[founders]), problem_of[founders])
    initial_fit = pop_fit[founders].copy()
    spent = n_problems
    exposures = np.ones(n_problems, dtype=np.int64)
    history: list[dict] = []

    progress_interval = (max(1, budget // 50) if progress_every is None
                         else int(progress_every))
    next_progress = progress_interval

    museum_z: list[np.ndarray] = []
    museum_p: list[np.ndarray] = []
    museum_keys: set[bytes] = set()
    births_since_event = 0
    next_event_births = capacity
    consolidations = 0
    genes_blocks = [(0, gene_dim)]

    def consolidation_corpus() -> tuple[np.ndarray, np.ndarray] | None:
        living = np.flatnonzero(alive)
        if consolidate == "breeders":
            tried = living[pop_attempts[living] > 0]
            if len(tried) == 0:
                rows = np.array([f * slots for f in range(n_problems)])
            else:
                rates = pop_wins[tried] / pop_attempts[tried]
                rows = tried[rates >= np.median(rates)]
        else:  # champions
            rows = np.array([f * slots for f in range(n_problems)])
        rows = rows[alive[rows]]
        if len(rows) == 0:
            return None
        flat = decode(pop_g[rows]).detach().cpu().numpy()
        for z, p in zip(pop_g[rows, :latent], flat):
            key = z.tobytes()
            if key not in museum_keys:
                museum_keys.add(key)
                museum_z.append(z.copy())
                museum_p.append(p.astype(np.float32))
        return np.stack(museum_z), np.stack(museum_p)

    while spent < budget:
        count = min(children, budget - spent)
        living = np.flatnonzero(alive)
        if crossover_mode == "bpe":
            genes_blocks = _learn_genes(pop_g[living])
        parents = rng.choice(living, size=count)
        # Mate = one uniform draw from the whole population; crossover is rare
        # by default (crossover_rate) — most children are pure mutations.
        mates = rng.choice(living, size=count)
        target = problem_of[parents]
        gains = problem_gain[target]

        child_g = pop_g[parents].copy()
        # Crossover fires only when the randomly-drawn mate is genome-COMPATIBLE
        # (among the closest crossover_rate fraction this generation). This
        # both keeps crossover rare AND blocks cross-objective mixing — a random
        # mate solving a different problem is genome-distant and never crosses.
        if crossover_rate >= 1.0:
            sexual = np.ones(count, dtype=bool)
        elif crossover_rate <= 0.0:
            sexual = np.zeros(count, dtype=bool)
        else:
            if compat_distance == "genes":
                pd, md = pop_g[parents, :latent], pop_g[mates, :latent]
            elif compat_distance == "latents":
                pd, md = pop_g[parents, latent:], pop_g[mates, latent:]
            else:
                pd, md = pop_g[parents], pop_g[mates]
            g_dist = np.sqrt(((pd - md) ** 2).mean(axis=1))
            thresh = np.quantile(g_dist, crossover_rate)
            sexual = g_dist <= thresh
        if crossover_mode == "uniform":
            take = rng.random((count, gene_dim)) < 0.5
        elif crossover_mode == "bpe":
            take = np.zeros((count, gene_dim), dtype=bool)
            for s, e in genes_blocks:
                take[:, s:e] = (rng.random(count) < 0.5)[:, None]
        elif crossover_cuts == "single":
            cuts = rng.integers(1, gene_dim, size=count)
            take = np.arange(gene_dim)[None, :] >= cuts[:, None]
        else:
            # Boundary-aware one-point: genes and latents are different
            # objects, so each half gets its own cut (or is inherited whole).
            take = np.zeros((count, gene_dim), dtype=bool)
            if crossover_cuts in ("separate", "genes_only"):
                cz = rng.integers(1, latent, size=count)
                take[:, :latent] = np.arange(latent)[None, :] >= cz[:, None]
            if crossover_cuts in ("separate", "latents_only"):
                cc = rng.integers(1, coeff_dim, size=count)
                take[:, latent:] = (np.arange(coeff_dim)[None, :]
                                    >= cc[:, None])
        take = take & sexual[:, None]
        child_g = np.where(take, pop_g[mates], child_g).astype(np.float32)
        latents_child = (rng.random(count) < 0.5) if auto_ratio else None
        child_g = mutate(child_g, gains, latents_child)

        child_fit = score(decode(child_g), target)
        spent += count

        wins = child_fit >= pop_fit[parents] - 1e-12
        if ratio_adapts and latents_child is not None:
            lat, gen = latents_child, ~latents_child
            if lat.any() and gen.any():
                if float(wins[lat].mean()) > float(wins[gen].mean()):
                    ratio = min(ratio * config.gain_step, 32.0)
                else:
                    ratio = max(ratio / config.gain_step, 1 / 32.0)
        np.add.at(pop_attempts, parents, 1)
        np.add.at(pop_wins, parents, wins.astype(np.int64))
        for f in np.unique(target):
            picks = np.flatnonzero(target == f)
            step = (config.gain_step
                    if wins[picks].mean() > config.win_target
                    else 1 / config.gain_step)
            problem_gain[f] = float(np.clip(
                problem_gain[f] * step, 1e-2, config.gain_limits[1]))
            exposures[f] += len(picks)

            rows = np.arange(f * slots, (f + 1) * slots)
            live = rows[alive[rows]]
            cand_fit = np.concatenate([pop_fit[live], child_fit[picks]])
            cand_g = np.concatenate([pop_g[live], child_g[picks]])
            cand_attempts = np.concatenate([
                pop_attempts[live], np.zeros(len(picks), dtype=np.int64)])
            cand_wins = np.concatenate([
                pop_wins[live], np.zeros(len(picks), dtype=np.int64)])
            keep = np.argsort(-cand_fit)[:slots]
            fill = rows[:len(keep)]
            pop_fit[rows] = -np.inf
            alive[rows] = False
            pop_fit[fill] = cand_fit[keep]
            pop_g[fill] = cand_g[keep]
            pop_attempts[fill] = cand_attempts[keep]
            pop_wins[fill] = cand_wins[keep]
            alive[fill] = True

        births_since_event += count
        if (consolidate != "off" and births_since_event >= next_event_births
                and spent + n_problems <= budget):
            births_since_event = 0
            next_event_births *= 2
            corpus = consolidation_corpus()
            if corpus is not None:
                decoder.fold(*corpus, config.stall_window, config.stall_tol)
                consolidations += 1
                # Reseed each problem's worst member onto the improved backbone
                # (coefficients zeroed), re-score honestly.
                reseed_rows, reseed_fams = [], []
                for f in range(n_problems):
                    rows = np.arange(f * slots, (f + 1) * slots)
                    live = rows[alive[rows]]
                    if len(live) < 2:
                        continue
                    worst = live[np.argmin(pop_fit[live])]
                    reseed_rows.append(int(worst))
                    reseed_fams.append(f)
                if reseed_rows:
                    rows = np.array(reseed_rows)
                    fams = np.array(reseed_fams)
                    pop_g[rows, latent:] = 0.0        # coefficient-zero
                    pop_attempts[rows] = 0
                    pop_wins[rows] = 0
                    pop_fit[rows] = score(decode(pop_g[rows]), fams)
                    spent += len(rows)
                    np.add.at(exposures, fams, 1)
                    for f in np.unique(fams):
                        frows = np.arange(f * slots, (f + 1) * slots)
                        live = frows[alive[frows]]
                        order = live[np.argsort(-pop_fit[live])]
                        pop_fit[frows[:len(order)]] = pop_fit[order]
                        pop_g[frows[:len(order)]] = pop_g[order]
                        pop_attempts[frows[:len(order)]] = pop_attempts[order]
                        pop_wins[frows[:len(order)]] = pop_wins[order]

        history.append({
            "evaluations": int(spent),
            "mean_best_fitness": float(np.mean(
                [pop_fit[f * slots] for f in range(n_problems)])),
            "consolidations": consolidations,
            "latent_ratio": float(ratio),
        })
        if progress is not None and spent >= next_progress:
            champs = np.array([f * slots for f in range(n_problems)])
            phenos = decode(pop_g[champs]).detach().cpu().numpy()
            progress(int(spent), int(budget), phenos, pop_fit[champs].copy())
            while next_progress <= spent:
                next_progress += progress_interval

    problems = []
    for f in range(n_problems):
        row = f * slots
        pheno = decode(pop_g[row:row + 1])[0]
        problems.append(ProblemResult(
            best_phenotype=pheno.detach().cpu().numpy(),
            best_fitness=float(pop_fit[row]),
            initial_fitness=float(initial_fit[f]),
            evaluations=int(exposures[f])))
    return MultiResult(problems=problems, evaluations=int(spent),
                       history=history, decoder=decoder.get_params(),
                       consolidations=consolidations)
