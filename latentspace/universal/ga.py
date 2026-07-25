"""THE universal genetic algorithm — Daniel's specification, 2026-07-21.

One solve function and no others: solve(fitness_fns, output_shape, epochs).

The two ideas that make this GA unusual are the only two it keeps. First,
no individual ever IS a solution: an individual is genes (the input the
shared decoder network reads) plus latents (a vector that bends the shared
network's behavior for that individual alone), and solutions are always
computed by decoding. Second, there is exactly ONE decoder network for the
whole run; when an individual's bending has proven useful, it is applied
DIRECTLY into the network's weights — exact arithmetic, no training — so
the environment itself absorbs discoveries over time.

Genes and latents are different concepts and are never conflated: they are
stored separately, crossed by different functions, mutated by different
functions, and no operator treats the pair as one string of numbers.

Fitness is organized as SHARES (Daniel's environment rule): the combined
fitness mass of the whole environment is always 1, every fitness function's
population collectively owns an equal slice of it, and individuals split
their function's slice by within-function rank. Selection and survival both
run on shares, so a function whose population swells dilutes its members
and self-corrects, and a function down to one struggling member concentrates
its whole slice there. Overtaking is impossible by construction. Extinction
is still allowed (an empty function owns nothing until speciation
re-seeds it).

Every operator is a replaceable function:

  selection          — which two parents breed each child
  gene_crossover     — how two parents' genes combine
  latent_inheritance — which parent's latents a child receives (whole)
  gene_mutation / latent_mutation — how each space is perturbed
  speciation         — how individuals move onto other fitness functions
  fold_selection     — whose latents get applied into the shared decoder

Population starts at TWO random individuals, both assigned the first
fitness function. There are no champions, no reserved slots per problem,
and no fixed generation size. A best-ever record per function is kept as
pure bookkeeping (never bred from) so the solver returns an answer for
every function even after extinctions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .conditional import build_conditional_decoder


# ---------------------------------------------------------------- results

@dataclass
class ProblemResult:
    best_phenotype: np.ndarray | None   # None if the function was never tried
    best_fitness: float
    initial_fitness: float
    evaluations: int


@dataclass
class GAResult:
    problems: list[ProblemResult]
    evaluations: int
    epochs: int
    history: list[dict] = field(repr=False, default_factory=list)
    decoder: np.ndarray | None = field(repr=False, default=None)

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


# --------------------------------------------------------------- shares

def fitness_shares(scores: np.ndarray, fn_idx: np.ndarray) -> np.ndarray:
    """Each living function owns 1/(functions alive) of the total fitness
    mass; its members split that slice by within-function rank (linear
    ranking on raw scores — raw comparison is legal inside one function).
    Returns one weight per individual; the weights sum to 1."""
    weights = np.zeros(len(scores))
    alive = np.unique(fn_idx)
    slice_mass = 1.0 / len(alive)
    for f in alive:
        members = np.flatnonzero(fn_idx == f)
        order = np.argsort(np.argsort(-scores[members]))   # 0 = best
        rank_weight = (len(members) - order).astype(np.float64)
        weights[members] = slice_mass * rank_weight / rank_weight.sum()
    return weights


# ------------------------------------------------------- default operators

def make_species_selection(outcross_rate=0.05):
    """The default: parent one is drawn proportionally to fitness share;
    parent two comes from the SAME function (species breed within
    themselves), also share-proportional, except a rare outcross draws it
    from the whole population instead. Rare cross-species mixing is the
    same law the earlier campaign measured for crossover generally: the
    partner must be compatible most of the time, genuinely different only
    rarely. Fully-mixed pairing was measured to floor both mutation dials
    (cross-species chimeras almost never beat their parents) and to double
    the evaluation bill (every mixed child is scored twice)."""
    def select(weights, fn_idx, rng, n_pairs):
        pop = len(weights)
        p = weights / weights.sum()
        a = rng.choice(pop, size=n_pairs, p=p)
        b = np.empty(n_pairs, dtype=np.int64)
        outcross = rng.random(n_pairs) < outcross_rate
        for i in range(n_pairs):
            kin = np.flatnonzero(fn_idx == fn_idx[a[i]])
            pool = np.arange(pop) if (outcross[i] or len(kin) < 2) else kin
            if len(pool) == 1:
                b[i] = pool[0]
                continue
            pw = weights[pool] / weights[pool].sum()
            b[i] = rng.choice(pool, p=pw)
            while b[i] == a[i] and len(pool) > 1:
                b[i] = rng.choice(pool, p=pw)
        return a, b
    return select


def share_selection(weights, fn_idx, rng, n_pairs):
    """Fully-mixed share-proportional pairing — kept as a research arm;
    see make_species_selection for why it is not the default."""
    pop = len(weights)
    p = weights / weights.sum()
    a = rng.choice(pop, size=n_pairs, p=p)
    b = rng.choice(pop, size=n_pairs, p=p)
    if pop > 1:
        clash = a == b
        while clash.any():
            b[clash] = rng.choice(pop, size=int(clash.sum()), p=p)
            clash = a == b
    return a, b


def uniform_selection(weights, fn_idx, rng, n_pairs):
    """No selection pressure at reproduction — kept as the control arm."""
    pop = len(weights)
    a = rng.integers(0, pop, n_pairs)
    b = rng.integers(0, pop, n_pairs)
    if pop > 1:
        clash = a == b
        while clash.any():
            b[clash] = rng.integers(0, pop, int(clash.sum()))
            clash = a == b
    return a, b


def one_point_gene_crossover(genes_a, genes_b, rng):
    """One cut per child across the gene vector only."""
    n, dim = genes_a.shape
    cuts = rng.integers(1, dim, n)
    take = np.arange(dim)[None, :] >= cuts[:, None]
    return np.where(take, genes_b, genes_a).astype(np.float32)


def coin_flip_latent_inheritance(latents_a, latents_b, rng):
    """Each child receives one parent's latents whole — latents are never
    spliced (Daniel's rule: a latent vector is a coherent bending of the
    shared decoder and half of one is not half as useful)."""
    pick_b = rng.random(len(latents_a)) < 0.5
    return np.where(pick_b[:, None], latents_b, latents_a).astype(np.float32)


def make_gaussian_mutation(rate=0.1, sigma=0.12):
    """Masked gaussian noise scaled by a self-tuning step dial. The dial is
    owned by the loop (one per space, updated by success rate); the operator
    just applies it."""
    def mutate(values, rng, dial):
        mask = rng.random(values.shape) < rate
        noise = rng.normal(0.0, sigma * dial, values.shape)
        return (values + mask * noise).astype(np.float32)
    return mutate


def make_random_speciation(rate=0.02):
    """Each living individual is re-assigned to a uniformly random function
    with probability `rate` per epoch. Re-assignment triggers an honest
    re-scoring on the new function."""
    def speciate(fn_idx, n_functions, epoch, rng):
        moves = rng.random(len(fn_idx)) < rate
        new = fn_idx.copy()
        new[moves] = rng.integers(0, n_functions, int(moves.sum()))
        return new
    return speciate


def largest_niche_champion_fold_selection(weights, fn_idx):
    """Apply the latents of the best member of the function with the most
    members — the most-proven niche donates its bending to the shared
    decoder. Returns a single index."""
    fns, counts = np.unique(fn_idx, return_counts=True)
    f = fns[np.argmax(counts)]
    members = np.flatnonzero(fn_idx == f)
    return int(members[np.argmax(weights[members])])


# --------------------------------------------------------------- the loop

def solve(fitness_fns, output_shape, epochs=1_000, architecture="auto",
          genes=64, latents=64, children=16, population_cap=32,
          device="auto", selection=None,
          gene_crossover=one_point_gene_crossover,
          latent_inheritance=coin_flip_latent_inheritance,
          gene_mutation=None, latent_mutation=None,
          speciation=None,
          fold_selection=largest_niche_champion_fold_selection,
          fold="on", fold_every=32, fold_optimizer="adam",
          fold_lr=0.5, directions="frozen", direction_every=16,
          direction_sigma=0.1, fresh_basis_rate=0.1,
          win_target=0.2, dial_step=1.15,
          founding="per_function", progress=None, progress_every=None,
          init_decoder=None, seed=None) -> GAResult:
    """Maximize every fitness function over phenotypes of `output_shape`.

    fitness_fns: one callable or a list. Each takes a torch tensor
        (B, *output_shape) with values in [0, 1] and returns B fitness
        values, higher better. `epochs` counts loop iterations; each epoch
        breeds `children` children. Evaluation counts (one per scoring) are
        tracked and reported for honest cross-method comparison.
    genes / latents: sizes of the two spaces. Genes are the decoder's
        input; latents bend the shared decoder per individual.
    directions: "frozen" (default). "sparse" replaces low-rank bending
        with a per-individual SPARSE WEIGHT PATCH (Daniel, 2026-07-22):
        the individual's seed picks `latents` coordinates of the decoder's
        weight vector and its latents are the values added there, so folds
        can reach any weight instead of being trapped in a frozen random
        subspace forever. Locations inherit with the latents; a
        `fresh_basis_rate` fraction of children draw new ones. "evolve" trials perturbations of the
        shared low-rank vocabulary as a (1+1) evolution strategy —
        FALSIFIED as built (apple, 171k evals: 0.01268 vs frozen 0.01222;
        ~560 trials, essentially all rejected, because a random
        perturbation of the whole vocabulary almost never survives a
        population-mean vote over 32 co-adapted individuals — it froze
        itself and paid a 12% trial tax). Kept for iteration; the
        designed refinements are one-direction-at-a-time proposals, a
        share-weighted acceptance signal, and trialing right after folds
        when the vocabulary is least load-bearing.
    fold: "on" treats proven individuals' latents as measured update
        directions for the shared decoder (Daniel's framing: evolution is
        computing gradients). Default: every `fold_every` epochs the
        selected donor's latents feed an Adam accumulator and the
        processed step is absorbed into the decoder by exact arithmetic —
        momentum keeps the cross-species consensus direction, the second
        moment damps dimensions donors disagree on. Measured (3 paired
        seeds): Adam flips frequent folding from harmful (raw every-32:
        0.053) to the best configuration (0.026, better than the raw
        doubling schedule on every seed, tying the legacy champion
        engine). The donor's phenotype is preserved exactly for any step
        size; everyone else shifts and is honestly re-scored.
        fold_optimizer="raw" absorbs whole bendings; "off" disables
        folding; fold_every=None restores the doubling schedule.
    The step dials for the two mutation spaces are global (dense feedback
    every epoch), start at 1.0, and self-tune by success rate with ties
    counted as successes. All other operators are the module-level defaults
    and are replaceable via the parameters above.
    """
    fns = [fitness_fns] if callable(fitness_fns) else list(fitness_fns)
    if not fns:
        raise ValueError("at least one fitness function is required")
    n_fns = len(fns)
    output_shape = tuple(int(s) for s in output_shape)
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(0, 2 ** 31)))
    if directions == "sparse":
        from .sparse import build_sparse_decoder
        decoder = build_sparse_decoder(
            architecture, genes, output_shape, latents, device)
    else:
        decoder = build_conditional_decoder(
            architecture, genes, output_shape, latents, device)
    if directions == "individual":
        from .conditional import attach_seeded_directions
        attach_seeded_directions(decoder)
    if init_decoder is not None:
        # Warm start: the shared decoder begins as a previous run's, so a
        # new problem inherits the family's structure instead of starting
        # from a random prior. `GAResult.decoder` is the vector to pass in.
        decoder.set_params(init_decoder)

    selection = selection or make_species_selection()
    gene_mutation = gene_mutation or make_gaussian_mutation()
    latent_mutation = latent_mutation or make_gaussian_mutation()
    # Speciation (migration between functions) defaults OFF: measured
    # 2026-07-21, seeding every function plus NO background migration beat
    # both the grown-coverage start and seeded-with-migration on all seeds
    # (migration churn pays re-scorings to disrupt working species). The
    # designed successor is EVENT-DRIVEN recolonization on extinction,
    # relevant when the function count dwarfs the population; random
    # migration remains available for research via make_random_speciation.

    # Best-ever bookkeeping per function (never bred from).
    best_score = np.full(n_fns, -np.inf)
    founder_score = np.full(n_fns, np.nan)
    best_pheno: list[np.ndarray | None] = [None] * n_fns
    fn_evals = np.zeros(n_fns, dtype=np.int64)
    spent = 0

    def score(genes_arr, latents_arr, fn_of, seeds=None) -> np.ndarray:
        """Decode once, score each phenotype on its own function, update
        the best-ever records, count evaluations."""
        nonlocal spent
        phenos = (decoder.decode_seeded(genes_arr, latents_arr, seeds)
                  if seeds is not None
                  else decoder.decode(genes_arr, latents_arr))
        values = np.empty(len(fn_of), dtype=np.float64)
        for f in np.unique(fn_of):
            picks = np.flatnonzero(fn_of == f)
            out = fns[int(f)](phenos[picks])
            v = np.asarray(out.detach().cpu().numpy()
                           if torch.is_tensor(out) else out, dtype=np.float64)
            values[picks] = v
            fn_evals[f] += len(picks)
            if np.isnan(founder_score[f]):
                founder_score[f] = float(v[0])
            top = int(np.argmax(v))
            if v[top] > best_score[f]:
                best_score[f] = float(v[top])
                best_pheno[int(f)] = (phenos[picks[top]]
                                      .detach().cpu().numpy().copy())
        spent += len(fn_of)
        return values

    # Founders, random in BOTH spaces. "two" (the spec's default): a single
    # pair on the first function, coverage grows through speciation.
    # "per_function": two founders on every function from the start — the
    # comparison arm for whether seeding every niche beats growing into
    # them.
    if founding == "per_function":
        population_cap = max(population_cap, 2 * n_fns)
        n0 = 2 * n_fns
        pop_genes = rng.standard_normal((n0, genes)).astype(np.float32)
        pop_latents = rng.standard_normal((n0, latents)).astype(np.float32)
        pop_fn = np.repeat(np.arange(n_fns), 2)
    else:
        n0 = 2
        pop_genes = rng.standard_normal((2, genes)).astype(np.float32)
        pop_latents = rng.standard_normal((2, latents)).astype(np.float32)
        pop_fn = np.zeros(2, dtype=np.int64)
    seeded = directions in ("individual", "sparse")
    pop_basis = (rng.integers(0, 2 ** 31, n0) if seeded
                 else np.zeros(n0, dtype=np.int64))
    pop_score = score(pop_genes, pop_latents, pop_fn,
                      pop_basis if seeded else None)

    gene_dial, latent_dial = 1.0, 1.0
    next_fold = 32
    adam_m = np.zeros(latents, dtype=np.float64)
    adam_v = np.zeros(latents, dtype=np.float64)
    adam_t = 0
    # Direction evolution ((1+1)-ES on the shared low-rank vocabulary, with
    # an Adam memory over ACCEPTED changes so proposals drift along what
    # has historically worked — the round-50 pattern one level deeper).
    dir_dial = 1.0
    dir_dim = len(decoder.direction_vector()) if directions == "evolve" else 0
    dir_m = np.zeros(dir_dim, dtype=np.float64)
    dir_v = np.zeros(dir_dim, dtype=np.float64)
    dir_t = 0
    history: list[dict] = []

    for epoch in range(int(epochs)):
        weights = fitness_shares(pop_score, pop_fn)

        # --- reproduction
        a, b = selection(weights, pop_fn, rng, children)
        child_genes = gene_crossover(pop_genes[a], pop_genes[b], rng)
        if seeded or latent_inheritance is coin_flip_latent_inheritance:
            # (basis, latents) travel as one unit — the latents only mean
            # anything relative to their basis, so seeded mode owns this
            # choice; custom operators apply in the shared-vocabulary modes.
            pick_b = rng.random(children) < 0.5
            child_latents = np.where(pick_b[:, None], pop_latents[b],
                                     pop_latents[a]).astype(np.float32)
            child_basis = np.where(pick_b, pop_basis[b], pop_basis[a])
        else:
            child_latents = latent_inheritance(pop_latents[a],
                                               pop_latents[b], rng)
            child_basis = pop_basis[a].copy()
        if seeded and fresh_basis_rate > 0:
            fresh = rng.random(children) < fresh_basis_rate
            n_fresh = int(fresh.sum())
            if n_fresh:
                child_basis[fresh] = rng.integers(0, 2 ** 31, n_fresh)
                child_latents[fresh] = 0.0

        # --- mutation: the two spaces are perturbed by different operators
        # with independent self-tuning dials; each child mutates exactly one
        # space so every dial's feedback is uncontaminated by the other.
        mutates_latents = rng.random(children) < 0.5
        gi = np.flatnonzero(~mutates_latents)
        li = np.flatnonzero(mutates_latents)
        if len(gi):
            child_genes[gi] = gene_mutation(child_genes[gi], rng, gene_dial)
        if len(li):
            child_latents[li] = latent_mutation(child_latents[li], rng,
                                                latent_dial)

        # --- scoring and function adoption. Same-function parents pass the
        # function down; mixed parents have the child scored on both and it
        # adopts the function on which it beats its parent by more.
        fa, fb = pop_fn[a], pop_fn[b]
        child_fn = fa.copy()
        mixed = fa != fb
        child_score = np.empty(children, dtype=np.float64)
        same = np.flatnonzero(~mixed)
        if len(same):
            child_score[same] = score(
                child_genes[same], child_latents[same], child_fn[same],
                child_basis[same] if seeded else None)
        for i in np.flatnonzero(mixed):
            s_a = score(child_genes[i:i + 1], child_latents[i:i + 1],
                        fa[i:i + 1],
                        child_basis[i:i + 1] if seeded else None)[0]
            s_b = score(child_genes[i:i + 1], child_latents[i:i + 1],
                        fb[i:i + 1],
                        child_basis[i:i + 1] if seeded else None)[0]
            take_a = (s_a - pop_score[a[i]]) >= (s_b - pop_score[b[i]])
            child_fn[i] = fa[i] if take_a else fb[i]
            child_score[i] = s_a if take_a else s_b

        # --- dial updates: the reference is the parent on the child's
        # adopted function (raw scores compare legally within one
        # function); ties count as successes.
        ref = pop_score[a].copy()
        adopted_b = mixed & (child_fn == fb)
        ref[adopted_b] = pop_score[b][adopted_b]
        success = child_score >= ref - 1e-12
        if len(gi):
            rate = float(success[gi].mean())
            gene_dial *= dial_step if rate > win_target else 1 / dial_step
            gene_dial = float(np.clip(gene_dial, 1e-3, 1e4))
        if len(li):
            rate = float(success[li].mean())
            latent_dial *= dial_step if rate > win_target else 1 / dial_step
            latent_dial = float(np.clip(latent_dial, 1e-3, 1e4))

        # --- population cap: everyone competes on fitness share; the
        # lowest shares are removed; a function may go extinct.
        all_genes = np.concatenate([pop_genes, child_genes])
        all_latents = np.concatenate([pop_latents, child_latents])
        all_basis = np.concatenate([pop_basis, child_basis])
        all_fn = np.concatenate([pop_fn, child_fn])
        all_score = np.concatenate([pop_score, child_score])
        keep = np.argsort(-fitness_shares(all_score, all_fn))[:population_cap]
        pop_genes, pop_latents = all_genes[keep], all_latents[keep]
        pop_basis = all_basis[keep]
        pop_fn, pop_score = all_fn[keep], all_score[keep]

        # --- speciation: individuals drift onto other functions over time;
        # a move is honest (re-scored on the new function immediately).
        if speciation is not None:
            new_fn = speciation(pop_fn, n_fns, epoch, rng)
            moved = np.flatnonzero(new_fn != pop_fn)
            if len(moved):
                pop_fn = new_fn
                pop_score[moved] = score(
                    pop_genes[moved], pop_latents[moved], pop_fn[moved],
                    pop_basis[moved] if seeded else None)

        # --- fold: a proven individual's latents are treated as a measured
        # update direction for the shared decoder (Daniel: "we are
        # essentially applying gradients"). fold_optimizer="raw" absorbs
        # the donor's whole bending; "adam" feeds each event into an
        # Adam-style accumulator (momentum + per-dimension normalization)
        # and absorbs the processed step. Either way the absorbed step is
        # subtracted from the donor's latents, which preserves the donor's
        # phenotype EXACTLY for any step size (the bending is linear in the
        # latents); everyone else feels the shift, so the population is
        # honestly re-scored.
        if fold_every is not None:
            fold_due = fold != "off" and (epoch + 1) % int(fold_every) == 0
        else:
            fold_due = fold == "doubling" and epoch + 1 >= next_fold
            if fold_due:
                next_fold *= 2
        if fold_due:
            donor = fold_selection(fitness_shares(pop_score, pop_fn), pop_fn)
            g = pop_latents[donor].astype(np.float64)
            if fold_optimizer == "adam":
                adam_t += 1
                adam_m = 0.9 * adam_m + 0.1 * g
                adam_v = 0.999 * adam_v + 0.001 * g * g
                m_hat = adam_m / (1 - 0.9 ** adam_t)
                v_hat = adam_v / (1 - 0.999 ** adam_t)
                step = (fold_lr * m_hat
                        / (np.sqrt(v_hat) + 1e-8)).astype(np.float32)
            else:
                step = pop_latents[donor].copy()
            if seeded:
                decoder.absorb_seeded(step, int(pop_basis[donor]))
            else:
                decoder.absorb(step)
            pop_latents[donor] -= step
            pop_score = score(pop_genes, pop_latents, pop_fn,
                              pop_basis if seeded else None)

        # --- direction evolution: trial a perturbation of the shared
        # vocabulary itself. Accept if the living population, re-scored
        # under the new directions, is on average better (per-individual
        # relative change, so no function's scale dominates); reject
        # restores the old vocabulary and the cached scores exactly. The
        # trial's re-scoring is charged to the budget either way.
        if directions == "evolve" and (epoch + 1) % int(direction_every) == 0:
            base_dir = decoder.direction_vector()
            scale = max(float(base_dir.std()), 1e-4)
            noise = rng.standard_normal(dir_dim)
            step = direction_sigma * scale * dir_dial * noise
            if dir_t > 0:
                drift = (dir_m / (1 - 0.9 ** dir_t)) / (
                    np.sqrt(dir_v / (1 - 0.999 ** dir_t)) + 1e-8)
                step = step + 0.5 * direction_sigma * scale * dir_dial * drift
            decoder.set_direction_vector(
                (base_dir + step).astype(np.float32))
            old_scores = pop_score.copy()
            trial_scores = score(pop_genes, pop_latents, pop_fn)
            gain_rel = np.mean((trial_scores - old_scores)
                               / np.maximum(np.abs(old_scores), 1e-12))
            if gain_rel >= 0:
                pop_score = trial_scores
                dir_dial = min(dir_dial * dial_step, 1e3)
                dir_t += 1
                dir_m = 0.9 * dir_m + 0.1 * step
                dir_v = 0.999 * dir_v + 0.001 * step * step
            else:
                decoder.set_direction_vector(base_dir)
                pop_score = old_scores
                dir_dial = max(dir_dial / dial_step, 1e-3)

        if progress is not None and (epoch + 1) % (
                progress_every or max(1, int(epochs) // 50)) == 0:
            progress(epoch + 1, int(epochs), int(spent),
                     [None if b is None else b.copy() for b in best_pheno],
                     best_score.copy())
        history.append({
            "epoch": epoch,
            "evaluations": int(spent),
            "population": int(len(pop_fn)),
            "functions_alive": int(len(np.unique(pop_fn))),
            "functions_tried": int(np.isfinite(best_score).sum()),
            "gene_dial": float(gene_dial),
            "latent_dial": float(latent_dial),
            "direction_dial": float(dir_dial),
            "mean_score": float(pop_score.mean()),
        })

    problems = [ProblemResult(
        best_phenotype=(None if best_pheno[f] is None
                        else best_pheno[f].reshape(output_shape)),
        best_fitness=float(best_score[f]),
        initial_fitness=(float(founder_score[f])
                         if np.isfinite(best_score[f]) else float("nan")),
        evaluations=int(fn_evals[f]),
    ) for f in range(n_fns)]
    return GAResult(problems=problems, evaluations=int(spent),
                    epochs=int(epochs), history=history,
                    decoder=decoder.get_params())
