"""Lazy-population multi-target evolution: no panels, no scheduler.

Every individual inherits its fitness function (target) from its parent.
The whole population is alive at once — one small family per target — but
compute only happens to individuals that reproduction selects: a child is
decoded once and scored against its OWN inherited target only. Sleeping
families cost nothing; their cached fitness never goes stale because the
targets are static. Mating may cross families (genome mixing is free; only
scoring costs), and the child stays in the base parent's family.

Parent selection is weighted random — the only policy knob:
  uniform: every living individual equally likely.
  improve: family weight tracks recent improvement per child (EMA), with a
           uniform floor so stale families keep getting probed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.demo_image_species_conditional_lora import (
    LATENT,
    SHAPE,
    decode_conditional,
    initialize_conditional_decoder,
)
from benchmarks.demo_image_species_vector import (
    ReferenceSpeciesView,
    load_targets,
)
from benchmarks.round28_anchor_conv import ConvRGB
from latentspace.universal.architectures import resolve
from benchmarks.legacy_engines.explorer import ExplorerConfig, _Template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument("--policy", choices=("uniform", "improve"),
                        default="uniform")
    parser.add_argument("--burst", type=int, default=1,
                        help="children per selected family; 1 = per-individual "
                             "sampling, >1 samples families and breeds clutches")
    parser.add_argument("--family-size", type=int, default=3)
    parser.add_argument("--children", type=int, default=192)
    parser.add_argument("--budget", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--coefficient-dim", type=int, default=64)
    parser.add_argument("--start-gain", type=float, default=512.0)
    parser.add_argument("--mating-radius", type=float, default=30.0)
    parser.add_argument("--crossover-until", type=int, default=10**12,
                        help="disable all crossover once this many evals have "
                             "been spent (tests WHEN crossover earns its keep)")
    parser.add_argument("--de-f", type=float, default=0.5,
                        help="Differential-Evolution step scale F in "
                             "child = parent + F*(a-b); only used by --mate de*")
    parser.add_argument("--de-rate", type=float, default=0.5,
                        help="fraction of children that get the DE step "
                             "(DE is meant to apply broadly, unlike crossover)")
    parser.add_argument("--adaptive-bands", type=int, default=5,
                        help="number of equal-occupancy distance bands the "
                             "adaptive mate bandit chooses among")
    parser.add_argument("--adaptive-scope", choices=("global", "family"),
                        default="global",
                        help="one shared band-bandit, or one per fitness fn")
    parser.add_argument("--adaptive-temp", type=float, default=0.15,
                        help="softmax temperature over band win-rates "
                             "(lower = greedier once a band looks best)")
    parser.add_argument("--adaptive-ema", type=float, default=0.9,
                        help="EMA retention for per-band win-rate credit")
    parser.add_argument("--flip-rate", type=float, default=0.02,
                        help="per-locus dominance-flip rate for --mate dominant "
                             "(surfaces a reserved/recessive allele into "
                             "expression, where selection can test it)")
    parser.add_argument("--stall-after", type=int, default=96,
                        help="children a family may go without meaningful "
                             "improvement before rescue crossover kicks in "
                             "(--mate stall*)")
    parser.add_argument("--stall-prob", type=float, default=0.5,
                        help="crossover probability for a STALLED family's "
                             "children (--mate stall*)")
    parser.add_argument("--stall-tol", type=float, default=0.001,
                        help="relative best-MSE drop that counts as "
                             "improvement and resets a family's stall counter")
    parser.add_argument("--cap-rate", type=float, default=0.05,
                        help="hard ceiling on the fraction of children that "
                             "cross per generation (--mate capped)")
    parser.add_argument(
        "--mate", choices=("lottery", "nearest", "gated-nearest", "gated-kin",
                           "gated-far", "gated-fittest", "gated-target",
                           "gated-breeder", "de", "de-gated", "adaptive",
                           "dominant", "dominant-plus", "stall", "stall-only",
                           "capped"),
        default="lottery",
        help=("lottery (default): one uniform draw gated by --mating-radius "
              "— crossover stays a rare event (roughly 3-7 percent), which "
              "the A/B showed is load-bearing. nearest: always cross with "
              "one of the k nearest genomes — measured 1.3-2.8x WORSE (it "
              "forces crossover on every child and inbreeds with siblings); "
              "kept for ablations. All gated-* rules fire on exactly the "
              "lottery's rare event and only change WHO the partner is, "
              "chosen among in-radius candidates: gated-nearest = random "
              "one of the k nearest (may be a sibling); gated-kin = same "
              "but siblings excluded; gated-far = the most distant "
              "in-radius stranger; gated-fittest = the in-radius stranger "
              "with the most error removed on its own target; gated-target "
              "= the in-radius stranger whose family target image is most "
              "similar to the parent's; gated-breeder = the in-radius "
              "stranger with the best smoothed record of producing winning "
              "children. No candidate -> keep the lottery mate, so "
              "crossover frequency is identical across arms. stall = the "
              "lottery PLUS forced crossover for families that have gone "
              "--stall-after children without improving (rescue on demand); "
              "stall-only = rescue crossover exclusively, no baseline "
              "trickle — tests whether crossover's timing can be steered "
              "by need instead of chance."))
    parser.add_argument("--live", action="store_true",
                        help="matplotlib window: targets|evolved + progress")
    parser.add_argument("--live-targets", type=int, default=10,
                        help="how many families the live view tracks")
    parser.add_argument("--improve-floor", type=float, default=0.2,
                        help="fraction of selection weight spread uniformly")
    parser.add_argument("--improve-ema", type=float, default=0.8)
    parser.add_argument("--reports", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    names, target_arrays = load_targets(args.targets)
    n_targets = len(names)
    _require_mps()
    device = "mps"
    targets = torch.as_tensor(target_arrays, device=device)
    flat_targets = targets.reshape(n_targets, -1)
    config = ExplorerConfig()
    _seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    template = _Template(resolve(
        lambda latent, shape: ConvRGB(latent, shape), LATENT, SHAPE), device)
    founder_theta = template.init_theta(int(rng.integers(0, 2**31)))
    model = initialize_conditional_decoder(
        "mixed", args.coefficient_dim, founder_theta, device)

    def decode(z: np.ndarray, coefficients: np.ndarray) -> torch.Tensor:
        return decode_conditional(model, z, coefficients, device)

    def score_own(phenotypes: torch.Tensor,
                  target_ids: np.ndarray) -> np.ndarray:
        """MSE of each phenotype against its own target only."""
        with torch.no_grad():
            flat = phenotypes.reshape(len(phenotypes), -1)
            own = flat_targets[torch.as_tensor(target_ids, device=device)]
            mse = ((flat - own) ** 2).mean(dim=1)
        return mse.cpu().numpy().astype(np.float64)

    # One founder per family. Slots beyond the founder start empty.
    slots = args.family_size
    capacity = n_targets * slots
    pop_z = np.zeros((capacity, LATENT), dtype=np.float32)
    pop_c = np.zeros((capacity, args.coefficient_dim), dtype=np.float32)
    # Shadow (recessive) latent: carried and inherited but never expressed;
    # crossover deposits foreign alleles here, a dominance-flip surfaces them.
    pop_s = np.zeros((capacity, LATENT), dtype=np.float32)
    pop_mse = np.full(capacity, np.inf, dtype=np.float64)
    pop_target = np.repeat(np.arange(n_targets), slots).astype(np.int64)
    alive = np.zeros(capacity, dtype=bool)
    # Breeder record per individual (children fielded / children that won):
    # read by --mate gated-breeder, maintained for every arm (no RNG cost).
    pop_attempts = np.zeros(capacity, dtype=np.int64)
    pop_wins = np.zeros(capacity, dtype=np.int64)
    # Children a family has gone without meaningful improvement — the need
    # signal for --mate stall*.
    stalled_children = np.zeros(n_targets, dtype=np.int64)

    founder_rows = np.arange(n_targets) * slots
    pop_z[founder_rows] = rng.standard_normal(
        (n_targets, LATENT)).astype(np.float32)
    if args.mate in ("dominant", "dominant-plus"):
        # Seed the reservoir only when it is used, so other arms keep their
        # exact RNG stream (and reproduce the pre-shadow baselines).
        pop_s[founder_rows] = rng.standard_normal(
            (n_targets, LATENT)).astype(np.float32)
    alive[founder_rows] = True
    spent = 0
    batch = 512
    for start in range(0, n_targets, batch):
        rows = founder_rows[start:start + batch]
        phenos = decode(pop_z[rows], pop_c[rows])
        pop_mse[rows] = score_own(phenos, pop_target[rows])
    spent += n_targets
    initial_mse = pop_mse[founder_rows].copy()
    exposures = np.ones(n_targets, dtype=np.int64)

    target_d2 = None
    if args.mate == "gated-target":
        # Pairwise distance between the family TARGET IMAGES themselves:
        # semantic relevance, which genome distance only proxies.
        with torch.no_grad():
            target_d2 = (torch.cdist(flat_targets, flat_targets)
                         .pow(2).cpu().numpy().astype(np.float64))

    # Adaptive mate bandit: per-band running win-rate, neutral (0.5) at start
    # so band choice begins uniform (== random) and concentrates as evidence
    # accrues. Global = one row; family = one row per fitness function.
    n_bands = args.adaptive_bands
    band_rows = n_targets if args.adaptive_scope == "family" else 1
    band_score = np.full((band_rows, n_bands), 0.5, dtype=np.float64)

    family_gain = np.full(n_targets, float(args.start_gain))
    family_rate = np.zeros(n_targets, dtype=np.float64)
    at3k_best = np.full(n_targets, np.nan)

    def family_weights() -> np.ndarray:
        """Selection weight per living individual, by policy."""
        if args.policy == "uniform":
            return alive.astype(np.float64)
        fam = family_rate / max(family_rate.max(), 1e-12)
        weight = args.improve_floor + (1 - args.improve_floor) * fam
        return alive * weight[pop_target]

    def mutate(values: np.ndarray, gain: np.ndarray,
               sigma: float) -> np.ndarray:
        mask = rng.random(values.shape) < config.genome_mutation_rate
        noise = rng.normal(0, 1.0, values.shape) * sigma * gain[:, None]
        return (values + mask * noise).astype(np.float32)

    generation = 0
    trace: list[dict] = []
    report_interval = max(1, args.budget // max(args.reports, 1))
    next_report = 0
    n_view = min(args.live_targets, n_targets)
    view = (ReferenceSpeciesView(
        names[:n_view], np.asarray(target_arrays[:n_view]), args.budget)
        if args.live else None)

    def update_view() -> None:
        rows = np.array([f * slots for f in range(n_view)])
        phenos = decode(pop_z[rows], pop_c[rows]).detach().cpu().numpy()
        hall = [{
            "image": phenos[i].reshape(*target_arrays.shape[1:]),
            "score": -float(pop_mse[rows[i]]),
        } for i in range(n_view)]
        view.update(spent, hall)

    while spent < args.budget:
        generation += 1
        count = min(args.children, args.budget - spent)

        living = np.flatnonzero(alive)
        if args.burst > 1:
            # Sample FAMILIES, breed a clutch per family: sibling competition
            # and meaningful win-rate statistics, still no panels or counters.
            if args.policy == "uniform":
                fam_w = np.ones(n_targets)
            else:
                fam = family_rate / max(family_rate.max(), 1e-12)
                fam_w = args.improve_floor + (1 - args.improve_floor) * fam
            fam_w = fam_w / fam_w.sum()
            n_fam = max(1, count // args.burst)
            chosen = rng.choice(n_targets, size=n_fam, replace=False, p=fam_w)
            parents = np.empty(count, dtype=np.int64)
            for k, target in enumerate(chosen):
                rows = np.arange(target * slots, (target + 1) * slots)
                members = rows[alive[rows]]
                lo = k * args.burst
                hi = count if k == n_fam - 1 else (k + 1) * args.burst
                parents[lo:hi] = rng.choice(members, size=hi - lo)
        else:
            weights = family_weights()[living]
            weights = weights / weights.sum()
            parents = rng.choice(living, size=count, p=weights)

        # Distance band each crossover child's partner was drawn from
        # (-1 = no crossover); used to credit the adaptive bandit later.
        chosen_band = np.full(count, -1, dtype=np.int64)

        if args.mate == "nearest":
            # Search the gene space: each parent's k nearest living genomes
            # (k = family size), one chosen at random. Relevant, not optimal.
            live_z = pop_z[living]
            d2 = (np.square(pop_z[parents]).sum(axis=1)[:, None]
                  + np.square(live_z).sum(axis=1)[None, :]
                  - 2.0 * pop_z[parents] @ live_z.T)
            self_col = np.searchsorted(living, parents)
            d2[np.arange(count), self_col] = np.inf
            k = min(args.family_size, len(living) - 1)
            nearest = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
            pick = nearest[np.arange(count), rng.integers(0, k, count)]
            mates = living[pick]
            sexual = np.ones(count, dtype=bool)
        else:
            # One uniform draw, gated by genome distance: crossover is rare.
            mates = rng.choice(living, size=count)
            z_dist = np.sqrt(
                ((pop_z[parents] - pop_z[mates]) ** 2).mean(axis=1))
            sexual = z_dist <= args.mating_radius
            if args.mate.startswith("gated-") and sexual.any():
                # Same crossover events as the lottery; only the partner
                # changes, chosen among in-radius candidates by rule.
                idx = np.flatnonzero(sexual)
                rows_i = np.arange(len(idx))
                live_z = pop_z[living]
                d2 = (np.square(pop_z[parents[idx]]).sum(axis=1)[:, None]
                      + np.square(live_z).sum(axis=1)[None, :]
                      - 2.0 * pop_z[parents[idx]] @ live_z.T)
                self_col = np.searchsorted(living, parents[idx])
                d2[rows_i, self_col] = np.inf
                radius2 = (args.mating_radius ** 2) * pop_z.shape[1]
                d2[d2 > radius2] = np.inf
                if args.mate != "gated-nearest":
                    # Stranger-only: a parent's own family is off-limits,
                    # so the arms differ purely in WHICH stranger is chosen.
                    kin = (pop_target[living][None, :]
                           == pop_target[parents[idx]][:, None])
                    d2[kin] = np.inf
                if args.mate in ("gated-nearest", "gated-kin"):
                    k = min(args.family_size, len(living) - 1)
                    order = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
                    choice = order[rows_i, rng.integers(0, k, len(idx))]
                    ok = np.isfinite(d2[rows_i, choice])
                elif args.mate == "gated-far":
                    score = np.where(np.isfinite(d2), d2, -np.inf)
                    choice = np.argmax(score, axis=1)
                    ok = np.isfinite(score[rows_i, choice])
                elif args.mate == "gated-fittest":
                    prog = 1.0 - (pop_mse[living]
                                  / initial_mse[pop_target[living]])
                    score = np.where(np.isfinite(d2), prog[None, :], -np.inf)
                    choice = np.argmax(score, axis=1)
                    ok = np.isfinite(score[rows_i, choice])
                elif args.mate == "gated-breeder":
                    # The in-radius stranger with the best smoothed record of
                    # PRODUCING WINNING CHILDREN — the mate-selection analogue
                    # of the breeders-beat-champions consolidation result
                    # (select for descendants, not for own fitness).
                    quality = ((pop_wins[living] + 1.0)
                               / (pop_attempts[living] + 2.0))
                    score = np.where(np.isfinite(d2), quality[None, :],
                                     -np.inf)
                    choice = np.argmax(score, axis=1)
                    ok = np.isfinite(score[rows_i, choice])
                else:  # gated-target: most similar family target image
                    sim = -target_d2[pop_target[parents[idx]][:, None],
                                     pop_target[living][None, :]]
                    score = np.where(np.isfinite(d2), sim, -np.inf)
                    choice = np.argmax(score, axis=1)
                    ok = np.isfinite(score[rows_i, choice])
                mates[idx[ok]] = living[choice[ok]]
            elif args.mate == "adaptive" and sexual.any():
                # Same rare crossover events as the lottery. The partner's
                # distance BAND is chosen by a bandit that starts uniform
                # (random) and shifts toward whichever band has been yielding
                # winning children. Bands are this generation's equal-occupancy
                # distance quantiles, so "band" means relative distance and no
                # radius cap applies — the bandit may learn near OR far is best.
                idx = np.flatnonzero(sexual)
                rows_i = np.arange(len(idx))
                live_z = pop_z[living]
                d2 = (np.square(pop_z[parents[idx]]).sum(axis=1)[:, None]
                      + np.square(live_z).sum(axis=1)[None, :]
                      - 2.0 * pop_z[parents[idx]] @ live_z.T)
                d2[rows_i, np.searchsorted(living, parents[idx])] = np.inf
                kin = (pop_target[living][None, :]
                       == pop_target[parents[idx]][:, None])
                d2[kin] = np.inf
                zdist = np.sqrt(np.maximum(d2, 0.0) / LATENT)
                pool = zdist[np.isfinite(zdist)]
                if pool.size >= n_bands:
                    edges = np.quantile(
                        pool, np.linspace(0, 1, n_bands + 1)[1:-1])
                else:
                    edges = np.array([])
                for j in range(len(idx)):
                    cols = np.flatnonzero(np.isfinite(zdist[j]))
                    if cols.size == 0:
                        continue
                    cand_band = np.digitize(zdist[j, cols], edges)
                    present = np.unique(cand_band)
                    row = (pop_target[parents[idx[j]]]
                           if band_rows > 1 else 0)
                    sc = band_score[row, present]
                    w = np.exp((sc - sc.max()) / max(args.adaptive_temp, 1e-6))
                    b = present[rng.choice(len(present), p=w / w.sum())]
                    pick = cols[cand_band == b]
                    mates[idx[j]] = living[rng.choice(pick)]
                    chosen_band[idx[j]] = b
            elif args.mate == "capped":
                # Frequency-capped lottery: same radius rule, same uniform
                # partners, but the EVENT RATE is bounded at --cap-rate per
                # generation. The 15-seed campaign showed the lottery's
                # "rare" event runs at 45-90 percent on ~1/4 of seeds (the
                # radius is a fixed constant, genome spread is not); this arm
                # isolates whether bounding the frequency alone recovers the
                # collapse seeds, separating rarity CONTROL from the stall
                # arms' need-conditioning.
                limit = max(1, int(round(args.cap_rate * count)))
                idx = np.flatnonzero(sexual)
                if len(idx) > limit:
                    keep = rng.choice(idx, size=limit, replace=False)
                    sexual = np.zeros(count, dtype=bool)
                    sexual[keep] = True
            elif args.mate in ("stall", "stall-only"):
                # Rescue on demand: crossover timing conditioned on family
                # NEED. The crossover-until sweep showed crossover's average
                # value is dominated by rare late rescues of collapsing
                # lineages; here a family that has gone --stall-after children
                # without improving gets forced crossover (probability
                # --stall-prob) with a compatible stranger, instead of waiting
                # for the lottery to roll one. "stall" keeps the lottery's
                # baseline trickle underneath; "stall-only" removes it, so
                # crossover happens exclusively on demand.
                if args.mate == "stall-only":
                    sexual = np.zeros(count, dtype=bool)
                is_stalled = (stalled_children[pop_target[parents]]
                              >= args.stall_after)
                forced = (is_stalled & ~sexual
                          & (rng.random(count) < args.stall_prob))
                idx = np.flatnonzero(forced)
                if len(idx):
                    rows_i = np.arange(len(idx))
                    live_z = pop_z[living]
                    d2 = (np.square(pop_z[parents[idx]]).sum(axis=1)[:, None]
                          + np.square(live_z).sum(axis=1)[None, :]
                          - 2.0 * pop_z[parents[idx]] @ live_z.T)
                    d2[rows_i, np.searchsorted(living, parents[idx])] = np.inf
                    kin = (pop_target[living][None, :]
                           == pop_target[parents[idx]][:, None])
                    d2[kin] = np.inf
                    radius2 = (args.mating_radius ** 2) * pop_z.shape[1]
                    for j in rows_i:
                        in_radius = np.flatnonzero(d2[j] <= radius2)
                        if len(in_radius):
                            # any compatible stranger serves (the rescue
                            # finding: the partner only needs DIFFERENCE)
                            mates[idx[j]] = living[rng.choice(in_radius)]
                        elif np.isfinite(d2[j]).any():
                            # no stranger in radius — take the nearest one,
                            # the most compatible different partner available
                            mates[idx[j]] = living[int(np.argmin(d2[j]))]
                        else:
                            continue
                        sexual[idx[j]] = True

        if spent >= args.crossover_until:
            # Causal probe: crossover off past this point, pure mutation only.
            sexual = np.zeros(count, dtype=bool)
            chosen_band[:] = -1

        gains = family_gain[pop_target[parents]]
        base_z = pop_z[parents].copy()
        base_c = pop_c[parents].copy()
        base_s = pop_s[parents].copy()  # shadow carried; never mutated
        if args.mate in ("dominant", "dominant-plus") and sexual.any():
            # Dominance/diploidy-lite: on a crossover event the mate's foreign
            # alleles go into the child's SHADOW (recessive, unexpressed); a
            # rare dominance flip later surfaces a reserved allele into
            # expression, where selection keeps it only if it wins. "dominant"
            # ALSO turns off expressed crossover (gentle-integration test);
            # "dominant-plus" keeps normal crossover and just ADDS the
            # reservoir (isolates the reservoir's marginal effect).
            cuts = rng.integers(1, LATENT, size=count)
            take = np.arange(LATENT)[None, :] >= cuts[:, None]
            base_s = np.where(take & sexual[:, None],
                              pop_z[mates], base_s).astype(np.float32)
        if args.mate.startswith("de"):
            # Differential Evolution: replace block-swap crossover with a
            # difference-vector step, child = parent + F*(a - b). The step
            # is self-scaling (large while the population is spread, small as
            # it converges) and never builds a fragile two-genome chimera.
            F = args.de_f
            apply = rng.random(count) < args.de_rate
            idx = np.flatnonzero(apply)
            if len(idx):
                if args.mate == "de":
                    # Classic DE: two random distinct donors, no gate.
                    a = rng.choice(living, size=len(idx))
                    b = rng.choice(living, size=len(idx))
                    ok = a != b
                else:  # de-gated: two random in-radius strangers (the band)
                    rows_i = np.arange(len(idx))
                    live_z = pop_z[living]
                    d2 = (np.square(pop_z[parents[idx]]).sum(axis=1)[:, None]
                          + np.square(live_z).sum(axis=1)[None, :]
                          - 2.0 * pop_z[parents[idx]] @ live_z.T)
                    self_col = np.searchsorted(living, parents[idx])
                    d2[rows_i, self_col] = np.inf
                    radius2 = (args.mating_radius ** 2) * pop_z.shape[1]
                    kin = (pop_target[living][None, :]
                           == pop_target[parents[idx]][:, None])
                    compat = (d2 <= radius2) & ~kin
                    keys = np.where(compat, rng.random(d2.shape), -np.inf)
                    top2 = np.argpartition(keys, kth=-2, axis=1)[:, -2:]
                    a_col, b_col = top2[:, 0], top2[:, 1]
                    ok = (np.isfinite(keys[rows_i, a_col])
                          & np.isfinite(keys[rows_i, b_col]))
                    a = living[a_col]
                    b = living[b_col]
                sel = idx[ok]
                aa, bb = a[ok], b[ok]
                base_z[sel] = (pop_z[parents[sel]]
                               + F * (pop_z[aa] - pop_z[bb])).astype(np.float32)
                base_c[sel] = (pop_c[parents[sel]]
                               + F * (pop_c[aa] - pop_c[bb])).astype(np.float32)
            sexual = apply  # for reporting: fraction that recombined
        elif sexual.any() and args.mate != "dominant":
            # Normal expressed crossover (skipped by plain "dominant", which
            # routes foreign alleles into the shadow instead).
            cuts = rng.integers(1, LATENT, size=count)
            take = np.arange(LATENT)[None, :] >= cuts[:, None]
            base_z = np.where(take & sexual[:, None],
                              pop_z[mates], base_z).astype(np.float32)
            cuts_c = rng.integers(1, args.coefficient_dim, size=count)
            take_c = (np.arange(args.coefficient_dim)[None, :]
                      >= cuts_c[:, None])
            base_c = np.where(take_c & sexual[:, None],
                              pop_c[mates], base_c).astype(np.float32)
        child_z = mutate(base_z, gains, config.genome_mutation_sigma)
        child_c = mutate(base_c, gains, config.genome_mutation_sigma)
        child_s = base_s  # reserved alleles preserved intact (no mutation)
        if args.mate in ("dominant", "dominant-plus"):
            # Dominance flip: swap expressed <-> shadow at a few loci so a
            # reserved allele gets expressed and tested by selection.
            flip = rng.random(child_z.shape) < args.flip_rate
            child_z, child_s = (np.where(flip, child_s, child_z),
                                np.where(flip, child_z, child_s))
            child_z = child_z.astype(np.float32)
            child_s = child_s.astype(np.float32)
        child_target = pop_target[parents]

        phenos = decode(child_z, child_c)
        child_mse = score_own(phenos, child_target)
        spent += count

        parent_mse = pop_mse[parents]
        wins = child_mse <= parent_mse + 1e-12
        np.add.at(pop_attempts, parents, 1)
        np.add.at(pop_wins, parents, wins.astype(np.int64))

        # Credit the adaptive bandit: each band that was used is nudged toward
        # whether its child beat the parent (EMA of win-rate per band).
        if args.mate == "adaptive":
            for child_pos in np.flatnonzero(chosen_band >= 0):
                b = int(chosen_band[child_pos])
                row = (pop_target[parents[child_pos]]
                       if band_rows > 1 else 0)
                r = 1.0 if wins[child_pos] else 0.0
                band_score[row, b] = (args.adaptive_ema * band_score[row, b]
                                      + (1 - args.adaptive_ema) * r)

        # Per-family: win-rate gain control, improvement rate, survival.
        touched = np.unique(child_target)
        for target in touched:
            picks = np.flatnonzero(child_target == target)
            win_rate = float(wins[picks].mean())
            step = (config.gain_step if win_rate > config.win_target
                    else 1 / config.gain_step)
            family_gain[target] = float(np.clip(
                family_gain[target] * step, 0.3, config.gain_limits[1]))

            rows = np.arange(target * slots, (target + 1) * slots)
            best_before = pop_mse[rows][alive[rows]].min()
            cand_mse = np.concatenate(
                [pop_mse[rows][alive[rows]], child_mse[picks]])
            cand_z = np.concatenate(
                [pop_z[rows][alive[rows]], child_z[picks]])
            cand_c = np.concatenate(
                [pop_c[rows][alive[rows]], child_c[picks]])
            cand_s = np.concatenate(
                [pop_s[rows][alive[rows]], child_s[picks]])
            cand_att = np.concatenate(
                [pop_attempts[rows][alive[rows]],
                 np.zeros(len(picks), dtype=np.int64)])
            cand_win = np.concatenate(
                [pop_wins[rows][alive[rows]],
                 np.zeros(len(picks), dtype=np.int64)])
            keep = np.argsort(cand_mse)[:slots]
            fill = rows[:len(keep)]
            pop_mse[rows] = np.inf
            alive[rows] = False
            pop_mse[fill] = cand_mse[keep]
            pop_z[fill] = cand_z[keep]
            pop_c[fill] = cand_c[keep]
            pop_s[fill] = cand_s[keep]
            pop_attempts[fill] = cand_att[keep]
            pop_wins[fill] = cand_win[keep]
            alive[fill] = True

            improvement = max(0.0, best_before - pop_mse[rows[0]])
            if improvement > args.stall_tol * max(best_before, 1e-12):
                stalled_children[target] = 0
            else:
                stalled_children[target] += len(picks)
            per_child = improvement / len(picks)
            family_rate[target] = (args.improve_ema * family_rate[target]
                                   + (1 - args.improve_ema) * per_child)

            before = exposures[target]
            exposures[target] += len(picks)
            if before < 3000 <= exposures[target]:
                at3k_best[target] = pop_mse[rows[0]]

        if spent >= next_report or spent >= args.budget:
            best = np.array([
                pop_mse[t * slots] for t in range(n_targets)])
            removed = 100 * (1 - best / initial_mse)
            trace.append({
                "e": int(spent),
                "generation": generation,
                "mean_removed_pct": float(np.nanmean(removed)),
                "worst_removed_pct": float(np.nanmin(removed)),
                "mean_gain": float(family_gain.mean()),
                "zero_exposure_targets": int(np.sum(exposures <= 1)),
                "sexual_fraction": float(sexual.mean()),
                "living": int(alive.sum()),
                "band_score": band_score.mean(axis=0).round(3).tolist(),
                "stalled_families": int(
                    (stalled_children >= args.stall_after).sum()),
            })
            print(f"  {spent:>7} evals  mean removed "
                  f"{trace[-1]['mean_removed_pct']:.1f}%  "
                  f"unvisited {trace[-1]['zero_exposure_targets']}",
                  flush=True)
            if view is not None:
                update_view()
            while next_report <= spent:
                next_report += report_interval

    best = np.array([pop_mse[t * slots] for t in range(n_targets)])
    result = {
        "method": "lazy_population_inherited_fitness",
        "policy": args.policy,
        "targets": [str(t) for t in args.targets],
        "target_count": n_targets,
        "family_size": args.family_size,
        "children": args.children,
        "budget": args.budget,
        "seed": args.seed,
        "start_gain": args.start_gain,
        "mating_radius": args.mating_radius,
        "improve_floor": args.improve_floor,
        "torch_version": torch.__version__,
        "initial_records_mse": {n: float(v)
                                for n, v in zip(names, initial_mse)},
        "final_records_mse": {n: float(v) for n, v in zip(names, best)},
        "records_removed_pct": {
            n: float(100 * (1 - b / i))
            for n, b, i in zip(names, best, initial_mse)},
        "at3k_records_mse": {
            n: (None if np.isnan(v) else float(v))
            for n, v in zip(names, at3k_best)},
        "target_exposures": exposures.tolist(),
        "trace": trace,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=1))
        print(f"wrote {args.output}")
    if view is not None:
        print("run finished — close the live window to exit")
        view.plt.show(block=True)
    return result


if __name__ == "__main__":
    run(parse_args())
