# Peer review: the multi-fitness / scaling phase (2026-07-19)

Scope: the uncommitted work after round 50 — the multi-fitness conditional
decoder experiments recorded in
`benchmark_results/image_conditional_lora_findings.md`, the raw JSON beside it
(`image_fitness_scaling_*`, `image_rotating_scaling_cifar_*`,
`image_always_shared_*`, `clip_*`), and the uncommitted changes to
`latentspace/universal/{explorer,solver,architectures}.py`. Method: every
statistic I could re-derive was recomputed from the raw JSON independently of
the findings document; code changes were read against the round 42–50 claims
they encode; the expensive runs (600k-evaluation arms, library reproduction)
were NOT rerun — where a claim rests on one of those, this review flags what a
rerun would need to show rather than disputing the number.

Verdict in one paragraph: **the numbers are honest — every value I recomputed
reproduces exactly — but several claims stated as confirmations do not clear
the campaign's own significance bar, one headline claim is missing its
control arm, and the mechanism-ranking chain (fold → retirement → species →
succession) rests on single-seed 600k runs whose deltas are smaller than the
measured seed noise at 30–60k.** Items 1–3 below are the ones that need
re-evaluation before anything more is built on top of them.

## What was verified and reproduces exactly

| Claim (findings doc) | Recomputed from raw JSON | Status |
|:---|:---|:---|
| Scenic core-four per-seed 4→32 changes −5.2% / −10.7% / −13.5% | −5.2% / −10.7% / −13.5% | exact |
| Scenic core-four means and seed SDs (all four rows) | match to the digit | exact |
| CIFAR shared-first-four table, all nine target counts, means and SDs | match to the digit | exact |
| CIFAR 4→8 transfer −21.7% | −21.7%, paired t = −10.35 | exact, and significant |
| Dilution fit beyond 32 targets: exponent 0.269, R² = 0.921 | 0.269, R² = 0.921 | exact |
| Isolated four-target run ≈ 0.11255 MSE at ~5,580 exposures | 0.11208 at e = 6,000 (nearest trace point), three seeds | confirms |
| 168-target 60k per-seed core wins −16.4% / −3.3% / −9.6% | −16.4% / −3.3% / −9.6% | exact |
| Explorer code implements rounds 42–50 as described (crossover with averaged decoders, tie-counts-as-success step control, pooled fitness-signed mutation memory, batched vmap decode) | read in full | faithful |

## Findings that need re-evaluation

### 1. Several "confirmed" transfer claims fail the campaign's own significance standard

The campaign's stated threshold at n = 3 seeds is |t| ≥ 4.303 (df = 2), used
explicitly in rounds 30 and 32 to withhold claims. Recomputed paired t-values
for the transfer results:

| Comparison | mean change | paired t (n=3) | clears 4.303? |
|:---|---:|---:|:---|
| CIFAR core, 4 → 8 targets | −21.7% | −10.35 | **yes** |
| CIFAR core, 4 → 16 targets | −12.1% | −1.60 | no |
| CIFAR core, 4 → 32 targets | −19.9% | −3.12 | no |
| Scenic core-four, 4 → 32 targets | −10.1% | −3.30 | no |
| CIFAR core, 168 targets @ 60k vs 4 targets @ 30k | −9.7% | −2.62 | no |

Only the 4→8 result is confirmed under the house standard. "All three seeds
improve" is true for the others but is a sign test with n = 3 (p = 0.25 under
the null) — the same evidence grade the campaign has previously labeled a
potential check, not a confirmation. The direction is consistent everywhere
and the effect is probably real; the fix is cheap: two more seeds per arm
(n = 5, critical t = 2.776) would settle 4→32 and the 60k result either way.
Until then these should be labeled potential checks.

### 2. The headline "168 targets viable at 2x budget" claim is missing its control arm

The claim compares the 168-target run at 60k total evaluations against the
4-target run at 30k total evaluations — the 168-target arm received twice the
total search compute. Because all targets share one population and one
decoder, total compute is the resource that matters, not per-target
exposures; "only 41% as many direct exposures per target" is accurate but not
the skeptic's question. The missing arm is **4 targets at 60k**, which does
not exist in `benchmark_results/` (checked). If the isolated run at 60k lands
below 0.0398, the headline inverts. The exposure-matched framing
(0.0509 vs 0.112 at equal exposures) survives this objection and is the
defensible version of the claim; the "beats isolated at 2x budget" framing
does not, yet.

Same issue, milder form, in the scenic study: the 4-function and 32-function
arms share a 60k budget, so that comparison is fair — the problem is specific
to the rotating-panel study's 60k extension.

### 3. The mechanism-ranking chain is built on single-seed differences smaller than measured seed noise

The sequence of design decisions — living-mean fold → retirement fold →
genotype species → lineage succession → z-only radius 30 — was each made on
one seed-3 run at 600k, with deltas of 0.9%–5.4% in mean MSE. The same
document's multi-seed tables measure seed coefficient of variation at 6–12%
at 30–60k budgets (e.g. SD 0.0044 on mean 0.0378 at 32 targets). Variance at
600k is unmeasured and plausibly smaller, but nothing in the record shows it
is small enough to rank arms separated by 1–5%. Concretely at risk:

- retirement fold vs living-mean fold (0.9% mean difference, called a
  tradeoff);
- lineage-succession memory vs species+fold memory (5.4%, used to conclude
  folding still wins on mean);
- z-only radius 30 vs radius 40 (3.5% mean vs tail tradeoff, decided the
  long-run configuration).

The document already flags "this remains a seed-3 result" for the warm-started
arm; the flag applies equally to the ordering of every arm after it. The
trajectories in the doc show arms separating by 20k–60k evaluations, so
paired 3-seed reruns at 60–120k of just the final two or three contenders
would establish the ordering at a fraction of a 600k run each. The
seed-overfitting risk is not hypothetical: every design decision in this
phase was tuned and accepted on the same seed.

### 4. The legacy bank's win over the living population has an untested trivial explanation

Named as required in the findings doc itself and still missing: the no-fold
legacy-bank ablation. The bank keeps the best-ever state per target; a
quality-diversity archive with no folding, no retirement weighting, and no
succession machinery does the same thing. Until the ablation runs, "the bank
beats its living population on 20/32 targets" is equally consistent with
"keeping an elitist archive per target is good" — a result the QD literature
(MAP-Elites) would predict with none of this phase's mechanisms. The same
external-baseline gap applies more broadly: this phase has independently
reconstructed archives, speciation, and niche-balanced selection; a standard
MAP-Elites loop over the same decoder at matched budget is now the honest
baseline, not only the internal reference sweep.

### 5. New hand-tuned constants, in a campaign whose repeated lesson is that fixed constants get falsified

Compatibility radius 30 (chosen from a sweep at one budget, on one seed,
while the doc itself notes genotype scale grows dramatically within a run —
median pairwise distance ~60 by 20k), age 10, panel width 32,
eight-generation panel blocks, softmax retirement temperature 1.0. Rounds
29–31 established that a constant measured in one configuration (the mutation
sigma) was ~100x wrong elsewhere and the fix was a self-tuning rule. The
direct analogue here: set the compatibility radius as a quantile of the
current population's pairwise-distance distribution rather than a fixed
number. At minimum, the radius sweep should be repeated at a second budget
before radius 30 hardens into a default.

### 6. Provenance gaps

- **The CLIP experiments are entirely undocumented.** 22 result files
  (`clip_species*`, `clip_curriculum*`, `clip_islands*`, including 600k-eval
  runs with animation frames) have no writeup in any findings file. Whatever
  was learned there currently exists only in JSON and in memory.
- **FINDINGS.md ends at round 50.** The entire multi-fitness phase — the most
  recent and among the most interesting work — lives in an untracked file
  under `benchmark_results/`.
- **Torch version inconsistency.** The multi-fitness JSONs record
  `torch_version: 2.6.0` while the CLIP runs record `2.12.0` and FINDINGS.md
  states PyTorch 2.12 for the campaign. If two environments are in use, MPS
  numerics can differ across versions; worth one line in the findings doc
  saying which env produced which results.

## What this review did not re-run

The 600k arms, the library-reproduction check (image 0.00318 vs 0.00336), and
anything requiring MPS wall-clock. Nothing above disputes those numbers; the
issues raised are about inference (significance, controls, ordering), not
arithmetic — the arithmetic all checks out.

## Addendum (2026-07-20): reruns executed — outcomes

The rerun list below was executed (44 runs, `benchmark_results/review_*.json`;
CLIP items excluded; full writeup appended to FINDINGS.md). Outcomes:

1. **4-target 60k control — run. The "168 targets at 2x budget" claim is
   falsified**: 168@60k (0.0425) is +3.7% worse than matched-compute 4@60k
   (0.0409, t = 0.99 tie) and not significantly better than half-compute
   4@30k (t = 2.06). Rotation is compute-neutral, not compute-positive.
2. **Seeds extended to 5 — the mid-scale transfer claims are CONFIRMED**:
   4→8 (t = 4.59), 4→16 (t = 6.27), 4→32 (t = 4.56), all significant at the
   n = 5 threshold; the 16-target anomaly was noise/runner artifact.
3. **No-fold legacy-bank ablation — run. Suspicion confirmed**: plain
   archive-only retirement is nominally BEST (0.02504) and most seed-stable;
   folding (0.02706) and lineage succession (0.02615) are statistical ties
   with it (|t| ≤ 0.93). The bank's value is the archive itself.
4. **Mechanism ordering rerun at 60k, 3 paired seeds — three-way tie**; the
   600k single-seed orderings remain unresolved and should not drive design.
5. **MAP-Elites external baseline — still open** (the no-fold arm serves as
   an internal approximation).
6. **New: maturity-confound experiment** (pixel-shuffled and duplicate-anchor
   objective pools): ~84% of the matched-exposure transfer effect reproduces
   with structureless noise objectives; duplicated anchors beat distinct real
   images; real-vs-noise is not seed-significant. Initially attributed to
   population maturation — **corrected by the content-gradient follow-up
   (Daniel's objection)**: iid uniform-noise objectives produce ZERO effect
   and collapse the shared step controller (maturation-alone falsified),
   while constant mean-color objectives reproduce the FULL effect. The
   transfer is real but its content is the palette — low-level learnable
   statistics — not image structure; unlearnable objectives are poison.
   See FINDINGS.md section 3b.

Also established: the 23:03 runner edit changed results ~10% (same config,
same seed), so pre- and post-edit studies must not be cross-compared; the
current runner is bit-deterministic on reruns.

## Closing note (2026-07-21)

The validation arc that began with this review ran to ground. Final state:
the multi-fitness phase's headline effects decomposed into a cold-start
artifact (fixed by hot-starting the step controller), a free-multi-scoring
artifact (fixed by the inherited-fitness lazy architecture,
`benchmarks/demo_image_lazy_population.py`), and one real, now cleanly
measured effect: at equal compute per problem, one shared population beats
separate runs at every budget tested (+16pp scarce → +4pp generous),
double that for related problems — a head start from crossbreeding shared
early-stage structure, converging at long budgets. Full record:
FINDINGS.md §3a–3i.

## Recommended reruns, in priority order

1. **CIFAR 4-target at 60k, 3 seeds** — the missing control for the headline
   claim. Cheapest item on the list and it gates the strongest sentence in
   the phase.
2. **Two more seeds (4 → 5 total) on CIFAR n = 4, 32, and 168@60k** — settles
   the transfer significance table above.
3. **No-fold legacy-bank ablation** (archive-only, no retirement weighting)
   at 60–120k, 3 seeds — attributes the bank's win.
4. **Paired 3-seed rerun at 60–120k of the final contenders** (species+fold
   vs lineage succession) — establishes the mechanism ordering.
5. **MAP-Elites baseline** with the same decoder and budget — the external
   yardstick for the whole phase.
6. **Write the CLIP results up or delete them** — undocumented results decay.
