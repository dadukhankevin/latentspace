# What we learned: from a universal decoder to a learned genetic code

This document is the narrative record of a eighteen-round experimental
campaign (2026-07-14/15, Apple M3 Pro, PyTorch 2.12, ten seeds and exact
evaluation budgets throughout). It explains what the project originally claimed, how
that claim died, what replaced it, exactly how the surviving method works,
and what remains open. The chronological lab record with every table lives in
[`benchmark_results/mps_rounds1to4_findings.md`](benchmark_results/mps_rounds1to4_findings.md);
raw per-seed JSON sits beside it; the scaling-law charts are in
[`benchmark_results/family_scaling.svg`](benchmark_results/family_scaling.svg)
and [`benchmark_results/mlp_vs_pca_scaling.svg`](benchmark_results/mlp_vs_pca_scaling.svg).

## The original claim, and how it died

The GeneSpace/latentspace thesis: evolve a universal latent vector; a single
co-evolving neural decoder maps it to any phenotype; change only the fitness
function and output shape. Tested honestly, every part of that failed:

- **Discrete problems** (TSP, deceptive traps, HIFF, NK landscapes): every
  latent variant landed nearer random search than a direct GA. A fresh
  sigmoid decoder emits every output at ~0.5, so all bits sit at the
  threshold and one latent mutation flips a large arbitrary set — the dense
  random MLP destroys locality (rounds 1, plus the earlier TSP studies).
- **The one apparent win was an artifact.** Latents beat direct search on
  Rastrigin only because sigmoid initialization concentrates phenotypes at
  0.5 — which is unshifted Rastrigin's optimum. Shifting the optimum
  (standard BBOB practice) inverted the win into a 2× loss (round 2).
- **Co-evolutionary training never helped.** Across nine trainer designs,
  three studies, and every objective: no confirmed improvement over a frozen
  decoder. The reason is structural, not a tuning failure — see the
  self-referentiality principle below.
- **The ancestor GeneSpace repo's demos** follow the same pattern: its TSP-12
  demo loses to a direct GA at matched budget, and its image demo sits
  exactly at the gray-canvas initialization artifact after 25,000 matched
  evaluations (round 5).

## The idea that survived

The salvageable core was never "a random network can solve anything" — it
was "problem structure should live in the decoder." Made precise:

> **A latent decoder earns its overhead if and only if its output manifold
> matches the structure of the problem's good solutions — and that structure
> can be *learned* from experience across a problem family.**

Established in three steps:

1. **Oracle test (rounds 3–4).** On objectives whose optima lie on a
   low-dimensional manifold, a hand-built matched decoder (low-frequency DCT
   basis) beat both an unstructured MLP (10/0) and direct search — and the
   advantage vanished exactly when the structure was absent. The outer
   optimizer mattered as much: CMA-ES in the same latent beat the latent GA
   10/0. On rugged landscapes with structured optima, matched decoder +
   latent CMA-ES beat everything at every budget.
2. **Learning test (round 6).** The same structure the oracle was *given*
   can be *learned* — but only from the right corpus. Elites from a single
   run encode only "the direction to this target" and transfer to nothing.
   Elites pooled across many instances of a family span the family manifold.
3. **Scaling law (round 7).** Transfer quality improves lawfully with the
   number of pretraining instances K, across problem domains (1-D signals,
   2-D images), dimensions (256, 1,024), and fitness topologies (unimodal,
   multimodal) — and refuses to appear on a structureless control family.
   Parity with direct search is crossed between K = 16 and 32 everywhere;
   at K = 128 the learned code beats direct search 1.3–3.3× and approaches
   its oracle bound.

## The method that works, step by step

One decoder per problem family. Fit it once from cheap experience; reuse it
on every future instance.

```
PRETRAIN (once per family; costs amortize across all future instances)
1. Sample K practice instances of the family (K > the family manifold's
   dimension; more is lawfully better — we used up to 128).
2. On each, run a cheap decoder-free direct GA (~2,000 evaluations).
3. Keep each run's ~10 best phenotypes. Pool them: the elite corpus.
4. Fit the decoder to the corpus in logit space:
     linear:  mean + top-32 PCA directions            (closed form, robust)
     neural:  32-bottleneck autoencoder, trained on the corpus PLUS ~5,000
              synthetic samples drawn from the fitted PCA decoder
              ("PCA scaffolding" — round 9; without it the network underfits
              and no amount of extra training rescues it)
5. Freeze the decoder.

SOLVE (per new instance)
6. decode(z) = sigmoid(mean + z @ basis)   [or the trained network]
7. Run CMA-ES over the 32-dimensional latent z. Every candidate decodes to
   a plausible family member; the search space is "the neighborhood of good
   solutions," not raw phenotype space.
```

Why each piece is there:

- **Fitness only selects the corpus.** Nothing backpropagates from the
  objective; the decoder is fit by supervised reconstruction. A good
  phenotype vector carries thousands of informative numbers; a reward
  carries one. Every reward-driven trainer lost to this.
- **Logit space** makes the sigmoid exactly invertible during fitting, so
  the squash never distorts the regression.
- **CMA-ES, not a fixed-sigma GA**, in the latent: it adapts step size and
  gene correlations online — the same representation scored ~35% better
  under CMA-ES than under the latent GA.
- **PCA before the network.** Closed-form PCA is the optimal flat map and
  needs no tuning; the network ties it only when trained on PCA-synthesized
  data plus the real elites, and is worth deploying only where the family
  manifold is genuinely curved. As of round 9 the network ties but has not
  yet beaten the flat map anywhere — its case rests on curvature we have
  not yet pushed hard enough to expose.

## The self-referentiality principle

The single deepest lesson, found independently three times:

> **A decoder trained on data produced by searching through itself cannot
> learn anything it does not already express.**

- Distillation trainers regress the decoder toward its own outputs on
  higher-ranked genes → point collapse, no new structure (rounds 1–5).
- Refitting PCA on its own search products returns the identical subspace,
  to six decimal places — the new elites lie in the old span (round 6).
- Even a warm-started *nonlinear* decoder, periodically refit on the best
  candidates CMA-ES finds and re-anchored, is only whisper-better (7/3, CI
  spanning zero): those candidates are its own decoded outputs, on-manifold
  by construction. Refitting teaches density, never geometry (round 10).

Corollary: within-run decoder learning can only pay if fed **off-manifold
data**. Round 12 tested the weight-space version of that channel (Daniel's
proposal): temporarily mutate the decoder's weights — a mutant net's outputs
are off the base manifold by construction — and distill back whatever beats
the base decoder on real evaluations. Calibration first: weight mutation has
a textbook mutational fitness landscape (43% of mutant outputs beat their
parent at noise scale 0.003, ~0% at 0.03+), so only small steps are viable.
One-step mutants gain ~0.5% each and distilling them changed nothing (null).
But compounding them in a (1+1)-ES walk on the weights — mutate, keep the
champion by real evaluations, mutate the champion — and distilling the base
from the *champion's* outputs produced the first within-run training that
ever significantly beat the frozen decoder (blob 8/2, CI excluding zero),
and it stops the floor degradation that plain online refitting causes. The
fix for self-distillation is having a teacher that is not the student: a
weight-mutant whose superiority was paid for in objective evaluations.

The effect is real but small (~1.4%), and the representation floor barely
moves: the walk improves the manifold *around the incumbent*, where its
evaluations are, and cannot discover missing geometry that nothing in the
channel points at. The unconstrained escalation — direct phenotype-space
local search written back into the corpus — remains untested (open
problem #1).

Round 13 generalized the teacher-that-is-not-the-student idea (Daniel's
proposal): **two decoders, where neither is ever trained on its own
outputs** — A refits only on elites found by searching through B and vice
versa — plus a subtle per-epoch weight decay *away* from each other (2% of
the current weight difference) to keep them from converging. Search
alternates decoders per epoch; the global best transfers across manifolds
by re-encoding. Controls factorized the two ingredients:

- **The repulsion is the active ingredient, not the cross-training.** On
  blobs, cross-only ties self-only (who trains whom doesn't matter), but
  adding repulsion beats everything the campaign has ever run at latent 32:
  10/0 vs frozen (−3.5%), 10/0 vs online, 10/0 vs the round-12 ES walk
  (2.5× its effect). Self-training + repulsion lands statistically on top
  of cross + repulsion.
- **The win comes from search-time diversity, not decoder quality.** The
  repelled arms have the *worst* representation floors of any arm
  (0.065 → 0.082) and still the best results: maintaining two genuinely
  different manifolds to search is worth more than either manifold's
  fidelity. Repulsion along θ_A − θ_B between two independently
  initialized nets is a large, coherent, persistent perturbation — a
  structured diversity force, not epsilon noise.
- **Cross-training is a stabilizer, not a driver.** On smooth — where the
  decoder can already nearly express the target — repulsion is poison
  (self+repel loses 0/10 to plain self), but the cross rule buffers most
  of the damage (cross+repel beats self+repel 9/1). Verified-good elites
  from the other decoder pull a perturbed net back toward useful geometry;
  its own elites do not.
- Splitting the search budget across two frozen decoders costs nothing
  (dual-frozen ties single-frozen): re-encoding the best phenotype into
  the other decoder's latent transfers progress across manifolds cleanly.

The obvious knob this exposes: repulsion strength should scale with the
floor gap — hard family, push apart; easy family, leave alone.

## One decoder across families (round 11)

Rounds 6–10 used **one decoder per family**. Round 11 tested the stronger
vision — one neural decoder shared across families — the way it should be
framed: the *manifolds* stay per family (each family's practice elites get
their own PCA scaffold), but a single autoencoder trains on the union of
every family's real + synthetic corpus. Two 256-dimensional families
(smooth 1-D signals, 16×16 blob images), so one output head serves both.

The shared decoder won on every axis measured:

- **No interference.** At matched capacity (latent 32) the universal
  decoder ties or slightly beats the per-family decoders on both families
  (8/2 and 7/3, CIs spanning zero). Sharing costs nothing.
- **No family ID needed.** Starting CMA-ES at the solving family's mean
  elite code versus at the origin makes no measurable difference: the
  shared code space puts both families' elites near the origin, and the
  fitness signal alone finds the right region. The decoder never has to be
  told which problem it is solving.
- **Cross-family data is real signal, not just harmless.** At latent 64,
  the per-family blob decoder improves (10/0) but the per-family smooth
  decoder collapses (0/10 — excess dimensions fit noise in a small corpus
  and CMA-ES wastes budget in them). The shared decoder at latent 64 gets
  the best blob result of the entire campaign (0.0029, beating the
  per-family-64 control 8/2) *and* stays healthy on smooth (9/1 over the
  per-family-64 control). Other families' data acts as regularization for
  excess capacity and as genuine transfer where the family can absorb it.

So the split-by-family design is dead as a requirement: there is no
measured reason to give each family its own network. What remains open is
scale and heterogeneity — these were two same-dimension families; spanning
different output shapes needs conditioning or masking, and whether the
scaling law survives dozens of families in one code is a pretraining-scale
question, which is exactly where every other field's version of this
problem ended up.

## The universality pivot (rounds 14–17)

Daniel then imposed the project's true constraint in final form: a
realistic universal GA gets ONE fitness function and a budget. No practice
problems. And no operator may ever touch the phenotype — mutation and
crossover exist only for genomes and decoder weights (plain tensors), so
the same code runs whether the output is a signal, an image, a 3-D model,
or a program. The single modality-specific element allowed is the
decoder's *architecture* (CNN for images, 1-D conv for signals,
transformers for sequences, ...), which is part of the problem interface
in the same way the fitness function is.

- **Round 14** re-ran the original package algorithm unmodified — plus the
  round-13 dual-repulsion mechanism injected as one `decoder_update`
  layer — with no pretraining. All latent arms plateaued together, 5–7×
  behind a direct GA. The dual mechanism amplifies decoders that already
  know something; it cannot create knowledge.
- **Round 15** tested per-individual decoders (Daniel's proposal: every
  individual = its own genome + its own decoder weights, children mutate
  both). First from-scratch method to beat the frozen-random-decoder
  plateau. Used as a practice-problem harvester it preserved the scaling
  law's direction but its elites were too weak to teach; iterating the
  loop (round 15c: re-harvest warm-started from the current decoder,
  refit, repeat) improved the corpus past the direct GA's while the
  decoder stalled at self-consistency. That decomposition yielded the
  campaign's second law: **a teacher's value is the independence of its
  errors, not their size.** Independent errors cancel under compression
  and reveal the family manifold; a shared bias (one warm start feeding
  every harvest) is preserved forever.
- **Round 16** moved the same insight inside a single run: explore with
  per-individual decoders (independent lineages = independent-ish errors),
  then compress the best 200 vetted phenotypes into a PCA decoder and
  spend the remaining budget on CMA-ES in its latent. Small gain (~5%).
- **Round 17** pulled the architecture lever: convolutional per-individual
  decoders (untrained conv nets are biased toward locally-coherent
  outputs — the deep-image-prior effect). Largest single-fitness lever
  found: image 0.0994 → 0.0765, signal 0.0729 → 0.0676.
- **Stacked** (conv exploration 60% → distill → CMA-ES 40%), 10 paired
  seeds: **image 0.0208 vs direct GA 0.0582 — 10/0, CI [−0.0437,
  −0.0309], 2.8× better. The first fully universal method in the campaign
  to beat the hand-matched traditional GA.** On the signal family it ties
  the GA (4/6, CI spans zero) with high seed variance — two flop seeds;
  best seeds beat the GA.

The single-fitness story therefore flipped: universality no longer costs
2–5×; with the right decoder architecture and within-run distillation it
wins outright on image-structured problems and draws on signals, with
variance control as the open engineering problem.

**Round 18** replaced the stack's magic 60/40 split with a stall rule
(exploration ends when its best loss improves <1% over 10 generations,
with a 10×latent evaluation reserve for CMA-ES) and tested a stratified
harvest (per-lineage cap on distilled solutions). The stall rule wins:
exploration actually stalls after ~350–900 evaluations, not 3,000, and
handing the surplus to CMA-ES beats the fixed split 8/2 on smooth (CI
excludes zero) while keeping the image 10/0; the smooth-vs-GA comparison
improved to a genuine 6/4 tie. The switch point varies per seed on the
image (350–3,200) — it adapts, it didn't just find a new constant. The
lineage cap helped nothing (diluting the top solutions cost more than the
independence it bought) and defaults to off. These defaults are packaged
in `latentspace.universal.solve` (round-18-validated; six CPU tests plus
an MPS parity check reproduce the benchmark numbers).

## Scoreboard

| Approach | Verdict |
|:---|:---|
| Random MLP decoder + latent GA ("one algorithm") | Falsified everywhere tested |
| Within-run decoder training (all nine trainers) | Never beat frozen; structurally cannot |
| Hand-matched decoder + latent CMA-ES | Strong wins where structure exists |
| Family-pretrained PCA decoder + CMA-ES | Beats direct search on fresh instances; lawful scaling in K |
| Family-pretrained neural decoder | Ties PCA when trained with PCA scaffolding; headroom unproven |
| Online refinement (warm-started, supervised, re-anchored) | Safe but geometrically inert without off-manifold data |
| One decoder shared across families (per-family scaffolds, union corpus) | Never worse than per-family, best-ever blob result at latent 64; needs no family ID at solve time |
| Weight-mutation channel, one-step (round 12) | Null — accepted improvements are epsilon-off-manifold |
| Weight-mutation ES walk + champion distillation (round 12) | First within-run training to significantly beat frozen (small effect); floor-protective |
| Dual decoders, cross-training alone (round 13) | Null vs self-training — who trains whom doesn't matter |
| Dual decoders + weight repulsion (round 13) | Biggest within-run win yet on the hard family (10/0 vs everything, −3.5%); repulsion is the active ingredient; harmful on the easy family, where cross-training buffers it |
| Original algorithm + dual repulsion, from scratch (round 14) | Null — the mechanism amplifies existing knowledge, cannot create it |
| Per-individual decoders: genome + own weights (round 15) | First from-scratch universal method to beat the frozen plateau; weak alone |
| Self-teaching bootstrap over practice problems (round 15c) | Corpus climbs past the GA's, decoder stalls at self-consistency — teacher value = error INDEPENDENCE, the second law |
| Modality-shaped decoder architecture (round 17) | Largest single-fitness lever: conv decoders −23% on images with identical evolution |
| Conv exploration + within-run distillation (rounds 16–17 stack) | **Beats the traditional GA 10/0 on the image problem — first fully universal win; ties on signals, variance to fix** |

## Open problems, in priority order

1. **Variance control, remaining half.** Round 18's adaptive switch fixed
   the split (smooth is now a 6/4 tie with the GA instead of a loss) but
   seed variance on signals persists; the lineage-cap fix was null. Next
   suspects: repulsion across exploration lineages (round 13's mechanism),
   and restarting exploration after exploitation stalls (round-trip
   phases).
2. **The architecture zoo.** Round 17 tested one convolutional shape.
   Transformer/GRU decoders for sequences, permutation-aware decoders for
   tours, and the question Daniel posed: is there a universally-best
   default architecture?
3. **Discrete problems.** Still never won by anything latent. The
   architecture-prior result reframes it: the failure may have been the
   dense-MLP prior, not latent search itself.
4. **The two regimes need one story.** The single-run universal method
   (rounds 16–17) and lifetime corpus accumulation (rounds 6–13) are the
   same system at different timescales: bank every real solve's vetted
   elites, share one decoder across everything (round 11), warm-start
   exploration from it — while keeping harvest errors independent
   (round 15c's law). Designing that loop so it climbs instead of
   echo-chambering is the central open design problem.
5. **Heterogeneous output shapes.** One shared decoder across different
   output sizes/types needs conditioning, masking, or shape-specific heads
   over a shared trunk; the scaling law needs retesting at many-family
   corpus scale.
6. **API redesign.** The first-class flow is now clear:
   `solve(fitness_fn, output_shape, architecture)` — per-individual
   decoder exploration, within-run distillation, CMA-ES exploitation, and
   an optional persistent corpus/decoder that accumulates across solves.

## Reproduction

```bash
python -m benchmarks.round1_deceptive --self-test          # objectives + CMA-ES validation
python -m benchmarks.round7_scaling   --output ...          # the scaling law
python -m benchmarks.round9_mlp_training --output ...       # PCA scaffolding for the network
python -m benchmarks.round10_online_refine --output ...     # online refinement
python -m benchmarks.round11_universal --output ...          # one decoder across families
python -m benchmarks.round12_weight_mutation --output ...    # weight-mutation channel
python -m benchmarks.round13_dual_decoder --output ...       # dual decoders + repulsion
python -m benchmarks.round14_original_dual --output ...      # original algorithm from scratch
python -m benchmarks.round15_individual_decoders --output ...# per-individual decoders
python -m benchmarks.round15b_universal_harvest --output ... # universal practice harvester
python -m benchmarks.round15c_bootstrap --output ...         # self-teaching loop (stalls)
python -m benchmarks.round16_single_fitness --output ...     # explore-then-distill
python -m benchmarks.round17_architecture_prior --output ... # conv architecture prior
python -m benchmarks.round18_adaptive --output ...           # adaptive switch (packaged defaults)
python -m benchmarks.plot_family_scaling                    # regenerate the charts
```

Environment notes: neural runs require Apple MPS and verify it; the CMA-ES
baseline is a from-scratch Hansen-tutorial implementation validated against a
16-d sphere (pip in this homebrew Python 3.14 is broken — libexpat mismatch —
so pycma/matplotlib were unavailable; charts are hand-generated SVG).
