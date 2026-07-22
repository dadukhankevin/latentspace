# What we learned: from a universal decoder to a learned genetic code

This document is the narrative record of a twenty-round experimental
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

**Round 19** swept the latent size {8-128} through the packaged API:
a cliff below ~32 (latent 16 gives back half the image win — the space
must exceed the intrinsic variety of good solutions), a broad plateau at
32-128, and 64 as the new default (best means on both problems, half the
image-family seed variance, the tightest GA win recorded). For truly
massive problems the latent should scale with solution variety — never
with raw output size — and `distill_top` must scale with it, since the
distilled space is fit from that many solutions. Decoupling the genome
size from the distilled dimension is the designed follow-up.

Two more controls sharpened the map (rounds 19b/19c). Swapping the exploit
phase's CMA-ES for a fixed-sigma genotype GA loses 0/10 on both problems
(3.7-4.6x) — the distilled directions differ hugely in fitness
sensitivity, and adaptive step sizes are what CMA-ES is for; it stays on
merit, and it is itself an evolution strategy over decoder inputs. And
pure direct CMA-ES on raw solution values splits the world 10/0 each way:
27x better than the stack on the 256-d unimodal curve (its home terrain —
the stack should not be used there), 2.3x worse on the 1,024-d image
(full-covariance adaptation starves; the learned decoders' compressed
directions win). With round 4's multimodality result, the domain map:
direct CMA-ES owns low-dimensional smooth continuous problems; the
universal stack owns high dimensions, multimodality, and everything that
is not a flat float vector.

**Round 20** tested re-entering exploration (cycling the two phases on
symmetric stall rules, with warm re-entry through weight noise). Falsified
at every tested budget: 0/10 on both problems — interrupting CMA-ES's
endgame costs more than renewed exploration returns, and even a 150k-eval
run never actually reached the representation floor that would make
re-entry rational. The one-way conveyor stays the default;
`phases="cycle"` remains as an experimental option, and the principled
future trigger is CMA step-size collapse (floor actually reached), not a
stall heuristic.

**Round 20b** then flipped the long-budget half of the story: a
deterministic counterfactual of the 150k-eval color-photo run (hand-off
disabled) showed the switch itself was the mistake there — CMA-ES sprints
~25k evaluations, then flattens at its frozen gene space's ceiling while
never-switched decoder evolution compounds to 2.3x better (single seed).
The ceiling is concrete: the photo's small leaf region, worthless to
fitness early, was absent from the distilled archive's variation, so the
compressed genes literally could not express it (leaf error 5x the run's
typical error). Regime map: short budgets, distill+exploit wins 4x; long
rich runs, the evolving decoders win 2.3x. Neither one-way order is right
everywhere — a still-untested design is a rate-based scheduler that
interleaves both forces and gives budget to the better measured
improvement rate.

**Round 21** attacked the discrete frontier: 50-city traveling salesman
under random keys (the decoder emits one priority per city; the fitness
function argsorts priorities into a tour — no operator ever touches a
tour), with new sequence decoders (GRU, LSTM, transformer) added to the
registry against the MLP, at matched parameter counts. Potential check,
3 seeds, 5,000 evaluations: a traditional GA mutating tours by segment
reversal reached 8.0; CMA-ES directly on the raw 50 priorities (no
decoder) 9.0; every decoder arm 15.7–17.9 — roughly the midpoint between
random (~25) and greedy nearest-neighbor (~6.9). Two findings. (1) The
sequence priors did nothing: the MLP was the best decoder and all four
were within noise, because city index order is arbitrary — there is no
neighbor structure in the priority vector for recurrence or locality to
exploit. (2) The decoder itself was the handicap: direct CMA on the
identical encoding beat every decoder arm by ~7 points, and an
explore-only diagnostic (no distill/CMA phase) landed at the same ~16,
so exploration — not distillation — is where the stack sticks. This
sharpens the architecture-prior law into a two-sided one: a decoder's
output correlations are a prior, and a prior mismatched to the solution
structure is negative knowledge (images: 24x win; permutations: ~2x
loss). The open route is a problem-conditioned decoder — e.g. a
transformer reading the city coordinates as tokens with the genome as
context, so its prior lives in the city geometry rather than in index
space.

**Round 22** ported the original package's ordering insight (its
PermutationTrainer refused to regress on raw key values and used a
pairwise ordering loss instead) into the universal stack's distill
phase: elites re-represented as normalized ranks — and, in a second arm,
as ranks of the rotation/reversal-canonicalized route — before the PCA.
Null, 3 seeds: value distillation 15.71, rank 15.57, canonical rank
15.68, all within noise, with full elite diversity (200 unique routes in
every top-200). The round-21 diagnostic had already implied this: the
full stack only matched explore-only, so no representation of a ~16
archive had room to help. The structural reading is a boundary on the
error-independence law: distillation's power comes from independent
errors cancelling under averaging, and pixel averages of decent images
are better images, but averages of decent tours are not better tours —
permutations are not closed under the mixing that PCA + CMA search
performs. Exploration (stuck at ~16 vs the tour GA's 8) remains the
binding constraint, which points at changing what the decoder can
express (problem-conditioned architectures) rather than how its output
is compressed.

**Round 23** changed the encoding itself: the decoder emits a 50x50
edge-score matrix and the fitness function constructs the tour by a
greedy walk (from city 0, always to the highest-scoring unvisited
city), on the theory that edges restore both mutation locality and
meaningful averaging (an edge-frequency map is ant colony
optimization's pheromone matrix). Falsified, 3 seeds: solver arms
17.3-17.8 and a no-decoder GA on raw matrices 18.8 — all WORSE than
random keys' 15.7, against the tour GA's 8.0. The averaging theory was
never even reached, because exploration again produced nothing worth
distilling: a greedy walk is a chaotic decode (one changed early score
reroutes the entire remaining trajectory), so mutation locality got
worse, not better. The sharpened law from three TSP rounds: the tour
GA's advantage is that segment reversal changes EXACTLY TWO edge
lengths — a minimal fitness-relevant perturbation — while any decoder
mutation under either encoding scrambles many tour edges at once.
Mutation-to-fitness locality, not representation of elites, is what a
permutation encoding must buy. Untested route: problem-conditioned
decoders (city coordinates as decoder input), where an untrained
network already computes spatially coherent scores — the deep-image-
prior mechanism, which is exactly what conv decoders provided on
images.

**Round 24** tested that route: a problem-conditioned decoder — a small
transformer whose input tokens are the 50 city coordinates (held in a
buffer evolution cannot touch), genome added as a context vector, one
priority per city out, random-keys fitness unchanged. This is a third
interface tier: tier 0, fitness function only; tier 1, modality-shaped
architecture (conv on images); tier 2, architecture reading the same
public instance data the fitness function reads (never any answer).
First real movement on the discrete frontier, 3 seeds: 15.71 -> 11.08
mean best tour, winning 3/3 paired seeds over the tier-0 MLP — but
still losing 0/3 to the tour GA's 8.00. The mechanism check confirmed
the deep-image-prior analogue directly: the best of 32 completely
UNTRAINED (genome, weights) draws averages ~13.1 for the conditioned
transformer vs ~22.2 for the MLP — the untrained spatial prior alone
beats 5,000 evaluations of tier-0 evolution (~15.7). The sobering half:
evolution then only improved the prior's ~13 to ~11, so
mutation-to-fitness locality is still the open deficit; the prior moved
the starting line, not the climb rate.

**Round 25 — the discrete frontier falls.** The climb-rate deficit had
a structural cause: round 24's genome entered as ONE global context
vector, so every genome mutation shifted all 50 priorities at once.
Round 25 restructured the genome's entry point into the same
transformer: the 64 genes are read as 8 spatial ANCHORS (2 genes: a
position in the unit square; 6 genes: a feature vector), and each city
draws its conditioning from the anchors near it (softmax over negative
squared distance, bandwidth 0.15). A mutation to one anchor now edits
priorities in one REGION of the tour — the decoder-side analogue of
segment reversal's two-edge locality. Result: 10-seed paired
confirmation vs the tour GA, **9/1, mean 7.85 vs 8.07, t = 2.47 —
significant. The first discrete win in the campaign, and the last
unconquered problem class.** The trajectory decomposition shows the
mechanism did exactly what it was built to do: the anchor field's
untrained prior matches the global-context version (~12), but
exploration now descends 12 -> ~8.3 (vs 13.8 -> 11.9 global) and runs
1.5-2x longer before stalling because it keeps finding improvements.
The parallel law, now with two data points: give the decoder the
problem's own geometry (tier 2), and give the GENOME spatially local
influence over the output — convolution did both jobs for images;
coordinate tokens + anchor fields do them for tours. Caveats: the margin is modest (2.7%) at 50 cities, and no comparison
against problem-specific local search (a 2-opt hill climber would
likely still win at these budgets — the claim is "beats the
traditional GA," not "beats TSP heuristics"). The mutation-only
baseline concern was tested and retired (round 25e): adding order
crossover made the tour GA WORSE at both sizes (8.07 -> 8.88 at 50
cities, 21.01 -> 22.05 at 100; population-32 crossover of mediocre
parents is disruptive while inversion hill-climbs), so the anchor
field beat the stronger GA variant.

The city-count sweep (3 seeds per size, same 5k budget) then mapped the
regime, and it is the campaign's oldest pattern again: at 20 cities the
tour GA wins (3.66 vs 4.25 — the problem is small enough for native
operators to nearly solve, cf. direct CMA owning low-d smooth); at 50,
the anchor field edges ahead 2.7% (9/1); at 100 cities it wins by 33%
(confirmed at 10 seeds: 14.09 vs 21.01, 10/10, t = 15.7) —
segment reversal fixes two edges per accepted move and starves at 50
evaluations per city, while the anchor field's spatial prior and
regional edits scale. Decoders win where search is hard; the advantage
COMPOUNDS with problem size. Also notable at 100 cities: the distill ->
CMA phase contributed large gains (e.g. 16.3 -> 13.2 after exploration
stalled) — the first time the exploit phase has worked on TSP,
plausibly because anchor-decoder priorities are spatially smooth enough
that averaging elites is finally meaningful.

**Round 26 — one genome grammar for every modality.** Strip the word
"city" out of round 25's design and nothing left in it is about TSP:
read the genome as K anchors, each with a location in the space the
solution lives in and a message; every site draws conditioning from the
anchors near it. That sentence is modality-free, so it is testable as a
universal genetic code — the same 64 genes, the same 8 anchors, the same
0.15 bandwidth, on the two problems the campaign had already won with a
different decoder. The decoders are a translation, not a redesign: round
25 built city tokens as `embed(coords) + anchor_conditioning`, mixed them
with a transformer, and read one logit per city; round 26 gives every
pixel (or sample) `embed(coords) + anchor_conditioning`, mixes with a
convolutional trunk, and reads one logit per site. Only the trunk is
modality-shaped, which the campaign already allows.

The first run looked like a loss (image 0.031 vs the conv decoder's
0.0155; curve merely tying the traditional GA) and was bimodal across
seeds. The trajectory instrumentation explained it with a 6/6
correlation: every bad anchor run explored until evaluation 4384, every
good one handed off at 352. **The adaptive scheduler was punishing the
anchor field for working.** Exploration ends when exploration stops
improving — the conv decoder's exploration does essentially nothing
(0.094 -> 0.0935, stalls instantly) and donates the budget to distill ->
CMA, where its real gains happen; the anchor field's exploration
genuinely climbs (0.090 -> 0.068, the same productive descent that won
TSP), so the stall detector kept feeding it budget and starved the
exploit phase. The reserve (10 x latent = 640) is not enough for CMA to
converge. This is open problem #4 arriving as a blocker: the detector
asks "is exploration still improving?" when it must ask "is exploration
improving FASTER than exploitation would?"

Forcing both arms onto the same split (0.07 — which is what the conv
decoder picks for itself, so the comparison is generous to the
incumbent) settles it. On the same 3 seeds, fixing the split left the
conv decoder essentially untouched (curve 0.0258 -> 0.0281, image
0.0155 -> 0.0166) while the anchor field went from 0.0145 -> 0.0037 and
0.0309 -> 0.0071 — the split rescued anchor rather than broke conv.
Confirmed at 10 paired seeds:

| Objective | traditional GA | conv decoder | anchor field | anchor vs conv |
|:---|---:|---:|---:|:---|
| smooth1d_256 (curve) | 0.01404 | 0.02141 | **0.00320** | 10/10, 6.7x, t = 5.52 |
| blob2d_1024 (image) | 0.05816 | 0.01614 | **0.00803** | 9/10, 2.0x, t = 3.87 |

Against the traditional GA the anchor field is 10/10 on both (4.4x,
t = 13.6 on the curve; 7.2x, t = 35.1 on the image). One 64-gene grammar
with unchanged constants now drives tours, images and curves, and beats
the specialized decoder on each — the universal genetic code has loci:
genes 0-1 mean WHERE, 2-7 mean WHAT, in every problem.

Mechanism note: the win arrives at a DIFFERENT phase than on TSP. There,
anchors fixed the climb rate during exploration. Here exploration barely
moves under the fixed split (0.075 -> 0.0746) and nearly all the gain
comes from distill -> CMA, so what the grammar buys on images and curves
is a phenotype distribution that distills into a far better latent space
— 64 genes controlling 8 localized sources is close to a native
parameterization. Same grammar, two different routes to the win.

Two live confounds. (1) blob2d's target IS three Gaussian blobs and an
anchor field IS a sum of localized sources, so its 2.0x may be family
match rather than generality; the curve result (a DCT low-frequency
signal, nothing blob-like, and the larger margin at 6.7x) is the honest
evidence, but a photograph target would settle it. (2) The anchor
decoder carries 7,265 weights against conv's 23,745, and fewer weights
independently helps weight evolution — inseparable from locality here,
because they are the same design decision.

**Round 27 — the goal restated, and the apple without CMA.** Daniel's
ruling, now standing: the deliverable is a neural-decoder GA that BEATS
CMA-ES, not a stack that relies on it. CMA-ES belongs in the baseline
arms next to the traditional GA; the champion arm is pure decoder
evolution; a hybrid winning somewhere is a finding about the baseline's
strength, not the method.

Under that framing, the published apple run (96x96 RGB photo, 150k
evaluations) was rerun with two arms on the identical target. The
champion: pure per-individual anchor evolution — genome + private
decoder weights, no distill, no CMA. The baseline: CMA-ES over the 64
genes of one untrained anchor decoder with frozen weights, which is also
the experiment that separates what the recorded hand-off ceiling
(0.0112, flattening ~25k) actually blamed — CMA itself, or the frozen
gene space it searched.

Results (single run, matching the recorded single-run references):

| Arm | apple MSE at 150k |
|:---|---:|
| traditional GA (recorded) | 0.1200 |
| CMA-ES on frozen anchor genes | 0.0513 |
| distill -> CMA hand-off stack (recorded) | 0.0112 |
| **pure anchor evolution (no CMA)** | **0.0080** |
| pure conv evolution (recorded) | 0.0049 |

Three findings. (1) **The ceiling belongs to frozen gene spaces, not to
CMA** — the frozen anchor space collapses by 5k evaluations (0.0594) and
crawls to 0.0513, 4.6x WORSE than the distilled PCA space, because the
distilled basis at least encoded image knowledge harvested from
exploration; 8 anchors steering a random frozen trunk can describe a
smooth blobby field, not a photograph. The expressive capacity lives in
the decoder weights — precisely what every CMA arm freezes and pure
evolution mutates. This also kills the tempting collapse of the whole
stack to "CMA on anchor genes." (2) **Pure neural evolution beats every
CMA variant on the photo**: direct-on-pixels (unlearnable at 27,648-d),
frozen-gene CMA (6.4x worse), and the hand-off hybrid (1.4x worse). The
crossover is budget-dependent and now mapped: at 5k pure anchor
evolution is barely ahead of the traditional GA (0.113 vs 0.120 —
consistent with the 5k blob check, where it loses to everything with a
CMA in it), it passes the hand-off stack near 50k (0.0112), and it is
still descending at 150k (0.0090 -> 0.0080 over the last 50k). Round
20b's law generalizes across grammars: freezing helps when evaluations
are scarce and caps you when they are not. (3) **The anchor grammar
loses to conv on a photograph** — 0.0080 vs the recorded 0.0049, 1.6x.
Round 26's blob-confound worry was justified at long budget: anchors
win on blobs, curves and tours, but a photo's fine texture rewards
convolution's translation-invariant local filters over 8 localized
sources. The grammar is universal in reach, not yet supreme on every
modality; the conv champion keeps the image crown at long budget.

**Round 28 — can the anchor grammar take the photo crown from conv?**
Round 27 left pure conv evolution ahead of pure anchor evolution on the
apple (0.0049 recorded vs 0.0080), with two candidate explanations: the
anchor trunk's missing capacity, or its missing multiscale texture
machinery. Round 28 varied only the trunk behind the identical genome
entry (8 anchors, unchanged constants), all arms pure evolution, no CMA:
anchor_flat (round 27's decoder, 7.5k weights), anchor_wide (32
channels, depth 4, 38k), anchor_pyramid (anchors paint a coarse 24x24
field, upsample+conv stages refine to 96x96 — conv's multiscale
machinery with the genome still entering only through anchors, 5.2k),
and conv_rgb (the conv champion rebuilt in this harness, dense genome
entry, 47k), staged at 50k then 150k.

Answers. **Capacity is not the gap and actively hurts**: anchor_wide
was 2x worse than anchor_flat at 50k (0.0230 vs 0.0112) — mutations
spread across 5x the weights climb slower. **Multiscale helps but does
not close it**: anchor_pyramid is the best anchor decoder (0.0074 at
150k, vs flat's 0.0080) with the fewest weights of any arm, and its
advantage is stable across seeds (0.00739 / 0.00717). **Conv keeps the
photo crown**: conv_rgb reproduced the recorded reference in this
harness (0.00457 vs recorded 0.00493) and beats the best anchor by
1.6x. The two decoders own different halves of the budget: the anchor
prior dominates early (3.4x ahead of conv at 50k — 0.0105 vs 0.0271;
conv's dense 47k-weight entry climbs slowly), but conv compounds
hardest late and overtakes at ~68k evaluations, still descending at
150k while the pyramid flattens. A photograph's fine texture rewards
translation-invariant filters everywhere; 8 localized sources cannot
carry the endgame detail. Standing map, all pure evolution: anchors own
short-to-mid budgets and win outright on blobs, curves and tours; conv
owns the long-budget photo endgame. The universal-grammar claim stands
for reach, not supremacy — and the crossover being budget-dependent
makes trunk choice (not grammar choice) the remaining scheduling
question on images.

**Rounds 29-30 — the last hand-tuned constant, and the biggest lever
found so far.** Daniel's proposal: mutate exactly the same things, but
set the mutation MAGNITUDE from measured phenotype movement against a
target, rather than from constants in parameter space.

Round 29 probed the premise, spending zero fitness evaluations (parent
and child phenotypes are already decoded for scoring, so displacement is
free to measure — which is the point). Three findings, each worse than
expected. (1) **An untrained decoder is nearly disconnected from its
genome**: output mean 0.515, std 0.002, against a target std of 0.246 —
generation zero is a population of near-identical blank canvases, and a
genome mutation at the shipped sigma moves the phenotype 0.0004 RMS.
Evolution's first job is inflating weights ~100x just to be able to draw.
(2) **Sensitivity drifts within a run, architecture-dependently**: 3.5x
for the anchor decoder, 1.2x for conv — so no fixed schedule fixes it
either. (3) The decisive one: the explorer's `weight_sigma` range
[0.003, 0.02] carries a comment justifying it ("~0% of mutant outputs
beat their parent at 0.03+") that is **architecture-specific and false
for the decoders we now use**. Measured success-vs-sigma on the blob
image at initialization:

| decoder | 0.02 | 0.1 | 0.5 | 2.0 |
|:---|---:|---:|---:|---:|
| mlp | 38% | 28% | 0% | 0% |
| conv2d | 70% | 64% | 52% | 4% |
| anchor | 50% | 38% | 50% | 40% |

The comment is roughly true for the MLP it was measured on and wildly
false elsewhere: the anchor decoder still wins 40% at sigma 2.0 — 100x
the shipped ceiling — where a mutation moves the phenotype 0.22 RMS
instead of 0.0004. By the 1/5th rule (steps are right-sized at ~20%
success), 40-70% means our steps were far too small, everywhere, for
years of rounds.

Round 30 built the controller: one gain multiplies both channels' sigmas,
adapted each generation so mean child-parent phenotype RMS displacement
tracks a target (0.05). Against the shipped explorer and against the
classic Rechenberg alternative (adapt the same gain from success rate
toward 1/5), 3 seeds, 5k evaluations, pure evolution throughout:

| Setup | fixed (shipped) | displacement | success | best vs fixed |
|:---|---:|---:|---:|:---|
| blob2d anchor | 0.0608 | **0.00474** | 0.00526 | 12.8x |
| blob2d conv | 0.0806 | 0.00848 | **0.00771** | 10.5x |
| smooth1d anchor | 0.0622 | **0.00065** | 0.00071 | 95.1x |

**This is the largest single lever the campaign has found**, and it
rewrites the regime map. Displacement-targeted pure evolution at 5k now
beats everything previously ahead of it on the blob: the distill+CMA
stack (0.0071), direct CMA on raw values (0.0438 at these seeds; 0.0161
in the round-26 harness), and the traditional GA (0.0562). On the smooth
curve it scores 0.000654 against direct CMA-ES's 0.000749 on the same
seeds — **beating CMA on the one problem round 19c said CMA owns
outright (27x better than the stack)**, and closing the honest gap round
26 left open by never running that arm. The "short budgets belong to
CMA" law of rounds 20b/27 was never about CMA: it was our mutation
constant being ~100x too small.

Honest limits. **Displacement targeting does NOT beat the classic
success-rate rule** — 4/9 paired wins, |t| <= 1.84 everywhere against a
4.303 threshold at n=3; the two are indistinguishable here, and the
success rule needs no target constant while displacement needs one
(0.05, unswept). The defensible claim is "adaptive step size, by either
rule, is worth ~10-95x", not "displacement is the right signal." Both
controllers independently drove the win rate from ~50% to 11-16%,
converging near the 1/5th rule's prescription from opposite directions —
which is the real evidence that step size, not the signal choice, was the
broken thing. Final gains land at 3-9x the shipped sigma and differ by
architecture (conv ~8-9x, anchor ~3-6x), so no single replacement
constant would work — the adaptation itself is the deliverable.

**Round 31 — the apple record falls to the win-rate rule, after two
controller failures.** Rerunning the apple with round 30's displacement
controller exposed its long-budget flaw: blazing start (error 0.014 by
5k evaluations — the published champion needed >100k to get there) then
deadlock — the child win rate hit 0% by evaluation 15,000 and stayed
there for 135,000 evaluations, because the controller holds phenotype
displacement at a constant 0.05, which is a sledgehammer once the image
is nearly right. Annealing the target to the remaining error (0.3 x RMS)
failed the same way for a structural reason: too-big steps stop
learning, so the error stops shrinking, so the error-tied target stops
shrinking — a deadlock the displacement signal cannot detect (final
0.0087 vs fixed conv's 0.0046). Both its constants (0.05 target, 0.3
fraction) were hand-picked — the very disease under treatment.

The signal that cannot deadlock is Rechenberg's win rate itself (grow
the step when >1/5 of children beat their parents, shrink when fewer —
no constants tied to problem scale). On the apple it produced the
campaign's cleanest trajectory: gain 18.8x at evaluation 700
(sledgehammer, win rate 28%), shrinking through 1.0 around ~10k, down to
0.12x — 8x FINER than the shipped sigma — in the endgame, win rate
pinned at 17-26% throughout. It spans a 150x step-size range in one run
with no problem-specific numbers. **Result: all-time apple record,
0.004566, in 120,256 evaluations — beating fixed conv's 0.004567 with
20% less budget and still descending at the early stop.** The same run
that owns the fast start owns the endgame; there is no remaining regime
where fixed sigma wins. Verdict, refining round 30: adaptive step size
is worth 10-95x at short budgets AND the long-budget record, but the
signal must be the win rate; displacement targeting (constant or
error-annealed) is falsified at long budget. Weight decay (sensitivity
drift is real, round 29) remains untested, one variable at a time.
Run artifacts: mps_round31_{anchor_earlystop,conv_annealed,conv_success}
.json, each with animation frames.

**Round 32 — TSP under win-rate control: two findings, one correction.**
Every TSP number was measured with the fixed mutation constant rounds
29-31 falsified, so the regime map was rerun (3 seeds, same instances;
the tour GA never touches the decoder and reproduced its recorded scores
exactly, so it is a fixed yardstick).

**(a) The 1/5th rule's core assumption fails on step-function fitness.**
TSP's decoder emits priorities and the fitness function argsorts them
into a tour, so shrinking the step past a threshold does not make
children *slightly* better — it makes them *identical*. Win rate then
falls toward 0 instead of rising toward 50%, the controller reads "few
wins, shrink", and shrinking makes more ties. Measured on 100-city TSP:
the gain hit its floor by generation 45, ties reached 75%, and the best
tour froze at 17.03 for the last 2,400 evaluations. **A tie means the
mutation changed nothing — evidence the step is too SMALL.** Counting
ties as successes (`<=`, now in `explorer.py`) removes the spiral: the
gain stabilizes at 0.03-0.07 and the win rate parks at 16-25%, on
target. Continuous phenotypes never tie, so the fix is a no-op there.

**(b) What helps exploration hurts distillation.** At 100 cities,
win-rate control improved pure exploration in 3/3 seeds (17.03 -> 15.48
mean) — but made the FULL stack worse (14.08 -> 15.09), because the
distill -> CMA phase collapsed from contributing 2.95 to contributing
0.39. The oversized fixed step was accidentally scattering lineages into
a diverse archive; PCA feeds on exactly that diversity. This is the
error-independence law (a teacher's value is the independence of its
errors) reappearing as a tension: right-sized steps converge lineages,
correlate their errors, and starve the compressor. Distillation wants
worse exploration.

**(c) The correction.** Under the standing ruling that CMA-ES is a
baseline and not a component, the honest headline is pure evolution.
There the rerun is a clean win: **pure anchor evolution beats the
traditional tour GA by 27% at 100 cities (15.48 vs 21.06), improved from
9% under the old constant.** But the recorded "+33% at 100 cities" was a
full-stack number and the trajectory shows CMA carried most of it
(exploration only reached 18.42; CMA delivered 14.66). That claim
overstated the decoder GA. At 50 cities the anchor field now LOSES to
the tour GA (8.12 vs 8.00) where the stack won 9/1, and at 20 cities it
still loses (4.46 vs 3.66) — small tours belong to segment reversal, and
that was never a step-size artifact. Standing discrete map: pure decoder
evolution wins big and growing at 100+ cities, loses below ~50.

**Round 33 — the CMA baseline on tours, and how far the advantage
scales.** Round 32 left two gaps: every TSP claim raced only the
traditional tour GA (the sole direct-CMA-on-tours number was 50 cities,
where CMA lost 8.97 to 8.00, after which it was quietly assumed
irrelevant), and there was no data above 100 cities. Both filled, 3
seeds, 5k evaluations, decoder GA = PURE evolution (no distill, no CMA).

| cities | tour GA | CMA-ES | decoder GA | vs GA | vs CMA | wins |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 21.06 | 27.01 | **14.91** | 1.41x | 1.81x | 3/3 |
| 200 | 56.67 | 83.59 | **31.65** | 1.79x | 2.64x | 3/3 |
| 400 | 143.11 | 186.34 | **63.39** | 2.26x | 2.94x | 3/3 |

**Both margins widen monotonically with problem size**, and CMA-ES loses
to the traditional tour GA at every size tested — at 100+ dimensions its
covariance is unlearnable inside this budget, the same wall it hit on the
apple's 27,648 pixels. The mechanism is structural: segment reversal
fixes exactly two edges per accepted move, so the tour GA starves as
cities grow; direct CMA's search space grows with the city count; the
decoder GA searches **the same 64 genes at 400 cities as at 20**. The
problem grows, its search space does not.

Also: **the CMA phase no longer earns its budget on tours.** Pure
evolution now beats the full stack at 100 cities (14.91 vs 15.09) and the
stack's edge at 200/400 is +0.52/+0.83 — a rounding error next to the
2-3x the decoder GA takes on its own, and reversed from the 2.95 CMA
contributed under the old broken constant (round 32). Fixing the
mutation step dissolved the stack's reason to exist here.

**The small-instance gap closed to a tie (round 33b, 10 seeds).** Rounds
25/32 reported the decoder GA LOSING at 20 and 50 cities, but those runs
used the full stack; round 33 showed the CMA phase actively hurts on
tours. Rerun as the pure decoder GA, 10 paired seeds:

| cities | tour GA | decoder GA | wins | paired t | verdict |
|---:|---:|---:|---:|---:|:---|
| 20 | 3.880 | 3.999 | 3/10 | -1.49 | statistical tie |
| 50 | 8.070 | 8.177 | 5/10 | -0.69 | statistical tie |

Neither margin clears the 2.262 threshold, so the earlier "small tours
belong to segment reversal" claim is retired: it was the CMA phase's
damage, not the decoder GA's failure (20 cities: 4.46 with the stack ->
4.00 pure; the 50-city loss was 5/10 seeds, i.e. a coin flip). **Full
discrete map, decoder GA vs the traditional tour GA: ties at 20 and 50,
then wins 3/3 at 100 (1.41x), 200 (1.79x) and 400 (2.26x), with the
margin widening monotonically.** There is no size at which the tour GA
is significantly better. A GA that has never heard of a tour now matches
or beats a GA hand-built for tours at every size tested — and beats
CMA-ES everywhere too.

**The standing honest caveat, unchanged and important:** greedy
nearest-neighbor construction still beats everything on this board
(10.63 / 13.01 / 18.76), and its lead *grows* with size — 1.4x at 100
cities, 3.4x at 400. The claim is "a GA that knows nothing about routing
beats a GA hand-built for routing, and beats CMA-ES," not "this is a good
TSP solver." Problem-specific constructive heuristics and local search
(2-opt) remain far ahead, and always were.

**Round 34 — do anchors still help once mutation self-tunes?** Daniel's
question: the anchor grammar and win-rate step control attack the same
problem (mutation-to-fitness locality) from two directions, so is the
grammar now redundant? Three decoders, identical evolution (pure decoder
GA, win-rate control, no distill/CMA), decomposing the two tier-2 ideas:
`mlp` never sees the city coordinates; `city_context` reads them as
transformer tokens with the genome added to every token as ONE global
vector; `anchor_field` reads them with the genome entering as 8 spatial
anchors.

| cities | tour GA | mlp | city_context | anchor_field | coords worth | anchors worth |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 8.00 | 17.09 | 11.47 | **7.82** | 1.49x | 1.47x |
| 100 | 21.06 | 37.18 | 23.01 | **14.91** | 1.62x | 1.54x |
| 400 | 143.11 | 171.90 | 99.39 | **63.39** | 1.73x | 1.57x |

**Not redundant — orthogonal, and both grow with problem size.** The
prediction that the controller would shrink the anchor advantage was
wrong in direction: the gap was 1.41x on record (11.08 vs 7.85, old
constant + stack) and is 1.47-1.57x now. The reason is mechanical: no
single step size can make a global-context mutation LOCAL. When one
vector is added to every city's token, small steps move all priorities a
little and large steps move them all a lot — there is no setting that
edits one region. Anchors change what a mutation can EXPRESS; the
controller changes how far it reaches. Different axes.

Load-bearing: **`city_context` loses to the tour GA at 50 and 100 cities**
(0.70x, 0.92x) and only passes it at 400 (1.44x). Every discrete win the
campaign has belongs to the anchor grammar specifically — instance
conditioning alone is not enough, and step control does not rescue a
global genome.

The trajectory decomposition names the mechanism exactly. Evolution's
multiplier on its own starting point: `anchor_field` climbs 1.56x / 1.49x
/ 1.53x at 50/100/400 while `city_context` climbs only 1.21x / 1.18x /
1.17x — flatter than even the coordinate-blind `mlp` (1.30x / 1.23x /
1.14x). Coordinate conditioning buys a better PRIOR (city_context starts
at 13.85 vs mlp's 22.29 at 50 cities); only anchors buy a better CLIMB.
This is round 24 -> 25's original finding surviving intact under a fixed
mutation step: the prior moves the starting line, the grammar decides
whether evolution can walk.

**Round 35 — the apple crown under win-rate control: a new record, and
the modality boundary is real.** Round 28's anchor-vs-conv photo verdict
was measured with the falsified mutation constant, so both arms were
rerun with the win-rate controller, pure decoder GA, same 150k budget and
target.

| run | apple MSE |
|:---|---:|
| published conv, fixed step | 0.004929 |
| in-harness conv, fixed step (round 28) | 0.004567 |
| conv, win-rate step, stopped early at 120k (round 31) | 0.004566 |
| **conv, win-rate step, full 150k (round 35)** | **0.004005** |
| anchor pyramid, win-rate step (round 35) | 0.009974 |

**New all-time apple record: 0.004005** — win-rate control is worth 1.14x
over the fixed step for the same decoder and budget, on top of setting
the record 20% early in round 31. And the anchor grammar got WORSE under
a correct step (0.0074 -> 0.0100), widening conv's lead from 1.6x to
**2.49x**. The modality boundary is not a mutation artifact: anchors win
curves, blobs and tours; convolution owns photographs. The trajectories
show why — the anchor pyramid leads early (0.0184 vs 0.0235 at 5k) then
flattens hard (0.0106 -> 0.0100 over the last 50k) while conv compounds
to the end (0.0050 -> 0.0040). Eight regional sources paint broad fields
well and cannot carry fine texture; translation-invariant filters can.

**Round 36 — one shared decoder + per-individual low-rank adapters
(Daniel's LoRA proposal).** Instead of every individual carrying full
private weights, share one backbone and store only a small per-individual
modification: `weights = backbone + P @ adapter`, with `P` a fixed random
projection and `adapter` a rank-r vector. Implemented on the flat weight
vector rather than per-layer, so it works for any architecture and keeps
universality (Li et al.'s intrinsic-dimension form). Crucially this is
NOT round 27's frozen decoder — the weights still evolve, they are merely
compressed, which is distillation's compression without distillation's
ceiling. Memory was never the point (~1MB per population); the search
space is: full weights make evolution perturb ~7,500 numbers per
individual.

| setup | arm | evolved dims | MSE | vs full |
|:---|:---|---:|---:|---:|
| blob image | full_weights | 7,329 | **0.00526** | 1.00x |
| | subspace_64 | 128 | 0.00665 | 1.26x |
| | subspace_256 | 320 | 0.00771 | 1.46x |
| | subspace_16 | 80 | 0.00923 | 1.75x |
| curve | full_weights | 5,489 | **0.00072** | 1.00x |
| | subspace_256 | 320 | 0.00097 | 1.36x |
| | subspace_64 | 128 | 0.00191 | 2.67x |
| | subspace_16 | 80 | 0.00254 | 3.55x |

**The mechanism works but costs quality: full weights win both.** A 57x
smaller search space costs 26% on the image (rank 64) and a 44x smaller
one costs 36% on the curve (rank 256). Nothing collapsed — the
error-independence canary (elite phenotype spread) held at 0.024-0.028 vs
full's 0.029 on the image and 0.016-0.017 vs 0.018 on the curve, so the
shared backbone did NOT correlate the population's errors; the cost is
expressiveness, not diversity.

The rank response is **non-monotonic on the image** (16: 1.75x, 64: 1.26x,
256: 1.46x) — an inverted U with an optimum near rank 64. That is round
28's capacity law arriving from a new direction: too few dimensions cannot
express the solution, too many dilute each mutation across more numbers
and slow the climb. The curve's optimum is higher (256 best, still
climbing) — consistent with it being the problem where full weights won by
95x, so it has the most structure to lose under compression. Optimal rank
tracks how much weight variety the problem needs, exactly like `latent`
tracks solution variety (round 19).

Verdict: a real, tunable compute/quality dial with a clean interpretation,
not a free win. Worth revisiting if per-individual weights ever become a
bottleneck (large decoders, big populations), or as a diversity mechanism
where the backbone itself evolves. As a default it loses to full private
weights on both problems tested.

**Round 37 — should evolution choose WHERE it mutates the decoder?**
Daniel's follow-up: let each individual modify a different part of the
base decoder, and let that choice mutate — the anchor grammar
transplanted into weight space (an anchor has a location and a message;
an adapter would have a location in weight space and a value). All arms
keep full private weights and vary only which coordinates a mutation may
touch, so this isolates concentration and locality from round 36's
expressiveness ceiling; every arm can still reach all of weight space.

| setup | arm | MSE | vs full | site overlap |
|:---|:---|---:|---:|---:|
| blob image | full_weights | **0.00533** | 1.00x | — |
| | sparse_random (k=256, fresh each time) | 0.01160 | 2.18x | — |
| | sparse_evolved (k=256, inherited) | 0.01175 | 2.21x | 0.70 |
| curve | full_weights | **0.00072** | 1.00x | — |
| | sparse_random | 0.00183 | 2.56x | — |
| | sparse_evolved | 0.00338 | 4.73x | 0.77 |

**Both falsified. Diffuse beats concentrated, and WHERE carries no
inheritable information.** Restricting mutation to 256 of ~7,500
coordinates costs 2.2x on the image and 2.6x on the curve regardless of
how those coordinates are chosen; inheriting the location is a tie on the
image (2/3 seeds, diff -0.00015) and actively 1.8x WORSE on the curve.
The high site overlap among surviving elites (0.70-0.77 vs ~3% by chance)
is shared ancestry, not selection finding meaningful weights — if it were
signal, sparse_evolved would beat sparse_random, and it does not.

**Why the anchor trick does not transplant: weight space has no metric.**
Anchors work because they MANUFACTURE locality in a space that has one —
cities and pixels have positions, so editing a region is meaningful.
Weight 500 is not "near" weight 501, and a neural network is a
distributed representation: no weight subset owns a patch of output. So
editing 256 weights does not edit a region of the image, it perturbs
everything badly. Locality helps exactly where locality exists.

**A related result that needs no experiment: merging adapters into the
backbone is a mathematical no-op.** With weights = `backbone + P @ a`,
merging the mean (or best) adapter and re-centering gives
`backbone' = backbone + P@ā`, `a' = a - ā`, so
`weights' = backbone + P@ā + P@(a - ā) = weights` — bit-for-bit
unchanged, and mutation is applied at absolute scale so it is unchanged
too. The variants that are NOT no-ops are the ones falsified here:
merge-then-redraw the projection (rounds 36 vs 37 argue against it — a
FIXED rank-64 subspace cost 1.26x while freshly-drawn sparse directions
cost 2.2x, so evolution benefits from accumulating progress in a
CONSISTENT basis, and redrawing more often is the wrong direction), and
merge-then-reset-adapters (an elitist restart collapsing 32 lineages onto
one point — the second law's exact failure mode).

Standing verdict on the compressed-weight family (rounds 36-37), **scoped
correctly after Daniel's scaling objection**: full private weights win
every comparison *at the sizes tested* (~7.5k params, ~1MB per
population), and at those sizes the direction optimizes a resource we are
not short of. But the verdict does NOT scale. At 1B params, 32 full
copies is 128GB; at 1T it is impossible — and search efficiency dies
before storage does (round 28: mutations diluted across more weights
climb slower; at 1e9 weights evolution's sample efficiency collapses).
**Above that line, per-individual low-rank adapters stop being a
compression trick and become the only way the method exists**: one
backbone, per-individual state = genome + small adapter, batched through
a single model. Round 36 also tested the WORST case for that regime — a
frozen RANDOM backbone, where a low-rank slice is a slice of nothing. At
scale the backbone would be pretrained, and pretrained networks have low
intrinsic dimension (the reason LoRA works at all) — the deep-image-prior
story taken to its conclusion: a learned manifold for free, evolution
searching the small honest space around it, genome still 64 floats,
universality invariant intact. Open question for that frontier: does a
pretrained backbone invert the 1.26x quality penalty measured against a
random one? Needs hardware this campaign does not have.

**Round 38 — how many survivors does the collapsed population need?**
Round 37's probe had shown the population collapses to one ancestral
lineage by generation 4, so Daniel's read was to embrace it: if the
survivors are becoming the same decoder, keeping many is waste. Sweeping
the survivor count (everything else fixed: pure decoder GA, win-rate
control, 32 children/generation, 5k):

| survivors | image | curve | TSP-100 |
|---:|---:|---:|---:|
| 1 | **0.00398** | **0.00098** | 18.55 |
| 2 | 0.00414 | 0.00121 | 18.01 |
| 4 | 0.00483 | 0.00132 | 16.83 |
| 8 | 0.00526 | 0.00299 | 16.64 |
| 16 (shipped) | 0.00771 | 0.00229 | **14.91** |

**The answer was problem-dependent — the worst kind of knob.** Smooth
problems wanted a single champion (1.9x/2.3x better than the shipped 16);
the rugged step-function tour landscape wanted many survivors (1.25x the
other way). Survivor count tracks landscape ruggedness: one lineage
hill-climbs a smooth surface, but only multiple independent bets escape
plateaus. A correction to round 37's framing fell out later (round 40):
"elites are near-clones 1.3% apart" is wrong in the units that matter —
they sit ~3 MUTATION STEPS apart, operationally distinct.

**Rounds 39 and 41 — four attempts to steer the survivor count
closed-loop. All four failed, and the failure generalizes.** The bar:
match the better fixed setting on every problem without being told which
problem it is.

* *Rank credit* (do non-champion survivors produce more than their share
  of champion-beating children?) — collapsed to 1 survivor everywhere.
  Structural downward bias: the rank-0 parent IS the champion, so it is
  the likeliest single parent to beat it. First version had the worse
  bug, a reusable one: **"did the child beat its own parent" is a corrupt
  signal** — the bar is lower the worse the parent, so it pays lineages
  for being mediocre (measured: drove survivors to ~19 on the image,
  where 1 is 1.9x better).
* *Stall response* (champion improving => narrow; stuck => widen) — got
  the direction right on all three problems but reached 16.73 on TSP vs
  fixed-16's 14.91, because...
* **...culling is irreversible.** Shrinking from 16 survivors to 12
  extinguishes 4 lineages; re-widening later just keeps more copies of
  whoever is left. TSP's early generations LOOK smooth (the champion
  climbs, survivors are fitness-spread), so every rule culls during the
  climb; the plateaus that need the diversity arrive after it is gone.
* *Monotone annealing* (start wide, only contract) — best of the family:
  1.33x BETTER than any fixed count on the image (0.00300, an all-time
  image best that still stands), but 0.62x on the curve and 0.94x on TSP.
* *Fitness-spread detection* (round 41; round 40's probe found survivors'
  relative fitness spread separates smooth from rugged by 10-100x — tied
  survivors that are genotypically far apart ARE the plateau signature) —
  failed for the same reason, measured directly: in the first quarter of
  the run, TSP's spread (2.55e-2) reads HIGHER than the image's
  (1.38e-2); the 100x separation only emerges after ~1,250 evaluations,
  when culling is already done.

**The two-sided law: the evidence for keeping diversity only becomes
legible after the moment you needed to act on it.** Two independent
signal families (performance credit, fitness geometry) hit the same wall.
A fixed survivor count of 16 wins on TSP by refusing to decide.

**Round 40 — where does survivor diversity live? (Falsified its own
premise, found the real signal.)** Genotype distances, in mutation-step
units so the threshold is constant-free (two lineages closer than one
step are operationally the same lineage): image 3.1 steps, curve 2.6,
TSP 2.3 — **no discrimination**; a genotype-diversity controller would
answer "~10 survivors" everywhere, wrong on both ends. Adding decoder
distance changes nothing (it tracks the genome). The column that DID
separate the problems 10-100x was relative fitness spread — predicted
blind on plateaus, actually the plateau detector — but see round 41: it
arrives too late to use.

**Round 42 — the decoder GA never had crossover.** Nobody removed it:
`universal/explorer.py` was written mutation-only from day one, while its
docstring claimed otherwise, and the working `Crossover` layer sits in
`evolver.py` — the traditional-GA baseline this campaign beats. Every
prior result was really a (mu + lambda) evolution strategy. Daniel's
call: add one-point genome crossover, decoder inherited whole from the
fitter parent (round 37's co-adaptation result argues weight vectors must
not be blended... see round 45 for the correction), every child still
mutated after. At 16 survivors: image 1.17x better, curve 1.51x, TSP-100
0.89x (worse). And the survivor sweep CHANGED SHAPE: with crossover, 8
survivors became best-or-near-best on all three problems at once
(0.00403 / 0.00115 / 15.50) where without it 8 was near-worst on smooth.
**Round 38's problem-dependence was substantially an artifact of having
no recombination operator: many lineages only pay if something can
combine them.**

**Round 43 — mate selection and cut placement (Daniel's diagnosis: don't
mix bad with good).** Uniform mate choice means the average mate is rank
~7.5 and the operator grafts middling genes into the best pairs.
Tournament selection (k=3) at 8 survivors: image 0.00342 (1.22x over
uniform — at the time the best fixed-configuration image result), curve
tie, **TSP worse** (17.73 vs 16.12) — on a plateau ranks are nearly
meaningless, so rank pressure concentrates on an arbitrary winner and
burns the diversity. Grammar-aligned cuts (cut only on anchor
boundaries so no anchor is chimeric): **falsified**, 16.49 vs 16.44
free — slicing through an anchor is not what hurts. Verdict: uniform
mates as the universal default; rank bias is a smooth-problem
optimization.

**Round 44 — Daniel's classical reproduction scheme vs the rigid
conveyor.** Standing population of 32, rank-biased pair selection, random
family sizes (1-3), some individuals mutation-only, some idle, trim the
worst. **The rigid loop won everywhere at 5k** — image 1.26x, curve
1.24x (both outside noise), TSP leaning the same way — reproduction
randomness is evaluation waste at short budgets; a child granted to a
mediocre individual by lottery is an evaluation the champion did not get.
Incidental: the rigid crossover arm put up 14.80 on TSP-100, nominally
the campaign's best tour score.

**Round 45 — attribution, decoder inheritance, and the vmap payoff.**
The 30-run benchmark took 59 seconds after wiring in the batched decode
(one `torch.func.vmap` call decodes the whole population; sequential
per-individual decode was ~96% of wall clock and 7x slower — measured
bit-identical). With that speed, 10 paired seeds everywhere (threshold
t = 2.262):

| | mutation only | + crossover (fitter decoder) | + crossover (averaged decoders) |
|:--|---:|---:|---:|
| image | 0.00568 | 0.00415 (t=3.3) | **0.00336 (t=5.6)** |
| curve | 0.00247 | **0.00103 (t=4.5)** | 0.00109 (t=5.3) |
| TSP-100 | 16.78 | 16.52 (ns) | 16.27 (ns) |
| TSP-200 | 33.77 | 32.01 (ns) | 32.20 (ns) |
| TSP-400 | 69.80 | 65.84 (t=2.7) | **64.19 (t=4.0)** |

**Three verdicts.** (1) *Crossover is real*: significant on image and
curve; "neutral on tours" was a small-instance artifact — the t-statistic
climbs monotonically with city count and crosses significance at 400
(8% better), exactly Daniel's prediction that it should help and the
weirdness was elsewhere. Mechanism consistent with building blocks: at
400 cities each anchor governs ~50 cities, single mutations are
relatively smaller edits, and recombining two parents' anchor layouts
covers distance mutation cannot. (2) *Averaging the parents' decoders is
the best inheritance rule* (Daniel's proposal; top arm on 4 of 5
problems, significant on image and TSP-400): round 37's co-adaptation
catastrophe does not apply WITHIN a run, because survivors are
lineage-collapsed and a few steps apart — averaging near-clones cancels
mutation noise instead of breaking function. Cross-RUN averaging remains
catastrophic. (3) The mutation-only baseline at 10 seeds confirms
crossover's smooth-problem gains are the operator's, not the extra
mutation riding on it.

**Shipped to the library after round 45**: crossover="average" (one
genome cut, uniform mates, averaged decoders, off switch retained),
elite=8, batched vmap decoding, and a docstring that now tells the truth.
Library reproduces the benchmark (image 0.00318 vs 0.00336). Round 46
reruns the 150k apple through `solve()` itself — the first published
number produced by the shipped framework rather than a benchmark-local
loop: **0.003248, a new record**, 1.23x past the previous 0.004005.

**Round 47 — equal WALL-CLOCK, the honest weak spot.** Every prior
comparison held evaluations equal, which assumes evaluations are scarce —
true for the method's target domain (expensive black-box scoring), false
on the apple's microsecond MSE. Protocol: time the decoder GA's 150k run
exactly (205.1s), give the pixel GA that wall-clock, and — strongest-
baseline rule — also give the pixel GA our win-rate step control.
**The decoder GA wins on time too**: 0.003248 vs fixed-sigma pixel GA
0.0232 (7.2x) and win-rate pixel GA 0.0110 (3.4x). Two predictions wrong
in instructive ways: the pixel GA got only 2.3x more evaluations (343k),
not 20-50x — mutating 27,648 pixels means ~20MB of random numbers per
generation, a cost that GROWS with output size, while the decoder GA
mutates the same 64+7.5k numbers at any resolution (and vmap batching
closed most of the rest); and the win-rate rule halved the pixel GA's
error (0.0232 -> 0.0110, gain annealed to 0.16x — same mechanism, same
benefit), so the strongest baseline earned its keep and still lost.
Throughput overhead is <1ms/evaluation, so any fitness costing ~1ms makes
the time budget and the evaluation budget the same test.

**Round 48 — the gradient ceiling.** Adam with real gradients, same
wall-clock: on raw pixels it reaches our record in **0.59 seconds** and
then machine-precision zero; training OUR OWN ConvRGB (weights + genome,
deep-image-prior style) it reaches the record in 1.61s and converges to
**0.000011**. Two boundary facts: (1) the black-box premium is ~350x in
time on differentiable problems — if gradients exist, this method is the
wrong tool, full stop; the domain is simulators, physical measurements,
and discrete structures (nobody can backprop an argsort). (2) The
decoder's EXPRESSIVENESS CEILING is 0.000011 — evolution's 0.003248 uses
~0.3% of it, so the search, not the architecture, is the bottleneck, and
the architecture demonstrably CAN paint the green leaf evolution never
found. Untuned learning rates; the result is too lopsided for tuning to
matter.

**Rounds 49-50 — do the decoders have an optimizer? (No — then Daniel
built one out of fitness.)** Weight changes were memoryless noise under
one global gain; nothing watched how the network changed over time.
Round 49 tried two Adam analogues and both LOST everywhere (momentum on
accepted steps 0.78-0.97x, per-parent NES gradient + Adam 0.76-0.97x).
Three deficiencies diagnosed, the third a bug: winners-only memory
discards 80% of the data (win rate is pinned at 20%); 4 children per
parent is a hopeless gradient estimate alone; and children were born with
ZEROED optimizer state, so the accumulator reset as fast as selection
replaced parents. Daniel's round-50 design fixed all three: **every
child's birth mutation, signed and scaled by its fitness change —
failures included, a step that hurt is a measured bad direction — feeds
one shared Adam-style accumulator** (the population as a distributed
gradient sensor; legal within a run because the decoders are
lineage-collapsed near-clones ~3 mutation steps apart), and new mutations
drift along the accumulated direction at half a mutation step. Per-lineage
inheritance of the same signal was mixed-to-worse; POOLING is what makes
the estimate usable. Confirmed at 10 paired seeds: **image 1.38x
(t=3.88, the campaign's best image method at 0.00246)**, curve and
TSP-100 not significant either way (curve +1.07x, TSP -0.975x lean).
Mechanism boundary as ever: the estimator needs fitness DIFFERENCES to
carry direction; TSP's ties and steps feed it noise. On the apple at
150k, same seed and loop: baseline 0.00274 -> signed memory **0.00178 —
the first sub-0.002 apple, 1.54x paired** (1.83x vs the round-46
record, cross-loop). Shipped as `mutation_memory="shared"` (default on;
"off" retained). One day's arc: 0.004005 -> 0.00178, and the ceiling
measurement says that is still only ~0.6% of what the architecture can
express.

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
| Sequence decoders (GRU/LSTM/transformer) on TSP random keys (round 21) | Falsified — all tie the MLP, and every decoder loses ~2x to a tour GA and to direct CMA on the same encoding; index-space priors are negative knowledge for permutations |
| Ordering-aware distillation: rank / canonical-rank elites (round 22) | Null — all within noise of value PCA; averaging decent tours does not make better tours, so PCA+CMA mixing is structurally inert for permutations; exploration is the binding constraint |
| Edge-matrix encoding + greedy-walk construction (round 23) | Falsified — worse than random keys (17.3+ vs 15.7 vs tour GA 8.0); greedy-walk decoding is chaotic, so mutation locality worsens; the tour GA wins because segment reversal changes exactly two edge lengths |
| Problem-conditioned decoder: city coordinates as transformer tokens (round 24) | First movement on the discrete frontier — 15.7 -> 11.1, 3/3 over the tier-0 MLP; the UNTRAINED spatial prior (~13) already beats 5,000 evals of tier-0 evolution; still loses 0/3 to the tour GA (8.0) because evolution only adds ~2 to the prior |
| Anchor-field genome: 8 spatial anchors conditioning the city transformer (round 25) | **Beats the traditional tour GA 9/1, t = 2.47, on 50-city TSP — the campaign's first discrete win.** Localizing genome influence fixed the climb rate (explore 12 -> 8.3 vs 13.8 -> 11.9 global); margin modest (2.7%) at this operating point. **SUPERSEDED by round 32** — measured with the falsified mutation constant, and the trajectory shows CMA-ES carried most of the margin |
| Apple crown rerun under win-rate control (round 35) | **New all-time record 0.004005** (conv, 150k) — win-rate control worth 1.14x over the fixed step at the same budget, after already setting the old record 20% early. Anchor pyramid got WORSE under a correct step (0.0074 -> 0.0100), widening conv's lead 1.6x -> **2.49x**: the modality boundary is REAL, not a mutation artifact. Anchors lead early (0.0184 vs 0.0235 at 5k) then flatten (0.0106 -> 0.0100 over the last 50k) while conv compounds (0.0050 -> 0.0040) — regional sources cannot carry fine texture |
| Evolving WHERE the decoder is mutated (round 37) | **Both halves falsified.** Restricting mutation to 256 of ~7,500 coords costs 2.18x (image) / 2.56x (curve) however chosen — diffuse beats concentrated. Inheriting the location ties on the image (2/3 seeds) and is 1.8x WORSE on the curve, so WHERE carries no inheritable information; the 0.70-0.77 elite site overlap is shared ancestry, not signal. **Weight space has no metric** — anchors work by manufacturing locality where a metric exists (cities, pixels); a network is a distributed representation, so no weight subset owns a patch of output. Corollary needing no run: merging adapters into the backbone is a mathematical NO-OP (`backbone + P@ā` with `a - ā` leaves weights bit-for-bit identical) |
| Shared backbone + per-individual low-rank adapters (round 36) | **Works, but costs quality — full private weights win both problems.** `weights = backbone + P @ adapter` (flat-vector LoRA, architecture-agnostic): 57x smaller search space costs 26% on the image (rank 64: 0.00665 vs 0.00526), 44x smaller costs 36% on the curve (rank 256: 0.00097 vs 0.00072). Error-independence canary HELD (elite spread 0.024-0.028 vs full 0.029) — the shared backbone did not correlate errors; the cost is expressiveness. Rank response is an inverted U on the image (optimum ~64) = round 28's capacity law again: too few dims cannot express, too many dilute each mutation. Optimal rank tracks the problem's weight variety, as `latent` tracks solution variety. A real compute/quality dial, not a default |
| Anchor grammar ablation under win-rate control (round 34) | **Anchors are NOT subsumed by adaptive mutation — the two are orthogonal and both grow with size.** Pure decoder GA, coords worth 1.49x/1.62x/1.73x and anchors a FURTHER 1.47x/1.54x/1.57x at 50/100/400. Global-genome `city_context` LOSES to the tour GA at 50 and 100 (0.70x, 0.92x): every discrete win belongs to the anchor grammar specifically. Mechanism: no step size can make a global mutation local — anchors change what a mutation can express, the controller changes how far it reaches. Evolution's climb multiplier: anchors 1.5x, global context only 1.2x (flatter than the coordinate-blind MLP) — conditioning buys a PRIOR, anchors buy a CLIMB |
| Small-instance TSP, decoder GA alone vs the tour GA (round 33b) | **The "small tours belong to segment reversal" claim is RETIRED** — it was the CMA phase's damage, not the decoder GA's. Pure, 10 paired seeds: 20 cities 3.999 vs 3.880 (t = -1.49) and 50 cities 8.177 vs 8.070 (5/10 wins, t = -0.69) — both statistical ties at the 2.262 threshold. Combined with round 33: no size where the tour GA is significantly better; ties at 20/50, wins 3/3 at 100/200/400 with margins widening |
| TSP scaling with the CMA baseline finally on the board (round 33) | **The decoder GA beats BOTH the tour GA and CMA-ES at every size >= 100 cities, 3/3 seeds, and both margins widen: 1.41x/1.81x at 100, 1.79x/2.64x at 200, 2.26x/2.94x at 400.** CMA-ES loses to the traditional tour GA at every size (covariance unlearnable at 100+ dims). Mechanism: segment reversal fixes 2 edges per move so the tour GA starves as cities grow; CMA's search space grows with cities; the decoder GA searches the same 64 genes at 400 cities as at 20. The CMA phase no longer earns its budget (pure beats the stack at 100; +0.52/+0.83 at 200/400). Caveat: greedy nearest-neighbor still beats everything and its lead GROWS (1.4x -> 3.4x) |
| Win-rate step control on TSP (round 32) | **Two findings + a correction.** (1) The 1/5th rule's assumption fails on step-function fitness: shrinking the step yields TIES, not small gains, so win rate falls to 0 and the controller death-spirals (gain floored by gen 45, 75% ties, learning frozen). Fix: ties are evidence the step is too SMALL — count them as successes; win rate then parks at 16-25%. (2) Right-sized steps converge lineages, correlate errors and STARVE the compressor: pure exploration improved 3/3 (17.03 -> 15.48) while the full stack got worse (14.08 -> 15.09) as distill+CMA's contribution collapsed 2.95 -> 0.39 — the error-independence law as a tension. (3) Correction: pure anchor evolution beats the tour GA by 27% at 100 cities (15.48 vs 21.06), but LOSES at 50 (8.12 vs 8.00) and 20 (4.46 vs 3.66). Discrete wins live at 100+ cities |
| Anchor-field genome as a UNIVERSAL grammar: same 64 genes / 8 anchors / 0.15 bandwidth on images and curves (round 26) | **One grammar beats the modality-specialized decoder on both: curve 10/10 (6.7x, t = 5.52), image 9/10 (2.0x, t = 3.87); 10/10 vs the traditional GA on both.** Genes 0-1 mean WHERE and 2-7 mean WHAT in every problem tested — tours, images, curves. Win arrives via distill -> CMA here vs via climb rate on TSP. Confounds: blob target matches the anchor prior (curve is the clean evidence); 3x fewer weights than conv |
| Adaptive stall scheduler, on a decoder whose exploration works (round 26) | **Falsified as budget-neutral** — it penalizes productive exploration: conv explore stalls instantly and donates budget to CMA, anchor explore genuinely climbs so the detector starves the exploit phase (4384/5000 spent exploring, reserve 640 too small to converge), turning a 2-7x win into a loss. Fixing the split rescues it. Promotes open problem #4 (rate-based scheduling) to blocking |
| CMA-ES on frozen untrained anchor genes, apple at 150k (round 27) | Falsified as a stack replacement — collapses by 5k evals, caps at 0.0513 (4.6x worse than the distilled hand-off): the ceiling belongs to ANY frozen gene space, not to CMA; capacity lives in decoder weights |
| Pure anchor evolution, no distill/no CMA, apple at 150k (round 27) | **Beats every CMA variant on the photo — 0.0080 vs hand-off 0.0112 (1.4x), frozen-gene CMA 0.0513 (6.4x), GA 0.1200 (15x); crossover vs the hand-off near 50k, still descending at 150k.** Loses to recorded pure conv evolution (0.0049, 1.6x): the anchor grammar is universal in reach, conv keeps the photo crown — round 26's blob confound confirmed real at long budget |
| Fixed mutation sigma in parameter space (rounds 29-30) | **Falsified as universal — the last hand-tuned constant.** The shipped [0.003, 0.02] range was measured on the MLP; viable ceilings differ >10x by architecture (mlp dies at 0.5, conv at 2.0, anchor still 40% win at 2.0). Success rates of 40-70% vs the 1/5th rule's 20% mean steps were ~100x too small everywhere. Untrained decoders emit near-flat output (std 0.002 vs target 0.246), so sensitivity must travel ~100x within a run; it drifts 3.5x (anchor) vs 1.2x (conv) |
| Adaptive step size from measured feedback, pure evolution (round 30) | **Largest lever in the campaign: 10-95x over the shipped explorer** (blob anchor 0.0608 -> 0.0047; blob conv 0.0806 -> 0.0077; curve 0.0622 -> 0.00065). Pure decoder evolution now beats the distill+CMA stack (0.0071), direct CMA (0.0161-0.0438) AND beats direct CMA on the smooth curve — its own home turf — 0.000654 vs 0.000749. The "short budgets belong to CMA" law was an artifact of the bad constant. Daniel's displacement rule ties the classic success-rate rule (4/9 paired wins, \|t\| <= 1.84): adaptivity is what matters, not the signal |
| Displacement-targeted mutation at long budget (round 31) | Falsified — 0% child win rate from evaluation 15k to 150k (constant 0.05 target = endgame sledgehammer); error-annealed variant deadlocks structurally (big steps stop learning -> error stalls -> error-tied target stalls); both need hand-picked constants |
| Win-rate (1/5th rule) mutation control, pure evolution (round 31) | **All-time apple record: 0.004566 in 120k evaluations — beats fixed sigma's 0.004567 with 20% less budget, still descending at the stop.** Gain arc 18.8x -> 0.12x (150x range) with win rate self-pinned at 17-26%; no problem-scale constants; owns fast start AND endgame — fixed sigma wins no remaining regime |
| Anchor trunk variants on the apple, pure evolution (round 28) | Capacity falsified as the gap (38k-weight anchor 2x WORSE than 7.5k at 50k — mutation dilution); multiscale pyramid is the best anchor decoder (0.0074 at 150k, seed-stable 0.00739/0.00717, only 5.2k weights) but conv keeps the crown: in-harness conv_rgb 0.00457 (validates the recorded 0.0049), 1.6x ahead. Anchors dominate early (3.4x at 50k), conv overtakes at ~68k and is still descending at 150k while the pyramid flattens |
| Fixed survivor count (round 38) | **Problem-dependent, the worst kind of knob**: smooth wants 1 survivor (1.9x/2.3x over the shipped 16), rugged TSP wants 16 (1.25x over 1). Survivor count tracks landscape ruggedness. Also corrects round 37: elites are ~3 MUTATION STEPS apart, not "near-clones" — 1.3% in raw weight space was the wrong unit |
| Closed-loop survivor control, four designs (rounds 39-41) | **All falsified against a strict bar (match the best fixed setting everywhere, untold).** Rank credit collapses to 1 (the champion is the likeliest to beat itself); stall response and fitness-spread detection read TSP's early climb as smooth — the spread signature separating rugged from smooth (10-100x, round 40) only emerges after ~1,250 evals, when culling is irreversible. **Law: the evidence for keeping diversity arrives after the moment you needed it.** Reusable bug: "child beat its own parent" is a corrupt signal (lower bar for worse parents). One keeper: monotone annealing scored 0.00300 on the image — 1.33x better than ANY fixed count, still the all-time image best |
| Genotype diversity as the survivor signal (round 40) | Falsified — in mutation-step units the survivors sit 2.3-3.1 steps apart on image, curve AND TSP alike (dedupe to ~10 everywhere), so genotype distance cannot tell the problems apart; decoder distance just tracks the genome. The discriminating column was relative fitness spread — the signal predicted useless — but it arrives too late to steer (round 41) |
| Crossover, added for the first time (round 42) | **The decoder GA was mutation-only from day one — the docstring claimed crossover that never existed; all prior results are really a (mu+lambda) ES.** One-point genome crossover + fitter parent's decoder, at 16 survivors: image 1.17x, curve 1.51x, TSP-100 0.89x. With crossover the survivor sweep changes shape: 8 becomes best-or-near-best on ALL THREE problems — round 38's problem-dependence was substantially the missing operator: many lineages only pay if something can combine them |
| Mate selection and cut placement (round 43) | Tournament mates: 1.22x on the image (0.00342), tie on curve, WORSE on TSP (rank pressure on a plateau concentrates on an arbitrary winner) — uniform is the universal default. Grammar-aligned cuts (never slice an anchor): **falsified**, ties free cuts — chimeric anchors were not the TSP problem |
| Classical stochastic reproduction (round 44) | Falsified at 5k: rank-biased pairs, random family sizes, mutation-only individuals, idling — the rigid conveyor (32 children every generation from hard truncation) wins 1.26x/1.24x/lean. Reproduction randomness is evaluation waste at short budgets. Untested at long budgets |
| Crossover attribution + decoder inheritance at 10 seeds (round 45) | **Crossover is significant on image (t=5.6) and curve (t=5.3), and its TSP benefit GROWS with city count — ns at 100/200, significant at 400 (t=4.0, 8%)**: "neutral on tours" was a small-instance artifact, as Daniel predicted. **Averaging the parents' decoders is the best inheritance rule** (top arm 4/5 problems): within-run survivors are lineage-collapsed near-clones, so the mean cancels mutation noise — round 37's co-adaptation catastrophe applies across runs, not within. Batched vmap decode: bit-identical, 7x, a 30-run TSP benchmark in 59s |
| Shipped defaults after rounds 42-45 | `crossover="average"`, 8 survivors, batched decode. Library reproduces the benchmarks (image 0.00318 vs 0.00336). Known open trade: monotone annealing still beats fixed-8-with-crossover on the image (0.00300 vs 0.00336) and the two are untested together |
| Equal wall-clock vs the pixel GA (round 47) | **Decoder GA wins on TIME too, on the worst-case (free) fitness**: 0.003248 vs 0.0232 fixed-sigma (7.2x) and 0.0110 win-rate-adaptive (3.4x) pixel GA at identical 205.1s. The pixel GA got only 2.3x more evals — per-pixel mutation cost GROWS with output size (20MB of randoms/gen at 96x96) while the decoder GA mutates the same 64+7.5k numbers at any resolution. Overhead <1ms/eval, so a fitness costing ~1ms makes time and evaluation budgets the same test |
| The gradient ceiling (round 48) | **Adam reaches our record in 0.59s (pixels) / 1.61s (our own decoder) and converges to 0 / 0.000011** — the black-box premium is ~350x on differentiable problems; the method's domain is exactly where gradients don't exist. New reference number: ConvRGB's expressiveness ceiling is 0.000011, so evolution uses ~0.3-0.6% of it — the search is the bottleneck, not the architecture, and the architecture CAN paint the leaf |
| Fitness-only Adam analogues, winners-only (round 49) | **Both falsified**: momentum on accepted steps (0.78-0.97x) and per-parent NES+Adam (0.76-0.97x) lose everywhere. Diagnosis: winners-only discards 80% of the data; 4 samples/parent cannot estimate a 7.5k-dim gradient; and (bug) children were born with zeroed optimizer state so the memory reset as fast as selection replaced parents. Selection + win-rate gain already do what momentum-on-winners does |
| Fitness-signed mutation memory, pooled (round 50, Daniel's design) | **Significant: image 1.38x at 10 seeds (t=3.88, best image method at 0.00246); apple 0.00274 -> 0.00178 paired — the first sub-0.002 apple, 1.83x past the round-46 record.** Every child's birth noise, signed by its fitness change — FAILURES INCLUDED — feeds one shared Adam-style accumulator; mutations drift along it. Pooling is load-bearing (per-lineage variant mixed-to-worse); ties/steps starve it (curve, TSP ns both ways). Law: failures are gradient samples; pool them and inherit the memory. Shipped `mutation_memory="shared"` |

## Open problems, in priority order

1. **Rate-based scheduling — now BLOCKING, promoted by round 26.** The
   stall detector ends exploration when exploration stops improving, which
   silently assumes exploration and exploitation are interchangeable uses
   of a marginal evaluation. They are not. Round 26 showed the failure is
   not hypothetical: on a decoder whose exploration actually works (the
   anchor field), the detector spends 4384 of 5000 evaluations exploring,
   leaves CMA the 640-evaluation reserve, and converts a 2-7x win into a
   loss. The conv decoder is only well served because its exploration is
   useless and stalls at once. The fix has been designed since round 18
   and never run: schedule by MEASURED IMPROVEMENT RATE per evaluation for
   each force, and give the next block of budget to whichever is currently
   steeper. Until then `explore_fraction` must be set by hand for any
   decoder that explores productively, and the default is actively wrong
   for the campaign's best genome grammar.
2. **Variance control, remaining half.** Round 18's adaptive switch fixed
   the split (smooth is now a 6/4 tie with the GA instead of a loss) but
   seed variance on signals persists; the lineage-cap fix was null. Next
   suspects: repulsion across exploration lineages (round 13's mechanism),
   and restarting exploration after exploitation stalls (round-trip
   phases). Note round 26's anchor field cut curve error 4.4x below the
   GA with LOWER spread than the conv decoder (0.0024 vs 0.0109) — the
   grammar may dissolve part of this problem rather than solve it.
3. **The architecture zoo — largely answered, one confound open.** Round
   17 tested one convolutional shape; round 21 added GRU/LSTM/transformer
   and showed sequence priors do nothing for permutations; rounds 24-26
   found what does work and it is not a zoo but a single grammar —
   condition the decoder on the instance data the fitness function reads,
   and route the genome in through structurally local anchors. That
   grammar now beats the specialized decoder on tours, images and curves
   with identical constants. Remaining: a photograph target, to separate
   the image win from the fact that blob2d's target and an anchor field
   are both sums of localized sources; and a design that separates
   locality from the anchor decoder's 3x smaller weight count.
4. **Discrete problems — WON at tier 2 (rounds 21-25), now needs range.**
   The anchor-field city transformer beats the tour GA 9/1 (t = 2.47) at
   50 cities / 5k evaluations. The regime is mapped (GA wins at 20
   cities; +2.7% at 50; +33% at 100, 10/10, t = 15.7), and the
   crossover-strengthened baseline was tested and retired (order
   crossover makes the tour GA worse at these budgets). Open: longer
   budgets, honest framing vs problem-specific local search (2-opt),
   and other discrete families (scheduling, assignment) where "anchor"
   must generalize beyond a METRIC space entirely. Round 26 showed the
   grammar spans 1-D and 2-D geometry unchanged, so the open edge is no
   longer dimensionality — it is problems with no distance to be local in
   (satisfiability, graph colouring, abstract permutations), where an
   anchor has no region to govern. The recipe as it stands: condition the
   decoder on the instance data the fitness function reads, and route the
   genome in through structurally local anchors.
5. **The two regimes need one story.** The single-run universal method
   (rounds 16–17) and lifetime corpus accumulation (rounds 6–13) are the
   same system at different timescales: bank every real solve's vetted
   elites, share one decoder across everything (round 11), warm-start
   exploration from it — while keeping harvest errors independent
   (round 15c's law). Designing that loop so it climbs instead of
   echo-chambering is the central open design problem.
6. **Heterogeneous output shapes.** One shared decoder across different
   output sizes/types needs conditioning, masking, or shape-specific heads
   over a shared trunk; the scaling law needs retesting at many-family
   corpus scale.
7. **API redesign.** The first-class flow is now clear:
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
python -m benchmarks.round21_tsp --output ...                # TSP random keys, sequence decoders
python -m benchmarks.round25_anchor_field --output ...       # anchor genome wins TSP
python -m benchmarks.round26_anchor_universal --explore-fraction 0.07 --output ...  # one grammar on images + curves
python -m benchmarks.round19_latent_sweep --output ...       # latent-size sweep (default 64)
python -m benchmarks.plot_family_scaling                    # regenerate the charts
```

Environment notes: neural runs require Apple MPS and verify it; the CMA-ES
baseline is a from-scratch Hansen-tutorial implementation validated against a
16-d sphere (pip in this homebrew Python 3.14 is broken — libexpat mismatch —
so pycma/matplotlib were unavailable; charts are hand-generated SVG).

## Peer review of the multi-fitness phase (2026-07-19)

An independent re-derivation pass over the post-round-50 multi-fitness work
(`benchmark_results/image_conditional_lora_findings.md` and its raw JSON) —
full review with recomputed statistics and a prioritized rerun list in
[`peer-review.md`](peer-review.md). This section records only the verdicts;
nothing in the reviewed results has been altered.

**Arithmetic verified.** Every recomputable statistic reproduces exactly from
the raw JSON: the scenic core-four per-seed changes, both scaling tables, the
dilution fit (exponent 0.269, R² 0.921), and the exposure-matched isolated
trajectory (~0.112 at ~5.6k evaluations). The uncommitted explorer code
faithfully implements rounds 42–50 as documented.

**Three inference issues found, none yet fatal, all cheap to settle:**

1. **Significance shortfalls under the house standard** (|t| ≥ 4.303 at
   n = 3). Only the CIFAR 4→8-target transfer is confirmed (t = −10.35).
   The 4→32 results (scenic t = −3.30, CIFAR t = −3.12), 4→16 (t = −1.60),
   and the 168-target-at-60k win (t = −2.62) are potential checks until two
   more seeds run.
2. **Missing control for the headline scaling claim.** "168 targets viable
   at 2x budget" compares 168@60k against 4@30k — half the total compute.
   No 4-target 60k run exists. The exposure-matched framing survives this
   objection; the 2x-budget framing awaits the control.
3. **The mechanism ordering (fold → retirement → species → succession) rests
   on single-seed 600k deltas of 0.9–5.4%**, while measured seed CV at
   30–60k is 6–12%. The ranking needs paired seeds at 60–120k before any
   arm hardens into a default. Also still missing: the no-fold legacy-bank
   ablation the phase's own writeup names as required (the bank's win is
   currently indistinguishable from "keep a per-target elitist archive",
   i.e. plain MAP-Elites), and any external quality-diversity baseline.

Provenance flags: the 22 `clip_*` result files (600k-eval runs included)
have no writeup anywhere; the multi-fitness JSONs record torch 2.6.0 while
the CLIP runs record 2.12.0 — which environment produced what should be
stated. This narrative record ends at round 50 while the newest phase lives
only in an untracked file.

## Validation experiments (2026-07-20): the peer-review reruns

Forty-four runs executed against the review's open items (CLIP work
excluded). All new results are in `benchmark_results/review_*.json`; no
existing file was modified. Every comparison below is internal to the
current runner version and paired by seed.

**0. Runner-version comparability — a scoping fact found first.** The
23:03 edit to `demo_image_fitness_scaling_rotating.py` (the panel-refresh
optimization) changed results, not just instrumentation: the identical
32-target seed-3 configuration scores 0.0350 mean archive MSE under the
pre-edit runner and 0.0384 under the current one (~10%, max per-target
drift 0.036). The current runner is otherwise perfectly deterministic — an
exact rerun reproduces every target MSE bit-for-bit. Consequence: numbers
from the rotating-scaling tables (pre-edit) and the matched-exposure
section (post-edit) must not be compared across studies, and all
validation runs below were re-based on the current runner.

**1. Mid-scale transfer is now CONFIRMED at five seeds.** Rerunning
4/8/16/32 targets at 30k (seeds 3–7, core-four archive MSE, threshold
t = 2.776 at n = 5): 4→8 −18.7% (t = 4.59), 4→16 −24.9% (t = 6.27),
4→32 −19.1% (t = 4.56) — all significant. The review's complaint that
these were underpowered is resolved in the claim's favor, and the old
16-target anomaly (worse than 8) disappeared on the current runner — it
was noise or a runner artifact, and 16 is now the best point. Beyond the
panel width the dilution regime still holds: 4→168 at 30k is +41% worse
(ns; one seed-7 outlier at 0.109 shows the >32-target tail variance is
real).

**2. The "168 targets at 2x budget" headline is FALSIFIED by its missing
control.** With the 4-target 60k arm finally run (5 seeds): the isolated
run keeps improving with budget (4@30k 0.0448 → 4@60k 0.0409, −8.6%,
t = 4.22, significant), and 168@60k (0.0425) is +3.7% WORSE than the
matched-compute 4@60k control (t = 0.99, tie) and only −5.2% vs the
half-compute 4@30k arm (t = 2.06, not significant). The defensible
statement is "168 targets at 2x total budget roughly MATCHES isolated
4-target quality on the shared targets" — rotation is cheap, not
profitable. The published stronger claim came from the pre-edit runner
with 3 seeds and no matched-compute arm.

**3. The matched-exposure transfer effect is mostly generic population
maturation, not teaching by other objectives.** Two new 168-target arms
isolate what the intervening work is made of, against the real-images arm
(which replicates its published 24.36% exactly). Percent of initial anchor
error removed at exactly 3,000 anchor exposures, 3 seeds:

| intervening 136 objectives | anchors' error removed | vs 32-target baseline |
|:---|---:|---:|
| none (32 targets total) | 0.27% | — |
| pixel-shuffled noise images | 20.45% | +20.2 pp |
| duplicates of the anchors | **29.08%** | +28.8 pp |
| distinct real images (published arm) | 24.36% | +24.1 pp |

Pixel-shuffled noise — objectives with NO image structure — reproduces
~84% of the published effect. Duplicated anchors (extra work on the same
objectives, uncounted as exposures) beat the distinct real images. At the
seed level, distinct-real vs noise is +3.9 pp with t = 0.87 — not
established; noise beat real outright in one seed. The only near-solid
ordering is duplicates > noise (+8.6 pp, t = 3.49). Reading: at a fixed
target-local exposure count, what matters is how much evolution of ANY
kind the population has done — step-size gain adapted (~6 at the
32-target baseline's milestone vs ~200-270 in the 168-target conditions),
weights inflated past the blank-canvas phase, shared decoder developed.
The "objectives as stepping stones/teachers" interpretation is
unsupported pending more seeds; the sample-efficiency effect itself is
real but its cause is maturation. (The earlier 4→8/16/32 result at equal
TOTAL budget is unaffected by this confound and stands as the genuine
transfer evidence.)

**4. Folding and succession do not beat a plain archive (no-fold
ablation + mechanism rerun).** Three arms, identical everything else
(scenic 32 targets, 60k, seeds 3–5, one shared mixed decoder from birth):
retirement folding, archive-only retirement (the previously-missing
no-fold ablation), and lineage succession. Bank mean MSE: no-fold
0.02504 (seed sd 0.00133), succession 0.02615 (0.00340), fold 0.02706
(0.00568) — a three-way statistical tie (all paired |t| ≤ 0.93), with
the PLAIN ARCHIVE nominally best and 3–4x more seed-stable. As the review
suspected, the legacy bank's value at this budget is "keep the best state
per target" — a standard quality-diversity archive; fitness-weighted
death folding and descendant-restricted succession add nothing detectable
at 60k on top of it. The 600k single-seed orderings among these
mechanisms should be treated as unresolved. (An external MAP-Elites
baseline is still worth running; the no-fold arm is the internal
approximation of one.)

Net effect on the phase's claims: the core positive-transfer result
(more objectives at equal total budget help the shared targets, 8–32
range) is now the best-established finding of the phase — significant at
five seeds. The two headline extensions (viable at 168 with 2x budget;
matched-exposure transfer as evidence of cross-objective teaching) are
respectively falsified and reattributed to population maturation. The
mechanism stack above the plain archive is currently unjustified at
short-to-mid budgets.

**3c. Separate or together? Solo runs win, and the transfer story inverts.**
Daniel's core question — to solve two problems, run the GA on each
separately or on both jointly — could not be answered from existing data
because the smallest condition ever run was 4 targets. New arms: true
single-problem runs (the target plus an identical self-copy, so population
mechanics exactly match the joint arms; the harness requires ≥2 targets),
a dedicated two-problem arm (whale + dolphin), and the fixed-total-budget
sweep extended down to n = 1 and 2 (seeds 3–5, 30k evaluations, current
runner; charts at the "Separate or together?" artifact).

- **Two problems, equal evaluations per problem** (which is also equal
  total fitness scorings): separate runs win, 0.0253 vs 0.0348 mean MSE.
  The joint run reshuffles quality rather than adding it: the whale
  target is 2.3x WORSE together (0.0191 → 0.0432) while the dolphin is
  16% better (0.0315 → 0.0264) — and giving the joint run twice the
  decodes (60k) barely helps the whale (0.0409). Cohabitation has a
  per-problem winner and loser, not a shared dividend.
- **The fixed-30k sweep, now from n = 1**: solo 0.0254 (best point on the
  entire curve), n = 2 0.0348, n = 4 0.0440 (the trough), 8–32 partial
  recovery (0.035–0.038, never back to solo), then dilution (n = 256:
  0.0553). **The round's "positive transfer 4→8/16/32" claims are
  arithmetically true but were measured against the interference trough,
  not against independence.** Adding objectives from 4 to 16 partially
  undoes harm that co-habitation itself caused; it never beats separate
  runs on quality.
- The honest decision rule: if fitness scoring is the expensive resource,
  solve K specific problems with K separate runs (optionally each with a
  few cheap palette-matched auxiliary objectives per 3b, if anything).
  The joint run's real economy is decode-sharing — one decode serves K
  scorings — so "together" wins only when the phenotype/simulation is the
  costly step and per-problem interference is a price worth paying.
  Caveat: this is a 2-problem, 3-seed, 30k result on one image family;
  whether solo's advantage survives at 600k budgets, or for problem pairs
  with genuinely shared structure (e.g. day/night variants of one scene),
  is untested.

Runs: `review_solo_t{0-3}_s{3-5}_{15000,30000}.json`,
`review_pair_t01_s{3-5}_{30000,60000}.json`.

Extension to 512 and 1,024 functions (same 30k budget, new 1,024-target
CIFAR extraction whose first 256 files are bit-identical to the existing
set): at 512, every function gets exactly 1,584 evaluations, and the
outcome is a scheduling lottery that repeats across all 3 evolution seeds
because the panel schedule is fixed (`panel_seed`) — the whale removed
72–77% of its error (turns came late, against a mature population;
solo-at-30k quality on 19x fewer evaluations) while the dolphin removed
0.2–2% (turns came early, against an infant population). WHEN a
function's evaluations happen matters more than how many it gets, once
turns are rare. At 1,024, starvation, not failure: 156 generations / 8
per panel = 20 panels x 32 = exactly 640 activation slots, and the
fairness-balanced scheduler grants one stint each to 640 functions, so
384 — including both measured targets, every seed, since the panel seed
is fixed — were never evaluated at all. Fixed-compute curve endgame: 73–79% (1–32 functions),
74→68% (64–256), ~38% lottery (512), 0% (1,024). Also worth flagging:
the fixed panel seed means "3 seeds" share one panel schedule — per-target
conclusions in the rotating regime are replicated against evolution
randomness but NOT against scheduling randomness; a panel-seed sweep
would separate them. Runs: `review_exposure_n{512,1024}_s{3-5}_30000.json`.

Scope note added after re-plotting in Daniel's requested form (percent of
a target's starting error removed at a fixed count of its own
evaluations, one curve over the number of functions in the run): below
the 32-function panel width the curve is flat within seed noise at both
the 3,000- and 30,000-evaluation milestones — "solo wins" above is a
fixed-total-budget, MSE-averaged statement dominated by one target (the
whale), and the two test problems disagree in direction. The two robust
facts are: no reliable per-evaluation benefit OR harm from company below
the panel width, and a large inherited-progress effect past it (0.4% →
66% at 3,000 own evaluations, 1 → 256 functions) paid for by compute
spent on the other functions between turns.

**3d. The warmup is the whole per-evaluation story — and one flag deletes
it.** The matched-exposure advantage of big pools traced to run age
(section 3b/3c work); run age traced to two warmup processes: the
win-rate step controller ramping from gain 1.0 toward the ~700x this
problem wants (15%/generation, and this harness's 192-child generations
make 3,000 evaluations only ~16 generations), plus fresh decoders' flat
near-zero-amplitude output (round 29). Direct test: solo runs (target +
self-copy) with `--start-gain` pre-set. Percent of starting error
removed at evaluation 3,000 / 30,000, whale+dolphin, 3 seeds: gain 1
(default) 0.4% / 73.3%; gain 8: 3.1% / 71.6%; gain 64: 51.2% / 85.5%;
gain 512: **67.4% / 84.5%**. The warm-started solo more than DOUBLES the
best pool condition's 3,000-evaluation result (n=256: 30.1%) with no
pool at all, improves the 30k final by ~12 points, and cures the
default solo's reliability failure (dolphin seed 5: 0.9% -> 81-83%
final). Verdict: the pools' ~75x per-evaluation advantage was a warmup
tax imposed by the cold-start default, purchasable for free with "start
hot, let the controller cool" — the mirror of Rechenberg ramp-up, and
the empirical vindication of the `initial_gain` escape hatch rounds
29-31 designed but never measured. NOT explained by this: the
equal-total-budget transfer at 4-32 functions (5-seed confirmed) — those
conditions share identical generation counts and gain schedules.
Follow-ups: sweep start-gain on the pooled configurations (they
presumably also improve, which may re-shrink every pool-vs-solo gap to
the 4-32 effect), and weight-amplitude warm initialization, the second
warmup process, still untested. Runs:
`review_solo_t{0,1}_g{8,64,512}_s{3-5}_30000.json`.

**3f. The lazy population (Daniel's Finch-inspired redesign): panels,
schedulers, and working sets all deleted — and the verdict splits on
which resource is scarce.** Design, from Daniel's spec: one big living
population, every individual inherits its fitness function (target) from
its base parent, a child is scored against ITS OWN target only, mating
may cross families (genome mixing is free, only scoring costs), survival
is elitist within each target's family (3 slots), per-family win-rate
gain control, hot-started. No panels, no rotation, no fairness counters,
no archive/re-entry — sleeping families cost nothing and their cached
fitness never goes stale. Implemented as
`benchmarks/demo_image_lazy_population.py` (~250 lines vs the 93KB
harness). Results, 30k decodes, 3 seeds, vs the rotating-panel harness
(both hot):

- Structural wins, every size: ZERO unvisited targets even at 1,024
  (rotating starves 384), and per-target progress is monotone — the
  rotating harness's archive can end WORSE than the random founders
  (worst target -31.5% at 1,024; -4.9% at 256), a genuine regression
  pathology the lazy design cannot exhibit.
- Equal DECODE budget: rotating wins the mean at 256+ (32.7% vs 22.1%),
  tie at 32 (49.9 vs 50.8 improvement-biased).
- Equal SCORING budget (the honest accounting when fitness is expensive
  — the method's stated domain): lazy 45.8% vs rotating 32.7% at 256
  (single seed, 960k scorings each). Rotating's mean advantage was its
  32x free scorings per decode, not its scheduling.
- Burst sampling FALSIFIED (my structured fix attempt: sample families,
  breed 24-child clutches): worse than the fully-mixed trickle (18.0 vs
  22.1 at 256) AND it broke coverage (300+ unvisited at 1,024). Daniel's
  fully-mixed per-individual selection beats both structured designs
  (panels, bursts). Plausible mechanism: interleaved singles are bred
  against an ever-fresher cross-family gene pool (sexual fraction 1.0 ->
  ~0.43 as families drift); a clutch breeds against one frozen moment.
- Improvement-biased parent weights vs uniform: within noise everywhere
  (50.8 vs 47.9 at 32; 19.9 vs 22.1 at 256) — the bandit knob has not
  yet earned its complexity.

Standing verdict: when fitness scoring is cheap relative to decoding,
score-everything harnesses exploit the free information and win means;
when scoring is the scarce resource, the lazy inherited-fitness design
wins on quality AND guarantees coverage, monotonicity, and warm state at
any target count. Runs: `review_lazy_n{32,256,1024}_{uniform,improve}[_burst24]_s{3-5}_30000.json`,
`review_lazy_n256_uniform_s3_960000.json`, `review_exposure_n1024_g512_s{3-5}_30000.json`.

**3g. The scaling experiments rerun under the lazy architecture: scaling
smooths out, and two conclusions flip.** All lazy, hot, uniform
selection, 3 seeds (128/256 per-problem points single-seed).

- Fixed 30k-decode budget, 1 -> 1,024 problems: a clean monotone dilution
  curve — 90.3% (solo) / 87.4 / 75.5 / 65.6 / 60.9 / 47.9 / 40.3 / 36.0
  / 30.3 / 27.1 / 22.1 / 13.7 / 8.5% — with none of the old harness's
  features: no sweet spot, no 96-collapse, no 512-lottery, no starvation,
  no below-founder regressions. The exotic scaling phenomena were harness
  artifacts. The lazy curve beats the rotating harness everywhere at or
  below 32 problems (e.g. 75.5 vs 56.6 at 4) and loses above (22.1 vs
  32.7 at 256) exactly where rotating's free 32x multi-scoring pays.
- The clean transfer test (impossible in the old harness): exactly 3,000
  evaluations per problem, budget = 3,000 x n, so the ONLY inter-problem
  channel is cross-family mating. Result: flat-to-slightly-down (79% at
  n=1-2, ~63-67% plateau for n=4-128; 44% at 256, single seed). **No
  transfer bonus. The old architecture's rising matched-exposure curve —
  and the surviving "extra objectives help" effect — are now fully
  attributed: cold-start warmup plus free multi-scoring.**
- The three-way FLIPS: same 4 problems, same 30k total, lazy: pool
  exactly the 4 = 75.5% (best, was worst); four separate quarter-budget
  runs = 70.4%; pool + 28 extras = 54.7% (worst, was best — extras now
  eat decode budget instead of riding free). Pooled-4 beats separate 3/3
  seeds (+5.1pp at 7.5k evals/problem; note at 3k evals/problem the
  comparison leans the other way — co-residence pays after families have
  something worth exchanging).

Standing recommendation, final form: **put the problems you care about —
and only them — in one lazy inherited-fitness population, hot-started.**
Simplest to run, best measured, and safe at any problem count. Runs:
`review_lazy_{n*,solo_t*,pp3k_*}` in benchmark_results/.

**3i. The definitive experiment (Daniel's framing): K problems, equal
compute per problem, separate runs vs one population — COMBINED WINS AT
EVERY BUDGET.** Ten CIFAR problems, identical in both arms, paired by
seed, lazy architecture, hot; per-problem budgets 1,500 to 30,000
evaluations (combined runs get 10x total so per-problem compute matches
exactly; children scored only on their inherited problem, so no scoring
subsidy). Mean % of starting error removed per problem:

| evals/problem | 10 separate runs | one population | delta | paired t |
|---:|---:|---:|---:|---:|
| 1,500 | 42.5 | 58.8 | +16.2 | 17.5 |
| 3,000 | 53.2 | 66.6 | +13.4 | 9.6 |
| 7,500 | 64.6 | 73.2 | +8.6 | 5.9 |
| 15,000 | 71.0 | 77.6 | +6.6 | 13.3 |
| 30,000 | 75.5 | 79.9 | +4.4 | 9.3 |

Ten RELATED problems (jittered variants of one image) roughly double the
gap again (80.0 -> 89.3 across the same budgets). Per-problem view: the
whale at its own 3,000th evaluation scores 79.0 alone, 84.5 among 9
unrelated problems, 85.6 among 9 variants. The advantage is a HEAD
START, not a ceiling change: at 30k evals/problem all arms converge
(~91% on the whale). Mechanism consistent with the content-gradient law:
early progress is made of what all images share (palette, broad shapes)
and crossbreeding trades it; late progress is problem-specific and the
curves close. **This also corrects 3g's "flat transfer" reading — that
table's n=1 baseline was the easiest target (composition confound);
properly paired, co-residence of even unrelated problems is a WIN at
equal per-problem compute, largest when evaluations are scarce.**
Standing recommendation, revised final form: **combine your problems in
one lazy inherited-fitness population — related or not — especially
when evaluations per problem are scarce.** Runs:
`review_ten_{solo_t0-9,combined,related}_s{3-5}_*.json`.

**Shipped to the library after 3i (2026-07-21):**
`latentspace.universal.solve_many(fitness_fns, output_shape, ...)` — the
lazy inherited-fitness population as a first-class API (one family per
fitness function, children scored on their inherited problem only,
fully-mixed uniform parent selection, cross-family genome crossover with
the decoder inherited whole across families and averaged only within a
family per rounds 37/45, per-family win-rate gain). Two new tests; all
nine pass. **The hot-start default was measured on the library's own
explorer and REJECTED**: with its 32-child generations, initial_gain=1.0
is already right (blob 0.00159) and 512 is 20x worse (0.0329) — the
hot-start rule from 3d is harness-specific, because warmup cost in
EVALUATIONS scales with generation size (192-child harnesses pay 6x more
per controller step). Documented on `ExplorerConfig.initial_gain`;
default unchanged. Also clarified for the record (Daniel's question):
the shared decoder in the rotating and lazy benchmark harnesses is
genuinely frozen — the exact LoRA folds of the species era changed
weights but provably no function (change of coordinates), the LoRA
directions were never learned, and the only arms that ever trained the
backbone were the consolidation/factorization ones (including the best
600k result). All learning in the current architecture lives in each
individual's personal numbers; consolidating it into shared weights is
the designed-but-unbuilt next step.

**3j. Consolidation into the universal decoder: first results — transfer
across runs works, and Daniel's breeders corpus beats the champions
corpus.** Design shipped in `latentspace.universal.multi`: periodically
train one shared core decoder B by supervised reconstruction on the
population's own discoveries (zero fitness evaluations), reseed each
family's worst member with B (champion untouched, reseed honestly
re-scored), doubling event schedule (no interval constant), training
stopped by the existing stall rule, museum of all past pairs kept
forever. Two corpus strategies: "champions" (each family's best) and
"breeders" (Daniel's: living parents with above-median child win rate —
genomes selected for producing good descendants, not for being good).
Race on 10 CIFAR problems, 30k, MLP decoder, 3 seeds:

- Within-run: off 4.0% mean error removed; breeders 5.4%; champions
  5.4% — both beat off 3/3 seeds (~35% relative), tie each other.
- **Across runs (the project's point): a core trained on 10 problems,
  carried to 5 NEVER-SEEN problems at 15k with no further consolidation:
  cold 4.9%, champions-core 8.7% (~1.8x), breeders-core 14.6% (~3x).
  Every warm seed beats every cold seed; breeders beats champions 3/3
  seeds.** Continual cross-run transfer learning is live in the library:
  `result.decoder` from one `solve_many` run, passed as `init_decoder`
  to the next, at zero evaluation cost.

Why breeders may win transfer: the corpus is larger and more diverse
(above-median breeders across all families vs one champion each), and
breeders occupy EVOLVABLE regions — the core learns where productive
search happens rather than what ten finished answers look like. The
teachers-not-the-student and error-independence laws are both satisfied
structurally (independent families, deviations paid for in real
evaluations). Limits: MLP decoder on RGB (weak; conv retest pending),
3 seeds, one problem family, within-run effect small, warm gains
seed-variable (9.8-21.3%). Tests: 11/11 pass.

**Compounding test (same day): transfer is persistent but does NOT
compound.** Five successive batches of 10 never-seen problems, breeders
core inherited and re-consolidated batch to batch, cold control on
identical problems and seeds. Chain vs cold, mean of 3 seeds, by
problems consolidated before the batch: 0: 5.6 vs 5.7 (sanity — no core
yet); 10: 16.5 vs 6.5 (2.5x); 20: 13.5 vs 6.9; 30: 10.0 vs 5.6; 40:
9.6 vs 4.2. All 12 post-first-batch stage-seed pairs beat cold; nothing
is forgotten (museum working); but the advantage plateaus at ~2x after
the FIRST batch — among unrelated images the shareable knowledge is
low-level structure and 10 problems' worth saturates it. The
head-start-not-ceiling law at lifetime scale. Compounding, if
reachable, needs related problem families or a stronger decoder (these
runs used the MLP fallback for RGB — the conv-registry follow-up is the
first thing to try). Charts on the artifact; raw values in the session
scratchpad (consol.json, compound.json).

**3l. The double-controlled view (Daniel's request): what one evaluation
buys, controlling for BOTH compute-per-problem and population size.**
Plot mean % removed per problem against evaluations-each-problem-received
(log x), one curve per population size (1, 4, 16, 64, 256, 1024), from
the fixed-30k lazy trajectories. Result: the curves for populations >= 16
COINCIDE (e.g. at ~24 evals/problem: 7.2% at n=256 vs 7.1% at n=1024; at
~190: 27.8% at n=16 vs 26.9% at n=64) and sit ~2x ABOVE the solo and
4-problem curves at matched evaluations (solo whale — an EASY target —
13.5% at 194 evals vs 27.8% for the 16-pool's harder mix). Initial read from that overlay ("saturates at ~16") was CORRECTED by the
fully controlled version (Daniel's spec: population on the x-axis, same
measured problems everywhere): dedicated runs at budget = 1,500 x n so
the SAME 10 problems get exactly 1,500 own evaluations inside
populations of 1/10/32/64/128/256/512/1024. **The curve is an inverted
U: 42.5% alone -> ~59% at 10-32 -> peak 62.8% at 64 -> declining to
44.3% at 1,024, never below solo.** Population is a per-evaluation
amplifier worth up to ~1.5x with an optimum near 32-64 neighbors;
extreme crowding hands the amplification back (a uniform-random mate
among 1,024 families is almost always too genetically distant to
recombine with usefully) but never goes negative. Both confounds
(target composition, per-problem compute) controlled simultaneously.
Runs: `review_pp1500_n{32-1024}_s{3-5}.json` + the ten-problem
trajectories. This decomposes the total-output curve honestly: up to
1.5x is amplification with a population-size optimum; the rest is
arithmetic.

Ablation of the peak (Daniel's "how are we doing selection?" probe,
which caught a wrong explanation): mate choice is ONE uniform-random
draw over the whole population, gated by genome RMS distance <= 30;
measured crossover frequency is 3-7% at the peak populations (32-64,
families drift apart fast) and HIGHER (21-35%) on the declining side —
the opposite of the "too distant to breed" story first offered. The
decomposition, same 10 problems at 1,500 evals each: solo giant
generations 42.5% / solo 4-child generations 55.1% / population-64
crossover-off 53.6% / population-64 crossover-on 62.8%. **Two-thirds of
the population amplification is GENERATION CADENCE** (the win-rate
step controller adapts once per generation; many small generations =
finer adaptation — free to any solo run via `children`), **one-third is
genuine crossbreeding** (+9pp from crossovers happening only 3-7% of
the time through a lottery mate rule — substantial headroom in mate
selection). Population-without-breeding equals cadence-matched solo
exactly. The 1,024 decline tracks frequent stranger-crossover, a
mate-rule artifact rather than a law. Runs:
`review_pp1500_n64_nocross_s*`, `review_solo_smallgen_t*_s*`.

Mate-SEARCH test (Daniel's proposal, honest implementation failure):
"search the gene space for a relevant mate, slightly randomized" was
implemented as always-cross-with-one-of-k-nearest (k = family size).
**Falsified hard: population 64 62.8 -> 49.2%, population 1,024 44.3 ->
15.8%** (`review_pp1500_n{64,1024}_nearest_s*`). Diagnosis: the
implementation bundled two changes — partner choice AND crossover
frequency (3-7% -> 100%). The lottery's implicit rule is "crossover with
probability = the parent's local in-radius density, partner uniform
among in-radius" — its RARITY is load-bearing (most children must be
pure step-size-calibrated mutations) and nearest-neighbors are mostly
siblings (inbreeding replaced the productive rare stranger-cross).
Defaults reverted to the lottery in both the benchmark runner
(`--mate lottery`) and `solve_many`. The unbundled partner-only test
(`gated-nearest`: crossover fires on exactly the lottery's rare event,
partner upgraded to a random one of the k nearest in-radius genomes)
then ALSO lost: population 64 62.8 -> 55.7% (2/3 seeds tie, one drops),
population 1,024 44.3 -> 35.0%. **Three-arm closure — the mate-selection
law: crossover wants partners that are COMPATIBLE BUT DIFFERENT, and
rarely.** The radius gate is the relevance filter; UNIFORMITY within it
supplies the difference (a compatible stranger with genuinely other
genes); rarity protects the calibrated-mutation engine. Searching harder
for the nearest compatible genome overshoots relevance into kinship —
near-clone crossover spends the event on a no-op. The lottery is
quietly optimal on all three axes; mate-selection headroom I predicted
does not exist in this form. Runs:
`review_pp1500_n{64,1024}_{nearest,gatednear}_s*`. Also shipped:
`--live` matplotlib view (targets|evolved grid + per-problem progress)
in the lazy runner.

Mate-selection round 2 (Daniel: "we intuited a strawman; how do advanced
GAs handle this? too-similar is useless, too-distant is meaningless").
Two families of experiments, all firing crossover on exactly the
lottery's rare event so only the recombination CHOICE changed, all
excluding a parent's own family so no sibling-inbreeding confound
remained. **(a) Partner-choice among in-radius strangers** (population 64,
3 seeds, mean percent error removed; lottery baseline 57.7): gated-far
(most-different compatible stranger) **57.9 — the only arm to tie**;
gated-fittest 48.2, gated-nearest 48.5, gated-target (family whose TARGET
IMAGE is most similar) 46.7, gated-kin (nearest stranger genome) 41.7.
Every "similar-partner" rule lost, and each lost by one seed detonating
(~27-32%) as crossover frequency ran away (climbing to 20-37% of children
instead of decaying to ~2%): pulling in similar partners makes the pool
look more self-compatible, triggering more crossover, homogenizing
further — a feedback loop that wrecks the win-rate mutation controller.
Only the most-DIFFERENT rule was stable and matched random — this is
negative assortative mating (Fernandes & Rosa), the one within-band
technique the GA literature endorses, and positive assortative mating's
known premature-convergence failure is exactly the collapse we saw.
**(b) A different OPERATOR — Differential Evolution** (child = parent +
F·(a−b), the difference between two random members used as a self-scaling
step). Classic DE with unrestricted donors CRATERED (population 64, 30k
budget, seed 3: 27-32% vs lottery 43.5) — with per-family objectives a
random difference vector is mostly other families' target-specific genes,
i.e. noise; this is Daniel's "too distant stops being meaningful" made
literal. Restricting donors to the in-radius band (`de-gated`, F=0.5
rate=0.5) recovered to a TIE: 43.1 vs 43.5 at population 64, and 42.9 vs
43.0 mean at population 256 (per-seed crossing over, no scale trend).
**Closure across THREE unrelated recombination paradigms — block-swap
crossover, partner search, and DE difference-vectors — every one ties the
random lottery the moment it respects the radius band, and degrades the
moment it doesn't. The compatibility band is the entire lever; within it,
recombination choice is a wash. There is no smart-mate headroom in this
system.** Daniel was right about the SHAPE (a band, both extremes bad),
wrong that a smart choice lives inside it. New arms in the runner:
`--mate gated-{kin,far,fittest,target}`, `--mate de[-gated]` with
`--de-f`/`--de-rate`. Runs: `review_pp1500_n64_{gatedkin,gatedfar,
gatedfittest,gatedtarget}_s*`, `review_pp1500_n256_degat_s*`.

Mate-selection round 3 — ADAPTIVE band (Daniel: "make it adaptive; start
random, track which distance BAND yields positive children, steer toward
it — per fitness function or globally"). Built a bandit over 5
equal-occupancy distance bands (quantile edges each generation, so band =
relative distance, no radius cap); crossover still fires only on the
lottery's rare event, so frequency is unchanged and only the partner's
BAND adapts. Per-band credit = EMA of whether that band's child beat its
parent; scope global (one bandit) or family (one per fitness fn). Starts
neutral (0.5) = uniform = random. Result (population 64, 3 seeds, vs
lottery 57.7): adaptive-global **57.4**, adaptive-family **57.8** — both
tie seed-for-seed. **This is the FOURTH mechanism to tie random, and the
band trace explains why ALL of them tie: the credit signal is only
nonzero for the first few hundred evaluations, then collapses to zero for
the rest of the run.** Global bandit's per-band win-rates, seed 3: at 256
evals [0.48, 0.29, 0.25, 0.34, 0.26] — it correctly found the NEAR band
wins most while the population is still random; by 9,664 evals [~0]; from
19k on, flat [0,0,0,0,0]. Mechanism: once the win-rate mutation
controller warms up, a parent sits near a local optimum that CALIBRATED
MUTATION found, and a crossover with any partner almost never beats it —
so crossover-child-beats-parent goes flat at zero. Every mate-selection
scheme (band, partner, DE, adaptive) has been optimizing a decision with
NO GRADIENT after warmup; they tie the lottery by necessity, not luck.
The one real dividend of adaptivity: ROBUSTNESS — it soft-preferred the
same near band that made hard gated-kin collapse a seed to 27%, but kept
exploring and never detonated (clean across all 3 seeds). Adaptivity buys
safety, not headroom. New runner arms: `--mate adaptive --adaptive-scope
{global,family}` (`--adaptive-bands/-temp/-ema`); band_score logged in
the trace. Runs: `review_pp1500_n64_adapt{global,family}_s*`.

CORRECTION to round 3's "zero gradient after warmup" mechanism (Daniel:
"your conclusion is crossover never works after a certain point?" — it
does not hold; caught by a causal test). Added `--crossover-until K`
(crossover off once K evals spent, pure mutation after) and swept K at
population 64. Crossover is worth +9.5pp on average (lottery 57.7 vs
no-crossover 48.2). WHEN it earns that, by seed (crossover-until-6000 vs
never vs always): seed 3 60.2 / 52.6 / 60.5 (banked EARLY — 6% of the run
suffices); seed 4 57.9 / 58.1 / 57.4 (crossover irrelevant this run);
**seed 5 37.7 / 34.0 / 55.2 — early-only sits at the NEVER level while
always reaches 55.2, so this seed's +18pp came from crossover firing
AFTER eval 6,000.** So "crossover is inert late" is FALSIFIED: in the
collapse-prone seed its value is LATE. Corrected mechanism: **crossover's
real job is RESCUE, not a head start.** Its average value is dominated by
preventing catastrophic collapse in unlucky lineages (a rare, large, late
event); in healthy lineages it is an early head-start or nothing. The
bandit's win-rate signal decayed to zero because it averages over
mostly-healthy families and washes out the rare late rescues. This
explains the mate-selection null BETTER than the zero-gradient story:
escaping a bad basin needs an injection of GENUINELY DIFFERENT genes, and
any sufficiently-different compatible stranger serves — exactly the
uniform lottery. Steering toward SIMILAR mates removes the escape route
(why gated-kin/target collapsed seeds); steering toward the single MOST
different (gated-far) is no better than random-different (any different
stranger works). FINAL CLOSURE of the mate-selection thread: the
compatibility band is the lever; within it, crossover is a sporadic
escape-from-collapse operator whose only requirement of a partner is
DIFFERENCE, which random-among-compatible already guarantees — so no
mate-selection rule beats the lottery, but the reason is rescue dynamics,
not a dead late-game gradient. Runs: `--crossover-until` sweep, seed 3
(3k/6k/12k/24k/48k), seeds 4-5 at 6k (not persisted to JSON — single-line
console outputs, protocol `--budget 96000 --children 192 --mate lottery`).

Mate-selection round 4 — DOMINANCE / diploidy-lite (Daniel: "could we
benefit from dominant and recessive genes?"). Motivation was strong: the
rescue finding says the population needs a reservoir of different genes,
and diploid-with-dominance is the classic GA mechanism for holding hidden
diversity under selection (Goldberg & Smith 1987). Implemented a shadow
(recessive) latent per individual: carried and inherited but never
expressed, crossover deposits the mate's foreign alleles into it, and a
per-locus dominance flip surfaces a reserved allele into expression where
selection tests it (`--mate dominant[-plus]`, `--flip-rate`; coefficients
stay haploid; shadow init gated so other arms keep their RNG stream and
reproduce baselines — lottery s3 still exactly 60.5). **Falsified, two
ways.** (a) `dominant` (gentle-integration: foreign alleles go recessive,
expressed crossover OFF): population 64, 96k — 43.0 / 51.6 / 27.2 (mean
40.6) vs lottery 57.7, WORSE than even no-crossover (48.2); flip rate
made it monotonically worse (s5: 0.0->33.6, 0.02->27.2, 0.05->23.7,
0.10->21.2) and flip=0 collapses to the no-crossover level — gentle
recessive integration is strictly worse than real crossover. (b)
`dominant-plus` (reservoir ADDED on top of full lottery crossover, the
confound-free test): 52.0 / 50.5 / 49.2 (mean 50.6) vs lottery 57.7 —
still ~7pp worse, and again monotone in flip rate (s5: 0.02->49.2,
0.005->54.1, ->0 recovers lottery). **The optimal amount of dominance is
zero.** Mechanism: the shadow stores FOREIGN alleles (mates optimized for
OTHER targets), so surfacing them injects target-mismatched genes into a
converging lineage — net-harmful, the same failure as classic-DE's
random difference vector. Dominance pays in biology and nonstationary GAs
because recessive alleles are once-useful or context-useful; here, with
static per-target fitness, the reservoir is a junkyard of wrong-target
genes and the flip is just disruptive mutation that selection must clean
up at the cost of evaluations. Untested flavor (self-memory reservoir:
store a lineage's own displaced alleles) — theory predicts it also fails
on a static landscape (displaced = worse), and the base rate for these
elaborations in this repo is ~zero, so not pursued. Runs: console outputs
under protocol `--budget 96000 --children 192 --mate dominant[-plus]`.

**Solidified into the library (the point of the whole arc).** Rounds 1-4
produced no new shippable mechanism — every mate-selection/DE/dominance
elaboration tied or lost to the simple rare lottery, so the deliverable
was confirming the lean design. Auditing what the library actually
embodied vs. what was validated surfaced ONE real, severe divergence: the
shipped `solve_many` did UNCONDITIONAL crossover — every child crossed
with a random mate, 100% of the time, no rarity gate — while the single
most-replicated finding of the arc is that crossover must be RARE. This
was a shipped bug, not a missing feature. The runner's rarity came from a
distance radius interacting with hot-start gains (large gains spread
genomes so a fixed radius fires ~5%); the library uses cold gains
(compact genomes), so a distance radius would gate nothing — and since
the radius's RELEVANCE role was already shown moot (only its rarity
mattered), the correct scale-independent port is a plain crossover
PROBABILITY. Added `crossover_rate` (default 0.05) to `solve_many` and
measured it: on 16 CIFAR images (3 seeds, 48k budget) always-crossing
(rate 1.0) removes **3.0%** of founder error, rate 0.05 removes **80.7%**,
rate 0.0 removes 81.1% — a ~27x defect fixed. Same shape on the curve
problems the tests use (rate 1.0 = 19.9%, rate 0.05 = 91.3%). Rare ≈ off
in these small runs (the runner's +9.5pp crossover dividend needs the
larger/longer collapse-prone regime), so 0.05 is the robust validated
default: it captures crossover's rescue value at scale without the
100%-crossover catastrophe. Shipped: `crossover_rate=0.05` default,
docstring updated, and a regression test (`test_rare_crossover_beats_
always_crossing`) asserting rare beats always by a wide margin so the
default cannot silently revert. All 14 tests pass. Nothing else from the
arc changes the library — `solve_many`, consolidation, `conv_image`, the
lottery mate rule, and cold-start gains were already correct.

**Crossover on TSP: it was in the WRONG SPACE all along (Daniel: "crossover
is key to a GA — if it doesn't help we're doing it wrong").** Ported the
rare-crossover knob into the single-problem explorer (`ExplorerConfig.
crossover_rate`) and swept it on TSP; rare didn't help (TSP-50 always 8.52
vs rate-0.05 8.56 vs OFF 7.98 — off marginally best and ties the tour GA;
null at 200/400 too). The decisive test isolated the operator in a proper
tour GA (identical segment-reversal mutation in every arm, only crossover
varies, 5 seeds, budget 20k): a POSITION-preserving crossover (Order
Crossover) HURTS monotonically (TSP-100 mutation-only 11.74 -> OX 14.25;
TSP-300 56.97 -> 61.68), but an EDGE-preserving crossover (Edge
Recombination, ERX) HELPS and helps MORE at higher rate and larger size —
TSP-100: 11.74 -> ERX-0.9 10.81 (-8%); TSP-300: 56.97 -> ERX-0.9 47.56
(-17%), the advantage compounding with problem size. **Diagnosis:** TSP
fitness is a sum over EDGES (city adjacencies), so useful recombination
must preserve edges. ERX does and wins big; OX preserves position not
edges and loses; and our decoder's latent one-point crossover, after the
argsort maps priorities -> tour, preserves NO edges at all — strictly
worse than OX. That is why crossover is inert-to-harmful in the decoder GA
on permutations: it operates in latent/priority space, which is
structurally disconnected from the tour edges the objective rewards. This
is the crossover-specific face of the standing "permutation space is
mismatched to latent decoders" result (round 21-22: index-space priors are
negative knowledge; averaging tours is meaningless). IMPLICATION for the
deliverable: on TSP the neural-decoder GA cannot capture crossover's
(large, compounding) value without a PHENOTYPE-space edge crossover, which
is domain-specific and breaks the universal-decoder abstraction. So the
honest options are (a) accept TSP as the decoder's weak domain — it still
beats CMA there (8.52 vs 8.98 at 50 cities) — or (b) add an edge-crossover
hook for permutation outputs. Not a bug in the crossover CODE; a
representation mismatch. `ExplorerConfig.crossover_rate` (default 1.0,
preserves behavior) shipped for future use; no default changed on the
strength of TSP alone.

**BPE / linkage crossover over the latent genome (Daniel's idea: tokenize
the population, discover 'genes', cross only over gene boundaries so a
sequence survives; "mutation can break a gene, crossover cannot"). Built
two ways, both FALSIFIED — and the failure localizes a deep architectural
fact.** The idea is right in theory — it is linkage learning / competent-GA
territory (LTGA, DSMGA, BOA), the principled universal answer to "the best
crossover is problem-structure-dependent" (edge-preserving for TSP, per-gene
uniform for images). Thesis kept intact: decoder and universal latent
genotype untouched; crossover operates on the latent and self-discovers
structure. (a) Contiguous BPE (quantize survivors' latents, merge high-MI
adjacent dims into genes, gene-level uniform crossover that never splits a
gene): images rate-0.9 = 2.9% error removed vs 81% for off, WORSE than plain
uniform (7.5%). (b) Order-agnostic linkage (fixing the obvious flaw — the
latent has no meaningful ordering, so genes must be ARBITRARY coadapted
dim-sets, found by an |correlation| graph + connected components): rate-0.9
= 4.6%, rate-0.3 = 72.5%, bigger genes 4.9% — never beats off, still
catastrophic at high rate. Fixing the gene-grouping did NOT rescue
crossover, so gene structure was never the problem. **Diagnosis — the latent
is not a shared coordinate frame.** Each individual carries its OWN
co-evolved decoder, so latent-dim meanings are decoder-relative; two
individuals solve the same target with different (latent, decoder) pairs,
and recombining their latent dims yields a vector neither decoder reads
correctly — no gene-grouping fixes a frame mismatch. Proof from data already
collected: the SAME uniform operator is the BEST on images in PIXEL space
(shared identity frame, +39.9%) and CATASTROPHIC in LATENT space
(per-individual decoders, +7.5%). Crossover recombines building blocks only
when individuals share a coordinate frame; the per-individual decoder
destroys it. **Constructive consequence for the thesis:** building-block
crossover is not dead, it is BLOCKED until a SHARED decoder exists — exactly
what the consolidation / universal-decoder line builds. Once individuals
share one decoder, latents become comparable and linkage crossover could
finally pay — a concrete reason to prioritize the shared universal decoder,
and a direct bridge from the crossover idea to the project's core thesis.
Until then off/rare crossover stays the Pareto default (unchanged). All
experimental modes reverted from the library (validated-negative); the
one-point rare-crossover default and `crossover_rate` stand. Runs: image
`solve_many` sweeps, crossover_mode in {uniform, bpe, linkage}.

**Diagnosis CONFIRMED (Daniel's correction): the genes were never the
problem — the PER-INDIVIDUAL decoder was.** Terminology, grounded in the
code (explorer.py:169,182 literally name z "genes"): GENES = z = the network
INPUT vector (the universal genotype); DECODER = theta = the network weights
(a separate evolving object, NOT a gene). Crossover on genes and on decoder
weights are already different operators (genes get one-point/BPE; theta is
averaged within-family / inherited whole across families). The BPE/linkage
failures above were genespace crossover failing because a crossed gene is
read by the parent's per-individual decoder, which was tuned to the parent's
genes. Decisive test: consolidate ONE shared decoder, freeze it, evolve
GENES ONLY against it with high-rate gene crossover. Result (4 targets, 2
seeds, gene crossover @rate 0.9): off 88.1%, uniform 88.2%, BPE 88.3% — the
catastrophe (per-individual uniform@0.9 = 7.5%) is entirely GONE. So gene
crossover is catastrophic ONLY under per-individual decoders; a shared
decoder makes it safe at any rate. This validates the whole chain: the genes
are universal and crossover-able; the per-individual decoder is what broke
the shared coordinate frame; and the UNIVERSAL SHARED DECODER (the
consolidation thesis) is precisely what unblocks crossover. HONEST LIMIT:
with the shared decoder crossover is now SAFE but not yet a WIN (88.2 approx
88.1 off) — the necessary precondition is met, but the building-block PAYOFF
needs a setting with real population diversity and epistatic structure to
exploit (single-target gene-inversion against a frozen decoder is nearly
separable, so mutation alone suffices). Architectural implication: "evolve
genes against a shared decoder" turns crossover from a landmine into a usable
operator — a concrete reason to center the shared decoder rather than treat
it as a periodic consolidation step. No library default changed; documented
as the validated next direction. Run: scratchpad shared_decoder_genecx.py.

**`solve_many` REBUILT onto the one-decoder architecture (Daniel: the library
had drifted to per-individual `pop_theta`; there must only ever be ONE shared
decoder, latents as LoRA — see the one-decoder-invariant).** New module
`latentspace/universal/conditional.py`: `ConditionalLoRADecoder` wraps ANY
resolved backbone (MLP / conv / etc.) by adding shared low-rank directions to
every Linear/Conv layer — `layer(x) = base(x) + scale*up(coeff*down(x))` — the
architecture-agnostic generalization of the image-only `ConditionalLoRAConvRGB`.
`solve_many` now holds one shared decoder; each individual's genome is
`[z | coefficients]` (network input + shared-LoRA gates), crossover and mutation
act on those genes, and `pop_theta` is gone entirely. Consolidation FOLDS
discoveries into the one backbone (trains it so coefficient-zero reproduces
champions) and reseeds worst members onto the improved backbone. `result.decoder`
is the shared decoder's params; `init_decoder` warm-starts it; new params
`coefficient_dim` (default=latent) and `crossover_mode` (one_point/uniform/bpe).
Torch seeded from the run RNG so the one decoder is reproducible; 14 tests pass.
The obsolete `test_rare_crossover_beats_always_crossing` (encoded the
per-individual artifact) was replaced with a crossover-modes-run test. HONEST
CAVEAT under test: with a random frozen backbone (consolidate="off") gene-only
evolution has limited reachability, so raw scores can be lower than the old
per-individual full-decoder evolution — the shared decoder must be TRAINED
(consolidation central, or a good init) to reach targets, which is the thesis.
Crossover behaviour on the real architecture (16 images, one shared decoder):
off 38.3 / one_point-0.05 41.8 / uniform-0.9 3.5 / bpe-0.9 3.9 (consolidate off;
similar with champions). High-rate crossover is STILL catastrophic here — so
the shared decoder did NOT by itself unblock it, and last turn's "shared decoder
fixes crossover" was CONFOUNDED (that test used a single target). Decisive
isolation (8 families, same shared decoder, budget 24k): population all solving
the SAME target — off 79.5, uniform-0.9 90.5 (crossover HELPS +11); population
solving DIFFERENT targets — off 45.1, uniform-0.9 2.6 (catastrophe). **The
crossover catastrophe is CROSS-OBJECTIVE MIXING, not the decoder.** When the
population shares an objective, high-rate crossover delivers the building-block
win Daniel predicted (+11pp); mixing a solution-for-A with a solution-for-B is
destructive because you cannot recombine solutions to different problems. So on
the multi-target `solve_many`, crossover must stay rare (cross-family mates
solve different targets); the building-block regime is a diverse population on a
SHARED objective. Corrects the round's earlier per-individual-decoder
frame-mismatch story: the real variable was same-vs-different objective, not
per-individual-vs-shared decoder (the shared decoder is still needed for a
common frame, just not sufficient). Also honest: multi-target raw performance
regressed vs the old per-individual full-decoder evolution (45% vs ~81% on 16
images at 48k) — one shared decoder is less per-individual-expressive, so the
backbone must be well-trained (consolidation central / good init) to reach many
targets. Default kept crossover_rate=0.05 (safe for the multi-target case).
Runs: scratchpad crossover_real_arch.py, same-vs-different verification.

**PROVEN decoder shipped into the library (Daniel: "ship what we've proven to
work really well; use the conditional conv decoder, generalize later").** My
first rebuild used a generic per-layer LoRA wrapper that UNDERperformed — 45% on
16 images vs the runner's proven `ConditionalLoRAConvRGB` at 70.5% (same task).
Root cause: I reinvented the decoder instead of reusing the tuned one (the
proven design has MIXED conditioning — half the coefficients are extra evolvable
decoder INPUTS for reachability, half gate LoRA — plus the conv deep-image-prior
backbone). Fix: moved the proven design into `latentspace/universal/conditional.py`
as `ConditionalLoRAConv`, generalized from the hardcoded (3,96,96) to any square
RGB `output_shape` (auto conv geometry), with the same interface as the generic
decoder. `build_conditional_decoder` picks conv-conditional for image-shaped
outputs and the generic LoRA wrapper for vector outputs (tests). solve_many now
gets **61.5% (cold, 32-child) to 65.5% (hot, 192-child)** on 16 native-32x32
images — recovering nearly all of the generic version's shortfall; the residual
vs 70.5% is the runner's 96x96 upscaling (smoother target, deeper decoder), not
the port. Crossover finding CONFIRMED on the proven decoder (8 families, 24k):
SAME target off 84.0 / uniform-0.9 93.5 (crossover HELPS +9.5); DIFFERENT
targets off 62.3 / uniform-0.9 1.2 (catastrophe) — same law, higher absolutes.
Q answered (is the decoder image-specific?): the LoRA-conditioning PATTERN is
general (works for any backbone); the conv backbone just works best on images
(spatial structure). Generalizing to MLP/transformer backbones with the same
mixed-conditioning pattern is the deferred next step. 14 tests pass; the library
no longer depends on benchmarks for its decoder. Runs: scratchpad image sweeps.

**Crossover rule FIXED — the multi-objective "breakage" was my regression, not a
law (Daniel: "we've had crossover help with multi-target before; this means our
crossover rule sucks").** He was right. The runner that produced the historical
+9pp multi-target crossover benefit gates crossover by GENOME COMPATIBILITY
(`sexual = z_dist <= mating_radius`): a randomly drawn mate only actually
crosses if its genome is close, so a mate solving a different problem is
genome-distant and never crosses. My rebuilt `solve_many` had replaced that with
a blind coin flip (`rng.random() < crossover_rate`), which crosses ANY mate
including cross-objective ones — rare AND indiscriminate, strictly worse. That
blind rule is what produced the "cross-objective crossover is catastrophic"
result (uniform-0.9 = 1.2%). Fix: `crossover_rate` is now SELECTIVITY — crossover
fires only when the drawn mate is among the closest `crossover_rate` fraction of
gene-distances that generation (a scale-adaptive port of the mating radius, which
does not transfer directly because cold-gain genomes are compact). Measured, 8
different targets, 24k: off 58.9 / 0.05 **61.1** / 0.3 60.0 / 0.6 32.5 / 0.9 26.9.
**Crossover HELPS on multi-objective again (+2.2 at the default)**, and the
high-rate catastrophe is defused (1.2 -> 26.9; the residual decline is correct —
a high rate means low selectivity, admitting incompatible mates). Supersedes the
earlier "crossover only works on a shared objective" claim: it works across
objectives too, provided mates are compatibility-gated. Default kept 0.05.

**3k. Consolidation retested at conv strength (library `conv_image`
architecture, ported from the round-28 champion, now the "auto" choice
for RGB shapes): within-run value CONFIRMED and LARGER; cross-run
transfer INVERTED.** Same problems, seeds, and budgets as 3j. (a)
Within-run, 10 problems at 30k: no consolidation 2.1% (cold conv
mutation-dilutes its 65k weights at ~3k evals/problem), breeders 10.0%,
champions 11.3% — a ~5x effect, champions over breeders 3/3 seeds
(reversing the MLP tie). (b) Cross-run transfer to 5 never-seen
problems: cold conv 23.0% vs warm-from-breeders-core 13.3%, warm-from-
champions 15.8% — **the MLP transfer result (4.9 -> 14.6) does not
survive: at conv strength the inherited core HURTS.** (c) Compounding
chain: seed noise, no consistent direction. Diagnosis, two parts: the
core's transferable content was mostly generic image-ness, which
convolution provides free through architecture (round 17's lever), so
what remains is training-image specifics that don't help strangers; and
warm-started founders all sit one mutation step from ONE core —
correlated founding errors, violating the error-independence law at
birth, where cold random founders are individually worse but diverse.
Designed fixes, untested: diversity-preserving warm start (half fresh
founders, half core — the round-20 re-entry recipe), and consolidation
across RELATED problem families where shareable content exceeds what
architecture encodes. Standing summary: **consolidation is a confirmed
~5x within-run mechanism on the strong decoder; as a cross-run transfer
vehicle it currently only substitutes for architecture the decoder
lacks.** Tests 12/12; artifact section updated to the two-sided result.

**3h. Correction after Daniel's challenge ("transfer doesn't help is a
hot take — think about confounds"): the null was over-claimed, and the
confound is structural.** Verified in code: BOTH recent harnesses (the
rotating runner and the lazy runner) contain no training machinery at
all — the shared decoder backbone is a frozen random network. So every
experiment since the species harness tested co-residence transfer with
the transfer-learning mechanism unplugged: a learned shared
representation was mechanically impossible, and the only open channel
was genome exchange through a frozen basis. The flat 3g transfer curve
is therefore a statement about UNRELATED problems in a substrate-free
system, nothing more. Direct test of the genetic channel with structure
present (3,000 evals/problem, lazy, 3 seeds): 32 jittered variants of
one image co-residing = **81.8%**, vs 79.0% alone, vs 62.6% with 32
unrelated problems, vs 59.9% with 10 same-class-but-different whale
photos. **Genetic transfer is real and tracks shared SOLUTION structure
steeply** — variants (+19pp over unrelated company, edges out solitude)
transfer; same semantic class with different pixels transfers nothing
(MSE relatedness is pixel-level, not categorical). The earlier
learned-substrate results (round 7's scaling law in K, round 11's shared
decoder) remain the proof that representation-level transfer works when
the substrate can learn. Designed next experiment: add zero-evaluation
supervised distillation of all families' elites into the lazy
architecture's backbone (round-9 style) and retest unrelated
co-residence — the content-gradient result predicts shared image
statistics would then transfer. Runs:
`review_lazy_pp3k_related32_s{3-5}_96000.json`,
`review_lazy_pp3k_whales11_s{3-5}_33000.json`.

**3e. The cold start fixed at EVERY pool size (Daniel's control): the
ordering stands, the illusion inverts, and one pooling effect survives.**
Hot-started (`--start-gain 512`) reruns of the full sweep, 2-512
problems, 3 seeds. (a) Fixed-budget finals, mean per problem: solo 84.5%
> 2: 68.3 > 4: 56.6 ≈ 8: 56.9 > ... > 512: 27.6 — monotone decline,
no crossover, and the hot start CURES the 96-problem collapse (25.5 ->
41.8, no failed seeds). (b) The matched-exposure curve INVERTS hot: at
each problem's 3,000th evaluation, solo 67.4% falls to ~28-33% for every
pool of 16+, instead of the cold 0.3 -> 30 rise. Company is a
per-evaluation cost at every size once the controller starts right. (c)
The 4->32 shared-target transfer SURVIVES hot (core four: 56.6 -> 67.0%,
t = +3.19 at 3 seeds, all seeds positive) — not a cold-start artifact.
(d) The decisive three-way at identical 30k total compute for the same 4
problems, all hot: pool of just the 4 = 56.6%; four separate
quarter-budget runs = 63.5%; pool of the 4 plus 28 extra objectives =
67.0% (2/3 seeds over separate, +3.5pp — suggestive). Final ruling:
never pool only the problems you care about (strictly worst); separate
hot runs are the robust default; a crowd of extra objectives may buy a
modest further edge and is the one live research question left from the
phase. Runs: `review_exposure_n{2-512}_g512_s{3-5}_30000.json`,
`review_solo_t{0-3}_g512_s{3-5}_7500.json`.

**3b. The content gradient (Daniel's objection, tested): "maturation" was
too strong — the effect is transfer of LOW-LEVEL learnable statistics,
and unlearnable objectives actively poison the run.** Daniel objected
that pixel-shuffled images still contain transferable content — they
preserve each donor's exact color histogram — so the noise arm could not
distinguish "any evolution matures the population" from "low-level
statistics teach." Three further 136-objective pools complete the
gradient (same design, 3 seeds; percent of initial anchor error removed
at exactly 3,000 anchor exposures, with the shared step-control gain at
that moment):

| intervening 136 objectives | teaches | error removed | gain at milestone |
|:---|:---|---:|---:|
| none (32-target baseline) | — | 0.3% | 6 |
| iid uniform-noise images | nothing learnable | **0.2%** | **0.3** |
| constant mid-gray images | trivial bias only | 12.5% (unstable: 2–23% by seed) | 77 |
| anchor duplicates | the anchors themselves | 29.1% | 119 |
| pixel-shuffled real images | exact palette, no structure | 20.5% | 208 |
| **constant mean-color images** | **palette only** | **26.3%** | 258 |
| distinct real images | everything | 24.4% | 268 |

Three conclusions, replacing 3's attribution (continued in 3c below). (1) **"Any intervening
evolution matures the population" is falsified by the uniform arm**: iid
noise targets are effectively unfittable, children stop beating parents,
the shared win-rate controller drives the gain to its floor (0.3), and
the anchors end BELOW the no-intervener baseline despite 8x the elapsed
evolution — a pool containing hopeless objectives strangles step control
for everyone. Objective quality is not additive; one poison class can
collapse the shared machinery. (2) **Daniel was right that the shuffled
arm's content was doing work — and the content that matters is just the
palette.** Constant mean-color targets, which contain no spatial
structure whatsoever, reproduce the full effect (26.3%, nominally ABOVE
the real images' 24.4%). Real images add nothing detectable beyond their
mean colors at this operating point. (3) The gain column tracks the
effect almost perfectly (0.3 → 77 → 208 → 258 → 268): learnable
intervening objectives keep the step controller alive and growing, and
what they teach the shared decoder needs only to be statistically
adjacent to the anchors. The refined law: **auxiliary objectives help
exactly insofar as they are learnable (keeping shared adaptation healthy)
and share low-level statistics with the targets of interest; their fine
structure is irrelevant at these budgets, and unlearnable ones are
poison.** The anchor-duplicates arm stays on top (29.1%) with only
modest gain (119) — direct on-target work still beats any teaching.
Results: `review_exposure_{gray,meancolor,uniform}168_s{3,4,5}_30000.json`.

**Mate-selection round 5 (2026-07-21, Daniel: "I have a hard time
believing we can't improve more upon mate selection") — the rounds-1-4
closure FALLS, on the axis none of those rounds varied: crossover
FREQUENCY, not partner choice.** The "no smart-mate headroom" verdict
was drawn at seeds 3-5. Extending the same lottery configuration to 15
paired seeds (population 64, `--budget 96000 --children 192`) shows
those three were an all-healthy sample: on 4 of 15 seeds (6, 13, 14,
15) the lottery COLLAPSES to 20-38% error removed vs 55-65% healthy,
and the signature is always the same — mid-run crossover frequency
45-90% instead of the healthy 1-4%. Mechanism: the lottery's rarity is
a fixed genome-distance radius (30), and when a seed's genome spread
stays compact the "rare" event fires constantly — the runaway
crossover -> homogenize -> more in-radius mates -> wreck the win-rate
step controller feedback loop from the gated-kin post-mortem, running
on the lottery itself (collapse-seed final gain ~14 vs ~40 healthy).
Rarity-by-fixed-radius is the same class of falsified constant as the
fixed mutation sigma of rounds 29-31, and the peer review's item 5
named the fix in advance. Three interventions, 15 paired seeds each:

| arm | mean | healthy delta | collapse delta | severe (<40%) seeds |
|:---|---:|---:|---:|:---|
| lottery (the closed default) | 50.4 | — | — | 4/15 |
| frequency-capped lottery (5% hard cap) | 53.1 | −0.4 | +11.1 | 2/15 |
| stall-triggered rescue (crossover only for families ≥96 children without improvement, p=0.5) | 53.3 | −1.1 | +14.0 | 2/15 |
| stall-triggered rescue, p=1.0 | 54.1 | −0.6 | +15.3 | 3/15, all mild |

Every intervention improves every collapse seed the lottery has and
ties on healthy seeds (deltas within noise). Seed-grain t = 1.39-1.71
— under the 2.145 house bar at n=15 because rare collapses dominate the
variance; per-target paired t ≈ 7, and the honest statistic is the
severe-failure rate itself. The three fixes are statistically tied with
one another (need-conditioned p=1.0 minus capped: +1.0, t = 0.52), so
**frequency CONTROL is the mechanism; need-conditioning is a refinement**
— it owns the single best rescue (seed 6: 23.5 -> 57.2, near-healthy,
where the hard cap only reaches 31.5) but does not separate on means.
Constant sensitivity (single-run probes on seeds 5-6): rescue
probability matters monotonically (1.0 > 0.5 > 0.25); stall-after 96
beat both 48 and 192 on seed 6 and was flat on seed 5.

The partner-choice half of the closure SURVIVES at better power:
most-different-partner extended to 5 seeds stays a tie (t = 0.18), and
the new breeder-quality rule — partner = the in-radius stranger with
the best smoothed record of producing winning children, the
breeders-corpus insight recast as a mate rule — detonated seed 5
(31.9 vs 55.2), the sixth partner-quality rule to fail by single-seed
collapse. Revised closure: **within the compatibility band, WHO you
cross with still doesn't matter; WHETHER the band's event rate is
bounded matters enormously, and the rounds-1-4 lottery only looked
optimal because three seeds never showed its failure mode.** Library
implication, checked: `solve_many`'s crossover gate is already a
QUANTILE of the current generation's gene distances, which bounds
frequency by construction — ported for scale-independence, now
validated as collapse-proof; no library change needed. The runner keeps
`--mate lottery` as its default for cross-study comparability; new arms
`--mate capped` (`--cap-rate`), `--mate stall`/`stall-only`
(`--stall-after`, `--stall-prob`, `--stall-tol`) and
`--mate gated-breeder` are shipped, and the baseline reproduction was
verified bit-identical after the runner edit. Runs:
`review_pp1500_n64_{capped,stallonly,stallonly_p1.0,breeder}_s{3-17}`,
`review_pp1500_n64_s{6-17}`, `review_pp1500_n64_gatedfar_s{6,7}`,
sensitivity `review_pp1500_n64_stallonly_{a48,a192,p0.25,p1.0}_s{5,6}`.

**The genes/latents operator matrix (2026-07-21, Daniel's directive:
"test every assumption in how to treat genes and latents"). Terminology:
GENES = z, the decoder's input; LATENTS = the LoRA coefficients gating
the shared decoder's directions. The shipped `solve_many` treated the
concatenation [z | coefficients] as one chromosome in all three
operators; the matrix varies one operator at a time.** Benchmarks: 16
native-32x32 CIFAR images through the conv conditional decoder at 24k,
and 8 smooth curves (64-d, generic LoRA decoder) at 12k (later 4k — 12k
is saturated at ~81% and cannot discriminate); metric = mean % of
founder error removed, 5 paired seeds, house bar t >= 2.776.

- **Mutation scale is the axis that matters, and its direction FLIPS
  with architecture.** Images: latents at 0.25x gene sigma is
  catastrophic (-16.2, t = -6.17); 2x is +4.4 and SIGNIFICANT
  (t = 4.41); the response is a broad 2-8x plateau (4x +3.1, 8x +3.6,
  16x +2.7 and fading). Mechanism: coefficients start at exact zero and
  must GROW, so they want harder kicks than std-1 genes. Curves at 4k:
  the sign reverses — 0.25x leans BETTER (+3.4, t = 1.96), 4x is
  nothing (+1.3, t = 0.97). No fixed ratio is universal — the fixed
  mutation-sigma law of rounds 29-31, now measured across the
  genes/latents boundary.
- **The self-tuning ratio ships as the default.** `latent_sigma_scale=
  "auto"`: each child mutates exactly ONE channel (coin flip), and a
  shared ratio multiplies the latents channel, moved by comparative win
  rate (latents-children beating genes-children -> ratio up) with the
  standard 1.15 step, clipped [1/32, 32]. Results: images +4.3 vs
  boundary-blind (t = 2.12) and statistically identical to the best
  hand constant (vs 2x: -0.1, t = -0.13); curves +4.5 (t = 1.81, all 5
  seeds positive) — where every fixed boost failed. Pooled across both
  benchmarks: +4.4 at n = 10 paired seeds, t = 2.96, significant. The
  alternation-only control (`"alt"`, ratio locked at 1) shows the gain
  is the adaptation, not the alternation: images +1.0, curves -5.6.
- **Crossover cuts and compatibility distance: boundary-blind survives.**
  Cut placement (single cut across the concatenation vs independent cuts
  per half vs one half inherited whole) is a tie everywhere, with
  genes-only crossover leaning harmful on images (-3.8, t(problem) =
  -2.40) — the latents carry crossable content. The z-only compatibility
  gate (the species-era harness rule) is +1.7 ns alone on images but
  INTERFERES with the auto ratio when combined (+1.3 combined vs +4.3
  auto-alone), so `compat_distance="all"` stays the default. All knobs
  remain available (`crossover_cuts`, `compat_distance`).

Runs: `gvl_{image,curve,curve4k}_{arm}_s{3-7}.json`, arms base /
compat_{genes,latents} / cuts_{separate,genes,latents} /
sigma_{quarter,2x,4x,8x,16x,alt,auto} / combo_harness /
combo_sigma4_compat / default_new. Tests extended to 16, all passing.

**CMA-ES removed from the library (Daniel's ruling, 2026-07-21: "we do
not want to use CMA-ES at all"), and the exploit phase demoted to an
option — the pure decoder GA is the shipped default.** The replacement
exploit is an actual GA over the distilled latent space (truncation
selection, uniform crossover at high rate — lawful there because every
individual solves the SAME objective through the SAME frozen decoder,
the measured building-block regime — and win-rate mutation), in
`latentspace/universal/exploit.py` with cma_minimize's exact interface.
`solve()` physically cannot run CMA-ES; cma.py remains only so
benchmarks can field it as an opponent arm. The ablation (blob-32 image,
curve-64, curve-256; 5k evaluations; 10 paired seeds; the CMA stack
rebuilt outside the library as the opponent):

| problem | pure explore (mean/median MSE) | GA exploit | CMA exploit |
|:---|---:|---:|---:|
| blob32 | **0.00124** / 0.00115 | 0.00529 / 0.00133 | 0.00507 / 0.00128 |
| curve64 | 0.000104 / 0.000101 | 0.000099 / 0.000097 | 0.000088 / 0.000086 |
| curve256 | 0.00872 / **0.00084** | 0.00599 / 0.00319 | 0.00530 / 0.00332 |

The decisive fact is the failure structure, not the means: on blob32 the
GA-exploit and CMA-exploit detonate on the SAME three seeds (8, 9, 11;
per-seed MSE correlation 0.967) while pure exploration passes all ten —
so the failures belong to the DISTILL phase (a lineage-collapsed archive
makes a bad latent space, and any optimizer confined to it is trapped;
round 32's "right-sized steps starve the compressor," now visible at
seed grain). The GA exploit ties CMA statistically on blob32 and
curve256 and loses only microscopically on curve64 (+0.000012 absolute,
t = 5.99) — dropping CMA costs nothing that matters, and the tiny
curve64 exploit benefit (GA-exploit beats pure 8/10, t = 2.99, at 5e-6
absolute) does not justify a phase that detonates 3/10 image seeds.
Shipped defaults: `solve(exploit=None)` resolves to "off" (pure decoder
evolution, fresh restart on stall) for single-phase runs and "ga" for
phases="cycle" (which needs the distilled hand-off); explicit
cycle+off raises. README rewritten to match ("all three phases are
load-bearing" retired — it described the pre-step-control stack). Runs:
`exploit_{blob32,curve64,curve256}_{ga,off,cma}_s{3-12}.json`.

**The API unified (Daniel: "solve and solve_many should be generalized —
one fitness function is just the one-problem case").** `solve(fitness,
output_shape, ...)` is now THE entry point: a single callable (or a bare
one-element list) routes to the per-individual engine that holds every
single-fitness record; a list routes to the shared-decoder population;
multi-only keywords (children, consolidate, crossover_rate, ...) route
even a one-element list to the multi engine so the knobs are honoured.
Both result types now read identically for the one-problem case
(`SolveResult.problems`, `MultiResult.best_phenotype`). Also shipped: a
zero-evaluation `progress` callback on the multi engine (champions
decoded at most every budget/50 evaluations) and
`benchmarks/demo_solve_live.py`, the live target|evolved window running
the LIBRARY instead of the benchmark harness. First parity check (16
CIFAR images, 48k, seed 3): the library with today's defaults — native
32x32, cold start, 32-child generations — reaches 69.7% mean error
removed, matching the harness demo's 69.8% run this morning at 96x96
with hot-start and 192-child generations, and well above the 61.5%
recorded for the library on this task before the operator matrix. At
48k the auto-ratio's edge over boundary-blind has closed on this seed
(69.7 vs 69.8 — the 24k advantage is a head start, consistent with the
head-start-not-ceiling law); its value is at scarce budgets. Tests: 17,
all passing.

**The family FALSIFIED as a mini-population (Daniel: "Finch had no
families concept — are we sure it's useful?"). What survives is one
champion lineage per problem.** The inherited-fitness core (child gets
its parent's problem, scored on it only) is the heavily validated part;
the 3-slot within-family elitism rode along as an implementation choice
and had never been isolated. Sweep of family_size {1, 2, 3, 5, 8} on
both benchmarks: quality degrades MONOTONICALLY with depth — image 58.2%
(1 slot) / 54.4 (3) / 45.1 (8); curve 66.6 / 60.0 / 50.9. Confirmation,
champion-only vs the shipped 3 slots: image 24k +4.0 (t = 4.83, n = 5),
curve +6.0 (t = 13.7, n = 5), image 48k +2.9 (t = 5.58, n = 3, clears
the stricter bar) — 13/13 paired seeds positive. Mechanism: parents are
drawn uniformly from the living population, so every extra slot means
more children bred from sub-champion parents, and the diversity those
slots hold never pays it back at any budget tested. Shipped:
family_size=1 default (the parameter remains for research; consolidation
reseeding needs >= 2 and no-ops at 1). The open structural question is
now sharper: not "how deep should families be" (answer: not at all) but
whether the guaranteed one-slot-per-problem floor plus a floating pool
biased toward improving problems beats the flat one-champion-each.
Runs: scratchpad family_sweep.py + confirmation, printed values in the
session log. Follow-up (Daniel: "why do we even have it as a concept?"):
the vocabulary is retired from the library. At one slot the concept
dissolves into the irreducible per-problem state the multi-problem
setting forces anyway — a problem tag on each individual, a champion per
problem, a mutation gain per problem — none of which is a "family" in
any GA sense. The parameter is renamed `slots_per_problem` (research
knob), internals renamed to problem_of/problem_gain, and the
mini-population framing removed from every docstring. The family concept
was implementation flourish from the original lazy-population build, not
part of the spec and not in Finch; measured, it lost.

**The 1,024-problem cold-start tax (found because Daniel called a 1.6M
live run "the worst we've had" — he was right in the way that matters).**
The run: 1,024 CIFAR problems, 1.6M evaluations (~1,560 per problem),
library defaults, live window — 14.2% mean error removed, vs 53-58% for
16-problem runs at the SAME per-problem budget. Probes at 1,024 x 160k,
seed 3 (the big run's own 160k trace point, 1.4%, is the deterministic
baseline): crossover off 2.8% (the always-on quantile gate is a real
but minor drag at extreme problem counts — it forces the closest 5% to
cross even when every mate is a distant stranger); **initial_gain 64:
12.5% — a 9x recovery**, nearly matching the whole cold 1.6M run on a
tenth of its budget; gain 512: 14.8%, BEATING the cold 1.6M run
outright. Mechanism: each problem's win-rate controller ramps on its own
child cadence — one child per ~32 generations at 1,024 problems — so
warmup cost scales with PROBLEM COUNT, not generation size; the
"cold start is already right" ruling was measured on the single-problem
explorer and does not transfer. Hot start also wins at 16 problems
(gain 64: +2.7, t = 5.0 at n = 3; gain 8/512 similar) and on curves
(gain 64: +8.0, all seeds), but 512 overshoots on curves (64.7 vs 74.6,
one seed to 50.4). Shipped: `initial_gain=64` as solve_many's default —
best-or-near-best on all three benchmarks, self-correcting downward,
while the single-problem explorer keeps its measured 1.0. Open, in
order: (a) the start is the FOURTH scale/architecture-dependent constant
of the day — the principled fix is one GLOBAL controller with dense
updates early handing off to per-problem gains (the pooling insight
again), designed but unbuilt; (b) even hot, 12-15% at 1,024 sits far
below per-problem-budget parity with small runs — the residual gap is
structural (candidates: the crossover floor, champion diversity at
extreme counts) and unexplained; (c) the quantile crossover gate should
be allowed to go silent when even the closest mates are distant (the
radius rule's one virtue). Tests 17/17.

# The redesign (2026-07-21, Daniel's specification): one GA, from scratch

Daniel's verdict on the accumulated system: "this has drifted
significantly from the original intent in countless ways... merely
working better than past worst versions of the same bad ideas." His
spec, now implemented verbatim in `latentspace/universal/ga.py` as the
ONLY public API (`solve(fitness_fns, output_shape, epochs)`):

- TWO random founders (genes AND latents random), both on the first
  fitness function; no champions, no per-problem slots, no fixed
  generation size — all three concepts deleted.
- Genes and latents permanently distinct: separate crossover (one-point
  on genes; latents inherited WHOLE from one parent — "half of one is
  not half as useful"), separate mutation operators with independent
  global win-rate dials (each child mutates exactly one space so each
  dial's feedback is clean).
- Capped population culled by standing = progress relative to the best
  score ever seen on the individual's own function (his ruling:
  comparable across easy and hard problems); extinction allowed.
- Speciation = modular re-assignment of individuals across functions
  over time (default: uniform-random at 2%/epoch, re-scored honestly);
  this is how coverage of the function list emerges.
- A fold step trains the one shared decoder on selected individuals'
  discoveries (default: best of each function present); doubling
  schedule.
- Every operator (selection, both crossovers, both mutations,
  speciation, fold selection) is a replaceable function — the design's
  purpose is fast iteration on each.
- Best-ever-per-function archive kept for reporting only (his ruling);
  epochs = loop iterations (his ruling), evaluations still counted and
  reported for honest comparison.

Legacy engines (the per-individual explorer stack and the
champion-per-problem population) are demoted to benchmark opponents,
importable from their modules but out of the API; they hold the records
the new design must beat. Tests: 7 new + 16 legacy, all passing.

First benchmark (16 CIFAR images, ~25k evaluations, 3 paired seeds, vs
the champion-engine at identical evaluations): the design WORKS
structurally — speciation reached all 16 functions from a 2-individual
start on every seed, accounting exact — and LOSES on quality by ~2.5x
(mean MSE 0.055-0.062 vs 0.020-0.035). Two defects found and fixed by
the first iteration loop: (1) the success rule compared children against
the BETTER of their two parents, which death-spiraled whichever
mutation space was temporarily behind (its dial floored on every seed);
reference changed to the adopted-function parent, dials now healthy,
latents running hotter than genes as the operator matrix predicted.
(2) folding changed the decoder under the population without re-scoring
it, so culling ran on stale scores; the population is now re-scored
after every fold. The remaining gap is the starting bar for operator
iteration; the untried levers, in rough order of expected effect:
selection pressure (uniform pairs is the weakest possible), informed
speciation (random re-assignment wastes migrations), fold cadence and
selection, and population allocation dynamics under progress-relative
culling.

**Round two of the redesign (same day): Daniel's fold correction and
fitness shares, plus the species-selection consequence.** Two spec
corrections from Daniel: (1) fold means applying a proven individual's
latents DIRECTLY into the decoder weights — exact arithmetic, no
training. Built as `absorb()` on both decoder classes (the low-rank
bending composes into base weights in closed form; the mixed decoder's
extra-input half folds into the bias, including the low-rank cross
term). Verified: donor's phenotype preserved to float32 precision after
its latents are zeroed. The round-37 no-op law was honored: versions
that compensate everyone are provably identity, so the semantics are
donor-preserved, everyone else shifts, population honestly re-scored.
(2) Fitness is SHARES: total environment fitness is always 1, each
living function's population owns an equal slice, members split their
slice by within-function rank; selection and culling both run on
shares, so overtaking is impossible by construction and my
founder-relative "standing" normalization was deleted. First measured
consequence: shares balance the population so well that fully-mixed
share-proportional pairing made ~94% of pairs cross-species — every
such child scored twice (evaluation bill 25k -> 45k for the same
epochs) and cross-species chimeras almost never beat their parents, so
both mutation dials floored and quality got WORSE (0.057-0.066). The
missing piece was the other half of the species concept: **species
breed within themselves.** Default selection is now assortative —
parent one share-proportional, parent two from the same function
(share-proportional) with a 5% outcross — the rare-stranger-crossover
law reappearing at the species level. Result (16 CIFAR, ~26k evals, 3
paired seeds): mean MSE 0.048/0.037/0.040 vs the pre-shares design's
0.055-0.062 — and the dials now climb to 100-350 ON THEIR OWN from a
cold start, which dissolves the hot-start problem exactly as the
global-controller design predicted (dense feedback every epoch is what
per-problem dials never had). Gap to the legacy champion engine: ~1.6x
(vs 2.5x at the first pass; seed 4 is nearly a tie, 0.0366 vs 0.0341).
Next levers: informed speciation, outcross-rate and fold-cadence
response curves, and mutation-operator refinement. Tests 23/23.

**Round three (Daniel's founding question + the outcross ablation;
defaults locked with his approval).** (1) Founding: seeding TWO
founders per function from the start, with background migration OFF,
beat both the grown-coverage default (0.0314 vs 0.0415 mean MSE, better
on all 3 seeds) and seeded-with-migration (0.0453 — the migration churn
ate the entire seeding advantage: each move pays a re-scoring to dump a
mismatched individual into a working species). New defaults:
founding="per_function" (population cap auto-raised to hold 2 per
function), speciation=None. Random migration's two legitimate jobs —
initial coverage and recolonizing extinctions — are obsolete under
seeding and near-impossible under shares respectively; the designed
successor is EVENT-DRIVEN recolonization, needed when the function
count dwarfs the population. (2) Cross-species crossover (outcrossing,
distinct from migration): 5% beats full isolation on all 3 seeds
(0.0317 vs 0.0408) with the rescue signature — isolation stalls 2/3
seeds at ~0.0475 while 5% lands all three tightly (0.027-0.034); 20%
swings back down (0.0350, plus every cross-species child double-scores:
27.0k evals vs 23.7k). The rare-stranger-crossover law reproduces
exactly in the new architecture; 5% stays the default. Cumulative state
of the redesign after one day: mean MSE 0.0317 vs the legacy champion
engine's 0.0259 on the same protocol — a 1.2x gap, from 2.5x at first
light. Tests 23/23.

**Round four — the fold optimizer (Daniel: "we are essentially applying
gradients... Adam might help right?"): PARITY with the legacy engine.**
The decoder previously had no optimizer — absorb was one full unscaled
step of a donor's bending. Adam over fold events (state in latent
space; the absorbed step is Adam's processed output; the donor stays
exactly preserved because subtraction of ANY absorbed step from its
latents is an identity on its phenotype), with the cadence confound
controlled: raw absorption at a fixed every-32 cadence is much WORSE
than the doubling schedule (0.0529 vs 0.0317 — frequent whole-bending
absorption is 43 disruptions per run), but the SAME cadence through
Adam is the best configuration the design has produced: 0.0256 mean
(0.0235/0.0295/0.0239), better than the previous default on all 3
seeds, lr 0.5 > lr 1.0 (0.0283). Mechanism: momentum bakes in the
cross-species CONSENSUS direction; the second moment damps dimensions
where donors disagree. Milestone: 0.0256 vs the legacy champion
engine's 0.0259 — a statistical tie with the record-holding engine on
its own benchmark, from 2.5x behind at the redesign's first pass, with
one seed won, one tied, one lost. Round 50's law (Adam-style
accumulation over evolution-measured directions) now operates at BOTH
levels of the system. Shipped defaults: fold_optimizer="adam",
fold_every=32, fold_lr=0.5. Untested: cadence and lr response curves
beyond these points, and the fold-selection function. Tests 23/23.

**Round five — the apple, live, one function: the redesign LOSES the
single-fitness flagship, 0.012218 at 153k evaluations vs the legacy
bars 0.004566 @ 120k and 0.00178 @ 150k (2.7x / 6.9x behind), with a
clear plateau from ~130k.** Mechanism, structural not tunable: with one
function the design's whole expressiveness is 128 evolvable numbers
steering FROZEN RANDOM low-rank directions; the fold moves what latents
can express into the backbone, but the reachable set is bounded by the
random directions' span — round 36's low-rank-costs-quality
measurement, amplified because exploration never touches the backbone.
The legacy engine evolves millions of private weights and keeps the
crown. The invariant-compatible lever for the gap: evolve the SHARED
LoRA directions themselves (one set, globally, via the same Adam-fold
pattern) so the vocabulary of bendings improves instead of staying
random — built same day and FALSIFIED as built: a (1+1)-ES over the
whole 3,580-weight vocabulary with population-mean acceptance rejected
essentially all ~560 trials (apple 0.01268 @ 171k vs frozen 0.01222 @
153k — a 12% trial tax and nothing else). Diagnosis: a random
perturbation of something all 32 co-adapted individuals depend on almost
never survives a population-wide vote — the chimera law again, one
level deeper. Designed refinements, untested: one-direction-at-a-time
proposals, share-weighted acceptance, trials timed right after folds.
Default stays frozen; the arm ships as directions="evolve".

**Round six — Daniel's counter-design: randomize the vocabulary PER
INDIVIDUAL (directions="individual"), no global directions at all. The
frozen ceiling BREAKS; the record stays distant.** Each individual
carries an integer seed; its low-rank basis is a frozen random function
of that seed (nothing per-individual is evolved weight — the invariant
holds); (basis, latents) inherit from one parent as a unit;
`fresh_basis_rate` of children are born with a new basis and zero
latents; the fold absorbs donors' bendings from ANY basis into the one
backbone. Apple, matched evaluations: 0.011152 vs frozen-shared
0.012218 at 153k (~9% better), still descending at 306k (0.010670)
where frozen was flat from 130k. Fresh-rate response is an inverted U
(5%: 0.0164, 10%: 0.0122@76k, 20%: 0.0142) — the rare-injection law
governing vocabularies now. Fresh-every-child (Daniel's literal
strongest form) is decisively falsified: 0.0978, 8x worse — zeroing
latents each birth destroys inheritance, and 4.6x slower (every child
its own decode group). Honest slope: doubling budget 153k->306k gained
4% — this escapes the random-span ceiling but does not close the 6x gap
to the 0.00178 record; the rank-32-vs-free-weights bottleneck remains
dominant. Ships as directions="individual" (not default pending the
16-image multi-function check). Runs: scratchpad apple probes, seed 3. Scoreboard tonight: parity on many
problems at a tenth the machinery; the old champion holds single-fitness
by a wide, explained margin. Run: benchmarks/demo_apple_live.py, seed 3.

**Scale probe (same day): 1,024 problems / 160k evaluations (156 per
problem) — the cold-start default fails at extreme problem counts, and
the hot-start rule turns out to be per-family, not per-harness.** Library
defaults, native 32x32, seed 3: 3.7% mean error removed (best 52.2%,
hardest 0.0%). Rerun with `initial_gain=64`: 11.3%; `initial_gain=512`:
12.4%, zero-progress problems 123 -> 68 of 1,024. Diagnosis: the
win-rate gain is per FAMILY, so each family's controller must warm up on
its own ~156 children — at 16 problems each family got 3,000 children
and cold start was measured correct (512 was 20x WORSE on the small
blob run); at 1,024 problems cold start is 3.4x worse. The 3d/3i rule
("warmup cost scales with generation size") needs amending: it scales
with EVALUATIONS PER FAMILY. Two data points is not a formula — the
default stays 1.0 with this documented; the principled fix is a
scale-aware or self-cooling initial gain, untested. Also noted: the
auto latents/genes ratio settled at 0.51 cold and exactly 1.00 in both
hot runs at this scale — the comparative signal is too thin at ~16
single-channel children per generation across 1,024 families to move
it; its measured wins are at 8-32 problems. For context the old
rotating-panel harness at 1,024 starved 384 targets outright; the lazy
library visits every family (guarantee held: worst family 0.0% but
visited). Runs: live `demo_solve_live --count 1024 --budget 160000`,
hot arms in-session (numbers above).


**CORRECTION to round six (the paired seeds arrived): the "frozen
ceiling breaks" claim is RETRACTED.** Completing the pairs the original
claim lacked: apple at ~153k evaluations, frozen vs per-individual
vocabularies at 10% fresh — seed 3: 0.012218 vs 0.011152 (individual
better); seed 4: 0.011262 vs 0.015383 (frozen better); seed 5:
0.016338 vs 0.013291 (individual better). Mean 0.013273 vs 0.013275 —
a dead tie, with seed noise (frozen alone spans 0.0113-0.0163, +-30%)
dwarfing every delta. On the 16-image multi-function benchmark the
per-individual vocabularies LOSE on all 3 seeds (0.0308 vs 0.0256,
~20%). Verdict: directions stay FROZEN everywhere;
directions="individual" remains a research arm. What survives of round
six: fresh-vocabulary-every-child is catastrophic (0.098, a real 8x
effect), and the 306k long run's 0.010670 is unpaired and unverified.
The meta-lesson, for the third time this campaign: the original claim
was one seed with an eval-matched probe ON THE SAME SEED — apple seed
variance makes every single-seed apple claim worthless; three paired
seeds minimum on that benchmark, forever. The single-function gap to
the legacy records (0.00178) is fully open again, bottleneck unknown.

**Round seven — per-individual SPARSE WEIGHT PATCHES (Daniel: "each
individual directly mutates the decoder's weights... just each one
selects a section and a modifier"). The first mechanism to move the
single-function number substantially.** Built as
`directions="sparse"` (latentspace/universal/sparse.py): an individual's
integer seed picks K coordinates of the decoder's flat weight vector and
its latents are the values added there — one shared backbone, no
per-individual weight matrices (K values + one int per individual), and
folding becomes an exact scatter-add that can reach ANY weight instead
of being trapped in the frozen low-rank span forever. Decode is one
vmapped call over per-individual materialized weights, so speed matches
the low-rank path.

Apple, ~153k evaluations, K=2048 at 10% fresh sites vs the frozen
low-rank default, PAIRED: seed 3 0.005921 vs 0.012218; seed 4 0.007374
vs 0.011262; seed 5 0.009625 vs 0.016338 — better on all three, mean
0.00764 vs 0.01327 (1.7x), margins well outside the +-30% seed band
that invalidated round six. Patch size is monotone on seed 3: K=128
0.0457 (too few), 512 0.0120, 2048 0.0059, 8192 0.0058. Fresh-sites
rate helps (0.3: 0.00719 vs 0.0: 0.00814) — consistent with round 37's
"WHERE carries no inheritable information" while still leaving
sparse-free strictly better than subspace-locked.

Decomposition (seed 3, the capacity control that matters): low-rank
latents=64 0.01222 -> low-rank latents=2048 0.00857 -> sparse K=2048
0.00592. So ~60% of the gain is CAPACITY (the shipped default starved
individuals at 64 evolvable numbers) and ~30%+ is genuinely FREE
PLACEMENT at matched capacity. The apple decoder has 47,155 weights, so
K=2048 touches 4.3% of them.

Status: the capacity control is SINGLE-SEED (the round-six lesson says
that is not enough on this benchmark) and K=8192 is single-seed; both
need pairs before defaults change. Nothing shipped as default yet.
Context for the remaining gap: legacy bars 0.004566 @ 120k and 0.001780
@ 150k — sparse K=2048/8192 is now within ~1.3x of the round-31 bar,
from 2.7x behind at round five. Runs: scratchpad sparse_apple.py,
sparse_capacity.py (stopped early for thermal reasons; partial).

**Round seven, part two — sparse patches on MULTI-FUNCTION: they LOSE,
and the disagreement localizes what the frozen vocabulary was doing.**
16 CIFAR images, seed 3, 1400 epochs (~25k evals), live: sparse K=2048
finished 57.0% mean error removed vs the frozen low-rank default's
65.6% on the identical configuration — despite LEADING early (10.4% vs
4.6% at epoch 56) and then being overtaken. Single seed, so the size is
provisional, but the direction matches the per-individual-vocabulary
result (also apple-neutral, multi-function-negative), which makes the
pattern harder to dismiss as noise.

Reading: free placement wins when ONE objective owns the whole decoder
(apple: 0.0063 vs 0.0122, 3 paired seeds) and costs when SIXTEEN species
share one backbone. Proposed mechanism, testable: each species folds its
freely-placed patch into coordinates ITS OWN seed chose, so absorbed
edits land in disjoint places and overwrite each other's accumulated
work; the frozen low-rank subspace, precisely by being a SHARED and
limited vocabulary, forces every species' folds into the same
coordinates where the Adam accumulator can average them into consensus.
The subspace was serving as a COORDINATION mechanism, not merely a
capacity constraint — which is a new fact about the architecture and
reframes rounds six and seven together.

Designed next arm (small change, separates the two variables for the
first time): sparse patches with SHARED sites — draw the coordinates
from one run-level seed instead of per-individual, keeping free
placement and full reachability while making every species edit the same
coordinates so folds compose instead of collide. If shared-site sparse
matches frozen on multi-function AND keeps the apple win, it is the
default. Run: benchmarks/demo_solve_live.py --directions sparse
--latents 2048.

**Round eight — TSP on the new universal engine: a clean, expected
loss.** First time the redesigned solve() has been pointed at
permutations (random-keys encoding: decoder emits N priorities in
[0,1], argsort -> tour). TSP-50, 3 seeds, ~20k evals: new GA 17.7 vs
traditional tour GA 6.0, direct CMA-ES 8.9, nearest-neighbor 7.3 —
loses to everything including greedy. Not a bug, two known reasons:
(1) random keys are the decoder GA's weak domain (round 21: index-space
priors are negative knowledge for permutations; small priority nudges
reorder no cities, so a decoder biased toward smooth neighboring outputs
searches a chaotic step landscape) — 17.7 matches the plain decoder GA's
historical ~15.7 at this size; (2) the decoder that WON TSP (the
anchor-field grammar reading city coordinates, round 25/33, beat the
tour GA 9/1 at 50 and pulled ahead from 100+) lives only in
benchmarks/legacy_engines + round25_anchor_field.py and was never ported
into the new library, so the universal default searches blind to the
instance geometry. Scoreboard: the redesign is strong on continuous
outputs (near the apple record) and out-of-the-box no better at TSP than
any latent decoder ever was. The open problem is unchanged and now the
biggest gap: port the anchor grammar and give solve() a channel for a
fitness function to hand its decoder the instance data it reads. Run:
benchmarks/tsp_new_ga.py.

**Round nine — EXPERIMENTAL: tokenized genome + full attention transformer
(Daniel's BPE fusion, benchmarks/experimental_bpe_transformer.py, touches
nothing in the library).** The elegant core: BPE learns which adjacent
genome symbols co-occur enough to be ONE token, so a co-adapted block
becomes atomic and a plain token cut CANNOT split it — "crossover can't
break a gene" becomes a property of the representation, not an enforced
rule. Decoder is a real hand-rolled transformer (multi-head self-attention
over the genome, then a learned pixel-query grid cross-attending to paint
the image), LoRA-gated per individual, arithmetic fold, species + shares
reused from the library. Built, correct, BPE vocab grows, folding works.

First look (4 CIFAR images, 400 epochs, seed 3), TWO sobering results:
(1) the transformer is a WEAK image decoder — 11.3% error removed where
the conv-LoRA engine clears ~50% at comparable per-image budget. Round 17
again: convolution's spatial prior was the biggest image lever ever found,
and a transformer painting pixels discards it. (A transformer may be the
RIGHT decoder for grid-less modalities — sequences, sets, graphs — where
its generality would pay; untested.) (2) BPE token crossover TIED plain
crossover (11.3 vs 11.1, one seed, within noise) — the building-block
payoff still absent, the THIRD independent "safe but not a win" for the
linkage idea across three architectures. Consistent mechanism: within a
species the population converges, so there is little co-adaptation
variance for BPE to detect. Caveat: this run carried the LoRA ceiling
(the one sparse patches broke) and 4 images is underpowered for the
crossover question. Designed final test before closing the linkage thread:
sparse-patch modifier + 16 images (diversity and power), then judge
token-vs-plain. If BPE ties there too, the linkage idea is closed in this
architecture — a real multi-falsification result, not a gap.