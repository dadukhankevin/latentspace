# Rounds 1–4: the universal-decoder claim, tested to destruction

Run on 2026-07-14 using an Apple M3 Pro, PyTorch 2.12.0 and Python 3.14 (a new
environment — earlier results in this directory were produced under PyTorch
2.6.0, so all baselines were re-run rather than compared across files). Ten
algorithm seeds (`0..9`) per result, exactly 5,000 objective evaluations unless
stated, all neural decoders verified exclusively on `mps`. Paired statistics
use same-seed differences with a t-based 95% confidence interval. The CMA-ES
baseline is a from-scratch (mu/mu_w, lambda) implementation following Hansen's
tutorial (pip cannot install pycma in this environment); it drives a 16-d
sphere to machine zero within the budget (`round1_deceptive.py --self-test`).

## Round 1 — deceptive, hierarchical, rugged (`round1_deceptive.py`)

Trap-5 (50 bits), HIFF-64, NK (N=32, K=4), Rastrigin-64. New baselines: a
direct bitstring GA and CMA-ES.

| Objective | Best direct | Best latent | Random search |
|:---|---:|---:|---:|
| trap5_50 | direct GA **7.8** | gradient 14.7 | 18.5 |
| hiff64 | bit GA **157** | guarded 273 | 290 |
| nk32 | bit GA **0.226** | fixed 0.313 | 0.327 |
| rastrigin64 | direct GA 215 | gradient **148** | 901 |

On every binary problem the latent variants sit closer to random search than
to the direct GA. Mechanism: a fresh sigmoid decoder emits outputs at
0.500 ± 0.026, so every output bit hovers at the 0.5 threshold and a small
latent mutation flips a large, arbitrary set of bits. The MLP's coupling is
real but aligned with nothing; direct 1/n bit flips are strictly better moves.
Combined with the earlier TSP studies, the conclusion is uniform: the dense
random MLP destroys locality on discrete problems.

## Round 2 — the Rastrigin win was an initialization artifact (`round2_shifted.py`)

Unshifted Rastrigin's optimum in phenotype coordinates is exactly 0.5 — and a
fresh sigmoid decoder concentrates phenotypes there. Measured directly: a
random latent population starts at loss ≈ 456 (best 272) versus ≈ 1188 (best
943) for uniform initialization. Following BBOB practice, round 2 hides each
optimum at a random interior point (fixed per instance seed).

| rastrigin64 | Unshifted | Shifted |
|:---|---:|---:|
| Direct GA | 214.9 | 215.0 |
| Latent gradient | **147.8** | 431.3 |
| Latent frozen | 168.2 | 430.8 |

The direct GA is translation-invariant; the latent advantage does not shrink —
it inverts into a 2× deficit (0/10 paired seeds versus the direct GA). Shifted
sphere and shifted Ackley agree. On shifted problems decoder training does
nothing (gradient vs frozen: +0.5 ± 29.0 on Rastrigin). The original 16-d
Rastrigin result in `mps_initial_5000.md` — the project's one encouraging
case — carries the same exposure and should be considered explained.

## Round 3 — decoder bias matched to solution structure (`round3_structure.py`)

Every earlier benchmark has full-rank solutions: nothing for a 32-d latent to
compress. Round 3 uses the first objective whose optimum lies on a genuine
low-dimensional manifold — `smooth1d_256`, matching a random 16-component
low-frequency signal — with `rough1d_256` (256 iid uniform targets) as the
no-structure control. Decoders span the bias spectrum: a fixed DCT-32 linear
expansion (oracle match), a learned Conv1D upsampling stack (generic
smoothness prior), and the standard MLP.

| Strategy | smooth1d_256 | rough1d_256 |
|:---|---:|---:|
| CMA-ES (direct, 256-d) | **0.00075** | **0.00084** |
| Latent DCT + GA | 0.0063 | 0.058 |
| Direct GA | 0.0140 | 0.0138 |
| Latent MLP + GA (frozen) | 0.080 | 0.059 |
| Latent Conv + GA (frozen) | 0.083 | 0.061 |

With structure present the DCT decoder beats the MLP 10/0 (−0.074 ± 0.003)
and the direct GA 9/1; on the structureless control the advantage vanishes
exactly as predicted. This is the first confound-free positive result for the
latent representation in the project — but note it is a hand-chosen fixed
representation; co-evolution contributes nothing (conv gradient ≈ conv frozen,
MLP gradient ≈ MLP frozen). The conv smoothness prior did not materialize
(4/6 versus MLP; possibly an architecture problem, unresolved). And direct
CMA-ES still beat everything by ~8×: MSE targets are unimodal spheres, its
home terrain, and the sigmoid pre-image of the target is not exactly in the
DCT-32 span, giving all latent methods a representation floor.

## Round 4 — a strong optimizer inside the latent space (`round4_latent_cma.py`)

CMA-ES run over the 32-d latent of the frozen DCT decoder (matched bias) and
of a frozen random MLP (compression without bias), versus direct 256-d
CMA-ES — on the structured sphere and on `rugged_smooth_256` (Rastrigin
ruggedness centred on the smooth target: optimum on the manifold, multimodal
everywhere). Budgets 1,000 and 5,000.

| Strategy | smooth @1k | smooth @5k | rugged @1k | rugged @5k |
|:---|---:|---:|---:|---:|
| CMA-ES direct | 0.0415 | **0.00075** | 3893 | 3122 |
| CMA-ES in DCT-32 | **0.0104** | 0.0011 | **2553** | **1798** |
| CMA-ES in MLP-32 | 0.0766 | 0.0757 | 4028 | 3832 |
| Direct GA | 0.0756 | 0.0140 | 4344 | 2366 |
| Latent DCT + GA | 0.0533 | 0.0063 | 3842 | 2454 |

Paired results, all 10/0 unless noted:

- Unimodal, low budget: latent CMA beats direct CMA 4× (−0.031 ± 0.005).
- Unimodal, high budget: direct CMA catches up and edges ahead (+0.00035 ±
  0.00106, 7/3 — the representation floor); the crossover is real.
- Rugged + structured: latent CMA wins at both budgets by ~1.7× (−1324 ± 200
  at 5k) — multimodality in 256 dimensions is where direct CMA-ES dies.
- Compression without matched bias is worthless: CMA in the random MLP latent
  is the worst non-random strategy everywhere (0/10 versus direct CMA).
- The outer optimizer matters: CMA in DCT-32 beats the latent GA in the same
  representation 10/0 (−656 ± 230 on rugged at 5k).

## What survives

1. **The universal claim is falsified as tested.** A co-evolving random-MLP
   decoder with a latent GA lost on every problem family tried — TSP,
   deceptive, hierarchical, rugged-binary, and shifted continuous — and its
   single apparent win was an initialization artifact.
2. **The decoder-as-seam thesis survives in a sharper form.** A decoder whose
   inductive bias matches the solution manifold produces large, replicable
   gains (10/0 sweeps), appearing exactly when the structure exists and
   vanishing when it does not.
3. **Decoder co-evolution has still earned nothing.** No trainer produced a
   confirmed improvement over its frozen counterpart on any objective in any
   round. The gains available so far come from prior structure and from a
   stronger latent-space optimizer, not from online learning.
4. **The latent search mechanics should not be a fixed-sigma GA.** CMA-ES in
   the same latent representation dominates it; the library's `Evolver` would
   benefit from a CMA-style latent backend.
5. The strongest configuration found: matched-bias frozen decoder + CMA-ES in
   latent space, on rugged landscapes with structured optima — the one regime
   where it beats every direct method at every tested budget.

## Round 5 — the GeneSpace image demo, with controls (`round5_image.py`)

The ancestor GeneSpace repository showcases evolving a 50×50 RGB image toward
a photo by MSE, with no baseline and roughly two million uncounted
evaluations. On the demo's actual target image, a fresh scale-1 GeneSpace
decoder population starts at MSE 0.064 — equal to a constant gray canvas
(0.0625) and 2.3× better than uniform random pixels (0.147): the round-2
initialization artifact again. Budget-matched (5 seeds at 5k, 3 seeds at 25k;
gray-canvas reference 0.0625):

| Strategy | MSE @5k | MSE @25k |
|:---|---:|---:|
| CMA-ES in 2D-DCT-192 latent | **0.0301** | **0.0156** |
| latentspace default (float-32 MLP, self-distill) | 0.0624 | — |
| Faithful GeneSpace recipe (binary-250, width 2000) | 0.0629 | 0.0627 |
| Direct pixel GA | 0.1101 | 0.0742 |
| Random search | 0.1392 | — |

After 25,000 evaluations the faithful GeneSpace recipe has improved 0.0003
beyond its own initialization — it sits exactly at the gray canvas and never
leaves it. The demo's visual plausibility is the initialization artifact plus
an enormous uncounted budget. Meanwhile the matched-bias configuration from
round 4 (2D-DCT decoder + latent CMA-ES) is 4× below the gray floor at 25k
and still descending, and the direct pixel GA struggles in 7,500 dimensions:
image matching genuinely is representation-friendly terrain — for a decoder
with spatial structure, not a random MLP.

## Round 6 — can training learn what the oracle was given? (`round6_learned_structure.py`)

Every existing trainer regresses the decoder toward its own output on the
current best individual — a self-referential point target. Round 6 replaces
that with the EDA view: fit the decoder as a generative model of the elite
set (linear instantiation: PCA of elite logits), then run CMA-ES in the
learned latent. All arms tested on `smooth1d_256` (family manifold: 16 DCT
components) with `rough1d_256` as the no-structure control, 10 seeds, 5,000
fresh evaluations. References: oracle DCT+CMA 0.0011, direct GA 0.0140,
random-MLP latents 0.078.

| Training corpus for the PCA-32 basis | smooth | rough (control) |
|:---|---:|---:|
| Own run's bootstrap (1k evals, counted) | 0.0478 | 0.0393 |
| Same + refit every 1k | 0.0478 | 0.0393 |
| One other instance (transfer) | 0.1109 | 0.1020 |
| 8 instances × 1k evals (pretrained) | 0.0638 | 0.0654 |
| 32 instances × 2k evals (pretrained) | 0.0119 | 0.0586 |
| 128 instances × 2k evals (pretrained) | **0.0042** | 0.0549 |

Findings:

1. **Within-run learning is exploitation, not structure.** The
   single-instance basis works equally well on the rough control — it learns
   "the direction toward this target," transfers to nothing (the one-instance
   transfer arm lands at random-search level), and loses to simply continuing
   the direct GA.
2. **Refitting from the search's own products is geometrically self-locking**
   (refit ≡ no-refit to six decimals): once search happens inside the learned
   span, every new elite lies in that span. This is the deepest form of the
   self-referentiality that afflicts all the distillation trainers.
3. **Cross-instance pretraining works and scales.** A fresh instance's target
   reconstructs inside the learned span at MSE 0.058 (8 instances), 0.0117
   (32), and search hits that floor exactly — the optimizer stops being the
   bottleneck. Eight instances fail because the family manifold is
   16-dimensional; you need more instances than manifold dimensions. At 128
   instances the learned code reaches 0.0042 on fresh instances — beating
   the direct GA 3.3× and closing most of the gap to the hand-built oracle —
   while the rough-family control stays broken at every scale (its instances
   span full rank; there is nothing shared to learn).

The practical recipe that emerges: **pretrain the decoder across a problem
family (with more instances than the manifold has dimensions), freeze it
within the run, and search its latent space with CMA-ES.** The "universal
shareable genetic code" is real but it is a pretraining artifact — no
single-run trainer tested here or in any earlier study can bootstrap it,
and the mechanism failure (self-locking span) suggests none can without an
off-manifold exploration channel.

## Round 7 — the scaling law, across problem types (`round7_scaling.py`)

K ∈ {8, 16, 32, 64, 128} pretraining instances, four families, ten seeds,
5,000 fresh evaluations per test instance. Values are fresh-instance loss
normalized to the direct GA at equal budget (1.0 = parity); chart in
`family_scaling.svg`, generated by `plot_family_scaling.py`.

| K | Smooth 1-D | Rugged smooth | Image 2-D | Rough control |
|---:|---:|---:|---:|---:|
| 8 | 4.62 | 1.57 | 1.27 | 5.06 |
| 16 | 1.66 | 1.09 | 1.01 | 4.67 |
| 32 | 0.85 | 0.92 | 0.87 | 4.24 |
| 64 | 0.48 | 0.83 | 0.57 | 4.07 |
| 128 | **0.30** | **0.79** | **0.41** | 3.97 |

- The law replicates across dimension (256 vs 1,024), domain (1-D signals vs
  2-D images), and fitness topology (unimodal vs multimodal): monotone
  improvement with K, parity crossed between K = 16 and 32 in every
  structured family.
- The rough control barely moves across 16× more pretraining — the gains are
  learned family structure, not a generic training effect.
- The rugged family converges to its oracle bound (1,878 at K = 128 versus
  1,798 for the hand-built DCT decoder): representation stops being the
  bottleneck; landscape difficulty in latent space takes over.

## Round 8 — a neural decoder on the identical corpus (`round8_mlp_pretrain.py`)

Same recipe as round 7 at K = 128, with the fitting step swapped: a
32-bottleneck autoencoder (256→128→32→128→256, 2,000 Adam steps on elite
logits) instead of PCA, searched by CMA-ES over its standardized code space.
A new curved family, `blob2d_1024` (three Gaussian blobs, 12 nonlinear
parameters), was added because PCA can only fit flat structure and the MLP
needs curvature to earn a win.

| 5k budget, 10 seeds | smooth1d_256 (flat) | blob2d_1024 (curved) | rough control |
|:---|---:|---:|---:|
| PCA-128 + CMA | **0.0042** | **0.0214** | 0.0549 |
| MLP-128 + CMA | 0.0164 | 0.0544 | 0.0495 |
| Direct GA | 0.0140 | 0.0582 | **0.0138** |

The MLP loses to PCA everywhere, including the curved family built for it.
Diagnosis (representation-floor probe): the autoencoder's own encode→decode
reconstruction of the test target is 0.030 (smooth) and 0.096 (blob) — worse
than what CMA-ES actually reached through it, so search is not the problem;
the learned map is simply underfit (final train loss 1.2–2.2 in logit MSE).
PCA solves the same fit in closed form. Notably PCA beats direct GA 2.7× even
on the curved family — a flat 32-plane through logit space captures enough
blob-blur structure to help. The neural decoder remains unearned: its case
now requires showing the fit gap closes with more data/steps and then
surpasses the flat map's curvature ceiling, in that order.

**8b — does the MLP keep the scaling law?** Yes. Sweeping K with the fixed
2,000-step recipe (chart: `mlp_vs_pca_scaling.svg`), values normalized to the
direct GA:

| K | smooth PCA | smooth MLP | blob PCA | blob MLP |
|---:|---:|---:|---:|---:|
| 8 | 4.62 | 4.08 | 0.79 | 1.18 |
| 16 | 1.66 | 2.25 | 0.67 | 1.27 |
| 32 | 0.85 | 1.92 | 0.54 | 1.19 |
| 64 | 0.48 | 1.50 | 0.45 | 1.07 |
| 128 | 0.30 | 1.17 | 0.37 | 0.94 |

Both decoders improve lawfully with K, but PCA's slope is steeper on both
families, so the gap widens rather than closes — with a fixed training
budget, more data leaves the network relatively more underfit while the
closed-form fit simply gets better. PCA's curved-family curve shows no
ceiling through K = 128 (0.79 → 0.37 and still falling), so the regime where
a neural decoder is necessary has not yet been reached.

## Rounds 9–10 — fixing the neural decoder, and the limit of online refinement

Round 9 (`round9_mlp_training.py`): the MLP's round-8 deficit was data, not
compute — 16× more Adam steps changed nothing, while augmenting the 1,280
real elites with 5,000 samples drawn from the fitted PCA decoder closed ~95%
of the gap (smooth 0.0164 → 0.0050 vs PCA 0.0042, statistical tie; blob
0.0544 → 0.0220 vs PCA 0.0214). Recipe: bootstrap elites → fit PCA → use it
to synthesize dense training data → train the neural decoder on both.

Round 10 (`round10_online_refine.py`): warm-start from that augmented fit,
then refit every ~1,000 evaluations on the best real candidates CMA-ES
discovers, re-anchoring CMA at the encoded best phenotype. Result: safe but
inert — online trends ahead of frozen 7/3 on both families but the CI
includes zero, and PCA still edges both. Diagnosis: candidates harvested from
the decoder's own outputs lie ON its manifold by construction, so refitting
teaches density, not geometry — the round-6 self-referentiality trap operates
even with a nonlinear decoder. Genuinely new geometry requires an
off-manifold exploration channel (e.g., short direct-space local search
around incumbents, fed back as training data).

## Round 11 — one decoder across families (`round11_universal.py`)

The universal-decoder question, tested the way Daniel framed it: manifolds
stay per family (each family's elites get their own PCA scaffold), but ONE
neural decoder trains on the union. Two 256-d families (SmoothTarget and a
16×16 blob variant) so a single output head serves both; K = 128 practice
instances each, round-9 recipe (real elites + 5,000 PCA-synthetic per
family, 8,000 Adam steps), 10 paired seeds, 5,000-evaluation budget.

Arms: `perfam` / `perfam64` (one decoder per family, latent 32/64),
`uni32` / `uni64` (one shared decoder), `uni32_blind` (shared, CMA-ES from
the origin instead of the solving family's mean elite code).

| arm | smooth1d_256 | blob2d_256 |
|:---|---:|---:|
| perfam (32) | 0.00498 | 0.00461 |
| perfam64 | 0.00870 | 0.00358 |
| uni32 | 0.00445 | 0.00442 |
| uni32_blind | 0.00449 | 0.00442 |
| uni64 | 0.00516 | **0.00287** |

Paired findings:

- **No interference.** uni32 vs perfam: 8/2 and 7/3 in the universal
  decoder's favor, CIs spanning zero. Sharing one decoder across families
  costs nothing at matched capacity.
- **Anchoring is unnecessary.** uni32 vs uni32_blind is a statistical tie
  (per-seed differences ~1e-7). The autoencoder places both families' elite
  codes near the origin of the shared space, and the fitness signal alone
  steers CMA-ES to the right family — the decoder never needs to be told
  which problem it is solving.
- **Capacity and sharing interact.** On blobs, latent 64 helps per-family
  (perfam64 beats perfam 10/0) and sharing adds more on top (uni64 beats
  perfam64 8/2, CI [−0.0011, −0.0003]). On smooth, latent 64 *hurts*
  per-family badly (perfam64 loses 0/10 — 48 excess latent dimensions fit
  noise in a 1,280-elite corpus and CMA-ES wastes budget in them), yet the
  shared corpus rescues it: uni64 beats perfam64 9/1 and matches perfam32.
  Cross-family data acts as regularization for excess capacity and as
  genuine signal where the family can use it.

Verdict: the shared decoder is never worse and sometimes better; there is
no measured reason to split neural decoders by family. Caveat: two
same-dimension families — spanning heterogeneous output shapes needs
conditioning/masking, and the corpus-scale version of this question
(many families, one code) remains open.

## Round 12 — weight mutation as the off-manifold channel (`round12_weight_mutation.py`)

Daniel's proposal for breaking the self-referentiality limit: temporarily
mutate the decoder's WEIGHTS. A weight-perturbed decoder has a different
output manifold, so its outputs are off the base manifold by construction —
yet still decoder-shaped. Distill the good ones back into the base weights.

Setup: round-10 scaffold (identical warm start and refit machinery; arms
differ only in refit data). Each epoch, ~8% of the evaluation budget goes to
mutant outputs at the best latent (+jitter). Two variants:

  * `wmut` — 12 independent one-step mutants; keep outputs that beat the
    base decoder's output at the same latent;
  * `wmut_es` — a (1+1)-ES walk: mutate, keep the champion if it wins on
    real evaluations, mutate the champion, repeat (2–5 accepted steps per
    epoch compound); then distill the base from the CHAMPION's outputs — a
    verified-better teacher rather than self-imitation.

Calibration finding: decoder weight mutation has a textbook mutational
fitness landscape — 43% of mutant outputs beat their parent at sigma_w
0.003, 24% at 0.01, ~0% at 0.03+. Only the small-step regime is viable.

| arm (10 seeds, 5,000 evals) | smooth1d_256 | blob2d_1024 | blob floor start→end |
|:---|---:|---:|:---|
| frozen_aug | 0.00498 | 0.02198 | 0.0646 → 0.0646 |
| online (round 10) | 0.00448 | 0.02178 | 0.0646 → 0.0781 |
| wmut | 0.00454 | 0.02171 | 0.0646 → 0.0775 |
| wmut_es | 0.00446 | 0.02168 | 0.0646 → 0.0773 |

Paired verdicts:

- `wmut` ties `online` everywhere — one-step mutations gain only ~0.5% per
  accepted sample, so the distilled data sits epsilon off the manifold and
  teaches epsilon of new geometry. Null.
- `wmut_es` is the first refitting arm to beat `frozen_aug` **with a CI
  excluding zero** on the curved family (blob 8/2, diff −0.0003), and on
  smooth it significantly protects the representation floor that plain
  online refitting degrades (floor_end 9/1 better than online, CI excludes
  zero). Against `online` directly, final loss is still a tie.
- New negative finding: refitting on incumbent-concentrated data actively
  RAISES the floor (0.065 → 0.078 on blob for every refitting arm) —
  round-10's "safe but inert" was generous; the refits trade target
  expressibility for density around the incumbent.

Verdict: the mechanism is validated — off-manifold data flows, and
distilling from a verified-better weight-mutant teacher is the first
within-run training that ever significantly beat frozen — but the effect
is ~1.4% where the blob representation floor is ~65% of the total loss.
The walk improves the manifold around the incumbent; it cannot discover
the target's missing geometry because nothing in the channel points toward
the target. The unconstrained version — direct phenotype-space local
search written back into the corpus (open problem #1) — remains the
designed escalation.

## Round 13 — dual decoders, cross-training, weight repulsion (`round13_dual_decoder.py`)

Daniel's generalization of round 12's teacher-not-student fix: TWO decoders,
neither ever trained on its own outputs (A refits only on elites found by
searching through B and vice versa), plus a subtle per-epoch weight decay
AWAY from each other (2% of the current weight difference, `REPEL = 0.02`)
so they cannot converge until cross-data becomes self-data. Both decoders
get the round-9 warm start and differ only in init seed; CMA-ES search
alternates decoders per epoch (A,B,A,B,A) and the global best phenotype is
re-encoded into whichever decoder searches next. Single-decoder baselines
(`frozen_aug`, `online`, `wmut_es`) are paired from round 12's JSONs — same
seeds, same warm-start machinery.

| arm (10 seeds, 5,000 evals) | smooth1d_256 | blob2d_1024 | blob floors A/B start→end |
|:---|---:|---:|:---|
| frozen_aug (round 12, single) | 0.00498 | 0.02198 | 0.0646 → 0.0646 |
| wmut_es (round 12, prev champion) | 0.00446 | 0.02168 | 0.0646 → 0.0773 |
| dual_frozen | 0.00451 | 0.02184 | 0.0646/0.0640 → unchanged |
| dual_self | 0.00446 | 0.02175 | → 0.0741/0.0728 |
| dual_cross | 0.00441 | 0.02170 | → 0.0741/0.0725 |
| dual_self_repel (13b control) | 0.00474 | 0.02130 | → mixed, elevated |
| **dual_cross_repel** | 0.00453 | **0.02120** | → 0.0802/0.0822 |

Paired verdicts (blob, the curved family):

- `dual_cross_repel` beats EVERYTHING ever run at latent 32: frozen 10/0
  (−0.00078, −3.5%), online 10/0, `wmut_es` 10/0 (−0.00048, 2.5× its
  effect), `dual_cross` 10/0, `dual_self` 9/1 — every CI well clear of
  zero. Biggest within-run training win of the campaign.
- Factorization: `dual_cross` vs `dual_self` is a TIE (7/3, CI spans zero)
  and `dual_self_repel` vs `dual_cross_repel` is a TIE (5/5) while
  `dual_self_repel` beats `dual_self` 9/1 — **the repulsion is the active
  ingredient; who trains whom does not matter on blob.**
- The repelled arms have the WORST floors (0.065 → 0.082) and the best
  results: the win comes from maintaining two genuinely different
  manifolds to search, not from decoder fidelity. Repulsion along
  θ_A − θ_B between independently initialized nets is a large coherent
  persistent perturbation, not epsilon noise.

Paired verdicts (smooth, the easy family):

- Repulsion is poison where the decoder already nearly expresses the
  target: `dual_self_repel` loses 0/10 to `dual_self`, `dual_cross_repel`
  loses 1/9 to `dual_cross`.
- But cross-training buffers the damage: `dual_cross_repel` beats
  `dual_self_repel` 9/1 (CI excludes zero). Verified-good elites from the
  OTHER decoder pull a perturbed net back toward useful geometry; its own
  elites do not. Cross-training is a stabilizer, not a driver.
- `dual_frozen` ties single frozen on both families: splitting the search
  budget across two manifolds costs nothing — re-anchoring transfers
  progress cleanly.

Verdict: dual decoders + repulsion is the strongest within-run mechanism
found so far, but for ensemble-diversity reasons, not the anti-self-
distillation reasons that motivated it. Open follow-ups: magnitude-matched
random-perturbation control (is the difference direction special?),
repulsion strength scheduled by floor gap, >2 decoders.

## Round 14 — the original algorithm from scratch, dual mechanism installed (`round14_original_dual.py`)

Daniel's constraint arrives: retry the ORIGINAL regime (random decoders, one
run, no pretraining, no CMA-ES, no phenotype operators beyond the reference
GA) with the round-13 dual-repulsion mechanism injected into the unmodified
package `Evolver` as one `decoder_update` layer. 3 seeds sufficed — spreads
are tiny, gaps enormous.

| arm (5,000 evals) | smooth1d_256 | blob2d_1024 |
|:---|---:|---:|
| direct_ga | 0.0146 | 0.0562 |
| latent_fixed (frozen random decoder) | 0.0795 | 0.1011 |
| latent_gradient (original co-evolution) | 0.0778 | 0.1007 |
| dual (cross-distillation hand-offs) | 0.0808 | 0.1021 |
| dual_repel (full round-13 mechanism) | 0.0801 | 0.1019 |

All latent arms are one plateau, 5–7× behind the direct GA. Verdict: the
dual mechanism amplifies decoders that already know something; it cannot
create knowledge. Bonus: the original Evolver runs a full budget in ~0.3s.

## Round 15 — per-individual decoders, harvest test, bootstrap stall (`round15_*.py`)

15a: Daniel's proposal — each individual owns its genome AND its decoder
weights; children mutate both (weight sigma in round-12's viable regime);
optional weight-PCA crossover (mix parent coefficients in the elite
population's principal subspace). 3 seeds: mutation-only 0.0723 / 0.0998 —
the FIRST from-scratch universal method to beat the frozen plateau
(smooth −9%); the weight-PCA crossover was neutral.

15b: use it as the practice-problem harvester in the round-7 protocol
(K instances × 2,000 evals, top-10 pooled, PCA-32, CMA solve). Scaling law
survives directionally (K=16: 0.0554 → K=128: 0.0354) but teachers are too
weak: corpus mean loss 0.062 vs the GA's 0.040 gives test solves 0.0354 vs
0.0040 — a 1.5× teacher gap amplified to ~9× downstream.

15c: the self-teaching loop — re-harvest all 128 practice problems warm-
started from the current decoder (distilled into each individual's weights
+ noise), refit, repeat ×3. Corpus quality climbs 0.062 → 0.050 → 0.034
(BETTER than the GA's 0.040) while the test solve stalls 0.0354 → 0.0345 →
0.0338, converging to corpus ≈ test (self-consistency fixed point).
**Second law: a teacher's value is the INDEPENDENCE of its errors, not
their size.** The GA's per-instance errors are uncorrelated and cancel
under compression, revealing the manifold; one shared warm start correlates
every harvest's bias and compression preserves it forever.

## Round 16 — explore-then-distill within one run (`round16_single_fitness.py`)

Single fitness function only. Explore with per-individual decoders (60% of
budget; 32 lineages ≈ independent-ish errors), compress the best 200 vetted
phenotypes (PCA-32), CMA-ES the latent (40%). Dense-MLP version: smooth
0.0695 (+5% over pure exploration), blob 0.0990 (tie) — selection
concentrates the archive into few lineages, eating the independence.

## Round 17 — the architecture prior (`round17_architecture_prior.py`)

Daniel's ruling: decoder ARCHITECTURE may match the modality (CNN, 1-D
conv, transformer, ...); evolution's operators stay universal. Per-
individual conv decoders (genome → 4×4 map → upsample+conv to 32×32; 1-D
analog for signals), identical evolution:

| arm (5,000 evals, 3 seeds) | smooth1d_256 | blob2d_1024 |
|:---|---:|---:|
| mlp_decoder | 0.0729 | 0.0994 |
| conv_decoder | 0.0676 | **0.0765** |

−23% on the image from architecture alone (deep-image-prior effect) — the
largest single-fitness lever found.

**Stack (conv exploration + distillation), 10 paired seeds
(`mps_round16c_confirmation_10seed.json`):** blob 0.0208 vs direct GA
0.0582 — **10/0, diff −0.0373, CI [−0.0437, −0.0309], 2.8× better: the
first fully universal method to beat the hand-matched traditional GA.**
Smooth: 0.0245 vs 0.0140, 4/6, CI spans zero — statistical tie with high
seed variance (best seeds beat the GA, two flop seeds 3–4× worse); variance
control is open problem #1.

## Round 18 — adaptive phase switch, stratified harvest (`round18_adaptive.py`)

Replace the stack's magic 60/40 explore/exploit split with a stall rule
(exploration ends when best loss improves <1% relative over 10 generations;
reserve 10×latent evals for CMA-ES) and test a per-lineage cap (10) on the
200 distilled solutions. 10 paired seeds; direct-GA pairs from
`mps_round16c_confirmation_10seed.json`.

| arm (10 seeds, 5,000 evals) | smooth1d_256 | blob2d_1024 | explore evals |
|:---|---:|---:|:---|
| fixed 60/40 | 0.0253 | 0.0213 | 3,008 |
| adaptive (stall switch) | **0.0203** | **0.0194** | 358 smooth / 864 blob (352–3,200) |
| adaptive + lineage cap | 0.0243 | 0.0205 | same switch |

Paired verdicts: adaptive beats fixed on smooth 8/2 (CI [−0.0094, −0.0008])
and edges it on blob 6/4 (CI spans zero); vs the traditional GA, blob stays
10/0 (CI [−0.0431, −0.0344]) and smooth improves from a losing tie to a
genuine 6/4 coin-flip (CI spans zero). Exploration truly stalls after a few
hundred evaluations — the 60/40 split was not just arbitrary but far too
generous to exploration — and the switch point varies 9× across blob seeds,
so the rule adapts rather than re-finding a constant. The lineage cap was
null-to-harmful (diluting top solutions costs more than the independence
buys); defaults to off.

These defaults were packaged as `latentspace.universal.solve` (explore →
distill → exploit with pluggable architectures); six CPU tests in
`tests/test_universal.py` and a 3-seed MPS parity check reproduce the
benchmark numbers through the public API.

## Round 19 — latent-size sweep through the packaged API (`round19_latent_sweep.py`)

Is 32 the right latent? Sweep {8, 16, 32, 64, 128}, 10 paired seeds, both
standing problems, run entirely through `latentspace.universal.solve`.

| latent | smooth1d_256 | blob2d_1024 |
|---:|---:|---:|
| 8 | 0.0463 | 0.0647 |
| 16 | 0.0239 | 0.0506 |
| 32 | 0.0203 | 0.0194 |
| 64 | 0.0199 | **0.0165** |
| 128 | 0.0249 | 0.0165 |

Cliff below 32 (blob: 16 loses to 32 on 0/10 seeds, giving back half the
win over the direct GA — the space must exceed the intrinsic variety of
good solutions, with headroom). 32/64/128 are pairwise indistinguishable
(all CIs span zero), but 64 has the best means on both problems, halves
blob seed variance (stdev 0.0026 vs 0.0052), and gives the tightest GA win
recorded (10/0, CI [−0.0444, −0.0389]). 128 pays a small noise tax on
smooth: fitting 128 directions from 200 distilled solutions. Package
default moved to 64; `distill_top` should scale >= ~3x latent. Untested:
decoupling genome size (exploration) from distilled dimension (exploit).

## Rounds 19b/19c — does the stack need CMA-ES, and does it beat pure CMA-ES? (`round19b_no_cma.py`)

19b: identical exploration and distilled decoder, but the exploit phase's
genotype evolution swapped from CMA-ES to a plain fixed-sigma GA (uniform
crossover, 0.1/0.12 mutation, the original evolver's rates). Catastrophic:
smooth 0.0742, blob 0.0893 — 0/10 vs the CMA exploit on both problems
(3.7-4.6x worse), and worse than never distilling at all. The distilled
space's directions have wildly different fitness sensitivities; adapting
per-direction step sizes is what CMA-ES is for, and a fixed-sigma GA
random-walks the sensitive axes. CMA-ES stays on merit — and it is itself
an evolution strategy over genotypes; no phenotype is touched.

19c: pure direct CMA-ES on the raw solution values (no decoder anywhere),
10 seeds (`mps_round19c_direct_cma.json`):

| method | smooth1d_256 (256-d) | blob2d_1024 (1,024-d) |
|:---|---:|---:|
| direct CMA-ES on raw values | **0.00075** | 0.04377 |
| universal stack (adaptive, latent 64 era: 32) | 0.02028 | **0.01938** |

Split verdict, both 10/0: on the low-dimensional unimodal curve, direct
CMA-ES is 27x better — continuous match-the-target at 256-d is its home
terrain and the stack should not be used there. On the 1,024-d image the
wall flips: full-covariance adaptation starves at 5,000 evals and the
learned decoders' 64 directions beat raw search 2.3x. Combined with round
4 (multimodal ruggedness kills direct CMA even at 256-d, matched-decoder
latent search wins 1.7x) the domain map is: direct CMA-ES for
low-dimensional smooth continuous problems; the learned-decoder stack for
high dimensions, multimodality, and everything that is not a flat float
vector.

## Round 20 — re-entering exploration: cycling falsified at tested budgets (`round20_cycle.py`)

Daniel's proposal: an elegant solver should hand the budget back and forth
between exploration and exploitation instead of running a one-way conveyor.
Implemented as `phases="cycle"` in the packaged solver: the exploit phase
gets the same stall rule as exploration (20 CMA generations without 1%
relative improvement); on stall, exploration re-enters with half its
population warm-started from the current distilled space decompressed into
decoder weights + noise (round 12's escape channel) and half fresh (round
15c's independence guardrail); the archive is cumulative and each cycle
re-distills.

Result, 10 paired seeds through the public API: cycling loses 0/10 on both
problems — smooth +0.0014 (CI grazes zero), blob +0.0030 (CI [+0.0010,
+0.0051], ~18% worse). Diagnosis: CMA-ES convergence has natural mid-run
plateaus before covariance adaptation pays off; a 20-generation patience
amputates its endgame, and the re-entered exploration (epsilon-scale weight
mutations around the incumbent manifold) cannot discover enough new
geometry in the remaining budget to repay the theft. The long-budget
regime offers no refuge either: the 150k-eval color-apple run's exploit
phase was still improving at the final evaluation (0.18% over the last
3,000) — the representation floor was never reached in any tested regime,
and re-entering exploration only makes sense once it is. `phases="cycle"`
stays in the package as a documented experimental option; the default
remains the one-way stack. The open question worth a future round: trigger
re-entry on a *converged* signal (CMA step-size collapse at the floor)
rather than a stall heuristic.

## Round 20b — the counterfactual: at long budgets the switch itself is the mistake

Daniel questioned the color-apple demo's chart: exploration looked like it
was on a fine trajectory when the stall rule handed off to CMA-ES. The
counterfactual (same seed, identical run, hand-off disabled — deterministic
prefix confirmed) settles it: CMA-ES sprints for ~25k evaluations
(0.0217 -> 0.0132 by 100k vs exploration's 0.0169), then flattens at the
ceiling of its frozen gene space (~0.011) while never-switched decoder
evolution keeps compounding to **0.00493 — 2.3x better at 150k**
(single seed, one problem). The stall rule fired during a slow patch, the
mirror image of round 20's failure: BOTH phases have bumpy progress curves
that stall heuristics misread.

Related measurement (Daniel's naked-eye catch): the target's leaf region
(3.5% of pixels) shows the frozen gene space's bias concretely — the
switched run's leaf error is 5x its own typical error and painted
anti-green (greenness -0.21 vs target +0.15), because no green-leaf
variation existed in the distilled archive; the traditional GA, unbiased
per-pixel, was slowly greening it. Small features that are worthless early
(0.0013 of total error) become the bottleneck late (~12% of final error) —
an accidental curriculum with no late-stage recovery under a frozen space.

Regime map after 20b: short budgets — distill+CMA is worth 4x (round 17);
long rich runs — evolving decoders outrun any frozen compression of
themselves. Round-21 design: a rate-based scheduler (interleave the two
forces, give budget to the better measured improvement rate), which would
have ridden the CMA sprint and returned to exploration when the rates
crossed. Counterfactual rerun with frames answered the leaf question, and it
CORRECTS the paragraph above: the leaf did NOT turn green even with no
compression and 150k evaluations of free evolution (leaf greenness -0.20,
identical to the switched run's -0.21, vs target +0.15) — despite the leaf
being ~21% of the run's remaining error by the end. The bias is NOT the
distillation's: it lives in the exploration lineages' inductive
bias/reachability — every individual descends from dark-blob-painting
ancestors and weight mutation never escaped that basin; only the unbiased
per-pixel GA was slowly greening it (-0.05). New open problem: rare local
features unreachable from the founding basin — candidate fixes are lineage
diversity mechanisms (decoder-family crossover, round-13 repulsion), not
more budget. Counterfactual final: 0.00493 (best result ever on this
problem, pure decoder evolution, no CMA-ES anywhere); demo artifact shows
its animation and the four-way finals.

Raw results: `mps_round1_deceptive_5000.json`, `mps_round2_shifted_5000.json`,
`mps_round3_structure_5000.json`, `mps_round4_latent_cma_1000.json`,
`mps_round4_latent_cma_5000.json`, `mps_round5_image_5000.json`,
`mps_round5_image_25000.json`, `mps_round6_learned_structure_5000.json`,
`mps_round6b_family_5000.json`, `mps_round6c_family32_5000.json`,
`mps_round6d_family128_5000.json`, `mps_round7_scaling_5000.json`,
`mps_round8_mlp_pretrain_5000.json`, `mps_round9_mlp_training_5000.json`,
`mps_round10_online_refine_5000.json`, `mps_round11_universal_5000.json`,
`mps_round11b_perfam64_5000.json`, `mps_round12_weight_mutation_5000.json`,
`mps_round12b_wmut_es_5000.json`, `mps_round13_dual_decoder_5000.json`,
`mps_round13b_self_repel_blob_s6to9.json` (13b smooth + blob seeds 0–5 were
printed to stdout only — the run was stopped before its JSON write — and are
transcribed in the round-13 stats; blob seeds 6–9 were rerun into the JSON),
`mps_round14_original_dual_potential.json`,
`mps_round15_individual_decoders_potential.json`,
`mps_round15b_universal_harvest_smooth.json`,
`mps_round15c_bootstrap_smooth.json`, `mps_round16_single_fitness.json`,
`mps_round16b_explore_distill_conv.json`,
`mps_round16c_confirmation_10seed.json`,
`mps_round17_architecture_prior.json`,
`mps_round18_adaptive_10seed.json`,
`mps_round19_latent_sweep.json`, `mps_round19b_no_cma.json`,
`mps_round20_cycle_5000.json`,
`mps_round19c_direct_cma.json`.
