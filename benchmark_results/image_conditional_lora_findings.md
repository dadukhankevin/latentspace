# Multi-fitness conditional decoder experiment

One matched seed (`3`), 32 real-image objectives, 48 survivors, 192 children,
fixed mating radius `0.3`, 60,000 objective evaluations. All conditional arms
use the identical private-decoder trajectory through evaluation 20,000; at
19,632 they are bit-for-bit matched at mean MSE `0.0292593`, worst MSE
`0.0741967`, and global gain `3.0590`.

At 20,000, the 48 current private functions are factorized using their
already-discovered phenotypes as labels. Real targets are used only for
fitness. The second phase evolves the original 64 genes plus aligned
conditional genes, uses the existing target-coverage selection, individual
age 10, global success controller, and inherited stagnation kicks. Three
200-step consolidation events at approximately 30k, 40k, and 50k train on
current adapted outputs while halving conditional values and applying a
coefficient-zero reconstruction loss.

| arm | individual conditional values | shared parameters | mean MSE | median MSE | worst MSE | persistent decoders |
|:---|---:|---:|---:|---:|---:|---:|
| reference inheritance + mergers | 0 | multiple | 0.024356 | 0.021251 | 0.062957 | 12 |
| private CNN mutation forever | full decoder | 48 x 47,155 | 0.023493 | 0.019276 | 0.074114 | 48 |
| conditional LoRA-32 | 32 | 92,819 | 0.018572 | 0.018164 | 0.043118 | **1** |
| conditional LoRA-64 | 64 | 138,483 | 0.018529 | **0.016538** | 0.049370 | **1** |
| ordinary extra-latent-64 control | 64 | 84,019 | **0.018256** | 0.018371 | **0.042781** | **1** |

Against the matched reference arm, LoRA-32 improves mean MSE by 23.7%, worst
MSE by 31.5%, and wins 29/32 targets. LoRA-64 improves mean by 23.9%, worst by
21.6%, and wins 28/32. The ordinary latent control improves mean by 25.0%,
worst by 32.0%, and wins 29/32.

The decisive finding is the shared conditional transition, not specifically
LoRA placement: the equal-size ordinary latent control slightly beats both
LoRA arms. LoRA-64's extra capacity improves median quality but weakens the
tail, reproducing the earlier non-monotonic rank/capacity law.

Coefficient-zero diagnostics show that consolidation is real rather than a
renaming of private decoders:

| arm | adapted mean | coefficient-zero mean | mean gap | coefficient-zero worst |
|:---|---:|---:|---:|---:|
| conditional LoRA-32 | 0.018572 | 0.019665 | 0.001093 | 0.045601 |
| conditional LoRA-64 | 0.018529 | 0.019997 | 0.001468 | 0.051837 |
| ordinary extra-latent-64 | 0.018256 | 0.018708 | **0.000452** | **0.043439** |

Thus even with all conditional values zeroed, one backbone retains most of the
gain. The latent control consolidates most completely. This is a one-seed
potential check, not a confirmed default; paired multi-seed confirmation and
ablations of consolidation/base-only loss are required.

## Full mixed run

The fixed-budget mix uses 32 ordinary extra-latent values and 32 learned LoRA
gates per individual. It ran for 600,000 objective evaluations with the live
view enabled, retaining private decoders through evaluation 120,000 and then
factorizing them into one shared conditional decoder. Four 200-step
consolidations ran at evaluations 220,032, 320,064, 420,096, and 520,128.

| full 600k arm | mean MSE | median MSE | worst MSE | persistent decoders |
|:---|---:|---:|---:|---:|
| reference sweep | 0.018688 | 0.017385 | 0.072594 | 30 |
| individual stagnation (previous best) | 0.017053 | 0.014466 | 0.062048 | 11 |
| mixed latent-32 + LoRA-32 | **0.012489** | **0.012585** | **0.024484** | **1** |

The mixed arm beats the previous best full run on 27/32 targets, reducing mean
MSE by 26.8% and worst-target MSE by 60.5%. It beats the reference sweep on
all 32 targets, reducing mean by 33.2% and worst by 66.3%.

With all 64 conditional values zeroed, the same shared decoder scores mean MSE
`0.014983` and worst MSE `0.029769`. The zero-conditional backbone by itself is
therefore 12.1% better in mean and 52.0% better in worst-target error than the
previous best full evolutionary run, although conditional state still provides
a meaningful 16.6% mean improvement over that backbone. Final conditional RMS
is `0.4560`; it fell after each consolidation and regrew during subsequent
evolution, so the outcome is a genuinely shared backbone plus useful personal
state, not a fully unconditional decoder.

This remains a seed-3 result. It establishes that the mix can work at full
budget and that one-decoder convergence need not sacrifice diversity or tail
quality; confirmation across paired seeds and a full-run latent-64 control are
the next tests needed to attribute the gain specifically to mixing latent and
LoRA conditioning.

## One decoder from the founders

The always-shared arm removes both the private-development phase and every
scheduled consolidation. All founders and descendants use one mixed decoder
with 32 extra-latent values and 32 LoRA gates. After every complete generation,
the mean LoRA gate among current survivors is subtracted from each individual
and folded algebraically into every backbone layer:

`B(x) + U diag(c) D(x) = [B + U diag(mean(c)) D](x) + U diag(c - mean(c)) D(x)`

This is an exact change of coordinates, not distillation. Across 3,124 folds
in the full run, maximum measured phenotype drift was `1.38e-14` MSE. The
extra-latent half remains private because its interaction with LoRA gates does
not admit the same simple fold in the current architecture.

Naive continuous replay was tested first and rejected. It improved mean MSE
from `0.0932` to `0.0895` by 10k, then accumulated shared-decoder drift and
collapsed to `0.1230` by 20k. An always-shared fixed-backbone control reached
`0.0440` at 20k, proving that mixed personal state itself fixes much of the old
shared-frozen failure. Exact mean folding reached `0.0497` at 20k but improved
its coefficient-zero backbone from the fixed control's `0.1001` to `0.0670`.

The exact-fold arm then ran live for the full 600,000 evaluations:

| full 600k arm | mean MSE | median MSE | worst MSE | persistent decoders |
|:---|---:|---:|---:|---:|
| reference sweep | 0.018688 | 0.017385 | 0.072594 | 30 |
| individual stagnation | 0.017053 | 0.014466 | 0.062048 | 11 |
| always shared + exact LoRA mean fold | 0.018304 | 0.018256 | 0.036635 | **1 from birth** |
| warm-started mixed conditional | **0.012489** | **0.012585** | **0.024484** | 1 after 120k |

Always-shared exact folding beats the reference sweep by 2.1% in mean and
49.5% in worst-target MSE. It is 7.3% worse in mean than individual stagnation
but improves that arm's worst target by 41.0%. Its trajectory catches the
private-development arm by about 120k (`0.02071` mean versus `0.02106`) and
then flattens near `0.019` before ending at `0.01830`.

The coefficient-zero diagnostic identifies the limitation: mean MSE is
`0.056065`, versus adapted `0.018304`, with conditional RMS `65.18`. Exact
folding safely absorbs only the population-common LoRA component; most useful
variation is target-specific and therefore remains personal. Because the LoRA
directions are random at birth and never learned, re-centering cannot create
new conditional geometry. The next always-shared design should retain exact
mean folding but trigger transactional learned-direction refreshes or
target-vetted temporary decoder probes when progress stalls. That is the
missing dynamic analogue of the warm run's one-time factorization.

## Death-triggered LoRA legacy

The next always-shared arm replaces living-population mean folding with a
retirement event. A retirement is a member of the previous persistent
48-survivor population that is absent after selection; rejected newborns are
excluded. Retirees are softmax-weighted within their assigned target niche
using normalized niche quality, positive lost-coverage margin, and lifetime
offspring success. Each represented niche contributes one equally weighted
legacy gate, so lineage population cannot dominate. The resulting gate is
folded once into every backbone layer and subtracted from every living genome.

One retired champion per target is also retained as `(z, extra latent, LoRA
gate)`. Every subsequent fold is subtracted from these stored gates, keeping
the entire 32-slot legacy bank exactly reproducible through the current single
decoder.

A 20k strength calibration favored fully folding the batched retirement legacy:

| arm at 20k | mean MSE | worst MSE | base-only mean |
|:---|---:|---:|---:|
| living-mean fold | 0.04966 | 0.08853 | 0.06697 |
| retirement fold, 25% | 0.07510 | 0.13220 | 0.08971 |
| retirement fold, 100% | **0.04232** | **0.08732** | **0.06158** |

The full 600k live run performed 3,124 batched folds over 86,351 retired
incumbents (27.6 per generation, representing 13.0 niches on average). Maximum
living-phenotype drift from a fold was `1.25e-14` MSE. The bank filled all 32
targets and its stored scores remained reproducible to within `2.98e-7` MSE.

| full 600k output | mean MSE | median MSE | worst MSE |
|:---|---:|---:|---:|
| living-mean fold, final population | **0.018304** | **0.018256** | 0.036635 |
| retirement fold, final population | 0.018474 | 0.018484 | 0.035190 |
| retirement fold, reproducible legacy bank | 0.017067 | 0.017170 | **0.027741** |
| warm-started mixed conditional | 0.012489 | 0.012585 | 0.024484 |

Retirement weighting makes the final living population 0.9% worse in mean but
3.9% better on the worst target than living-mean folding, winning 15/32 paired
targets. The legacy bank is materially stronger: it beats its final living
population on 20/32 targets, improves mean by 7.6%, and improves worst-target
MSE by 21.2%. It essentially ties the prior 11-decoder individual-stagnation
run in mean (`0.017067` versus `0.017053`) while reducing its worst error by
55.3%, using one shared decoder throughout.

The bank result demonstrates that age-forced exploration was discarding useful
but still reproducible specialists. It does not yet isolate whether
fitness-weighted death folding improves the bank beyond simply retaining one
champion per target; that requires a no-fold legacy-bank ablation. Base-only
mean remains high at `0.055918`, so death folding also does not solve learned
conditional geometry. Its clean contribution is a natural, parallel event
clock plus durable niche-balanced memory.

## Historical all-personal-state reproductive species

This experiment was originally described as "genotype-only," but that name is
now retired because it hid three distinct kinds of individual state. It used
the 64-value `z`, 32 extra decoder inputs, and 32 LoRA gates together. The
shared decoder itself was never included. An RMS-distance graph was built in
that 128-value personal-state space, and connected components were
reproductive species. This made membership transitive: A could mate with C
when A--B--C was connected even if A and C were farther apart than the direct
radius.

Fitness retains all of its previous responsibilities. Normalized 32-target
fitness profiles choose ecologically similar mates *inside* a genotype
component, assign target roles, weight retirements, and drive target-covered
survivor selection. Thus genotype answers whether two lineages may mix;
fitness answers which allowed mixing is useful. Exact LoRA folding subtracts
the same gate offset from every genome, so it leaves every genotype pairwise
distance unchanged.

A first ablation that used genotype distance for both eligibility and uniform
mate choice failed: radii `0.85` through `1.15` ended around `0.0866`--`0.0929`
mean MSE at 20k. It had removed the useful fitness-assortative preference. The
transitive graph plus fitness preference restored it. Small genetic radii still
failed because successful mutation expands the genotype scale dramatically;
the old run's retired specialists had median pairwise distance near `60` at
20k. A fixed-radius sweep at 60k gave:

| genotype radius | components | sexual reproduction | mean MSE | worst MSE |
|---:|---:|---:|---:|---:|
| 40 | 23 | 62% | 0.022848 | 0.046233 |
| **50** | **19** | **67%** | **0.022796** | 0.042979 |
| 55 | 8 | 85% | 0.023337 | 0.044223 |
| 60 | 10 | 90% | 0.024157 | **0.041256** |
| prior fitness-only trajectory | 17 fitness components | 89% | 0.023956 | 0.042080 |

Radius `50` was the best balance and ran live for 600k. The genotype graph was
one component during early development, split during the discovery burst, had
19 components by 60k, and finished with 20. Final component sizes were
`[12, 5, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`; sexual
reproduction remained 74%. All 32 fitness-defined target roles stayed covered.

| full 600k output | mean MSE | median MSE | worst MSE |
|:---|---:|---:|---:|
| prior retirement fold, living | 0.018474 | 0.018484 | 0.035190 |
| prior retirement fold, bank | 0.017067 | 0.017170 | 0.027741 |
| **genotype species, living** | **0.015689** | **0.015164** | 0.027512 |
| **genotype species, bank** | **0.014998** | **0.015150** | **0.024772** |
| warm-started mixed conditional | 0.012489 | 0.012585 | 0.024484 |

Against the prior retirement-fold run, the new living population wins 22/32
targets and improves mean by 15.1% and worst by 21.8%. Its bank wins 25/32,
improving mean by 12.1% and worst by 10.7%. The living population also beats
the old 11-decoder individual-stagnation run by 8.0% in mean and 55.7% in worst
MSE, though the warm-factorized conditional run remains 25.6% better in mean.

The full run made 3,124 batched death folds over 91,252 retirees, representing
14.94 fitness niches per fold on average. Maximum living-phenotype drift was
`1.37e-14` MSE, and the 32-slot bank remained reproducible within `4.17e-7`
MSE. This is the strongest always-shared-from-birth result so far: genotype
speciation materially improves diversity and quality without reintroducing
private decoders.

## Death is succession, not decoder assimilation

The retirement-fold design conflated removal from the living population,
preservation of useful behavior, and transfer into the shared decoder. An
exact LoRA fold is function-preserving: it adds one common gate offset to the
backbone and subtracts it from every personal gate. It is therefore primarily
a change of coordinates, not evidence that the decoder learned what a dead
individual knew. The new death experiments separate these operations.

Several literal alternatives failed and clarified the requirements:

- Uniform species-local replacement stalled near `0.0921` mean at 20k. It
  turned the population into conservative local hill climbing and prevented
  the large discovery jump.
- Permanently protecting target champions while stopping reproduction at age
  20 froze all target quality at `0.02110` mean / `0.03592` worst from roughly
  135k through 345k. A champion without descendants cannot be superseded
  incrementally.
- Giving protected elders 5% reproductive weight avoided a perfectly flat
  trace but still effectively plateaued near `0.02175` by 120k.
- Removing age turnover entirely was slightly worse than hard age-10 turnover
  at 60k (`0.02608` versus `0.02575` mean). Turnover itself is useful; arbitrary
  loss of lineage and conflating death with decoder updates are the mistakes.

The successful policy is **descendant-restricted lineage succession**:

1. An adult reaching age 10 remains eligible to reproduce that generation.
2. If it has an evaluated child as primary parent or mate, its fitness-role
   seat is restricted to those descendants for the succession event.
3. If it has no evaluated descendant, it receives a one-generation reprieve.
4. The retired phenotype is saved exactly in the external target archive.
5. No LoRA value is folded and no decoder parameter changes because of death.

At 20k, lineage succession reached `0.03737` mean / `0.07823` worst, compared
with `0.04336` / `0.08013` for archive-only global death and `0.04224` /
`0.08925` for the prior fold-based genotype run. At 60k it reached `0.02394` /
`0.04549`, improving archive-only hard death by 7.0% in mean and 13.6% in
worst-target MSE.

The full 600k live run exhibited the intended exploration-with-memory
dynamics. Successions occasionally caused sharp living-population regressions;
for example worst MSE rose from `0.02876` to `0.03953` around 285k, then the
descendant recovered to `0.02773` by 345k. A second late succession had not
fully recovered at the exact budget boundary, making the external memory the
appropriate stable result.

| full 600k result | mean MSE | median MSE | worst MSE |
|:---|---:|---:|---:|
| lineage succession, living | 0.017130 | 0.017830 | 0.035690 |
| lineage succession, retired archive | 0.015813 | 0.016983 | 0.025827 |
| **lineage succession, living + archive** | **0.015812** | **0.016983** | **0.025827** |
| genotype species + death folding, living | 0.015689 | 0.015164 | 0.027512 |
| genotype species + death folding, memory | 0.014998 | 0.015149 | 0.024772 |
| old retirement-fold memory | 0.017067 | 0.017170 | 0.027741 |

Lineage memory wins 20/32 targets against the fold-based living population. It
is only 0.8% worse in mean and improves worst-target MSE by 6.1%. Against the
fold-based living-plus-bank memory it is 5.4% worse in mean and 4.3% worse in
worst; against the older non-genotype retirement memory it improves mean by
7.3% and worst by 6.9%.

The run performed 1,649 explicit target-lineage successions with only five
reprieves. Across 3,125 selection events, 90,672 persistent adults retired
(29.0 per generation). The 32-target archive stayed reproducible within
`5.36e-7` MSE, and the maximum fold magnitude was exactly zero. The final
population had 19 genotype components and 80% sexual reproduction.

Coefficient-zero mean MSE is high (`0.1011`) because archive-only retirement
does not continually move the coordinate origin into the backbone. This shows
that the old base-only diagnostic was partly gauge-dependent: exact folding
can improve it without learning or changing any adapted phenotype. Future
decoder learning should be evaluated through actual generalization or
compression, separately from death and succession.

## Crossover eligibility: `z` alone beats adding the extra latent

The compatibility API now names exactly what it measures:

- `z_only` uses only the original 64 values passed to the decoder.
- `decoder_input` uses `z` plus the 32 extra-latent values in mixed mode.
- The 32 LoRA gates are inherited and mutated, but never enter either
  compatibility distance.
- Fitness remains an ecological mate preference inside each transitive
  input-space component; it does not alter component membership.

This replaces the misleading `genotype` name and removes LoRA gates from
crossover eligibility. On the previous archive, median pairwise RMS distance
was `69.2` in `z_only`, `78.0` in `decoder_input`, and `83.0` in the retired
all-personal-state definition, so each of the two current spaces received its
own radius sweep rather than sharing an accidentally unequal graph.

| compatibility space | radius | components at 60k | memory mean MSE | memory worst MSE |
|:---|---:|---:|---:|---:|
| **`z_only`** | **30** | 25 | **0.023212** | 0.040958 |
| `z_only` | 40 | 18 | 0.024024 | **0.036091** |
| `z_only` | 50 | 16 | 0.024222 | 0.044509 |
| `z_only` | 60 | 7 | 0.023847 | 0.047257 |
| `decoder_input` | 30 | 25 | 0.024139 | **0.041365** |
| `decoder_input` | 40 | 16 | 0.023855 | 0.045082 |
| **`decoder_input`** | **50** | 13 | **0.023618** | 0.048441 |
| `decoder_input` | 60 | 11 | 0.023847 | 0.047257 |

The best `z_only` mean is 1.7% lower than the best `decoder_input` mean. Its
best worst-target result is also slightly better. Radius 30 was selected for
the long run because mean quality is the primary measure and radius 40's
better tail cost 3.5% in mean.

At 600k, `z_only` radius 30 reached `0.014158` memory mean and `0.027437`
memory worst MSE. Against the preceding lineage-succession run, mean improves
10.5% while worst regresses 6.2%. The final living graph has 26 components and
57% sexual reproduction. Tighter species are therefore a real
mean-versus-tail tradeoff, not an unqualified win, but the extra decoder input
did not improve crossover eligibility.

## Fitness-function count: a positive but noisy scaling signal

The first scaling study used nested, family-balanced target sets of
`4 ⊂ 8 ⊂ 16 ⊂ 32`, 60k phenotype evaluations, and evolution seeds 3,
4, and 5. The four targets present in every run were Catalina, Tree, china,
and Valley. The 8-target set added flower, Peak, Dome, and Big Sur; the
16-target set added one dark/light or viewpoint variant from the larger image
families; the 32-target set used the complete photo set. Population size,
decoder, compatibility, and evaluation budget were fixed.

| active fitness functions | active-set memory mean | seed SD | same core-four mean | seed SD |
|---:|---:|---:|---:|---:|
| 4 | 0.035209 | 0.004300 | 0.035209 | 0.004300 |
| 8 | 0.033306 | 0.003300 | 0.033771 | 0.001856 |
| 16 | 0.026106 | 0.003549 | 0.034861 | 0.005464 |
| **32** | **0.025138** | **0.001986** | **0.031639** | **0.002430** |

The active-set column is partly confounded by which images enter at each
scale. The paired core-four comparison is the important result: all three
seeds improve from 4 to 32 functions (`-5.2%`, `-10.7%`, and `-13.5%`), for a
10.1% mean improvement. Core-four worst MSE improves 4.5%. Every core target
improves in the three-seed average:

| target | 4 functions | 32 functions | change |
|:---|---:|---:|---:|
| Catalina | 0.030798 | 0.025253 | -18.0% |
| Tree | 0.030374 | 0.027100 | -10.8% |
| china | 0.044537 | 0.042555 | -4.5% |
| Valley | 0.035126 | 0.031647 | -9.9% |

A power fit to the four core means gives an exponent near `-0.042`, or roughly
4% lower MSE per doubling, but the fit is weak (`R² ≈ 0.60`) and the 16-target
point is worse than the 8-target point. This is evidence for a scaling effect,
not yet a scaling law. Crucially, archive-only death does not train the shared
decoder, so the benefit must currently come from multi-objective population
structure: more niches preserve stepping stones, alter crossover pathways,
and improve search even for objectives already present.

For target counts larger than the survivor population, the next design should
not score every objective on every generation. Keep one persistent archive
slot per target, then activate a bounded panel made from the currently weakest
targets, the stalest targets, and a random rotating coverage sample. Selection
and ecological profiles use the active panel; archived champions preserve
inactive targets; results are normalized by target exposures as well as
phenotype evaluations. Before that larger rotating experiment, the 4/8/16/32
result should be repeated on a genuinely diverse stratified image dataset,
because these 32 scenic photos contain correlated variants and may overstate
the value of related auxiliary objectives.

## Fixed-compute scaling to 256 diverse images

The larger study implements the rotating design instead of increasing the
population. It uses a deterministic nested CIFAR-100 sequence: one image from
each shuffled class is selected before any class receives a second image.
Every run retains the same one mixed-64 shared decoder, 48 survivors, 192
children, `z_only` compatibility, age-10 lineage succession, and phenotype
budget. At most 32 targets are active at once.

For more than 32 targets, a balanced scheduler always chooses targets with the
fewest prior panel appearances and randomizes only ties. The best personal
state ever found for every target remains in a global archive. When a target
returns, its archived state can re-enter the living candidate pool. Panels
last eight generations. Thus every ordinary generation performs no more than
the old 32-target score work, while inactive objectives retain exact memory
and later resume rather than restart.

Three evolution seeds were run for 30k phenotype evaluations at
`4, 8, 16, 32, 64, 96, 128, 168, 256` targets: 810k primary evaluations in
total. All targets were visited and filled in every result.

| total targets | archive mean MSE | seed SD | shared first-four MSE | seed SD | mean first-four exposures |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.044086 | 0.001491 | 0.044086 | 0.001491 | 30,000 |
| **8** | **0.031329** | 0.001390 | **0.034517** | 0.002625 | 30,000 |
| 16 | 0.036680 | 0.002624 | 0.038768 | 0.004334 | 30,000 |
| 32 | 0.037796 | 0.004363 | 0.035314 | 0.004277 | 30,000 |
| 64 | 0.042462 | 0.004262 | 0.043586 | 0.004399 | 15,012 |
| 96 | 0.044092 | 0.001927 | 0.049923 | 0.001798 | 10,188 |
| 128 | 0.041922 | 0.000109 | 0.048932 | 0.002992 | 7,308 |
| 168 | 0.043350 | 0.000699 | 0.050889 | 0.001769 | 5,580 |
| 256 | 0.050368 | 0.007737 | 0.066326 | 0.016349 | 3,276 |

There are two different scaling regimes:

1. **Positive transfer, 4--32 targets.** Eight targets improve the shared
   first four by 21.7% versus training those four alone. Thirty-two retains a
   19.9% improvement. The earlier scenic-photo result therefore was not just
   duplication among near-identical variants; diverse CIFAR targets show the
   same multi-objective stepping-stone effect.
2. **Fixed-compute dilution, beyond 32.** Once panels rotate, quality declines
   as each objective receives a smaller share of the fixed budget. From 32 to
   256, shared-core MSE follows approximately `N^0.269` (`R² = 0.921`). This
   is a clear compute frontier. At 256, the core is 50.4% worse than at four
   targets and seed variance expands sharply.

The dilution result should not be confused with poor sample efficiency. At
168 targets, each core objective receives only 5,580 new-phenotype exposures,
yet reaches `0.050889` MSE. The isolated four-target trajectory is near
`0.11255` at the same exposure. Access to the other objectives therefore cuts
exposure-matched core MSE by 54.8%. Even 256 targets improve exposure-matched
core MSE by 42.4%. More objectives are useful teachers/stepping stones; the
30k total budget simply cannot exploit all of them equally.

Panel cadence does not explain the 168-target result:

| generations per panel | archive mean MSE | shared first-four MSE | worst MSE | refresh decodes |
|---:|---:|---:|---:|---:|
| 4 | **0.042609** | **0.048786** | 0.120756 | 2,820 |
| **8** | 0.043350 | 0.050889 | **0.119299** | 1,344 |
| 16 | 0.043586 | 0.051241 | 0.134934 | **571** |

Four-generation blocks slightly improve the mean but more than double archive
refresh work. Sixteen-generation blocks leave hard targets stale too long.
Eight remains the mean/tail/compute compromise. After this measurement, the
runner was optimized to reuse living phenotypes at panel switches and decode
only genuinely re-entering archived states.

Finally, the fixed-compute fit predicted that roughly 1.7 times more search
would recover four-target quality at 168 targets. A direct three-seed 60k run
confirmed and exceeded that prediction:

| result | total archive mean | shared first-four mean | mean target exposure |
|:---|---:|---:|---:|
| 4 targets, 30k | 0.044086 | 0.044086 | 30,000 |
| 168 targets, 30k | 0.043350 | 0.050889 | 5,580 |
| **168 targets, 60k** | **0.037916** | **0.039826** | **12,300** |

The 168-target core beats the isolated four-target core in every paired seed
at 60k (`-16.4%`, `-3.3%`, `-9.6%`), averaging 9.7% better despite receiving
only 41% as many direct exposures per target. At equal 12.3k exposure, it is
26.2% better. The practical result is that 168 objectives are viable with
only a 2x total phenotype budget, not the 5.25x budget that equal per-target
exposure with a 32-wide panel would require. The remaining scaling bottleneck
is scheduling finite attention, not decoder expressiveness.

## Exact matched-exposure transfer on 32 shared targets

The shared-first-four analysis above was useful but underpowered, and its
aggregate exposure interpolation did not measure the question precisely. The
benchmark now records each target's best-so-far MSE at exact target-local
fitness-evaluation milestones. When a candidate batch crosses a milestone,
only the candidate prefix visible by that exact count contributes to the
recorded quality. Panel-activation comparisons are counted as exposures too.

The exact experiment holds the same first 32 CIFAR targets fixed as anchors in
every condition, varies total fitness functions across
`32, 64, 96, 128, 168, 256`, and repeats seeds `3, 4, 5`. Quality is computed
per target as the percentage of that target's own initial MSE removed, then
paired and averaged across the 32 anchors. This removes target-set difficulty
from the comparison and avoids relying on a four-image mean.

At exactly 3,000 fitness evaluations per anchor target:

| total fitness functions | initial target error removed | paired gain vs 32 | 95% target-bootstrap CI | anchors better than 32 |
|---:|---:|---:|---:|---:|
| 32 | 0.271% | +0.000 pp | `[+0.000, +0.000]` | -- |
| 64 | 0.538% | +0.267 pp | `[+0.157, +0.396]` | 31/32 |
| 96 | 4.158% | +3.887 pp | `[+1.951, +6.166]` | 30/32 |
| 128 | 13.203% | +12.932 pp | `[+9.179, +17.116]` | 32/32 |
| 168 | 24.361% | +24.090 pp | `[+17.914, +30.733]` | 32/32 |
| 256 | 29.104% | +28.833 pp | `[+22.221, +35.569]` | 32/32 |

Every adjacent increase in objective count also has a positive paired
target-bootstrap interval at 3,000 exposures: `32→64 +0.267 pp`,
`64→96 +3.620 pp`, `96→128 +9.045 pp`, `128→168 +11.158 pp`, and
`168→256 +4.743 pp`. The transfer effect is therefore monotonic in this sweep,
not an artifact of the four original targets. It becomes large after roughly
96 total objectives.

This answers target-local sample efficiency, not total-compute efficiency.
Larger conditions perform more evolutionary work on other targets between two
evaluations of an anchor; that intervening work is the treatment whose transfer
is being measured. The conclusion is precise: after the same fitness function
has evaluated the same number of candidate phenotypes, it has improved much
more when the population also evolves against many other objectives.
