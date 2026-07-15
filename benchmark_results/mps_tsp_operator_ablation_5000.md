# TSP operator study: float, binary, and GeneSpace schedules

Run on 2026-07-14 using an Apple M3 Pro, PyTorch 2.6.0, five algorithm seeds
(`0..4`), exactly 5,000 objective evaluations per result, and the same nested
Euclidean TSP instances (`instance_seed=2026`). Every neural decoder was frozen
to isolate genetic search mechanics and verified exclusively on `mps`.

The study contains 25 controlled variants and 625 neural runs. Except for the
explicit scale-1 legacy diagnostic, population is 64, offspring is 64, decoder
width is 128, and selection pressure is fixed. Lower tour length is better.

## Matched float versus binary

Each entry is the best mutation strength tested for that representation at that
size, selected post hoc. Float sigma used `{0.05, 0.12, 0.25}`; binary bit-flip
rate used `{0.02, 0.05, 0.10}`. All use one shallow hidden layer and eight-point
crossover.

| Cities | Float-32 | Binary-32 | Float-250 | Binary-250 |
|---:|---:|---:|---:|---:|
| 8  | 2.711 | 2.745 | 2.711 | 2.729 |
| 12 | 3.334 | 3.489 | **3.188** | 3.454 |
| 16 | 4.156 | 4.674 | **3.922** | 4.346 |
| 24 | **6.191** | 7.565 | 6.208 | 7.245 |
| 32 | **9.166** | 10.992 | 9.488 | 10.254 |

Float genes outperform binary genes at both matched latent lengths beyond the
trivial eight-city case. A 250-dimensional float latent is strongest at 12–16
cities; the shorter 32-dimensional float latent is slightly better at 24–32.

For float genes, sigma `0.25` is generally strongest through 24 cities, while
`0.12` wins at 32. For binary-250, the gentlest tested rate (`0.02`, about five
flips per individual) is best from 12 cities onward. The original `0.05–0.10`
rates flip roughly 12.5–25 bits and are usually too disruptive.

## Schedule comparison: float-32

These rows fix float-32, sigma `0.12`, one shallow 128-wide layer, and eight-point
crossover.

| Cities | Mutate whole population | Mutate children only | Two-stage, no replacement | Two-stage, replacement |
|---:|---:|---:|---:|---:|
| 12 | **3.621** | 3.658 | 3.869 | 3.785 |
| 16 | **4.380** | 4.391 | 5.014 | 4.984 |
| 24 | **6.628** | 6.983 | 7.445 | 7.448 |
| 32 | 9.166 | **8.889** | 10.265 | 10.229 |

The current one-stage whole-population mutation is strongest through 24 cities.
Offspring-only elitism becomes useful at 32, but the clean two-stage GeneSpace
schedule consistently hurts float search.

## Schedule comparison: binary-250

These rows fix binary-250, bit-flip rate `0.05`, one shallow 128-wide layer, and
eight-point crossover.

| Cities | Mutate whole population | Mutate children only | Two-stage, no replacement | Two-stage, replacement | Legacy aliases |
|---:|---:|---:|---:|---:|---:|
| 12 | 3.660 | **3.224** | 3.315 | 3.315 | 3.712 |
| 16 | 4.782 | 4.399 | **3.962** | 4.248 | 4.818 |
| 24 | 7.690 | 7.578 | 7.049 | **6.801** | 8.059 |
| 32 | 11.234 | 10.601 | **10.339** | 10.405 | 11.339 |

Here the GeneSpace sequencing idea is real: evaluating/capping crossover
children before producing independent mutation children materially improves
binary search. Whether mutation parents are sampled with replacement is not
consistent across sizes. The clean copies matter; faithfully reproducing the
old duplicate-reference behavior is worse everywhere and reduces the run from
39 to about 28 generations. The full population-200 scale-1 legacy recipe gets
only about 12 generations and is worse again.

## Crossover and forced mutation

Four versus eight crossover points is a secondary effect. Four points are
slightly better for float search at 12–24 cities, while eight points are better
for binary two-stage search at every nontrivial size. Neither choice changes the
scaling conclusion.

Allowing a selected individual to receive zero mutations, as old GeneSpace did,
has mixed results and no reliable advantage. For binary-250 at practical flip
rates, a zero-flip event is already extremely unlikely.

## Best observed latent result versus direct GA

The latent entry is the best of all 25 variants selected separately at each
problem size, so it is an optimistic post-hoc envelope rather than one method.

| Cities | Direct GA | Best latent | Gap | Best latent variant |
|---:|---:|---:|---:|:---|
| 8  | 2.711 | 2.711 | 0.0% | several |
| 12 | 2.995 | 3.188 | +6.4% | float-250, sigma 0.12, one-stage |
| 16 | 3.703 | 3.922 | +5.9% | float-250, sigma 0.25, one-stage |
| 24 | 3.994 | 6.191 | +55.0% | float-32, sigma 0.25, one-stage |
| 32 | 6.312 | 8.889 | +40.8% | float-32, offspring-only |

This is a meaningful improvement over the earlier Finch gaps of 10.5%, 20.8%,
76.7%, and 58.4% at 12/16/24/32 cities. Operator and representation choices
nearly close the small/medium cases, but a very large gap remains at 24–32 even
after choosing the best variant after seeing the results.

## Decision

- Keep float genes as the general default.
- Keep whole-population one-stage mutation as the float default; retain
  offspring-only and two-stage schedules as explicit options.
- Use the clean two-stage schedule when experimenting with binary genespaces.
- Do not reproduce GeneSpace's in-place alias behavior.
- Treat crossover-point count and replacement as tunable, secondary details.

The remaining large-instance failure is still representational: a dense MLP
mutation can globally reorder city priorities and contains no edge or geometric
locality. The next high-value experiment remains a coordinate-aware,
permutation-aware decoder rather than another generic operator sweep.

Raw results are in `mps_tsp_operator_ablation_5000.json` and
`mps_tsp_operator_supplement_5000.json`.
