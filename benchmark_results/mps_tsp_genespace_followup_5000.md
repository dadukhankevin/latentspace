# TSP follow-up: elitism and the GeneSpace recipe

Run on 2026-07-14 using an Apple M3 Pro, PyTorch 2.6.0, five algorithm seeds
(`0..4`), and exactly 5,000 objective evaluations per reported result. Every
neural decoder was verified exclusively on `mps`. All methods use the same
nested Euclidean TSP instances (`instance_seed=2026`). Lower is better.

This follow-up tests the implementation differences found in the original
GeneSpace repository:

- true offspring-only mutation, preserving incumbent parents;
- 250-dimensional binary genes;
- shallow MLP decoders with widths 128, 512, and 2,000;
- the scale-1 GeneSpace recipe: population 200, width 2,000, exponential rank
  pressure 20, eight-point crossover, and 10% bit flips;
- guarded decoder updates every 20 generations rather than every five.

## Results

| Cities | Direct GA | Previous Finch | Elitist Finch | Best binary frozen* | Compact guarded/20 |
|---:|---:|---:|---:|---:|---:|
| 8  | 2.711 | 2.711 (+0.0%) | 2.745 (+1.2%) | 2.711 (+0.0%) | 2.745 (+1.2%) |
| 12 | 2.995 | 3.308 (+10.5%) | 3.723 (+24.3%) | 3.418 (+14.1%) | 3.896 (+30.1%) |
| 16 | 3.703 | 4.474 (+20.8%) | 4.348 (+17.4%) | 4.518 (+22.0%) | 4.306 (+16.3%) |
| 24 | 3.994 | 7.059 (+76.7%) | 7.197 (+80.2%) | 7.288 (+82.5%) | 7.163 (+79.3%) |
| 32 | 6.312 | 9.998 (+58.4%) | 10.454 (+65.6%) | 10.788 (+70.9%) | 10.126 (+60.4%) |

*The best binary frozen entry is selected post hoc at each size from widths
128/512/2,000 and the scale-1 recipe; it is not one consistent strategy.

Offspring-only mutation raised Finch from roughly 38 to 74 breeding generations
within the same budget, versus 78 for the direct GA. Despite that, it improved
only the 16-city mean and hurt the other sizes. Whole-population mutation was
therefore supplying valuable diversity, not merely wasting evaluations. It
remains the default; offspring-only mutation is retained as an explicit
ablation switch.

Binary 250-dimensional latents and a wide shallow decoder helped at 12 cities,
where width 2,000 reached 3.418 instead of the corrected compact frozen result
of 3.896. The advantage disappeared at 16 cities and reversed at 24 and 32.
Increasing decoder width therefore adds capacity but does not add the missing
tour structure.

Rare guarded training helped the compact representation modestly at 16 cities
(4.442 frozen to 4.306), but was neutral or harmful elsewhere. The original
GeneSpace ingredients do not close the scaling gap.

## Conclusion

The remaining failure is structural. A mutation in direct random-key space can
change one city's priority locally. A mutation in a dense latent code propagates
through the MLP and can reorder many cities at once; neither more decoder width,
binary genes, elitism, nor rarer training creates edge or permutation locality.

The next credible experiment is a custom TSP decoder that consumes city
coordinates and produces geometry-aware city or edge scores, ideally with a
deterministic repair or 2-opt refinement stage. Decoder-training strategies
should be revisited only after that representation exists.

Raw results:

- `mps_tsp_scaling_5000.json` — previous whole-population mutation result;
- `mps_tsp_scaling_elitist_5000.json` — offspring-only mutation;
- `mps_tsp_genespace_ablation_5000.json` — frozen GeneSpace recipe variants;
- `mps_tsp_rare_training_5000.json` — guarded updates every 20 generations.
