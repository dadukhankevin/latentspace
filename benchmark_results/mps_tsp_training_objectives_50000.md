# TSP decoder training: objectives, learning rates, and optimizers

Run on 2026-07-14 using an Apple M3 Pro and PyTorch 2.6.0. Every neural
decoder was verified exclusively on `mps`. The problem is the same 24-city
Euclidean TSP instance (`instance_seed=2026`), with exactly 50,000 objective
evaluations used for every reported result. Fitness probes used by
backtracking count against that budget.

The latent search configuration is fixed at the strongest current 24-city
recipe: 32 float genes, population and offspring 64, one 128-unit hidden layer,
mutation rate 0.1 with sigma 0.25, and eight-point crossover. Trainable
decoders update every 10 generations.

## Full matrix

The initial five-seed matrix tested four objectives at network learning rates
`3e-5`, `1e-4`, `3e-4`, and `1e-3` under standard Adam:

- raw-key good-to-best MSE;
- pairwise permutation ordering;
- permutation ordering with elite-output anchoring;
- anchored permutation ordering with backtracking factors
  `{1, 0.5, 0.25, 0.125}`.

The best objective from that stage was then tested with Adam `beta2=0.95` and
SGD with momentum 0.9 over the same learning-rate grid. Direct GA and frozen
latent baselines were included. The complete matrix contained 130 runs and
finished in 307.6 seconds.

The apparent five-seed winner was raw MSE with SGD momentum at `3e-4`, but its
gain was driven mostly by one seed. A second run added seeds 5 through 19 for
the three finalists and both baselines.

## Confirmed 20-seed comparison

Lower tour length is better.

| Method | Mean | SD | Mean gain vs frozen | Paired W/T/L |
|:---|---:|---:|---:|---:|
| Direct GA | **4.233** | 0.371 | 1.080 | 19/0/1 |
| Permutation + anchor + backtracking, Adam `3e-5` | **5.179** | 0.622 | **0.133** | 11/2/7 |
| Frozen latent | 5.312 | 0.596 | — | — |
| Raw MSE, SGD momentum `3e-4` | 5.316 | 0.755 | -0.003 | 10/2/8 |
| Raw MSE, Adam `1e-4` | 5.351 | 0.641 | -0.038 | 10/1/9 |

The two raw-MSE finalists regress to essentially the frozen mean over 20 seeds.
The permutation-aware backtracking trainer retains a 0.133 improvement (2.5%)
and closes 12.3% of the frozen-to-direct gap, but the paired 95% confidence
interval is `[-0.066, 0.332]` and therefore still includes no effect. This is a
promising signal, not a confirmed win.

Across the 20 backtracking runs, 739 decoder updates were proposed. The full
step was accepted 222 times, a smaller step 193 times, and all tested steps were
rejected 324 times. In total, 56.2% of proposals retained at least a partial
update. This confirms that update-size adaptation is doing real work: many
useful gradient directions are too large at their original step size.

## Conclusion

- Learning rate and optimizer choices can look decisive in five seeds but did
  not generalize over 20 matched seeds.
- Raw-key MSE is not a reliable improvement over a frozen decoder on 24-city
  TSP.
- Permutation-aware training plus anchoring and backtracking is the only tested
  configuration with a persistent positive mean signal, though more seeds or a
  stronger objective are needed before treating it as the default.
- Direct GA remains substantially better: the confirmed latent gap is 0.947 in
  tour length, or about 22.4% relative to direct GA.

Raw results are in `mps_tsp_training_objectives_50000.json` and
`mps_tsp_training_objectives_confirm_50000.json`.
