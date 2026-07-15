# TSP size scaling: Finch versus direct GA

Run on 2026-07-14 using an Apple M3 Pro, PyTorch 2.6.0, five algorithm seeds
(`0..4`), and exactly 5,000 objective evaluations per reported result. The
Finch decoder was verified exclusively on `mps`; the direct GA used NumPy on
CPU. Lower tour lengths are better.

The Euclidean TSP instances use one nested random family (`instance_seed=2026`):
larger instances append cities to the smaller instances. Finch used the same
guarded random non-RL trainer, 32-dimensional latent, population 64, and MLP
configuration at every size. The direct GA was selected before this study
because it tied for the strongest conventional result in the 12-city baseline.

| Cities | Direct GA mean ± sd | Finch mean ± sd | Finch gap vs GA | Exact optimum |
|---:|---:|---:|---:|---:|
| 8 | 2.7114 ± 0.0000 | 2.7114 ± 0.0000 | 0.0% | 2.7114 |
| 12 | 2.9947 ± 0.0000 | 3.3082 ± 0.2867 | +10.5% | 2.9947 |
| 16 | 3.7034 ± 0.2972 | 4.4737 ± 0.5641 | +20.8% | 3.3584 |
| 24 | 3.9942 ± 0.1009 | 7.0592 ± 0.3454 | +76.7% | — |
| 32 | 6.3122 ± 0.3940 | 9.9976 ± 0.6223 | +58.4% | — |

Held-Karp dynamic programming supplies exact references through 16 cities. Both
methods found the exact 8-city optimum in every seed. The direct GA found the
12-city optimum in every seed and the 16-city optimum in one seed; Finch found
the 12-city optimum in two seeds and no 16-city optimum.

## Reading the scaling result

- Finch does not scale competitively on this random-key TSP representation at a
  fixed 5,000-evaluation budget. Its mean gap to the direct GA grows from zero
  at 8 cities to 10.5%, 20.8%, and then roughly 60–77% at 24–32 cities.
- The normalized tour length (`length / sqrt(cities)`) stays roughly flat for
  the direct GA through 24 cities but rises sharply for Finch after 16 cities.
  That indicates representation/search degradation, not merely the geometric
  fact that longer instances have longer tours.
- Finch took roughly 0.29–0.42 seconds per run versus 0.008–0.013 seconds for
  the direct GA, a 29–55× wall-clock ratio on these very cheap objectives.
- This is a useful negative result for the universal-decoder claim. The decoder
  emits independent random keys and has no permutation or local-edge inductive
  bias, while direct crossover and mutation act directly on those keys. A
  permutation-aware decoder or tour-local refinement is the credible next TSP
  direction; trainer changes alone are unlikely to close this scaling gap.

This study holds total evaluations fixed to measure sample efficiency. A
separate evaluations-per-city study would answer a different question: how much
additional budget each method needs to maintain solution quality as size grows.

Raw per-seed runs, exact references, timing, normalized values, and relative
gaps are in [`mps_tsp_scaling_5000.json`](mps_tsp_scaling_5000.json).
