# Initial MPS strategy comparison

Preliminary benchmark run on 2026-07-14 using an Apple M3 Pro, PyTorch 2.6.0,
five seeds (`0..4`) and exactly 5,000 objective evaluations per reported result.
All 45 neural runs verified that decoder parameters were on `mps`; direct
non-neural methods used NumPy on CPU. Lower values are better.

| Objective | Strategy | Mean | Standard deviation |
|---|---|---:|---:|
| Target MSE | Differential evolution | 0.00001584 | 0.00001138 |
| Target MSE | Direct GA | 0.00004916 | 0.00000669 |
| Target MSE | Direct `(mu + lambda)` ES | 0.00021572 | 0.00003616 |
| Target MSE | Random search | 0.035814 | 0.006548 |
| Target MSE | Latent gradient | 0.055069 | 0.010055 |
| Target MSE | Latent fixed | 0.073059 | 0.006944 |
| Target MSE | Latent decoder ES | 0.074150 | 0.007287 |
| Rastrigin | Latent decoder ES | 7.5948 | 3.6635 |
| Rastrigin | Direct GA | 7.6077 | 2.6487 |
| Rastrigin | Latent fixed | 8.3602 | 3.0552 |
| Rastrigin | Latent gradient | 9.0992 | 6.0277 |
| Rastrigin | Direct `(mu + lambda)` ES | 36.946 | 20.700 |
| Rastrigin | Differential evolution | 90.823 | 4.569 |
| Rastrigin | Random search | 160.444 | 7.284 |
| TSP tour length | Direct GA | 2.994745 | 0.00000024 |
| TSP tour length | Direct `(mu + lambda)` ES | 2.994745 | 0.00000021 |
| TSP tour length | Differential evolution | 3.1293 | 0.3009 |
| TSP tour length | Latent decoder ES | 3.4296 | 0.2641 |
| TSP tour length | Latent fixed | 3.5238 | 0.4188 |
| TSP tour length | Latent gradient | 3.7712 | 0.2866 |
| TSP tour length | Random search | 3.9841 | 0.1653 |

The exact optimum of this 12-city TSP instance is `2.994744753`, calculated
separately with Held-Karp dynamic programming. The direct GA and direct ES found
it in every seed; differential evolution found it in four of five. The fixed
latent and decoder-ES variants each found it in one seed.

## Initial reading

- Direct search decisively wins the low-dimensional smooth target. Gradient
  refinement improves the fixed latent result by about 25%, but all latent
  variants remain behind random phenotype search at this budget.
- Rastrigin is the encouraging case. Decoder-weight ES and the direct GA are
  effectively tied on the five-seed mean, and all latent variants substantially
  beat the simple direct ES. Five seeds are not enough to claim a real advantage.
- Decoder self-distillation is problem-dependent: helpful on the smooth target,
  unstable on Rastrigin, and harmful on TSP with this configuration.
- Decoder-weight ES modestly improves the fixed decoder on Rastrigin and TSP,
  but not on target matching.
- Neural variants take roughly 0.25–0.40 seconds per run versus milliseconds for
  the simple direct methods on these extremely cheap objectives. Objective count
  is controlled, but compute cost is not equal.

These results do not support a broad superiority claim yet. They suggest that the
next tests should emphasize high-dimensional or structured phenotypes, where a
compressed learned representation has a plausible reason to earn its overhead.

Raw configuration, per-seed runs and aggregate values are in
[`mps_initial_5000.json`](mps_initial_5000.json).
