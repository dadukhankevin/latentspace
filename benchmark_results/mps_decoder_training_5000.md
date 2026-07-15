# MPS decoder-training comparison

Preliminary study run on 2026-07-14 using an Apple M3 Pro, PyTorch 2.6.0,
five seeds (`0..4`), and exactly 5,000 objective evaluations per reported
result. All 105 runs verified that decoder parameters were exclusively on
`mps`. Lower values are better.

The latent GA, MLP architecture, initialization-by-seed, update cadence, and
objectives were fixed. Only the decoder trainer changed. RL policy samples were
recorded as objective evaluations, so REINFORCE and advantage-weighted
regression received no free rollouts.

| Trainer | Target MSE | Rastrigin | TSP tour length |
|---|---:|---:|---:|
| Frozen | 0.07306 ± 0.00694 | 8.360 ± 3.055 | 3.524 ± 0.419 |
| Bottom-to-top | 0.05507 ± 0.01005 | 9.099 ± 6.028 | 3.771 ± 0.287 |
| Good-to-best | 0.06558 ± 0.01182 | 11.380 ± 2.470 | 3.548 ± 0.474 |
| Each-to-next | 0.07123 ± 0.00784 | **6.499 ± 3.026** | 4.077 ± 0.319 |
| Contrastive worst-negative | 0.06273 ± 0.01431 | 7.738 ± 4.266 | **3.428 ± 0.270** |
| REINFORCE | **0.03066 ± 0.00619** | 12.727 ± 5.898 | 3.901 ± 0.208 |
| Advantage-weighted | 0.03450 ± 0.00656 | 19.337 ± 10.819 | 3.663 ± 0.407 |

Values are mean ± sample standard deviation. The frozen control completed 39
generations, distillation/contrastive trainers 36, and RL trainers 33. That is
the intended consequence of charging decoder re-evaluation and RL exploration
to the common fitness budget.

## Reading the result

- REINFORCE reduced target MSE by 58% versus frozen and won the paired
  comparison in all five seeds. Advantage-weighted regression was second and
  also won all five. Both now slightly beat random phenotype search from the
  initial study, but remain far behind direct GA/ES on this small smooth task.
- Each-to-next reduced mean Rastrigin loss by 22% versus frozen and won all five
  paired seeds. Its 6.499 mean is also better than the initial direct-GA (7.608)
  and decoder-weight-ES (7.595) means, although five seeds are not enough for a
  superiority claim.
- Contrastive worst-negative training produced the best TSP mean and essentially
  tied the earlier decoder-weight ES result. The paired evidence is weak: two
  wins, two losses, and one tie versus frozen. Direct representation-specific
  methods still reach the exact optimum reliably.
- Good-to-best is too collapse-prone as a general rule. The naive Gaussian RL
  rules are similarly problem-dependent: strong on the smooth target, but poor
  on multimodal Rastrigin and discontinuous random-key TSP.

There is no universal trainer in this first pass. The useful next design is a
guarded or adaptive update: propose a decoder change with one of these trainers,
then accept, shrink, or roll it back using fitness on a held-out elite set. For
permutation problems, a permutation-aware stochastic decoder is more promising
than asking Gaussian output noise to discover rank swaps.

Raw configuration, per-seed runs, timings, and aggregate values are in
[`mps_decoder_training_5000.json`](mps_decoder_training_5000.json). The broader
baseline comparison is in [`mps_initial_5000.md`](mps_initial_5000.md).
