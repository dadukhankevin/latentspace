# Mixed decoder-objective study on MPS

Follow-up study run on 2026-07-14 using an Apple M3 Pro, PyTorch 2.6.0, five
seeds (`0..4`), and exactly 5,000 objective evaluations per reported result.
All 240 runs in this follow-up verified that decoder parameters were exclusively
on `mps`. Lower values are better.

The six component objectives were bottom-to-top, good-to-best, each-to-next,
contrastive worst-negative, REINFORCE, and advantage-weighted regression. A
micro-batch always used one objective; losses were never numerically averaged.
RL rollouts and guard probes consumed the common evaluation budget.

## Naive schedules

| Schedule | Target MSE | Rastrigin | TSP length |
|---|---:|---:|---:|
| Random one-at-a-time | 0.05216 | 16.786 | 3.611 |
| Round-robin | 0.04611 | 9.614 | 3.601 |
| Shuffled cycle | 0.05103 | 11.913 | 3.608 |
| Random non-RL | 0.06348 | 9.457 | 3.676 |
| Random three micro-batches | 0.02091 | 22.707 | 3.721 |
| Shuffled three micro-batches | 0.02142 | 15.110 | 3.760 |

Naive mixing did not combine the specialists' strengths. The apparent target
gain from three random micro-batches was an intensity effect: three dedicated
REINFORCE steps reached `0.00445`, and three AWR steps reached `0.00647`. The
same increased intensity was strongly harmful on Rastrigin and TSP.

## Adaptive guarded schedules

The guard snapshots decoder and optimizer state, proposes an update, and scores
the same top 25% latent probes before and after. It accepts when either probe
mean or probe best improves; otherwise it restores the proposal. Candidate
evaluations are retained as real search work and count against the budget.

| Guarded schedule | Target MSE | Rastrigin | TSP length |
|---|---:|---:|---:|
| Uniform random, all objectives | 0.05216 | 6.771 | 3.381 |
| Shuffled cycle, all objectives | 0.05103 | 5.619 | 3.521 |
| Uniform random, non-RL | 0.06348 | **4.506** | **3.308** |
| Random three, all objectives | **0.01899** | 6.694 | 3.522 |
| Cost-aware random, all objectives | 0.06061 | 6.482 | 3.317 |
| Frozen reference | 0.07306 | 8.360 | 3.524 |

Values are five-seed means. Standard deviations for guarded random non-RL were
`0.01003`, `2.191`, and `0.2867`; for cost-aware all-objective they were
`0.01073`, `1.515`, and `0.2946`.

## What the gate learned

- On the smooth target, guarded non-RL accepted all 35 proposals. Training
  directions are broadly aligned there, but dedicated REINFORCE remains the
  clear specialist.
- On Rastrigin, it accepted only 8 of 35 proposals. Different seeds retained
  different combinations of contrastive, bottom-to-top, each-to-next, and even
  good-to-best updates. Its `4.506` mean beats the prior best single objective
  (`6.499` each-to-next), decoder-weight ES (`7.595`), and direct GA (`7.608`)
  in this preliminary five-seed suite.
- On TSP, it accepted only 4 of 35 proposals, mainly contrastive moves. The
  `3.308` mean improves the previous latent best (`3.428`) but remains behind
  differential evolution (`3.129`) and the exact `2.994745` reached by the
  representation-specific direct GA/ES.
- Guarded non-RL is the first single configuration in this study to improve all
  three frozen means. The cost-aware all-objective mixer also improves all three
  while retaining a nonzero chance of selecting either RL rule.

The gain comes from both rejecting destructive representation changes and
allocation: a rejected proposal spends only the 16 elite probe evaluations,
whereas accepting a changed decoder requires re-evaluating the remaining
population. This is fair under the exact objective budget, but it is part of the
algorithm—not evidence that mixed gradient losses alone are synergistic.

Five seeds and three small objectives are not enough for a broad claim. The
result does support a concrete Finch 4 direction: trainers should be competing
proposal generators behind a fitness-based acceptance gate, rather than one
fixed loss or an unconditional blend.

Raw runs and schedules:

- [`mps_mixed_training_5000.json`](mps_mixed_training_5000.json)
- [`mps_mixed_intensity_controls_5000.json`](mps_mixed_intensity_controls_5000.json)
- [`mps_guarded_mixed_training_5000.json`](mps_guarded_mixed_training_5000.json)
- [`mps_guarded_cost_aware_training_5000.json`](mps_guarded_cost_aware_training_5000.json)
- [`mps_adaptive_mixed_training_5000.md`](mps_adaptive_mixed_training_5000.md)
