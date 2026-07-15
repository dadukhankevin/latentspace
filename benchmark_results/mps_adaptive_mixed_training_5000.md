# Adaptive no-rollback objective allocation on MPS

Study run on 2026-07-14 using an Apple M3 Pro, PyTorch 2.6.0, five seeds
(`0..4`), and exactly 5,000 objective evaluations per reported result. All 90
runs verified that decoder parameters were exclusively on `mps`. Lower values
are better.

This tests a non-stationary bandit interpretation of decoder training. Each
objective is an arm. After each kept micro-batch, the same elite probe is scored
and the arm receives a normalized combination of probe-mean and probe-best
improvement. Exponentially decayed values become softmax allocation
probabilities; a 15% exploration mixture prevents permanent lock-in. No decoder
or optimizer update is rolled back.

## Results

| Adaptive strategy | Target MSE | Rastrigin | TSP length |
|---|---:|---:|---:|
| All objectives, one micro-batch | 0.05103 | **11.913** | **3.608** |
| All objectives, three micro-batches | 0.02186 | 13.510 | 3.647 |
| Non-RL, three micro-batches | 0.05871 | 16.818 | 3.835 |
| Cost-aware all, three micro-batches | 0.03225 | 15.907 | 3.748 |
| No forced warm-up, three micro-batches | **0.02041** | 23.223 | 3.627 |
| No warm-up plus frozen arm | 0.02281 | 19.412 | 3.805 |

For comparison, guarded random non-RL scored `0.06348`, `4.506`, and `3.308`;
frozen scored `0.07306`, `8.360`, and `3.524`; dedicated three-step REINFORCE
scored `0.00445` on the target.

## Interpretation

- Dynamic allocation works mechanically. In a controlled test where an
  objective switches from rewarding increasing output to rewarding decreasing
  output, the learned preference reverses after the switch.
- It also reacts in the real runs. Smooth-target seeds tend to raise REINFORCE's
  probability, while rugged runs lower arms with negative recent probe rewards.
  The frozen arm rises from a uniform 14.3% prior to roughly 16–21% on Rastrigin
  and TSP.
- The reaction is too late. On Rastrigin and TSP, exploratory micro-batches can
  irreversibly damage the decoder before their probability falls. Once nearly
  every arm has negative reward, allocation can choose only the least harmful
  update. Removing forced warm-up or adding a no-op arm does not recover the
  guarded results.
- Immediate probe reward is also myopic. An update can improve the current elite
  probe yet harm the decoder manifold that later generations need. The rollback
  gate protects against at least the immediately visible damage; learned
  allocation alone cannot.
- On the smooth target, where almost every update direction is helpful,
  no-rollback allocation is competitive with random mixtures. It still trails
  a correctly chosen high-intensity REINFORCE specialist.

The conclusion is not that adaptive weighting is useless. Its probabilities
are valuable diagnostics and could guide proposal frequency. But on the current
tasks it works best *inside* a safety mechanism: adapt how often each trainer is
proposed, while retaining acceptance/rollback or a trust-region bound on how far
one kept update may move the decoder.

Raw results, choices, per-batch rewards, and final probabilities:

- [`mps_adaptive_mixed_training_5000.json`](mps_adaptive_mixed_training_5000.json)
- [`mps_adaptive_no_warmup_training_5000.json`](mps_adaptive_no_warmup_training_5000.json)
