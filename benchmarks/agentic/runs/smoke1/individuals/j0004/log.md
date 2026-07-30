# Work log — j0004 (MUTATE from dead-space-penalty parent, task: binpack)

## Structure exploited

Items are uniform on (0.1, 0.7), so a placement's leftover residual has
three qualitatively different fates: <= tau it is a cheap near-perfect
close; in (tau, 0.1) it is permanently unfillable dead space (no item is
smaller than 0.1); >= 0.1 it can still receive a future item. The parent
expressed this as a smooth penalty added to best-fit tightness and
flipped zero decisions (scored exactly best-fit). The mutation makes the
distinction lexicographic: tier strictly dominates, tightness (-residual)
only orders bins within a tier. Note the family collapses to exact
best-fit at tau = 0.1, so tau interpolates between "avoid dead space at
all costs" and plain best-fit — the calibration finds where the hard
rule helps rather than hurts.

## Iterations (canonical scorer, train seeds; scorer never edited, no --holdout)

| # | change | score |
|---|--------|-------|
| 0 | plain best-fit baseline (reference) | 0.9417321645639388 |
| 1 | 3 tiers, tau = 0.01 | 0.924071910660146 |
| 2 | tau sweep 0.02–0.09: 0.02 -> 0.9311; 0.03/0.04/0.05 -> 0.9348; 0.06 -> 0.9383; 0.07 -> 0.9455; 0.08/0.09 -> 0.9417 | best 0.9454968704462917 |
| 3 | fine sweep 0.062–0.078: plateau [0.070, 0.078] all 0.9454968704462917; 0.065/0.068 -> 0.9417; 0.062 -> 0.9383 | 0.9454968704462917 |
| 4 | shipped tau = 0.074 (plateau midpoint, robustness) | 0.9454968704462917 |

Stop rule: after the 0.070 optimum, successive tau probes (0.072, 0.075,
0.078) tied without improving — two-plus consecutive non-improving
changes, so calibration stopped and the plateau midpoint shipped.

## Interpretation

A tiny hard-tier threshold (tau ~ 0.01) is much worse than best-fit:
refusing every dead residual sacrifices near-perfect closes and piles up
loose bins. The win lives in a narrow band tau in [0.070, 0.078]: treat
residuals up to ~0.074 as acceptable closes, dodge only the expensive
dead band (0.074, 0.1), and stay best-fit otherwise. One train case
(seed 102) improves from 0.941 to 0.960; none regress.

Final canonical score for shipped artifact.py: 0.9454968704462917
(parent / plain best-fit: 0.9417321645639388).
