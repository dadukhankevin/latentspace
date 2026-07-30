# Work log — j0000 (binpack, FOUND)

## Structure exploited (step 2 statement)

Packing waste is the total leftover capacity across opened bins, and the
scorer's items are uniform on [0.1, 0.7] — so a leftover below 0.1 is
dead forever and a leftover barely above 0.1 is nearly dead. Per the
variation, the plan was to classify prospective leftovers (perfect /
dead / live) and shape priority around that classification, on top of a
best-fit backbone that greedily minimizes each placement's leftover.

## Iterations (canonical scorer, train seeds)

1. Plain best-fit baseline, `-(capacities - item)`: **0.941732**
2. Variation's simplest form — perfect-fit bonus, flat penalty on dead
   gaps (r < 0.1) proportional to stranded space: **0.890090**. Worse.
   Lesson: a small dead gap IS a near-full bin; penalizing it fights
   best-fit's main strength.
3. One change — penalize the truly awkward zone instead: step penalty
   (−0.3) on live-but-barely leftovers r in [0.1, 0.25): **0.938579**.
   Better than iter 2, still below baseline. (Note: any monotone
   transform of −r packs identically to best-fit; only non-monotone
   reorderings like this can differ.)
4. One change — smooth the awkwardness penalty: −0.35·exp(−(r−0.1)/0.1)
   for r ≥ 0.1, largest at the 0.1 boundary, decaying as leftovers grow
   usable: **0.938725**. Improved on iter 3, still below baseline.

Iterations 3 and 4 are two consecutive changes that failed to improve
the kept-best artifact, so per the base stopping rule I stopped and
shipped the best scorer: plain best-fit.

## Outcome

The distribution-aware corrections consistently landed just under plain
best-fit on the train seeds (0.9386–0.9387 vs 0.9417). Honest result
for this lineage: the variation's leftover-classification idea, followed
faithfully, converged back to best-fit. Shipped artifact: iteration 1.

Final canonical score: **0.9417321645639388** (score.json is the
scorer's verbatim output for the shipped artifact.py).
