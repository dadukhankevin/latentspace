# j0006 work log (MUTATE, binpack)

## Structure exploited

Items are uniform on [0.1, 0.7], so a bin leftover below 0.1 is
permanently unusable (dead), and a leftover barely above 0.1 is nearly
dead — only items in the razor-thin [0.1, leftover] window can use it.
The heuristic accepts tiny dead gaps (they close a bin cheaply) but
hard-avoids creating leftovers in the nearly-dead band [0.072, 0.13),
falling back to best-fit for genuinely live leftovers. The parent scored
dead space proportionally and never flipped a placement (tied plain
best-fit at 0.9417); the mutation makes the band hard-ordered and
extends it above the 0.1 minimum.

## Iterations (train-seed pilot mirrors the canonical pack loop; final
   numbers below marked "canonical" are the scorer's own output)

1. Family screen (pilot): best-fit baseline 0.941732; all monotone
   transforms of leftover tie it (same argmax). Sum-of-squares gap
   inventory (w=0.05) 0.938725 but beat BF on seed 103; tight-else-worst
   0.934814; proportional dead-penalty (parent-style, when it does flip
   decisions) 0.920377. No family beat BF outright.
2. Dead-gap accept/avoid tiers (pilot): accept leftover < d, avoid
   [d, 0.1). d=0.03/0.05 hurt (0.9348), d=0.07 improved to 0.945497
   (gain on seed 102, no losses). Kept.
3. Extend avoid band above 0.1 (pilot): avoid [0.07, hi). hi=0.12/0.13
   improved to 0.949408 (additional gain on seed 103). hi=0.15+ or
   0.17+ lose seeds. Kept hi=0.13.
4. Fine grid (pilot): plateau at 0.949408 across d in [0.07, 0.074],
   hi in [0.115, 0.14]. No further improvement — picked plateau-center
   d=0.072, hi=0.13. (First consecutive non-improvement.)
5. Second avoid band above 0.7 and reversed within-band ordering
   (pilot): both 0.949408, no change. (Second consecutive
   non-improvement — stop, per playbook.)

## Final (canonical scorer, verbatim)

{"task": "binpack", "score": 0.9494084350721421, "cases": [0.960784, 0.96, 0.958333, 0.943396, 0.924528], "errors": []}

Parent/best-fit canonical baseline: 0.9417321645639388.
