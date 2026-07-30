"""j0006 — online bin packing priority (banded dead-gap rule).

Items are uniform on [0.1, 0.7], so a bin leftover below 0.1 can never be
used again, and a leftover barely above 0.1 can only be used by items in
a razor-thin window. Three hard-ordered tiers on prospective leftover:

  1. leftover < 0.072  : near-fit. Take the tightest such bin — the tiny
                         stranded gap is a cheap price for closing a bin.
  2. 0.072 <= leftover < 0.13 : nearly-dead band. Hard-avoid — this
                         either strands close to the maximum unusable
                         space (< 0.1) or leaves a sliver only a
                         near-minimum item can use (0.1 .. 0.13).
  3. leftover >= 0.13  : live. Plain best-fit (smallest leftover wins).

Tiers are separated by large score offsets so tier order always wins,
while within-tier order stays tightest-first — so if only nearly-dead
bins are feasible the least-bad one is still chosen.
"""
import numpy as np

SNUG = 0.072      # below this, accept the gap and close the bin
NEAR_DEAD_HI = 0.13   # avoid leftovers in [SNUG, NEAR_DEAD_HI)


def priority(item, capacities):
    caps = np.asarray(capacities, dtype=np.float64)
    left = caps - item
    scores = -left                                   # tier 3: best-fit
    scores = np.where(left < SNUG, 1000.0 - left, scores)          # tier 1
    scores = np.where((left >= SNUG) & (left < NEAR_DEAD_HI),
                      -100.0 - left, scores)                        # tier 2
    return scores
