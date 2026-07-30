"""Online bin packing priority — lexicographic residual tiers.

Structure exploited: items are uniform on (0.1, 0.7), so a bin's
residual after a placement falls into one of three qualitatively
different classes:

  - near-perfect close: residual <= TAU. The bin is finished and the
    locked-in waste is at most TAU — cheap.
  - dead: residual in (TAU, 0.1). No future item can ever fit (all
    items are >= 0.1), so the entire residual is permanently wasted
    and it is large enough to matter.
  - usable: residual >= 0.1. A future item can still land here.

Unlike a smooth penalty added to best-fit tightness (which is easily
outvoted and ends up changing nothing), the class distinction here is
lexicographic: any near-perfect close outranks any usable placement,
which outranks any dead placement, regardless of tightness. Best-fit
tightness (-residual) only orders bins within the same tier. The one
calibrated constant is TAU, chosen by sweeping the canonical scorer
on the training seeds (plateau 0.070-0.078; midpoint shipped).

Pure, deterministic, numpy only.
"""
import numpy as np

TAU = 0.074   # near-perfect-close threshold (calibrated on train seeds)
DEAD = 0.1    # minimum possible item size: smaller residuals are dead
TIER = 10.0   # tier offset; larger than any tightness difference


def priority(item, capacities):
    caps = np.asarray(capacities, dtype=np.float64)
    r = caps - item
    score = -r
    score = score + TIER * (r <= TAU)                # perfect closes: top
    score = score - TIER * ((r > TAU) & (r < DEAD))  # dead residuals: bottom
    return score
