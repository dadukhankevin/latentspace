"""j0007 — online bin packing priority.

Policy: keep the portfolio of open-bin capacities extreme, not middling.
- Tight placements (prospective leftover < 0.1, the minimum item size)
  are scored by best-fit: smaller leftover is better.
- Placements that leave a live leftover (>= 0.1) are scored in reverse:
  a roomier surviving capacity beats a middling one, so live bins stay
  broadly fillable instead of drifting into awkward mid sizes.
- Any placement into the bin currently holding the maximum remaining
  capacity pays a flat penalty, reserving the largest slot for items
  nothing else can accept.

Pure, deterministic, numpy only.
"""
import numpy as np


def priority(item, capacities):
    caps = np.asarray(capacities, dtype=np.float64)
    leftover = caps - item
    # Tight branch: best-fit. Live branch: inverted (roomier is better).
    score = np.where(leftover >= 0.1, -0.3 + 0.2 * leftover, -leftover)
    # Reserve the current largest capacity unless it is the only option.
    score = score - 0.2 * (caps >= caps.max() - 1e-12)
    return score
