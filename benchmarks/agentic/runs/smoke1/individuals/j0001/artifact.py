import numpy as np

DEAD = 0.1  # items are uniform(0.1, 0.7): a residual in (0, DEAD) can never be filled
K = 1.0     # exchange rate: 1 unit of dead (unfillable) residual costs K units of live residual


def priority(item, capacities):
    """Dead-zone-aware best fit, graded by waste: prefer tight fits, and
    penalize unfillable residuals in (0, 0.1) in proportion to the
    capacity actually wasted, so near-perfect closes stay attractive."""
    r = capacities - item
    dead = (r > 1e-9) & (r < DEAD)
    score = -r - np.where(dead, K * r, 0.0)
    return score
