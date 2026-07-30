import numpy as np


def priority(item, capacities):
    """Best-fit: prefer the feasible bin with the smallest leftover.

    Exploits that packing waste is the sum of leftover capacities in
    opened bins: greedily minimizing each placement's leftover keeps
    bins near-full. Distribution-aware non-monotone corrections (dead
    gaps below the 0.1 minimum item size, awkward barely-live leftovers)
    were tried and all scored below plain best-fit on the train seeds;
    see log.md.
    """
    return -(capacities - item)
