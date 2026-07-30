"""j0002 — greedy TSP tour construction.

Distance-plus-stranding rule: from the current city, pick the unvisited
candidate c minimizing  d(current, c) - LAM * iso(c),  where iso(c) is the
distance from c to its nearest OTHER unvisited city. Isolated cities are
grabbed while we are already near them, instead of forcing an expensive
detour late in the tour. LAM = 0.0 reduces to exact nearest-neighbor;
LAM = 0.20 is the mid-point of the best plateau on the canonical train
seeds.
"""
import numpy as np

LAM = 0.20


def next_city(current, unvisited, coords):
    cand = coords[unvisited]
    d = np.linalg.norm(cand - coords[current], axis=1)
    if len(unvisited) == 1:
        return int(unvisited[0])
    dm = np.linalg.norm(cand[:, None, :] - cand[None, :, :], axis=2)
    np.fill_diagonal(dm, np.inf)
    iso = dm.min(axis=1)
    return int(unvisited[int(np.argmin(d - LAM * iso))])
