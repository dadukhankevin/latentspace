"""j0003 — isolation-aware greedy TSP construction.

Pick the unvisited city minimizing d(current, c) - W * iso(c), where
iso(c) is c's distance to its nearest OTHER unvisited city. Cities that
are about to become stranded get a bonus, so the tour collects outliers
in passing instead of paying a long detour for them at the end.
"""
import numpy as np

W = 0.2


def next_city(current, unvisited, coords):
    cand = coords[unvisited]
    d = np.linalg.norm(cand - coords[current], axis=1)
    if len(unvisited) == 1:
        return int(unvisited[0])
    dm = np.linalg.norm(cand[:, None, :] - cand[None, :, :], axis=2)
    np.fill_diagonal(dm, np.inf)
    iso = dm.min(axis=1)
    return int(unvisited[int(np.argmin(d - W * iso))])
