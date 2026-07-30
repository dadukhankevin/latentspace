"""j0005 — greedy TSP step pricing pair-depth strandedness.

Pick the unvisited city minimizing

    d(current, c) - W * isolation(c)

where isolation(c) is the MEAN of c's distances to its two nearest
OTHER unvisited cities (pair-depth loneliness: a remote pair looks
lonely even though its members are close to each other). Pure numpy,
deterministic, no I/O.
"""
import numpy as np

W = 0.4


def next_city(current, unvisited, coords):
    unvisited = np.asarray(unvisited)
    cand = coords[unvisited]
    d_cur = np.linalg.norm(cand - coords[current], axis=1)
    m = len(unvisited)
    if m <= 2:
        # 1 city: forced. 2 cities: the pair distance is identical for
        # both candidates, so isolation cancels — pure proximity.
        return int(unvisited[np.argmin(d_cur)])
    dmat = np.linalg.norm(cand[:, None, :] - cand[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    two_nearest = np.partition(dmat, 1, axis=1)[:, :2]
    iso = two_nearest.mean(axis=1)
    return int(unvisited[np.argmin(d_cur - W * iso)])
