# Work log — j0005 (MUTATE, tsp)

## Structure exploited

Nearest-neighbor's dominant failure is stranding remote cities that force
long end-of-tour detours. The parent priced single-city strandedness
(distance to nearest other unvisited). This mutation deepens the measure
by one level: isolation is the MEAN of a candidate's distances to its TWO
nearest other unvisited cities, so a remote PAIR — invisible to the
single-nearest measure because its members are mutually close — also
reads as lonely and gets collected while the tour is nearby.

## Iterations (canonical scorer, train seeds)

Selection rule: minimize d(current, c) − W · iso(c),
iso(c) = mean of two nearest other-unvisited distances.
W is the only knob; base loop: one change at a time, keep best, stop
after two consecutive non-improvements.

| # | change            | score     | verdict |
|---|-------------------|-----------|---------|
| 1 | initial, W = 0.4  | 1.0458038 | best    |
| 2 | W = 0.5           | 1.0290888 | worse (non-improvement 1) |
| 3 | W = 0.3           | 1.0190158 | worse (non-improvement 2) — stop |

Reverted to W = 0.4 and rescored; output captured verbatim in
score.json.

## Shipped

artifact.py with W = 0.4 — canonical score 1.0458037648565088
(parent: 1.0442762539183397; nearest-neighbor baseline: 1.0).
Cases: [1.026208, 1.125599, 0.935549, 1.057913, 1.08375].
