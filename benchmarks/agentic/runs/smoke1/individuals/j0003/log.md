# Work log — j0003 (FOUND, tsp)

Structure exploited: nearest-neighbor's dominant failure mode on uniform
random points is stranding isolated cities — it defers outliers whose only
close neighbor gets consumed, then pays a long detour at the end. The greedy
choice therefore subtracts W times a city's isolation (distance to its
nearest other unvisited city) from its distance-to-current, collecting
near-stranded cities in passing.

## Iterations (canonical scorer, train seeds)

| # | change            | score     |
|---|-------------------|-----------|
| 1 | initial, W = 0.5  | 0.9667752612868343 |
| 2 | W = 0.25          | 0.9978494713423796 |
| 3 | W = 0.15          | 1.0331019520178404 |
| 4 | W = 0.1           | 1.0050091932147667 (worse, reverted direction) |
| 5 | W = 0.2           | 1.0442762539183397 (best) |
| 6 | W = 0.22          | 1.0442762539183397 (tie — no improvement) |
| 7 | W = 0.18          | 1.0442762539183397 (tie — no improvement) |

Two consecutive changes failed to improve → stopped per base rule 4.
Shipped W = 0.2. Final canonical score: 1.0442762539183397 (beats the
nearest-neighbor baseline on all 5 train seeds; score.json is the scorer's
verbatim output for the shipped artifact).
