# Work log — j0002 (FOUND, tsp)

## Structure exploited

Nearest-neighbor's classic failure on uniform random points is stranding:
it skips past isolated cities to chase nearby ones, then must pay a long
detour to collect the stragglers at the end. I rank candidates by
d(current, c) − λ·iso(c), where iso(c) is c's distance to its nearest
other unvisited city, so isolated cities are picked up while the tour is
already near them.

## Iterations (canonical scorer, train seeds)

1. λ = 0.0 (pure nearest-neighbor sanity check) — score 1.0 exactly
   (ties baseline on every seed, harness verified).
2. Signed coarse λ sweep (the variation's single tunable), each value
   scored by the canonical scorer on a stamped copy:
   −0.60→0.9326, −0.40→0.9274, −0.30→0.9227, −0.20→0.9576, −0.10→0.9921,
   +0.10→1.0050, +0.20→1.0443, +0.30→0.9644, +0.40→0.9772, +0.50→0.9668,
   +0.60→0.9637, +0.80→0.9941, +1.00→0.9575.
   Negative λ (look-ahead bias) always hurts; positive stranding bonus wins.
3. Fine sweep near the peak: +0.12→1.0042, +0.14→1.0190, +0.16→1.0331,
   +0.18→1.0443, +0.22→1.0443, +0.24→0.9951, +0.26→0.9978, +0.28→0.9958.
   Plateau at [0.18, 0.22] with identical per-case scores; took λ = 0.20
   (mid-plateau) for robustness.
4. Change A — cap the isolation bonus at the candidate's own distance,
   iso' = min(iso, d): score 0.9966. Worse; reverted.
5. Change B — count the depot (city 0) as an isolation partner,
   iso' = min(iso, d(c, city0)): score 1.0442762539183397, an exact tie
   with the same cases. Not an improvement; kept the simpler rule.

Two consecutive changes (A, B) failed to improve — stopped per the base
playbook.

## Shipped

artifact.py with λ = 0.20. Canonical score (train): 1.0442762539183397,
cases [1.017022, 1.020263, 1.032732, 1.025708, 1.125655], no errors.
Beats the nearest-neighbor baseline on all 5 train seeds. Holdout never
run, per rules.
