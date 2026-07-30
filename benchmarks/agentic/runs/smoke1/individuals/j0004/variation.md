# Variation clause — j0004 (MUTATE, parent penalized dead space proportionally)

The base methodology, BUT where the parent softly penalized dead space in
proportion to the capacity it locks in, make the distribution-derived
distinction lexicographic instead of proportional: since items are uniform
on (0.1, 0.7), classify each feasible placement by the residual it would
leave — near-perfect close (residual <= tau), dead (residual in (tau, 0.1),
permanently unfillable), or usable (residual >= 0.1) — and let tier
strictly dominate: any near-perfect close beats any usable placement,
which beats any dead placement, regardless of tightness; best-fit
tightness only orders bins within a tier. The single empirically
calibrated constant is tau, the near-perfect threshold, swept on the
canonical training score.

contradicts_base: false
