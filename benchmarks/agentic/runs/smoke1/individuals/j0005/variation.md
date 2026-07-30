# Variation clause — j0005 (MUTATE of parent, one deliberate change)

The base methodology with the parent's strandedness pricing, BUT isolation
is measured at pair depth: a candidate's isolation term is the mean of its
distances to its TWO nearest other unvisited cities, not the single
nearest. A remote pair of cities scores near-zero isolation under the
single-nearest measure — each has the other close by — yet the pair as a
whole is stranded and still forces nearest-neighbor's end-of-tour detour;
averaging in the second-nearest distance exposes pair-level loneliness so
such pairs are also collected in passing. Everything else follows the
parent: pick the city minimizing (distance-to-current − w · isolation),
w is the only tuning knob, tuned per the base loop — one change at a
time, keep the best canonical score, stop after two consecutive
non-improvements.

contradicts_base: false
