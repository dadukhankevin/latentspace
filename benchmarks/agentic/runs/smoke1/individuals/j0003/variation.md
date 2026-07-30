# Variation — j0003 (FOUND, tsp)

The base methodology, BUT the greedy choice prices strandedness instead of
ranking candidates by proximity alone: for every unvisited city compute an
isolation term — its distance to the nearest OTHER unvisited city — and pick
the city minimizing (distance-to-current − w · isolation). Lonely outliers
are thereby collected in passing while the tour is nearby, rather than left
behind to force long end-of-tour detours, which is nearest-neighbor's main
failure mode. The single weight w is the only tuning knob and is tuned per
the base loop: one change at a time, keep the best canonical score, stop
after two consecutive non-improvements.
