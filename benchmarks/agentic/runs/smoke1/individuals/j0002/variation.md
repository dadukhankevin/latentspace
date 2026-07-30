# Variation — j0002

The base methodology, BUT rank candidate cities by a distance-plus-stranding
score instead of raw distance: from the current city, each unvisited candidate
c is scored d(current, c) − λ·iso(c), where iso(c) is c's isolation — the
distance from c to its nearest OTHER unvisited city. Positive λ grabs isolated
cities while we are already near them, avoiding expensive detours later;
negative λ recovers a look-ahead bias toward cities whose onward hop is cheap.
λ is the single tunable: sweep it over a coarse signed grid using the canonical
train scorer, keep the best value, and leave every other step of the base
methodology unchanged.
