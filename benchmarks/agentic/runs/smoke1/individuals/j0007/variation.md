# Variation clause — j0007 (MUTATE of parent scoring 0.9417321645639388)

The base methodology, BUT replace per-bin leftover shaping with a
portfolio rule over the whole set of open bins: keep the multiset of
remaining capacities extreme rather than middling. Tight placements
(prospective leftover under the 0.1 minimum item size) are scored by
best-fit as usual; placements leaving a live leftover are scored in
REVERSE — roomier survivors beat middling ones (score -0.3 +
0.2*leftover) — and any placement into the bin currently holding the
maximum remaining capacity pays a flat 0.2 penalty, reserving the
portfolio's largest slot for items nothing else can take. The one
deliberate change: the score is population-aware and non-monotone in
leftover (inverted on the live branch), instead of any refinement of
per-bin leftover classes.

contradicts_base: false
