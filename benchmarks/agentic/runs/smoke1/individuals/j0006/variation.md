# j0006 variation clause (MUTATE of parent, canonical parent score 0.9417321645639388)

The base methodology, BUT replace the parent's proportional dead-space
score-shaping with a hard three-tier band around the scorer's 0.1 minimum
item size, with the avoid region extended ABOVE that minimum: (1) if a
bin's prospective leftover is under ~0.07, treat the bin as closed and
take the tightest such fit — tiny stranded gaps are a cheap price for
closing a bin; (2) hard-avoid leftovers in the nearly-dead band
[0.072, 0.13), which either strand close to the maximum unusable space
(below 0.1) or leave a sliver only near-minimum items in a razor-thin
window can use (0.1 to 0.13); (3) otherwise plain best-fit. Tiers are
separated by large score offsets so the band actually flips placements
instead of collapsing to best-fit, while within-tier order stays
tightest-first so forced choices remain sensible.

contradicts_base: false
