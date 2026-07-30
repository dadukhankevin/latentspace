# Variation — j0000

The base methodology, BUT design the priority explicitly around the
item-size distribution the scorer generates (uniform on [0.1, 0.7])
rather than a distribution-agnostic packing rule: classify each bin's
prospective leftover capacity as perfect (~0), dead (below the 0.1
minimum item size, so no future item can ever use it), or live, and
shape the score so near-perfect fits are strongly preferred, dead gaps
are penalized in proportion to the space they strand, and live
leftovers fall back to best-fit.
