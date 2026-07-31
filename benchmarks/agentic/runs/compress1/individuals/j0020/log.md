Structure exploited: the input is text-like and repeated byte continuations are much more predictable after 2–4 preceding bytes. I use a sparse order-4 PPM tree with escape probabilities and exclusion, so long contexts are used when observed and shorter contexts receive only genuinely new symbols; an adaptive global row handles novel bytes.

Iteration 1: initial order-4 PPM with count-one updates, distinct-symbol escape mass, and exclusion. Canonical output: {"task": "compress", "score": -2.8851318359375, "bpb": 2.88513, "seconds": 1.76, "holdout": false, "errors": []}. Kept as the incumbent.

Iteration 2: tested a fifth-byte context by extending the same sparse PPM tree. The first invocation failed with IndexError because the row container was still sized for four contexts; this was repaired before the next invocation.

Iteration 3: repaired order-5 candidate round-tripped on direct tests but canonical output was {"task": "compress", "score": -2.904296875, "bpb": 2.9043, "seconds": 2.65, "holdout": false, "errors": []}. Restored the order-4 incumbent as the shipped artifact.
