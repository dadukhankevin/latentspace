Structure exploited: the input is text with repeated words, punctuation, and source fragments. The artifact combines context-adaptive coding for unmatched bytes with bounded LZ matches; the parser uses backward dynamic programming so distance and length field costs are paid when selecting a match, including future parse cost.

Iteration 1: implemented the first complete adaptive range-coded LZ codec with order-1/order-2 literal contexts, four distance classes, a four-entry distance history, and the DP parser. The initial range coder failed the canonical round-trip check; after replacing its low/range normalization with an inclusive arithmetic interval and fixing byte padding, the artifact round-tripped successfully.

Iteration 2 (final): canonical scorer output was {"task": "compress", "score": -3.7890625, "bpb": 3.78906, "seconds": 1.15, "holdout": false, "errors": []}. This is the shipped artifact and the third and final canonical scorer invocation.
