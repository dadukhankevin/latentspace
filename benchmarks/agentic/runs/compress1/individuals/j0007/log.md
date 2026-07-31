Structure exploited: natural-language bytes have recurring two-byte contexts, so a sparse order-two model should predict the next byte more sharply than order one. A weighted order-one backoff keeps unseen or rare bigrams from becoming overconfident, and both coder directions use the same integer distribution.

Iteration 1: BACKOFF=32; score -3.4329833984375 (3.43298 bpb).
Iteration 2: changed only BACKOFF to 16; score -3.392822265625 (3.39282 bpb), kept.
Iteration 3: changed only BACKOFF to 8; score -3.381591796875 (3.38159 bpb), kept as best.
