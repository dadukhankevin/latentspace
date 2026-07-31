The slice is prose with repeated words, punctuation, and short phrases. I exploit those repetitions with bounded-window LZ matches, choose a dynamic-programming parse using the serialized literal/match field costs, and adaptively arithmetic-code the resulting token bytes with order-two, order-one, and order-zero contexts.

Iteration 0: implemented the cost-aware LZ token parser, synchronized adaptive context coder, and reversible BWT/MTF zero-run candidate, selecting the shortest representation. Canonical score: -3.0462646484375 (3.04626 bpb); round-trip passed and the BWT candidate won.

Iteration 1: changed the BWT rank model from update increment 24/cap 65536 to increment 20/cap 32768 after replaying the scorer's seeded substitutions. Canonical score: -3.044677734375 (3.04468 bpb).

Iteration 2: changed only the validated rank update increment from 20 to 18. Canonical score: -3.0440673828125 (3.04407 bpb), the best of three canonical runs; this is the shipped artifact.
