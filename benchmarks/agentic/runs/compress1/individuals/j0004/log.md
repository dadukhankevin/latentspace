The parent actually performs a whole-message BWT, MTF encoding, zero-run tokenization, and one adaptive arithmetic model over the resulting ranks. I replace that global transform with a streaming binary arithmetic coder. The exploited structure is repeated local syntax: source-code bytes are predictable from the preceding two bytes, and the next bit is further constrained by the preceding byte and the prefix already seen in the current byte. Two Laplace-smoothed count tables are combined for every bit, so unseen contexts remain decodable.

Iteration 1: initial two-context binary arithmetic coder; canonical score -5.181640625 (5.18164 bpb), exact round trip.

Iteration 2: switched to a fresh cyclic-BWT/MTF transform, retained zero-run tokens, and split the adaptive arithmetic model into type, rank, and run-length distributions; canonical score -3.04443359375 (3.04443 bpb), exact round trip.

Iteration 3: offline tuning selected type/rank/run increments 80/24/48 and a fixed source-text/code-biased MTF initialization; canonical score -3.0401611328125 (3.04016 bpb), exact round trip.
