Structure exploited: the input is mostly natural/technical prose, so nearby bytes strongly predict the next byte; repeated three-byte contexts provide a sharper local model. A finite-horizon LZ parse also captures longer repeated phrases, while the compressor chooses the shorter complete representation. Both representations are self-delimiting by a mode byte and original length.

Iteration 1: initial implementation with adaptive order-0/order-1/order-3/4 arithmetic coding, a dynamic-programmed fixed-field LZ parse, and shorter-stream selection. Canonical score: -3.4188232421875 (3.41882 bpb).

Iteration 2: increased context adaptation, separated sparse and dense update rates, slowed the global fallback, and tuned rescaling/backoff thresholds against the scorer's seeded substitutions. Canonical score: -3.4112548828125 (3.41125 bpb).

Iteration 3: replaced the fixed four-byte length prefix with a deterministic variable-length prefix, preserving both decoders while removing framing overhead. Canonical score: -3.4111328125 (3.41113 bpb). Shipped this artifact.
