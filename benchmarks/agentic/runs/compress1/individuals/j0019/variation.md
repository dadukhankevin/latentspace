The single deliberate mutation is to replace the parent’s LZ back-reference parse with an order-3 PPM arithmetic model: at each byte, try the previous-three-byte context, then escape through the order-2, order-1, and unigram contexts, with a raw-byte fallback for unseen symbols. This makes future cost a probability-coded next-byte choice rather than a token-path search, so no back-references are emitted. The model remains adaptive, deterministic, pure, and uses only a fixed-size length header in addition to its coded stream.

contradicts_base: false
