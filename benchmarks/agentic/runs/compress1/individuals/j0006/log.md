Structure exploited: the global BWT clusters repeated contexts, MTF turns those clusters into zero-heavy ranks, and zero-run coding removes repeated rank-0 symbols. Encoding run lengths as r−1 lets the marker and all lengths fit the same 256-symbol arithmetic alphabet without a separate 257th symbol.

Iteration 0: BWT/MTF/zero-run pipeline with 4096-token local models; canonical score -3.3216552734375.
Iteration 1: order-2 byte-context trial; local round-trip score 5.120117 bpb, rejected before canonical scoring.
Iteration 2: interpolated order-1 context/unigram trial; local round-trip score 4.6138916015625 bpb, rejected before canonical scoring.
Iteration 3: returned to one global BWT/MTF/zero-run stream, tuned the adaptive increment to 12 with a 65536 cap, and used zero-based run lengths in a 256-symbol alphabet; canonical score -3.0889892578125.
