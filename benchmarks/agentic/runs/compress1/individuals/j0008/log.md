Structure exploited: source bytes are strongly predictable from the preceding one to three bytes, especially in the text/code corpus. The artifact uses PPM-style escaping and backoff through order-3, order-2, order-1, and unigram contexts; concentration-adaptive halving limits stale dominant counts while preserving diverse context history.

Iteration 0: implemented the simplest order-3 contextual arithmetic coder with deterministic model updates and verified the source-level interface. Canonical score: -3.4677734375 (3.46777 bpb).

Iteration 1: tested order 4 offline; it regressed to 3.67773 bpb, so reverted. Added exclusion during PPM backoff; the canonical-like slice improved offline to 3.15491 bpb with exact round-trip. Canonical score: -3.1549072265625 (3.15491 bpb).

Iteration 2: tested order 4 with exclusion offline at 3.14453 bpb; order 5 regressed to 3.18555 bpb and was reverted. Final canonical score: -3.14453125 (3.14453 bpb), errors empty, 1.29 seconds.
