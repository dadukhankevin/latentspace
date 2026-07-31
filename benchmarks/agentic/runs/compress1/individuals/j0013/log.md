Structure exploited: the input is source-like text with repeated substrings. A cyclic BWT groups equal contexts, move-to-front turns local symbol runs into small ranks and zero runs, and arithmetic coding models the resulting skew. The token prior is decaying because low MTF ranks and short zero runs are common before the adaptive counts have learned them.

Iterations (canonical scorer, no holdout):

1. Order-4 recent-byte arithmetic model with sparse-context backoff: score -4.9967041015625 (bpb 4.9967).
2. Fresh cyclic-BWT/MTF/zero-run arithmetic implementation with uniform token counts: score -3.0460205078125 (bpb 3.04602).
3. Same transform with decaying priors for literal ranks and run lengths: score -3.0421142578125 (bpb 3.04211). Shipped.
