Structure exploited: the input is prose and code, so short byte contexts recur with highly predictable continuations. The artifact uses sparse order-1 through order-4 continuation rows, excludes symbols already represented by longer contexts during backoff, and uses a printable-text prior at order 0 so cold-start and unseen contexts remain representable.

Iteration 1: implemented the deterministic sparse PPM4 arithmetic coder from scratch. Canonical score: -3.1346435546875 (bpb 3.13464).

Iteration 2: changed only the order-3/4 escape weighting from doubled weights to the local distinct-symbol mass at every order, after an offline slice-length comparison. Canonical score pending.

Iteration 2 canonical score: -2.8509521484375 (bpb 2.85095).

Iteration 3: kept the PPM model and changed the fixed printable prior to weight printable bytes and whitespace 64, then replaced the four-byte length prefix with an arithmetic-coded EOF category. Canonical score pending; run 3 is the final permitted evaluation.

Iteration 3 canonical score: -2.8460693359375 (bpb 2.84607, 0.66 seconds). This was retained as the best of the three canonical runs.
