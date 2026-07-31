The slice is repetitive technical prose and source text, so the previous byte is a strong predictor of the next byte. I use an adaptive order-one arithmetic coder with one 256-symbol frequency row per preceding byte. The deliberate variation is live per-context rescaling: a context's cap is `1024 + 640 * active_symbols`, bounded at 32768, so its memory depends on the concentration observed in this instance.

Canonical iterations (higher score is better; all are exact scorer output):

1. Initial order-one model, update 1, live cap `1024 + 128 * active`: `-4.4171142578125` (bpb `4.41711`).
2. Changed only the update mass to 17: `-4.0191650390625` (bpb `4.01917`).
3. Changed only the live-cap multiplier from 128 to 640: `-3.9698486328125` (bpb `3.96985`).

The final artifact is iteration 3, the best score observed.
