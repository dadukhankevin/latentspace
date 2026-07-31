The parent actually performs adaptive arithmetic coding with one byte of history. Each byte context starts uniform, the observed symbol gets a +17 update, and that row is halved when its total crosses a context-dependent cap; there is no order-two modeling or explicit global backoff.

The artifact uses the same lossless arithmetic-coding structure but exploits repeated two-byte sequences. Pair rows are dense and deterministic; before a pair has three observations, coding falls back to the previous-byte row, while both models learn and periodically halve their counts. This makes new or rare pairs inherit the stronger order-one prediction and lets recurrent pairs specialize.

Iterations:

- Initial implementation, pair activation after 3 observations: score -3.708740234375 (3.70874 bpb), round-trip passed.
- Changed only the activation threshold to 2 observations: score -3.7349853515625 (3.73499 bpb); retained the better threshold-3 result.
- Changed only the activation threshold to 4 observations: score -3.68994140625 (3.68994 bpb); retained this best result.
