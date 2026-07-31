# j0031 compression log

The input is ordinary, repetitive technical prose with strong short-range byte structure: punctuation, whitespace, identifiers, and recurring code fragments make recent byte contexts predictive. The artifact exploits this with sparse PPM contexts through order 4, exclusion-aware backoff, an adaptive root model, and a bounded age-aware escape mass. The decoder updates the same context timestamps and counts after each decoded byte.

## Iterations

1. Initial order-4 PPM with the age-aware escape prior (`AGE_STEP=8`, `ESCAPE_CAP=24`). Canonical score: `-3.7418212890625` (3.74182 bpb).
2. Reduced the escape cap to 4 while keeping the coder and update rules unchanged. Canonical score: `-3.051025390625` (3.05103 bpb).
3. Made the escape prior coarser and less aggressive (`AGE_STEP=32768`, `ESCAPE_CAP=20`). Canonical score: `-2.8524169921875` (2.85242 bpb).

The third artifact was selected. Offline audits also round-tripped empty input, repeated text, unrelated corpus slices, and random bytes.
