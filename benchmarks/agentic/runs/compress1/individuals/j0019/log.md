The artifact exploits repeated short byte contexts in the text: an adaptive order-3 PPM model gives highly predictable trigrams their own distribution, while escape symbols back off through shorter contexts and finally to a uniform raw byte. Arithmetic coding turns those probabilities into a compact, deterministic bitstream without needing the source corpus.

Iterations:

1. Initial order-2 PPM with escape frequency equal to the number of distinct symbols in each context: score `-3.4676513671875` (3.46765 bpb), round trip passed.
2. One-change trial using a single escape count per context: score `-3.6114501953125` (3.61145 bpb), worse; discarded.
3. Best candidate, restoring the initial escape rule and adding one order-3 context above the order-2 backoff: score `-3.206298828125` (3.2063 bpb), round trip passed. Shipped.
