Structure exploited: the input is text with recurring three-byte prefixes and repeated spans. The artifact stores unmatched bytes as literals and repeated spans as distance/length back-references; a backwards dynamic program prices the actual variable-width fields and chooses the cheapest path.

Iteration 0: implemented the locality-biased LZ candidate search, variable-width distance and length fields, and bit-packed lossless stream. Canonical score: -4.2264404296875 (4.22644 bpb).

Iteration 1: changed literal payloads from fixed eight-bit values to a canonical Huffman code whose lengths are included in the stream header, and priced those lengths in the same dynamic program. Canonical score: -3.9720458984375 (3.97205 bpb).

Iteration 2: changed literal coding to adaptive order-1 arithmetic coding, updating the context model over both literal and matched output bytes; the planner uses the corresponding probability estimates. Canonical score: -3.861083984375 (3.86108 bpb).
