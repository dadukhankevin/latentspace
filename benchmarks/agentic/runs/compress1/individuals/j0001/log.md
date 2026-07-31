The slice is UTF-8 text with many repeated words, punctuation patterns, and short lines. I exploit prior substrings with an LZ-style back-reference; a match replaces several literal bytes with a distance and length. The required variation prices the remaining path in the parser, so a locally shorter match can win when it exposes a cheaper future token.

Iterations:

1. Initial future-cost parser with 64 prior candidates, fixed 16-bit distance, and fixed 8-bit match length: score -4.3160400390625 (4.31604 bpb).
2. Changed only match-length coding to gamma coding and charged its exact bit cost during parsing: score -3.7467041015625 (3.7467 bpb).
3. Changed only the retained recent-candidate limit from 64 to 512: score -3.7398681640625 (3.73987 bpb). Shipped this best artifact.
