# j0029 log

Structure exploited: the input is sequential natural-language/source text with repeated local phrases. The artifact uses adaptive arithmetic coding over all 256 bytes, mixing a global text prior, order-1 evidence, sparse order-2 through order-4 context rows, and longer sparse order-6/order-8/order-10 phrase rows. Long rows are only present after an exact suffix has occurred; row aging and capped weights limit damage from stale or one-off evidence. The length header makes the arithmetic stream self-delimiting.

Iterations:

1. Fresh order-0/1-to-4 mixture plus order-6/order-8 phrase rows. Canonical output: `{"task": "compress", "score": -2.9949951171875, "bpb": 2.995, "seconds": 8.87, "holdout": false, "errors": []}`.
2. One change: added an order-10 continuation row for longer repeated phrases. Canonical output: `{"task": "compress", "score": -2.9903564453125, "bpb": 2.99036, "seconds": 8.52, "holdout": false, "errors": []}`. Kept this version.

Robustness checks: round-tripped empty input, ordinary text, a disjoint corpus slice, repetitive bytes, and random bytes locally; the canonical scorer also verified an exact round trip on its substituted 64 KiB slice.
