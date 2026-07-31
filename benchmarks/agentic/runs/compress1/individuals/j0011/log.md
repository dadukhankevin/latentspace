# Log

The scored bytes are structured prose. A global cyclic Burrows–Wheeler ordering groups equal contexts, while move-to-front ranks turn repeated BWT symbols into zero runs and small nonzero ranks. The artifact exploits this with a typed event stream: event type has a previous-type model, nonzero ranks use previous-rank backoff, and zero-run lengths use their own adaptive histogram.

Iterations:

1. Initial typed arithmetic coder with zero-run update 16: `{"task": "compress", "score": -3.02685546875, "bpb": 3.02686, "seconds": 0.56, "holdout": false, "errors": []}`.
2. Changed only the zero-run update increment from 16 to 32: `{"task": "compress", "score": -3.0267333984375, "bpb": 3.02673, "seconds": 0.63, "holdout": false, "errors": []}`. Kept.
3. Changed only the zero-run update increment from 32 to 24: `{"task": "compress", "score": -3.0267333984375, "bpb": 3.02673, "seconds": 0.47, "holdout": false, "errors": []}`. Tied the best; shipped this deterministic artifact.
