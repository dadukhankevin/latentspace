The artifact exploits global redundancy in the complete message: BWT sorts all cyclic rotations so equal left contexts become adjacent, MTF turns those clusters into small ranks, and zero-run tokens expose the resulting long runs. A single adaptive arithmetic histogram then carries state across the entire transformed solution.

Iterations:

- Offline sanity check: BWT + MTF + one shared order-0 arithmetic stream, without run tokens; 3.03918 bpb estimated on the canonical slice.
- Offline candidate: add global zero-run tokens; 2.98828 bpb estimated on the unperturbed canonical slice.
- Canonical run 1, BWT/MTF/zero-runs with update increment 32 and all 512 initial frequencies: `{"task": "compress", "score": -3.048095703125, "bpb": 3.0481, "seconds": 0.57, "holdout": false, "errors": []}`.
- Canonical run 2, changed only the shared arithmetic update increment to 19: `{"task": "compress", "score": -3.046142578125, "bpb": 3.04614, "seconds": 0.56, "holdout": false, "errors": []}`.
- Canonical run 3, changed only the initial histogram to assign zero mass to the two impossible token values: `{"task": "compress", "score": -3.0460205078125, "bpb": 3.04602, "seconds": 0.56, "holdout": false, "errors": []}`.

The final score is the best of the three canonical runs; the run budget was exhausted after two improving changes.
