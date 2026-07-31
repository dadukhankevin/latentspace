# Run log

The input is a long mixture of prose and source code, where recent byte
contexts recur but are sparse. Exclusion-based order-4 PPM gives those
contexts first claim on each continuation, while adaptive lower orders and
the order-0 floor cover phrase boundaries and altered bytes.

## Iterations

- Iteration 1: implemented the independent sparse order-4 PPM/arithmetic
  coder with 1024-count local row aging and a 32768-count global model.
- Iteration 2: changed only the local row-aging ceiling from 1024 to 2048;
  canonical output was `{"task": "compress", "score": -2.8837890625,
  "bpb": 2.88379, "seconds": 0.73, "holdout": false, "errors": []}`;
  this was a slight regression.
- Iteration 3: restored the 1024 threshold because it was the best-scoring
  tested artifact; final canonical output was `{"task": "compress", "score":
  -2.883544921875, "bpb": 2.88354, "seconds": 0.86, "holdout": false,
  "errors": []}`. This is the shipped version.
