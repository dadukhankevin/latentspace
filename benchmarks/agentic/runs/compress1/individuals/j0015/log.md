# Log

The global BWT/MTF/zero-run representation comes from Parent B: it clusters
equal local neighborhoods and exposes long runs, but I replace its flat shared
histogram because that model cannot exploit token-to-token syntax. The sparse
two-token predictor and independently adapting one-token backoff come from
Parent A, applied to the transformed token stream so BWT clustering makes its
contexts more reusable. The two parents' choices reinforce rather than
conflict; the transform changes the modeling domain, while the backoff model
controls local overfitting and remains exactly synchronized in both directions.

Canonical iterations (higher score is better; exact scorer output):

1. Initial combined BWT/MTF/zero-run stream, sparse second-order model,
   `_BACKOFF = 8`, `_ONE_STEP = 4`:
   `{"task": "compress", "score": -3.43701171875, "bpb": 3.43701, "seconds": 0.58, "holdout": false, "errors": []}`
2. Changed only `_BACKOFF` from 8 to 32:
   `{"task": "compress", "score": -3.34033203125, "bpb": 3.34033, "seconds": 0.84, "holdout": false, "errors": []}`
3. Changed only `_ONE_STEP` from 4 to 16; shipped this best artifact:
   `{"task": "compress", "score": -3.2772216796875, "bpb": 3.27722, "seconds": 0.88, "holdout": false, "errors": []}`
