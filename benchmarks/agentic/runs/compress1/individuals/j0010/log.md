The input is narrative prose mixed with source code, so nearby byte contexts recur in words, punctuation, indentation, and repeated syntax. The artifact exploits this with adaptive arithmetic coding: a fixed order-0 floor prevents unseen contexts from becoming impossible, dense order-1 rows learn byte transitions, and sparse order-2/3/4 rows capture repeated phrases. Long-context counts are weighted relative to the active order-1 mass, while local rescaling limits stale-context dominance.

Iterations (canonical scorer, canonical slice, no holdout):

1. Initial sparse order-0–4 hierarchy, with a learned order-0 row, order-1 increment 4, and fixed weights 1/4/16/64/256: `{"task": "compress", "score": -4.0850830078125, "bpb": 4.08508, "seconds": 1.76, "holdout": false, "errors": []}`
2. One change: make order 0 a fixed floor so its total cannot overwhelm context rows: `{"task": "compress", "score": -3.5806884765625, "bpb": 3.58069, "seconds": 1.7, "holdout": false, "errors": []}`
3. One change: restore strong order-1 updates (+17, backoff weight 8) and make order-2 weight dynamic, with 2x/4x bonuses for orders 3/4: `{"task": "compress", "score": -3.075439453125, "bpb": 3.07544, "seconds": 1.7, "holdout": false, "errors": []}`

The third artifact is the shipped best; all round-trip checks passed.
