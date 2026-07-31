# Log

The scored slice is structured technical prose and source code. A global BWT clusters equal contexts, MTF makes nearby symbols small ranks, and binary coding of zero-rank runs avoids paying for a large fixed run-length alphabet. The sentinel makes the transform unambiguous for every byte string while the shared adaptive arithmetic model stays synchronized in both directions.

Iterations:

- Local round-trip checks initially exposed and then fixed a sentinel-container bug; empty, repetitive, binary, and random inputs now pass.
- Canonical iteration 1: sentinel-safe BWT/MTF with binary zero-run digits and a shared 259-symbol arithmetic model, score `-3.0467529296875`.
- Offline replay swept the arithmetic update mass; 28 was the best local candidate.
- Canonical iteration 2: changed only the shared arithmetic update mass from 19 to 28, score `-3.044677734375`.
- Offline replay swept the rescale threshold; 42120 was the best local candidate with the current initialization.
- Canonical iteration 3: changed only the rescale threshold from 65536 to 42120, score `-3.0439453125` (best; shipped).
