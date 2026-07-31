# Run log

The slice is natural text and code, so repeated short byte phrases should
make order-2 through order-4 continuations much sharper than a unigram
model. The artifact uses sparse context rows, exclusion-based PPM escapes,
and a fixed order-0 floor so unseen or corrupted contexts remain lossless.

## Iterations

- Iteration 1: fresh sparse order-4 exclusion PPM with local row rescaling; first implementation failed the range-decoder round trip and scored `-99.0`.
- Iteration 2: repaired unsigned range handling by using a bitwise arithmetic coder; canonical output was `{"task": "compress", "score": -2.8917236328125, "bpb": 2.89172, "seconds": 0.82, "holdout": false, "errors": []}`.
- Iteration 3: one tuning change, increasing the local continuation-row rescale threshold from 256 to 1024 so repeated contexts retain evidence longer; canonical output was `{"task": "compress", "score": -2.883544921875, "bpb": 2.88354, "seconds": 0.94, "holdout": false, "errors": []}` and this is the shipped artifact.
