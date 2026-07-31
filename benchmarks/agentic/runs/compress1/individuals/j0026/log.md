The slice is repetitive experimental prose. A suffix-array BWT makes nearby
contexts cluster in its last column; move-to-front then turns those clusters
into many zero ranks and a small tail of low ranks. The artifact exploits this
with zero-run symbols, adaptive arithmetic coding, a present-symbol bitmap for
sparse text, and a terminator so the coded stream needs no rank-count field.
It also retains a full-alphabet transform and raw fallback for inputs where
the sparse representation is not worthwhile.

Iterations (canonical scorer, no holdout):

1. Fresh BWT/MTF/range-coded artifact, increment 18 and model limit 32768:
   score -3.0418701171875, bpb 3.04187, 0.36 s. Sparse branch selected.
2. Tuned the adaptive increment to 12 and rescale limit to 14000:
   score -3.04150390625, bpb 3.0415, 0.44 s. Kept as best.
3. Tuned the rescale limit to 18000 with increment 12:
   score -3.040771484375, bpb 3.04077, 0.45 s. Shipped as best.
