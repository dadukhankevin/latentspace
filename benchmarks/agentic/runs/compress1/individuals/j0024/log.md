# j0024 compression run

The slice is natural-language and code-like text with recurring short phrases,
so the artifact exploits repeated byte suffixes.  A fixed printable-text
prior handles the beginning and unseen contexts; dense order-1 rows learn
common continuations, while sparse order-2/3/4 rows sharpen predictions when
the same phrase recurs.  Local row aging limits stale long-context evidence.

Iterations (canonical scorer, no holdout):

1. Initial fixed-prior plus order-1/2/3/4 arithmetic coder: score
   `-3.088623046875`, bpb `3.08862`, round trip passed. **Best.**
2. Changed all sparse-context caps from 32 to 16: score
   `-3.102294921875`, bpb `3.10229`, round trip passed; rejected.
3. Restored order-2 cap 32 while keeping order-3/4 caps at 16: score
   `-3.10205078125`, bpb `3.10205`, round trip passed; rejected.

The shipped artifact is the first iteration, restored exactly after the two
non-improving changes.
