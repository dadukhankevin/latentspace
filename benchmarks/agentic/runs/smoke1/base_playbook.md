# Base playbook — version 1

This file is the EVOLVABLE shared methodology — the agentic substrate's
"base decoder weights." Every individual in the population is this
playbook plus one variation clause. Consolidation edits this file (and
bumps the version above); nothing else may. The rules of conduct live in
the `agentic-ga` skill and are NOT part of this file — they cannot be
evolved away.

## Methodology

1. Read the task's canonical `score.py` end to end before writing any
   code: the exact interface your function must expose, how instances
   are generated, and what the score rewards.
2. State, in one or two sentences in your work log, what structure of
   the problem your heuristic will exploit and why.
3. When the natural baseline is a greedy rule, ask what failure mode
   that rule creates for its own FUTURE self — capacity left nearly
   dead (binpack), remote cities left stranded for an end-of-tour
   detour (tsp) — and consider pricing that future cost into the
   present choice. Both audited wins so far had this shape; treat it as
   a strong prior to try early, not a law.
4. Know which score shapings can change behavior: any monotone
   transform of a greedy ranking key selects the same argmax and
   changes nothing. To actually flip decisions, use hard tier dominance
   — large score offsets that partition candidates into bands, with the
   plain greedy order (e.g. tightest-first) within each band — rather
   than proportional penalties (binpack).
5. Make lookahead measures deep enough to see grouped trouble: in tsp,
   a remote PAIR of cities is invisible to a single-nearest isolation
   measure because its members are mutually close, yet the pair still
   forces a detour. Averaging distance to the TWO nearest other
   unvisited cities exposed it; the winning rule picked the city
   minimizing (distance-to-current − w·isolation), with w left to the
   iteration loop to tune (best found: w=0.4; holdout 1.023 vs the 1.0
   nearest-neighbor baseline).
6. Binpack structure that held up under audit: the nearly-dead leftover
   region worth avoiding extends both BELOW and ABOVE the scorer's
   minimum item size, and a tiny leftover that effectively closes a bin
   is cheap, not costly — prefer the tightest such closing fit.
7. Expect constants tuned on training instances not to transfer: the
   binpack tier thresholds that won on train scored exactly plain
   best-fit on held-out seeds. Carry the structure, not the numbers —
   calibrate any thresholds against the canonical scorer's own instance
   generator, and leave tunable weights to the iteration loop.
8. Write the simplest version of that heuristic first. Pure function,
   numpy only, no I/O, no randomness, fast enough to run hundreds of
   times.
9. Run the canonical scorer on your artifact. Iterate: change one thing
   at a time, keep whichever artifact scored highest, and stop when two
   consecutive changes fail to improve.
10. Ship your best-scoring artifact and report its exact canonical
    score. Never report a number the scorer did not print.
