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
3. Write the simplest version of that heuristic first. Pure function,
   numpy only, no I/O, no randomness, fast enough to run hundreds of
   times. For sequential byte compression, use an adaptive sparse PPM
   model through order 4: try the longest learned byte context first,
   encode an unseen continuation with an adaptive escape, and back off
   through shorter contexts while excluding symbols already represented by
   higher contexts. Keep a global order-0 model so every byte remains
   representable, update tables only from prior input bytes, and encode the
   selected intervals with adaptive arithmetic coding.
4. Before shipping, verify your artifact is robust on data it was not
   developed against (for compression: round-trip a different slice of
   ordinary text and some random bytes). One individual scored well on
   the canonical input but its decompressor crashed on unseen data —
   an automatic audit failure regardless of score.
5. Run the canonical scorer on your artifact. Iterate: change one thing
   at a time, keep whichever artifact scored highest, and stop when two
   consecutive changes fail to improve.
6. Ship your best-scoring artifact and report its exact canonical score.
   Never report a number the scorer did not print.
