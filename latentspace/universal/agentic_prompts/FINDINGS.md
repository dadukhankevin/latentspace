
**Round forty-one, addendum — lineage exhaustion and native spawning
(2026-07-30, Daniel's design).** Two protocol refinements. First,
breeding agents may now declare their lineage EXHAUSTED: given a
parent whose idea they judge genuinely spent, they found fresh instead
of grinding out a token variation, reporting `fresh_start: true`. The
engine records the child as origin "refound" and marks the parent
non-breeding — it keeps its population slot, shares, and best-ever
archive record (so consolidation can still absorb it), it just stops
receiving offspring; a task whose every member is exhausted refounds
like an extinct one. The agent working a lineage is a better judge of
"dry" than any stall counter (the tensor loop's stall-immigrant
mechanism measured nothing at 10 seeds; this is the informed version).
Second, orchestration is runtime-NATIVE: the skill now instructs an
orchestrating session to spawn workers with its own subagent
mechanism (Claude's Agent tool, Codex's spawning) rather than
cross-CLI shell-outs — the server API is the only contract, and the
shell-out path (drive.py --agent-cmd) remains only for unattended runs
and deliberately mixed fleets. Both unmeasured design choices,
recorded as such.
