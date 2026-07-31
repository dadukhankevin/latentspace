#!/bin/zsh
# The GA-vs-solo comparison on the lm task (train-script optimization).
# Matched: 16 decoder-agent calls per side, <=3 canonical scorer runs
# per call, same scorer, same 60s wall-clock training budget, one
# global training lock. The GA side additionally spends consolidator
# and rewrite agent calls (orchestration overhead, counted in the
# report). GA first, then solo, strictly sequential — the budget is
# wall clock, so the two sides must never share the machine.
set -e
cd "$(dirname "$0")/../.."
AGENT='claude -p "$(cat {promptfile})" --permission-mode bypassPermissions'

python3 -m latentspace.universal.drive \
  --run benchmarks/agentic/runs/lm_ga \
  --tasks lm --tasks-dir benchmarks/agentic/tasks \
  --agent-cmd "$AGENT" --port 8799 \
  --rounds 4 --founders 4 --children 4 --population-cap 10 \
  --consolidate-every 2 --auto-consolidate \
  --max-parallel 3 --agent-timeout 3600 --seed 0

python3 -m latentspace.universal.solo \
  --run benchmarks/agentic/runs/lm_solo \
  --task lm --tasks-dir benchmarks/agentic/tasks \
  --agent-cmd "$AGENT" --experiments 16 --agent-timeout 3600
