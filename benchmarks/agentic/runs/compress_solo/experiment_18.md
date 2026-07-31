You are a research agent improving a training script — one experiment in a longer campaign (experiment 18 of 31).

Read, in this order:
- The campaign worklog (every prior hypothesis and result): /Users/daniellosey/Documents/latentspace/benchmarks/agentic/runs/compress_solo/WORKLOG.md
- The current champion script: /Users/daniellosey/Documents/latentspace/benchmarks/agentic/runs/compress_solo/champion.py (canonical score -3.361328125)
- The canonical scorer (run it, NEVER edit it): /Users/daniellosey/Documents/latentspace/benchmarks/agentic/tasks/compress/score.py

Run ONE experiment:
1. Form ONE hypothesis for improving the champion's score — something the worklog has not already tried and rejected. State it in one sentence.
2. Copy the champion to /Users/daniellosey/Documents/latentspace/benchmarks/agentic/runs/compress_solo/candidate.py and apply ONLY that change.
3. Score it: python3 /Users/daniellosey/Documents/latentspace/benchmarks/agentic/tasks/compress/score.py /Users/daniellosey/Documents/latentspace/benchmarks/agentic/runs/compress_solo/candidate.py   (never pass --holdout; at most 3 scorer runs in this experiment, tuning included).
4. Verdict: if the candidate's canonical score is strictly better than -3.361328125, replace the champion: cp /Users/daniellosey/Documents/latentspace/benchmarks/agentic/runs/compress_solo/candidate.py /Users/daniellosey/Documents/latentspace/benchmarks/agentic/runs/compress_solo/champion.py. Otherwise leave the champion untouched.
5. Append to /Users/daniellosey/Documents/latentspace/benchmarks/agentic/runs/compress_solo/WORKLOG.md — experiment number, hypothesis, exact scores printed by the scorer, KEPT or REVERTED. Never report a number the scorer did not print.
6. Append one JSON line to /Users/daniellosey/Documents/latentspace/benchmarks/agentic/runs/compress_solo/results.jsonl: {"experiment": 18, "hypothesis": "<one sentence>", "score": <candidate score>, "kept": <true|false>}

Your final message: one line — KEPT or REVERTED, and the score.