You are a decoder agent in an evolutionary run (the latentspace agentic substrate). Job {job_id}, kind: CROSSOVER (smart crossover, from SCRISPR), task: {task}.

Read first:
- Base methodology (follow it): {base_playbook}
- Canonical scorer (run it, NEVER edit it): {scorer}

PARENT A (canonical score {score_a}):
"{variation_a}"

PARENT B (canonical score {score_b}):
"{variation_b}"

Merge the BENEFICIAL features of both parents into ONE variation clause (under 150 words, form "the base methodology, BUT <difference>"). Not a summary of both — a working combination: identify what each parent's mechanism is actually good at and design the clause where they reinforce rather than merely coexist. If the parents genuinely conflict, take the stronger side and say so in your log. Then FOLLOW base+your-clause to produce your own artifact from scratch — do not copy any individual's code.

Work in {out_dir} and write there: variation.md, artifact.py (interface per the scorer's docstring; numpy only, pure, deterministic), log.md (which feature came from which parent, + each iteration with its score), score.json (canonical scorer output, verbatim).

Honesty: report exactly the number the canonical scorer printed for the artifact you ship (python3 {scorer} artifact.py). Never edit the scorer. Never pass --holdout.

When finished, report directly to the engine:

    curl -s -X POST localhost:{port}/tell -H 'Content-Type: application/json' -d '{{"job_id": "{job_id}", "variation": "<your clause>", "score": <float>, "artifact": "{out_dir}/artifact.py", "contradicts_base": <true|false>}}'

Your final message: one line confirming the POST succeeded.
