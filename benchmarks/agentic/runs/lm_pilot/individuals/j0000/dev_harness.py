"""Offline dev harness for j0000 — NOT the canonical scorer.

Trains an artifact with a short budget on the train split MINUS its
last 32768 bytes and evaluates on that held-back tail (a dev slice
inside the train split; the scorer's val/holdout splits are never
touched here). Takes the same GPU file lock the canonical scorer uses
so concurrent canonical runs on this machine are not corrupted.

Usage: python3 dev_harness.py <artifact.py> <budget_seconds>
"""
import fcntl
import importlib.util
import json
import sys
import time

import numpy as np

TASK = "/Users/daniellosey/Documents/latentspace/benchmarks/agentic/tasks/lm"
spec = importlib.util.spec_from_file_location("canon", TASK + "/score.py")
canon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canon)

DEV = 32768


def main():
    artifact, budget = sys.argv[1], float(sys.argv[2])
    train_full, _, _ = canon.splits()
    train_head = train_full[:-DEV]
    dev = train_full[-DEV:]
    with open(canon.LOCK, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        ns = {}
        with open(artifact) as f:
            exec(compile(f.read(), artifact, "exec"), ns)
        t0 = time.time()
        model_fn = ns["train"](train_head, budget, 0)
        train_time = time.time() - t0
        te0 = time.time()
        bpb = canon.bits_per_byte(model_fn, dev)
        print(json.dumps({"dev_bpb": round(bpb, 5),
                          "train_seconds": round(train_time, 1),
                          "eval_seconds": round(time.time() - te0, 1),
                          "budget": budget}))


if __name__ == "__main__":
    main()
