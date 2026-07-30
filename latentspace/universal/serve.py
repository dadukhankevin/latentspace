"""Live reporting server for the agentic substrate: agents tell the
engine directly instead of relaying results through the orchestrator.

    python3 -m latentspace.universal.serve --run benchmarks/agentic/runs/r2 \
        --tasks binpack tsp            # creates state.json if absent

One process holds the ONE AgenticGA instance; a lock serializes every
request, so concurrent agents can report the moment they finish without
racing each other or a shared state file (the file-relay pattern is only
safe because the orchestrator is the single writer — this server is the
single writer with the relay removed). State is saved to the run's
state.json after every mutating call, so a crash loses nothing.

The server binds localhost only and writes `server.json` (port, pid)
into the run directory so agents can discover it. JSON in, JSON out:

    GET  /summary               engine summary
    GET  /due                   {"due": bool} — consolidation due?
    GET  /batch                 per-task best-evers (consolidation input)
    GET  /stale                 individuals whose score predates the base
    GET  /contradictions        contradiction report per task
    POST /ask                   {"n": optional} -> jobs for one round
    POST /tell                  {job_id, variation, score, artifact?,
                                 contradicts_base?, log?} -> {"id": ...}
    POST /abandon               {job_id}
    POST /consolidated          {} -> survivors owing rewrites
                                (call AFTER editing the base playbook)
    POST /rewrite               {id, variation, contradicts_base?}
    POST /rescore               {id, score, artifact?}
    POST /audit                 {id, passed}

An agent reports its own result with one line:

    curl -s -X POST localhost:PORT/tell -d '{"job_id": "j0004", ...}'
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agentic import AgenticGA


class GAService:
    """The engine plus the lock and the save-after-every-mutation rule."""

    def __init__(self, ga, state_path):
        self.ga = ga
        self.state_path = state_path
        self.lock = threading.Lock()

    def call(self, name, body):
        ga = self.ga
        with self.lock:
            if name == "summary":
                return ga.summary(), False
            if name == "due":
                return {"due": ga.consolidation_due()}, False
            if name == "batch":
                return ga.consolidation_batch(), False
            if name == "stale":
                return ga.stale(), False
            if name == "contradictions":
                return ga.contradiction_report(), False
            if name == "ask":
                return ga.ask(), True
            if name == "tell":
                ind = ga.tell(body["job_id"], body["variation"],
                              body["score"], artifact=body.get("artifact"),
                              contradicts_base=body.get(
                                  "contradicts_base", False),
                              log=body.get("log"))
                return {"id": ind}, True
            if name == "abandon":
                ga.abandon(body["job_id"])
                return {"ok": True}, True
            if name == "consolidated":
                return ga.record_consolidation(), True
            if name == "rewrite":
                ga.tell_rewrite(body["id"], body["variation"],
                                contradicts_base=body.get(
                                    "contradicts_base"))
                return {"ok": True}, True
            if name == "rescore":
                ga.retell_score(body["id"], body["score"],
                                artifact=body.get("artifact"))
                return {"ok": True}, True
            if name == "audit":
                ga.mark_audited(body["id"], body.get("passed", True))
                return {"ok": True}, True
            raise KeyError(name)

    def handle(self, name, body):
        result, mutated = self.call(name, body)
        if mutated:
            self.ga.save(self.state_path)
        return result


GETS = {"summary", "due", "batch", "stale", "contradictions"}
POSTS = {"ask", "tell", "abandon", "consolidated", "rewrite", "rescore",
         "audit"}


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self, allowed):
            name = self.path.lstrip("/").split("?")[0]
            if name not in allowed:
                self._reply(404, {"error": f"unknown route {name!r}"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                self._reply(200, service.handle(name, body))
            except KeyError as e:
                self._reply(404, {"error": f"unknown id/job {e}"})
            except Exception as e:
                self._reply(400, {"error": repr(e)})

        def do_GET(self):
            self._route(GETS)

        def do_POST(self):
            self._route(POSTS)

        def log_message(self, fmt, *args):
            print(f"[serve] {args[0]}", flush=True)
    return Handler


def serve(run_dir, port=0, tasks=None, **ga_kwargs):
    """Build (or load) the run's engine and return a ready server.
    port=0 picks a free port. Caller runs .serve_forever()."""
    state_path = os.path.join(run_dir, "state.json")
    if os.path.exists(state_path):
        ga = AgenticGA.load(state_path)
    else:
        if not tasks:
            raise SystemExit("no state.json — pass --tasks to create a run")
        os.makedirs(run_dir, exist_ok=True)
        ga = AgenticGA(tasks=tasks, **ga_kwargs)
        ga.save(state_path)
    service = GAService(ga, state_path)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(service))
    with open(os.path.join(run_dir, "server.json"), "w") as f:
        json.dump({"port": server.server_address[1], "pid": os.getpid()}, f)
    return server


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--founders", type=int, default=2)
    p.add_argument("--children", type=int, default=4)
    p.add_argument("--population-cap", type=int, default=12)
    p.add_argument("--consolidate-every", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    server = serve(a.run, a.port, a.tasks, founders=a.founders,
                   children=a.children, population_cap=a.population_cap,
                   consolidate_every=a.consolidate_every, seed=a.seed)
    print(f"[serve] run={a.run} port={server.server_address[1]}",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
