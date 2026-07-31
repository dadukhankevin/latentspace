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
import html
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agentic import AgenticGA


class GAService:
    """The engine plus the lock and the save-after-every-mutation rule."""

    def __init__(self, ga, state_path):
        self.ga = ga
        self.state_path = state_path
        self.lock = threading.Lock()
        self.events = deque(maxlen=300)
        self.started = time.time()

    def event(self, text):
        self.events.append((time.strftime("%H:%M:%S"), text))

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
            if name == "tell":
                self.event(f"tell {result['id']} score={body['score']:.5f}"
                           f" ({body.get('variation', '')[:70]}...)")
            elif name == "ask":
                self.event(f"ask: {len(result)} jobs "
                           f"({', '.join(j['kind'] for j in result)})")
            elif name == "consolidated":
                self.event(f"CONSOLIDATED -> base v"
                           f"{self.ga.base_version}; "
                           f"{len(result)} rewrites owed")
            elif name == "rewrite":
                self.event(f"rewrite {body['id']}")
            elif name == "audit":
                self.event(f"audit {body['id']}: "
                           f"{'pass' if body.get('passed', True) else 'FAIL'}")
            elif name == "abandon":
                self.event(f"abandon {body['job_id']}")
        return result

    # ------------------------------------------------------ progress page

    def fitness_svg(self, individuals):
        """Best-ever fitness curve over evaluation order, dependency-free.
        Scores <= -90 (disqualified / failed) are excluded from the
        curve but counted on the x axis."""
        pts, curve, best, n = [], [], None, 0
        for ind in sorted(individuals, key=lambda i: i["id"]):
            n += 1
            if ind["score"] <= -90:
                continue
            pts.append((n, ind["score"], ind["id"]))
            best = ind["score"] if best is None else max(best,
                                                         ind["score"])
            curve.append((n, best))
        if len(pts) < 2:
            return ""
        ys = [p[1] for p in pts]
        lo, hi = min(ys), max(ys)
        pad = max((hi - lo) * 0.15, 1e-9)
        lo, hi = lo - pad, hi + pad
        W, H, ML, MB = 720, 240, 60, 26
        xmax = n + 1

        def X(x):
            return ML + (W - ML - 12) * x / xmax

        def Y(y):
            return 10 + (H - 10 - MB) * (hi - y) / (hi - lo)

        s = [f'<svg width="{W}" height="{H}" style="background:#161616;'
             'border:1px solid #333">']
        s.append(f'<text x="{ML}" y="{H-8}" font-size="10" fill="#888">'
                 'x &#8212; evaluation order &#183; y &#8212; score '
                 '(higher is better) &#183; line = best ever</text>')
        path = ""
        for i, (x, y) in enumerate(curve):
            path += (f"M {X(x):.1f} {Y(y):.1f} " if i == 0
                     else f"L {X(x):.1f} {Y(curve[i-1][1]):.1f} "
                          f"L {X(x):.1f} {Y(y):.1f} ")
        s.append(f'<path d="{path}" fill="none" stroke="#7ac" '
                 'stroke-width="2"/>')
        for x, y, lab in pts:
            s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" '
                     'fill="#888"/>')
            s.append(f'<text x="{X(x)+6:.1f}" y="{Y(y)-6:.1f}" '
                     f'font-size="9" fill="#9a9">{lab} {y:.4g}</text>')
        for gy in (lo + pad, hi - pad):
            s.append(f'<text x="{ML-6}" y="{Y(gy)+3:.1f}" font-size="9" '
                     f'fill="#888" text-anchor="end">{gy:.4g}</text>')
        s.append("</svg>")
        return "".join(s)

    def progress_html(self):
        with self.lock:
            ga = self.ga
            s = ga.summary()
            living = sorted((i for i in ga.individuals.values()
                             if i["alive"]),
                            key=lambda i: (i["task"], -i["score"]))
            best = {t: b for t, b in ga.best.items() if b is not None}
            curve = self.fitness_svg(list(ga.individuals.values()))
        rows = "".join(
            f"<tr><td>{i['id']}</td><td>{i['task']}</td>"
            f"<td>{i['score']:.5f}</td><td>{i['origin']}</td>"
            f"<td>{'yes' if i['scored_on_base'] < ga.base_version else ''}"
            f"</td><td>{'!' if i['contradicts_base'] else ''}</td>"
            f"<td class=v>{html.escape(i['variation'][:160])}</td></tr>"
            for i in living)
        bests = "".join(
            f"<tr><td>{t}</td><td><b>{b['score']:.5f}</b></td>"
            f"<td>{b['id']}</td>"
            f"<td class=v>{html.escape((b['variation'] or '')[:160])}</td>"
            f"</tr>" for t, b in sorted(best.items()))
        events = "".join(f"<tr><td>{ts}</td>"
                         f"<td class=v>{html.escape(e)}</td></tr>"
                         for ts, e in reversed(self.events))
        mins = (time.time() - self.started) / 60
        return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5"><title>agentic run</title><style>
body{{font-family:ui-monospace,monospace;margin:1.5em;background:#111;
color:#ddd}} table{{border-collapse:collapse;margin:.8em 0;width:100%}}
td,th{{border:1px solid #333;padding:.25em .5em;text-align:left;
font-size:.85em}} th{{background:#1d1d1d}} .v{{color:#9a9;max-width:44em;
overflow:hidden;white-space:nowrap;text-overflow:ellipsis}}
h1,h2{{font-size:1em;color:#7ac}} .k{{color:#c96}}</style></head><body>
<h1>agentic run &mdash; round {s['round']}, base v{s['base_version']},
population {s['population']}, {s['stale']} stale,
{s['open_jobs']} jobs in flight, {mins:.0f} min up</h1>
<h2>best ever per task</h2>
<table><tr><th>task</th><th>score</th><th>id</th><th>variation</th></tr>
{bests}</table>
<h2>fitness curve</h2>
{curve}
<h2>living population</h2>
<table><tr><th>id</th><th>task</th><th>score</th><th>origin</th>
<th>stale</th><th>contra</th><th>variation</th></tr>{rows}</table>
<h2>events (newest first)</h2>
<table>{events}</table></body></html>"""


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
            if name in ("", "progress") and allowed is GETS:
                page = service.progress_html().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
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
