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

PALETTE = ["#7ac", "#c96", "#9c7", "#b8a", "#8cc", "#ca8"]


def registry_path():
    """Global registry of every run this machine has served — one JSON
    line per server start. The hub (hub.py) reads it to show ALL
    evolution jobs, live and finished, on one page."""
    return os.environ.get(
        "LATENTSPACE_REGISTRY",
        os.path.expanduser("~/.latentspace/registry.jsonl"))


def register_run(run_dir, port):
    path = registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"run_dir": os.path.abspath(run_dir),
                            "port": port, "pid": os.getpid(),
                            "started": time.time()}) + "\n")


def curve_svg(series, points=None, xlabel="evaluations", w=720, h=260,
              labels=True):
    """Fitness-over-time chart, dependency-free. series maps a name to
    its best-so-far [(x, y), ...] step curve — one line per task or
    fitness function, same rendering for every kind of run. points are
    optional (x, y, label) markers (agentic individuals)."""
    series = {k: v for k, v in series.items() if len(v) >= 2}
    if not series:
        return "<p>curve appears after two scored points</p>"
    allx = [x for c in series.values() for x, _ in c]
    ally = [y for c in series.values() for _, y in c]
    if points:
        ally += [p[1] for p in points]
    lo, hi = min(ally), max(ally)
    pad = max((hi - lo) * 0.15, 1e-9)
    lo, hi = lo - pad, hi + pad
    ML, MB = (62, 28) if labels else (8, 8)
    xmax = max(allx) * 1.05 + 1e-9

    def X(x):
        return ML + (w - ML - 14) * x / xmax

    def Y(y):
        return 12 + (h - 12 - MB) * (hi - y) / (hi - lo)

    s = [f'<svg width="{w}" height="{h}" style="background:#161616;'
         'border:1px solid #333">']
    if labels:
        s.append(f'<text x="{ML}" y="{h-8}" font-size="10" fill="#888">'
                 f'x &#8212; {xlabel} &#183; y &#8212; best score so far '
                 '(higher is better)</text>')
    for i, (name, curve) in enumerate(sorted(series.items())):
        color = PALETTE[i % len(PALETTE)]
        path = ""
        for j, (x, y) in enumerate(curve):
            path += (f"M {X(x):.1f} {Y(y):.1f} " if j == 0
                     else f"L {X(x):.1f} {Y(curve[j-1][1]):.1f} "
                          f"L {X(x):.1f} {Y(y):.1f} ")
        s.append(f'<path d="{path}" fill="none" stroke="{color}" '
                 'stroke-width="2"/>')
        if labels:
            lx, ly = curve[-1]
            s.append(f'<text x="{X(lx)-4:.1f}" y="{Y(ly)-6:.1f}" '
                     f'font-size="10" fill="{color}" text-anchor="end">'
                     f'{html.escape(str(name))} {ly:.5g}</text>')
    if labels:
        for x, y, lab in (points or []):
            s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" '
                     'fill="#666"/>')
            s.append(f'<text x="{X(x)+5:.1f}" y="{Y(y)+11:.1f}" '
                     f'font-size="8.5" fill="#777">{lab}</text>')
        for gy in (lo + pad, hi - pad):
            s.append(f'<text x="{ML-6}" y="{Y(gy)+3:.1f}" font-size="9" '
                     f'fill="#888" text-anchor="end">{gy:.5g}</text>')
    s.append("</svg>")
    return "".join(s)


def agentic_curves(individuals):
    """Per-task best-ever step curves over tell order; disqualified
    scores (<= -90) count on the x axis but not in the curves."""
    series, points, best, n = {}, [], {}, 0
    for ind in sorted(individuals, key=lambda i: i["id"]):
        n += 1
        if ind["score"] <= -90:
            continue
        t = ind["task"]
        points.append((n, ind["score"], ind["id"]))
        if t not in best or ind["score"] > best[t]:
            best[t] = ind["score"]
        series.setdefault(t, []).append((n, best[t]))
    return series, points


def telemetry_curves(telemetry):
    series = {}
    for point in telemetry:
        x = point.get("evaluations") or point.get("epoch") or 0
        for name, score in (point.get("best") or {}).items():
            if score is None:
                continue
            cur = series.setdefault(name, [])
            y = max(score, cur[-1][1]) if cur else score
            cur.append((x, y))
    return series


class GAService:
    """The engine plus the lock and the save-after-every-mutation rule.

    Also the ONE telemetry sink for every kind of run: agentic runs
    feed it through tell/audit/consolidated, and the tensor solver
    feeds it through POST /telemetry (see live_progress below), so the
    /progress dashboard is the same page for every evolutionary problem
    this library runs. ga may be None (telemetry-only mode)."""

    def __init__(self, ga, state_path, run_dir=None):
        self.ga = ga
        self.state_path = state_path
        self.run_dir = run_dir
        self.lock = threading.Lock()
        self.events = deque(maxlen=300)
        self.telemetry = []
        self.started = time.time()

    def event(self, text):
        self.events.append((time.strftime("%H:%M:%S"), text))

    def call(self, name, body):
        ga = self.ga
        with self.lock:
            if name == "telemetry":
                point = {"epoch": body.get("epoch"),
                         "evaluations": body.get("evaluations"),
                         "best": body.get("best", {})}
                self.telemetry.append(point)
                if self.run_dir:
                    with open(os.path.join(self.run_dir,
                                           "telemetry.jsonl"), "a") as f:
                        f.write(json.dumps(point) + "\n")
                return {"ok": True}, False
            if ga is None:
                if name == "summary":
                    last = self.telemetry[-1] if self.telemetry else {}
                    return {"telemetry_points": len(self.telemetry),
                            "best": last.get("best", {})}, False
                raise KeyError(f"{name} needs an engine (telemetry-only "
                               "server)")
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

    def curve_svg(self, series, points=None, xlabel="evaluations"):
        return curve_svg(series, points, xlabel)

    def _agentic_curves(self, individuals):
        return agentic_curves(individuals)

    def _telemetry_curves(self):
        return telemetry_curves(self.telemetry)

    def progress_html(self):
        with self.lock:
            ga = self.ga
            if ga is None:
                curves = self._telemetry_curves()
                curve = self.curve_svg(curves, xlabel="evaluations")
                last = self.telemetry[-1] if self.telemetry else {}
                mins = (time.time() - self.started) / 60
                events = "".join(
                    f"<tr><td>{ts}</td><td class=v>{html.escape(e)}</td>"
                    f"</tr>" for ts, e in reversed(self.events))
                return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5"><title>latentspace run</title><style>
body{{font-family:ui-monospace,monospace;margin:1.5em;background:#111;
color:#ddd}} table{{border-collapse:collapse;margin:.8em 0;width:100%}}
td,th{{border:1px solid #333;padding:.25em .5em;text-align:left;
font-size:.85em}} .v{{color:#9a9}} h1,h2{{font-size:1em;color:#7ac}}
</style></head><body>
<h1>solver run &mdash; {len(self.telemetry)} progress reports,
{mins:.0f} min up &mdash; latest best: {html.escape(json.dumps(
    last.get('best', {})))}</h1>
<h2>fitness over time</h2>{curve}
<h2>events</h2><table>{events}</table></body></html>"""
            s = ga.summary()
            living = sorted((i for i in ga.individuals.values()
                             if i["alive"]),
                            key=lambda i: (i["task"], -i["score"]))
            best = {t: b for t, b in ga.best.items() if b is not None}
            agentic_series, agentic_points = self._agentic_curves(
                list(ga.individuals.values()))
            agentic_series.update(self._telemetry_curves())
            curve = self.curve_svg(agentic_series, agentic_points,
                                   xlabel="evaluation order")
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
         "audit", "telemetry"}


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


def serve(run_dir, port=0, tasks=None, telemetry_only=False, **ga_kwargs):
    """Build (or load) the run's engine and return a ready server.
    port=0 picks a free port. Caller runs .serve_forever().
    telemetry_only=True starts the same server with no engine — the
    dashboard for tensor solve() runs (see live_progress)."""
    os.makedirs(run_dir, exist_ok=True)
    if telemetry_only:
        ga, state_path = None, None
    else:
        state_path = os.path.join(run_dir, "state.json")
        if os.path.exists(state_path):
            ga = AgenticGA.load(state_path)
        else:
            if not tasks:
                raise SystemExit("no state.json — pass --tasks to "
                                 "create a run")
            ga = AgenticGA(tasks=tasks, **ga_kwargs)
            ga.save(state_path)
    service = GAService(ga, state_path, run_dir=run_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(service))
    server.service = service
    with open(os.path.join(run_dir, "server.json"), "w") as f:
        json.dump({"port": server.server_address[1], "pid": os.getpid()}, f)
    register_run(run_dir, server.server_address[1])
    return server


def live_progress(run_dir=None, port=0, names=None):
    """One dashboard for every run: pass the result as solve()'s
    progress= callback and open the printed URL.

        from latentspace.universal import solve, live_progress
        solve(fitness_fns, output_shape=(64, 64), epochs=10_000,
              progress=live_progress())

    Starts a telemetry-only reporting server (same /progress page the
    agentic substrate uses) in a daemon thread and returns a callback
    with solve()'s progress signature. names labels the fitness
    functions on the chart (default fn0, fn1, ...)."""
    import tempfile
    run_dir = run_dir or tempfile.mkdtemp(prefix="latentspace-live-")
    server = serve(run_dir, port=port, telemetry_only=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/progress"
    print(f"[latentspace] live progress: {url}", flush=True)

    def progress(epoch, epochs, evaluations, best_pheno, best_score):
        best = {}
        for i, score in enumerate(best_score):
            label = names[i] if names and i < len(names) else f"fn{i}"
            value = float(score)
            if value == value and abs(value) != float("inf"):
                best[label] = value
        server.service.handle("telemetry", {
            "epoch": int(epoch), "evaluations": int(evaluations),
            "best": best})
        server.service.event(f"epoch {epoch}/{epochs} "
                             f"evals={evaluations} best={best}")

    progress.url = url
    progress.server = server
    return progress


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
    p.add_argument("--telemetry", action="store_true",
                   help="no engine: dashboard-only server for solver runs")
    a = p.parse_args()
    server = serve(a.run, a.port, a.tasks, telemetry_only=a.telemetry,
                   founders=a.founders,
                   children=a.children, population_cap=a.population_cap,
                   consolidate_every=a.consolidate_every, seed=a.seed)
    print(f"[serve] run={a.run} port={server.server_address[1]}",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
