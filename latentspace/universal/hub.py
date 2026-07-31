"""The super-evolution hub: every evolution job on one page.

    python3 -m latentspace.universal.hub            # port 8800

Every reporting server registers itself in a global registry
(~/.latentspace/registry.jsonl, override with LATENTSPACE_REGISTRY) the
moment it starts — agentic runs and tensor solve() runs alike. The hub
reads that registry and renders one self-refreshing page with a card
per run: LIVE or FINISHED, the best score per task or fitness function,
a mini fitness curve, and a link to the run's own full dashboard while
it is alive.

The hub needs nothing from the runs to be running: agentic state
(state.json) and solver telemetry (telemetry.jsonl) are persisted in
each run directory, so finished runs stay on the board with their final
curves. However many problems, algorithms, agents, and machines are
evolving at once, this is the one page to watch.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .serve import agentic_curves, curve_svg, registry_path, \
    telemetry_curves


def load_registry():
    """Latest entry per run directory, newest first."""
    path = registry_path()
    if not os.path.exists(path):
        return []
    latest = {}
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if os.path.isdir(entry["run_dir"]):   # vanished tmp dirs
                    latest[entry["run_dir"]] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return sorted(latest.values(), key=lambda e: -e.get("started", 0))


def run_status(entry):
    """One run's card data, read from disk plus a liveness probe."""
    run_dir = entry["run_dir"]
    info = {"run_dir": run_dir, "name": os.path.basename(run_dir.rstrip("/")),
            "port": entry.get("port"), "live": False, "kind": "solver",
            "best": {}, "evaluations": 0, "series": {}, "points": []}
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{entry['port']}/summary",
                timeout=0.5) as r:
            json.loads(r.read())
        info["live"] = True
    except Exception:
        pass
    state_path = os.path.join(run_dir, "state.json")
    telem_path = os.path.join(run_dir, "telemetry.jsonl")
    try:
        if os.path.exists(state_path):
            state = json.load(open(state_path))
            info["kind"] = "agentic"
            inds = list(state.get("individuals", {}).values())
            info["evaluations"] = len(inds)
            info["series"], info["points"] = agentic_curves(inds)
            info["best"] = {t: (None if b is None else round(b["score"], 5))
                            for t, b in state.get("best", {}).items()}
        if os.path.exists(telem_path):
            telemetry = [json.loads(l) for l in open(telem_path)]
            info["series"].update(telemetry_curves(telemetry))
            if telemetry:
                info["evaluations"] = max(
                    info["evaluations"],
                    telemetry[-1].get("evaluations") or 0)
                if not info["best"]:
                    info["best"] = {k: round(v, 5) for k, v in
                                    (telemetry[-1].get("best") or {}).items()}
    except (json.JSONDecodeError, OSError):
        info["error"] = "unreadable run data"
    return info


def hub_html():
    cards = []
    for entry in load_registry():
        info = run_status(entry)
        badge = ('<span style="color:#7c7">&#9679; LIVE</span>'
                 if info["live"] else
                 '<span style="color:#777">&#9632; finished</span>')
        title = html.escape(info["name"])
        if info["live"]:
            title = (f'<a href="http://127.0.0.1:{info["port"]}/progress" '
                     f'style="color:#7ac">{title}</a>')
        best = " &#183; ".join(f"{html.escape(str(k))}: <b>{v}</b>"
                               for k, v in info["best"].items()) or "&#8212;"
        curve = curve_svg(info["series"], xlabel="evaluations",
                          w=340, h=120, labels=False)
        cards.append(
            f'<div class="card"><div class="head">{badge} '
            f'<span class="t">{title}</span> '
            f'<span class="k">{info["kind"]}, '
            f'{info["evaluations"]} evals</span></div>'
            f'<div class="best">{best}</div>{curve}</div>')
    body = "".join(cards) or "<p>no runs registered yet</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5"><title>super evolution</title><style>
body{{font-family:ui-monospace,monospace;margin:1.5em;background:#111;
color:#ddd}} h1{{font-size:1.05em;color:#7ac}}
.grid{{display:flex;flex-wrap:wrap;gap:14px}}
.card{{background:#181818;border:1px solid #333;padding:10px 12px;
border-radius:6px}}
.head{{margin-bottom:4px;font-size:.9em}} .t{{font-weight:bold}}
.k{{color:#888;font-size:.85em}} .best{{color:#9a9;font-size:.85em;
margin-bottom:6px}} a{{text-decoration:none}}</style></head><body>
<h1>super evolution &#8212; every run on this machine</h1>
<div class="grid">{body}</div></body></html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=8800)
    a = p.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            page = hub_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, fmt, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"[hub] http://127.0.0.1:{server.server_address[1]}/  "
          f"(registry: {registry_path()})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
