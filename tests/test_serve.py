"""The reporting server driven end to end over localhost HTTP —
concurrent tells included, since removing the single-writer relay is the
server's whole reason to exist."""
import json
import threading
import urllib.request

import pytest

from latentspace.universal.agentic import AgenticGA
from latentspace.universal.serve import serve


@pytest.fixture
def server(tmp_path):
    srv = serve(str(tmp_path), port=0, tasks=["alpha", "beta"],
                founders=2, children=4, consolidate_every=1, seed=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    port = srv.server_address[1]

    def call(name, body=None):
        url = f"http://127.0.0.1:{port}/{name}"
        if body is None:
            req = urllib.request.Request(url)
        else:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(), method="POST")
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    call.port = port
    yield call, tmp_path
    srv.shutdown()


def test_full_round_over_http(server):
    call, run_dir = server
    jobs = call("ask", {})
    assert len(jobs) == 4 and all(j["kind"] == "found" for j in jobs)

    # concurrent tells: every agent reports the moment it finishes
    def report(job):
        call("tell", {"job_id": job["job_id"], "variation": f"v-{job['job_id']}",
                      "score": float(len(job["job_id"])),
                      "artifact": f"/tmp/{job['job_id']}.py"})
    threads = [threading.Thread(target=report, args=(j,)) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = call("summary")
    assert summary["population"] == 4
    assert summary["tasks_alive"] == ["alpha", "beta"]

    assert call("due")["due"] is True
    batch = call("batch")
    assert set(batch) == {"alpha", "beta"}
    call("audit", {"id": batch["alpha"]["id"], "passed": True})

    survivors = call("consolidated", {})
    assert len(survivors) == 4
    call("rewrite", {"id": survivors[0]["id"], "variation": "deeper"})
    call("rescore", {"id": survivors[1]["id"], "score": 9.0})
    assert len(call("stale")) == 3          # one re-scored, three stale

    # state survived on disk after every mutation
    ga = AgenticGA.load(str(run_dir / "state.json"))
    assert ga.summary() == call("summary")
    # server.json advertises the port for agents
    assert "port" in json.load(open(run_dir / "server.json"))


def test_progress_page_renders(server):
    call, _ = server
    jobs = call("ask", {})
    call("tell", {"job_id": jobs[0]["job_id"], "variation": "v", "score": 1.0})
    import urllib.request
    with urllib.request.urlopen(
            f"http://127.0.0.1:{call.port}/progress") as r:
        page = r.read().decode()
    assert "living population" in page and "tell i0000" in page


def test_errors_are_json_not_crashes(server):
    call, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        call("nonsense", {})
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        call("tell", {"job_id": "no-such-job", "variation": "x", "score": 1})
    assert e.value.code in (400, 404)


def test_telemetry_only_dashboard_and_live_progress(tmp_path):
    import numpy as np
    from latentspace.universal import live_progress, solve

    cb = live_progress(run_dir=str(tmp_path / "live"))

    def fitness(phenotypes):        # solve()'s contract: batched
        return -(phenotypes.flatten(1) ** 2).mean(dim=1)

    solve(fitness, output_shape=(8,), epochs=4, children=4,
          population_cap=8, founders=2, device="cpu", seed=0,
          progress=cb, progress_every=1)
    with urllib.request.urlopen(cb.url) as r:
        page = r.read().decode()
    assert "fitness over time" in page and "fn0" in page
    # telemetry persisted alongside the run
    lines = open(tmp_path / "live" / "telemetry.jsonl").read().splitlines()
    assert len(lines) >= 2
    assert "best" in json.loads(lines[-1])
    cb.server.shutdown()


def test_agentic_page_uses_multiseries_curve(server):
    call, _ = server
    jobs = call("ask", {})
    for i, job in enumerate(jobs):
        call("tell", {"job_id": job["job_id"], "variation": f"v{i}",
                      "score": float(i)})
    import urllib.request
    with urllib.request.urlopen(
            f"http://127.0.0.1:{call.port}/progress") as r:
        page = r.read().decode()
    assert "alpha" in page and "beta" in page   # one labeled line per task
