import http.client
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from chaosai import config
from chaosai.dashboard import App, build_series, make_server
from chaosai.observer import Store
from chaosai.safety import KillSwitch, SafetyViolation


# ---------------------------------------------------------------- series
def add(store, t, kind="vote", ok=1, lat=10.0):
    store.execute("INSERT INTO probes VALUES (?,?,?,?,?,?)", (t, kind, ok, 200 if ok else 500, lat, None))


def test_series_buckets_per_second_per_journey(tmp_path):
    s = Store(tmp_path / "t.db")
    add(s, 100.1, "vote", 1, 10); add(s, 100.6, "vote", 0, 5000); add(s, 100.7, "vote", 1, 30)
    add(s, 100.2, "result", 1, 4)
    add(s, 102.5, "vote", 1, 12)
    out = build_series(s, now=104.2, window_s=5)                # whole seconds 99..103
    assert out["t"] == [99, 100, 101, 102, 103]
    v = out["vote"]
    assert v["n"] == [0, 3, 0, 1, 0]
    assert v["err"] == [None, 33.3, None, 0.0, None]            # no data = null, not 0
    assert v["p95"][1] == 30.0 and v["p50"][1] == 10.0           # the failed 5000 ms request is excluded from latency
    assert out["result"]["n"][1] == 1 and out["result"]["p95"][1] == 4.0


def test_series_excludes_the_in_progress_second(tmp_path):
    s = Store(tmp_path / "t.db")
    add(s, 104.5)
    assert build_series(s, now=104.9, window_s=5)["vote"]["n"] == [0, 0, 0, 0, 0]     # 104 is still being filled


def test_all_failed_second_has_error_100_and_null_latency(tmp_path):
    s = Store(tmp_path / "t.db")
    add(s, 100.1, ok=0, lat=3000); add(s, 100.2, ok=0, lat=3000)
    out = build_series(s, now=102.5, window_s=3)["vote"]
    assert out["err"][1] == 100.0 and out["p95"][1] is None


# ---------------------------------------------------------------- runner
class FakeInjector:
    def __init__(self):
        self.restored = 0

    def inject(self, *a, **k):
        return object()

    def rollback(self, h):
        return []

    def restore_all(self):
        self.restored += 1
        return []


class FakeOrch:
    """Blocks inside run() until released, so tests can observe the phases."""

    def __init__(self, fail=None):
        self.injector = FakeInjector()
        self.release, self.entered, self.fail = threading.Event(), threading.Event(), fail
        self.exps = []

    def run(self, exp, baseline_s, recovery_timeout_s):
        self.exps.append((exp, baseline_s, recovery_timeout_s))
        self.entered.set()
        if self.fail:
            raise self.fail
        h = self.injector.inject(exp.target, exp.fault, exp.params)
        self.release.wait(5)
        self.injector.rollback(h)
        st = lambda e, p: SimpleNamespace(n=5, error_rate=e, p95_ms=p)
        return SimpleNamespace(aborted=False, abort_reason=None, recovered=True, recovery_s=2.0, restarts={"redis": 1},
                               rollback_errors=[], notes=["crashed redis"], missing_signals=[],
                               baseline_by_journey={"vote": st(0.0, 12.0)}, during_by_journey={"vote": st(0.5, 300.0)})


def wait_for(cond, secs=3):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def make_app(tmp_path, orch=None):
    cfg = config.load(Path("none.yaml"))
    kill = KillSwitch(tmp_path / "KILL")
    orch = orch or FakeOrch()
    app = App(cfg, Store(tmp_path / "t.db"), orch.injector, kill, orch,
              lambda: {"redis": {"state": "running", "health": "healthy", "restarts": 0}})
    return app, orch, kill


def test_runner_tracks_phases_and_fault_window(tmp_path):
    app, orch, _ = make_app(tmp_path)
    code, body = app.inject({"target": "redis", "fault": "kill"})
    assert code == 202 and orch.entered.wait(2)
    assert wait_for(lambda: app.runner.snapshot()["phase"] == "fault")
    st = app.runner.snapshot()
    assert st["running"] and st["fault_start"] and st["fault_end"] is None
    orch.release.set()
    assert wait_for(lambda: not app.runner.snapshot()["running"])
    st = app.runner.snapshot()
    assert st["phase"] == "idle" and st["fault_end"] >= st["fault_start"]
    assert st["result"]["during"]["vote"]["error_rate"] == 0.5 and st["result"]["restarts"] == {"redis": 1}
    assert st["error"] is None


def test_button_experiments_use_config_defaults(tmp_path):
    app, orch, _ = make_app(tmp_path)
    app.inject({"target": "vote", "fault": "network-delay", "params": {"latency_ms": 900}})
    orch.entered.wait(2)
    exp, baseline_s, recovery = orch.exps[0]
    assert exp.params == {"latency_ms": 900} and exp.duration_s == 20 and baseline_s == 8 and recovery == 60
    assert exp.thresholds.error_rate == 0.5
    orch.release.set()


def test_second_experiment_while_running_is_refused(tmp_path):
    app, orch, _ = make_app(tmp_path)
    app.inject({"target": "redis", "fault": "kill"})
    orch.entered.wait(2)
    code, body = app.inject({"target": "db", "fault": "kill"})
    assert code == 409 and "already running" in body["error"]
    orch.release.set()


def test_refused_experiment_reports_the_guard_message(tmp_path):
    app, orch, _ = make_app(tmp_path, FakeOrch(fail=SafetyViolation("'ghost' is not labeled")))
    app.inject({"target": "ghost", "fault": "kill"})
    assert wait_for(lambda: app.runner.snapshot()["error"])
    st = app.runner.snapshot()
    assert "not labeled" in st["error"] and not st["running"] and st["phase"] == "idle"


def test_unexpected_exception_does_not_kill_the_thread_silently(tmp_path):
    app, orch, _ = make_app(tmp_path, FakeOrch(fail=RuntimeError("boom")))
    app.inject({"target": "redis", "fault": "kill"})
    assert wait_for(lambda: app.runner.snapshot()["error"])
    assert "RuntimeError: boom" in app.runner.snapshot()["error"]


# ---------------------------------------------------------------- HTTP
@pytest.fixture
def web(tmp_path):
    app, orch, kill = make_app(tmp_path)
    server = make_server(app, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SimpleNamespace(app=app, orch=orch, kill=kill, port=server.server_port)
    orch.release.set()
    server.shutdown()
    server.server_close()


def call(port, method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    h = {"Content-Type": "application/json", "X-Chaos": "1", **(headers or {})}
    c.putrequest(method, path, skip_host=host is not None)
    if host is not None:
        c.putheader("Host", host)
    for k, v in h.items():
        c.putheader(k, v)
    raw = body if isinstance(body, bytes) else json.dumps(body if body is not None else {}).encode()
    c.putheader("Content-Length", str(len(raw) if method == "POST" else 0))
    c.endheaders(raw if method == "POST" else None)
    r = c.getresponse()
    data = r.read()
    return r.status, (json.loads(data) if r.getheader("Content-Type", "").startswith("application/json") else data.decode())


def test_page_and_metrics_are_served(web):
    status, html = call(web.port, "GET", "/")
    assert status == 200 and "chaos-ai" in html and "Crash Redis" in html and "KILL SWITCH" in html
    status, m = call(web.port, "GET", "/api/metrics")
    assert status == 200 and set(m) >= {"series", "gauges", "events", "containers", "experiment", "kill_switch", "window_s"}
    assert len(m["series"]["t"]) == 120 and m["containers"]["redis"]["restarts"] == 0
    assert call(web.port, "GET", "/nope")[0] == 404


def test_inject_endpoint_starts_the_experiment(web):
    status, body = call(web.port, "POST", "/api/inject", {"target": "redis", "fault": "kill"})
    assert status == 202 and web.orch.entered.wait(2) and "id" in body
    assert call(web.port, "GET", "/api/metrics")[1]["experiment"]["running"] is True


@pytest.mark.parametrize("body, code", [
    ({"target": "redis"}, 400), ({"fault": "kill"}, 400), ({"target": "redis", "fault": "explode"}, 400),
    ({"target": 5, "fault": "kill"}, 400),
])
def test_inject_validates_input(web, body, code):
    assert call(web.port, "POST", "/api/inject", body)[0] == code


def test_non_json_body_is_rejected(web):
    assert call(web.port, "POST", "/api/inject", b"not json")[0] == 400
    assert call(web.port, "POST", "/api/inject", b"[1,2]")[0] == 400


def test_abort_engages_kill_switch_restores_and_blocks_new_faults(web):
    status, body = call(web.port, "POST", "/api/abort")
    assert status == 200 and web.kill.engaged and web.orch.injector.restored == 1
    assert call(web.port, "POST", "/api/inject", {"target": "redis", "fault": "kill"})[0] == 409
    assert call(web.port, "GET", "/api/metrics")[1]["kill_switch"] is True
    status, body = call(web.port, "POST", "/api/reset")
    assert status == 200 and not web.kill.engaged


def test_reset_refused_while_an_experiment_runs(web):
    call(web.port, "POST", "/api/inject", {"target": "redis", "fault": "kill"})
    web.orch.entered.wait(2)
    assert call(web.port, "POST", "/api/reset")[0] == 409


# ---------------------------------------------------------------- security
def test_post_without_the_custom_header_is_refused(web):
    status, body = call(web.port, "POST", "/api/inject", {"target": "redis", "fault": "kill"}, headers={"X-Chaos": "0"})
    assert status == 403 and web.orch.exps == []


def test_post_with_wrong_content_type_is_refused(web):
    assert call(web.port, "POST", "/api/abort", headers={"Content-Type": "text/plain"})[0] == 403
    assert not web.kill.engaged


def test_cross_origin_post_is_refused(web):
    status, _ = call(web.port, "POST", "/api/abort", headers={"Origin": "http://evil.example"})
    assert status == 403 and not web.kill.engaged


def test_same_origin_post_is_allowed(web):
    status, _ = call(web.port, "POST", "/api/reset", headers={"Origin": f"http://127.0.0.1:{web.port}"})
    assert status == 200


def test_foreign_host_header_is_refused_for_get_and_post(web):
    assert call(web.port, "GET", "/api/metrics", host="evil.example")[0] == 403          # DNS rebinding
    assert call(web.port, "POST", "/api/abort", host="evil.example")[0] == 403
    assert call(web.port, "GET", "/api/metrics", host=f"localhost:{web.port}")[0] == 200


def test_server_listens_on_loopback_only(web):
    s = make_server(web.app, 0)
    try:
        assert s.server_address[0] == "127.0.0.1"
    finally:
        s.server_close()


def test_watcher_start_snapshot_is_flagged_initial_but_real_changes_are_not(tmp_path):
    """The watcher re-records every container's state each time it starts. That must not look like a restart."""
    app, orch, _ = make_app(tmp_path)
    ev = lambda t, svc, kind, val: app.store.execute("INSERT INTO events VALUES (?,?,?,?)", (t, svc, kind, val))
    now = time.time()
    ev(now - 500, "redis", "restart_count", "2")          # an earlier session
    ev(now - 60, "redis", "restart_count", "2")           # new watcher start: same value re-recorded
    ev(now - 60, "redis", "state", "running")             # nothing earlier for this kind at all
    ev(now - 30, "redis", "restart_count", "3")           # a REAL restart
    got = {(e["kind"], e["value"], round(now - e["t"])): e["initial"] for e in app._events(now - 120, now)}
    assert got[("restart_count", "2", 60)] is True
    assert got[("state", "running", 60)] is True
    assert got[("restart_count", "3", 30)] is False
