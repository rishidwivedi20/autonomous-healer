"""Live dashboard: a local web page that shows every metric moving while a real fault is injected.

  GET  /                 the page (chaosai/static/dashboard.html)
  GET  /api/metrics      one JSON snapshot: per-second series for both journeys, pipeline gauges,
                         container table, container events, experiment state, kill switch
  POST /api/inject       {target, fault, params?, duration_s?}  start one experiment in the background
  POST /api/abort        kill switch: abort the running experiment, undo every fault
  POST /api/reset        undo everything, then clear the kill switch

Everything shown is read from the same SQLite store the CLI uses; nothing is synthesised. Injection
goes through the same Orchestrator, safety Guard and rollback as `chaos run`.

Security: bound to 127.0.0.1 only. Because a web page can make a browser POST to localhost, every
request must carry a Host of 127.0.0.1/localhost on this port, POSTs additionally need
`X-Chaos: 1` (which forces a CORS preflight we never grant) and a same-origin Origin if one is sent.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import Config
from .injector import InjectionError
from .models import Experiment, Thresholds
from .observer import Store, events as query_events, gauge_series, percentile
from .safety import SafetyViolation

STATIC = Path(__file__).parent / "static" / "dashboard.html"
JOURNEYS = ("vote", "result")
GAUGES = ("redis_queue_len", "db_votes")


# ---------------------------------------------------------------- data
def build_series(store: Store, now: float, window_s: int) -> dict:
    """Per-second stats for each journey over the last window_s whole seconds (the in-progress second is
    excluded). Latency percentiles cover successful requests only; a second with no data is null."""
    end = int(now)
    start = end - window_s
    buckets: dict[str, dict[int, list[tuple[int, float]]]] = {k: {} for k in JOURNEYS}
    for t, kind, ok, lat in store.query("SELECT t, kind, ok, latency_ms FROM probes WHERE t >= ? AND t < ?", (start, end)):
        if kind in buckets:
            buckets[kind].setdefault(int(t), []).append((ok, lat))
    ts = list(range(start, end))
    out: dict = {"t": ts}
    for kind in JOURNEYS:
        n, err, p50, p95 = [], [], [], []
        for s in ts:
            b = buckets[kind].get(s, [])
            n.append(len(b))
            err.append(round(100.0 * sum(1 for ok, _ in b if not ok) / len(b), 1) if b else None)
            lats = sorted(lat for ok, lat in b if ok)
            p50.append(round(percentile(lats, 0.50), 1) if lats else None)
            p95.append(round(percentile(lats, 0.95), 1) if lats else None)
        out[kind] = {"n": n, "err": err, "p50": p50, "p95": p95}
    return out


class TimedInjector:
    """Wraps the injector so the dashboard knows the exact moments a fault was applied and lifted."""

    def __init__(self, inner, runner: "Runner"):
        self._inner, self._runner = inner, runner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def inject(self, *a, **k):
        self._runner._mark("fault")
        try:
            return self._inner.inject(*a, **k)
        except Exception:
            self._runner._mark("baseline")      # nothing was applied
            raise

    def rollback(self, handle):
        errors = self._inner.rollback(handle)
        self._runner._mark("recovery")
        return errors


class Runner:
    """Runs one experiment at a time in a background thread and tracks its phase for the page."""

    def __init__(self, orch, cfg: Config):
        self.orch, self.cfg = orch, cfg
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.state: dict = {"running": False, "phase": "idle", "result": None, "error": None}
        orch.injector = TimedInjector(orch.injector, self)

    def _mark(self, phase: str) -> None:
        with self.lock:
            now = time.time()
            self.state["phase"] = phase
            if phase == "fault":
                self.state["fault_start"], self.state["fault_end"] = now, None
            elif phase == "recovery":
                self.state["fault_end"] = now

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    def start(self, target: str, fault: str, params: dict, duration_s: int) -> str:
        c = self.cfg
        with self.lock:
            if self.state["running"]:
                raise SafetyViolation("an experiment is already running")
            exp = Experiment(
                id=f"{time.strftime('%Y%m%d-%H%M%S')}-{target}-{fault}-{uuid.uuid4().hex[:4]}", target=target,
                fault=fault, params=params, duration_s=duration_s,
                thresholds=Thresholds(error_rate=c.safety.abort_error_rate, p95_ms=c.safety.abort_p95_ms,
                                      breach_seconds=c.safety.breach_seconds))
            self.state = {"running": True, "phase": "baseline", "id": exp.id, "target": target, "fault": fault,
                          "started_at": time.time(), "fault_start": None, "fault_end": None,
                          "duration_s": duration_s, "result": None, "error": None}
        self.thread = threading.Thread(target=self._run, args=(exp,), daemon=True, name="experiment")
        self.thread.start()
        return exp.id

    def _run(self, exp: Experiment) -> None:
        result, error = None, None
        try:
            r = self.orch.run(exp, baseline_s=self.cfg.dashboard.baseline_s, recovery_timeout_s=60)
            result = {
                "aborted": r.aborted, "abort_reason": r.abort_reason, "recovered": r.recovered,
                "recovery_s": r.recovery_s, "restarts": r.restarts, "rollback_errors": r.rollback_errors,
                "notes": r.notes, "missing_signals": r.missing_signals,
                "baseline": {j: {"n": s.n, "error_rate": s.error_rate, "p95_ms": s.p95_ms} for j, s in r.baseline_by_journey.items()},
                "during": {j: {"n": s.n, "error_rate": s.error_rate, "p95_ms": s.p95_ms} for j, s in r.during_by_journey.items()},
            }
        except (SafetyViolation, InjectionError) as e:
            error = str(e)
        except Exception as e:                                   # never let the thread die silently
            error = f"{type(e).__name__}: {e}"
        with self.lock:
            self.state.update(running=False, phase="idle", result=result, error=error)


class ContainerTable(threading.Thread):
    """Refreshes the container table once a second so HTTP requests never wait on Docker."""

    def __init__(self, table_fn):
        super().__init__(daemon=True, name="container-table")
        self.table_fn, self.data, self._stop_evt = table_fn, {}, threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.data = self.table_fn()
            except Exception:
                pass                                              # Docker hiccup: keep the last table
            self._stop_evt.wait(1.0)

    def stop(self):
        self._stop_evt.set()


class App:
    def __init__(self, cfg: Config, store: Store, injector, kill, orch, containers_fn):
        self.cfg, self.store, self.injector, self.kill = cfg, store, injector, kill
        self.runner = Runner(orch, cfg)
        self.containers_fn = containers_fn

    # ---- reads
    def _events(self, start: float, end: float) -> list[dict]:
        """Container events, each flagged `initial` when it only re-states what was already known: the
        watcher records a first snapshot every time it starts, which is not a state change and must not be
        drawn as a crash/restart."""
        out = []
        for e in query_events(self.store, start, end):
            prev = self.store.query(
                "SELECT value FROM events WHERE service=? AND kind=? AND t < ? ORDER BY t DESC LIMIT 1",
                (e.service, e.kind, e.t))
            out.append({**e.model_dump(), "initial": (not prev) or prev[0][0] == e.value})
        return out

    def metrics(self) -> dict:
        now = time.time()
        w = self.cfg.dashboard.window_s
        return {
            "now": now, "window_s": w,
            "series": build_series(self.store, now, w),
            "gauges": {g: [[round(t, 2), v] for t, v in gauge_series(self.store, g, now - w, now)] for g in GAUGES},
            "events": self._events(now - w, now),
            "containers": self.containers_fn(),
            "experiment": self.runner.snapshot(),
            "kill_switch": self.kill.engaged,
            "probe_rate_per_s": self.cfg.dashboard.probe_rate_per_s,
        }

    # ---- actions (each returns (http_status, body))
    def inject(self, body: dict) -> tuple[int, dict]:
        target, fault = body.get("target"), body.get("fault")
        if not isinstance(target, str) or not isinstance(fault, str):
            return 400, {"error": "target and fault are required strings"}
        if fault not in self.cfg.faults:
            return 400, {"error": f"unknown fault {fault!r}"}
        if self.kill.engaged:
            return 409, {"error": "kill switch is engaged; press Reset first"}
        params = {**self.cfg.faults[fault].params, **(body.get("params") or {})}
        duration = int(body.get("duration_s") or self.cfg.dashboard.duration_s)
        try:
            exp_id = self.runner.start(target, fault, params, duration)
        except SafetyViolation as e:
            return 409, {"error": str(e)}
        return 202, {"message": f"started {fault} on {target}", "id": exp_id}

    def abort(self) -> tuple[int, dict]:
        self.kill.engage("dashboard abort")
        try:
            errors = self.injector.restore_all()
        except Exception as e:
            return 500, {"error": f"kill switch is set, but restore failed: {e}"}
        return (500, {"error": "restore problems: " + "; ".join(errors)}) if errors else (200, {"message": "kill switch engaged; all faults undone"})

    def reset(self) -> tuple[int, dict]:
        if self.runner.snapshot()["running"]:
            return 409, {"error": "an experiment is still running; wait for it to stop"}
        try:
            errors = self.injector.restore_all()
        except Exception as e:
            return 500, {"error": f"restore failed, kill switch left engaged: {e}"}
        if errors:
            return 500, {"error": "kill switch left engaged; restore problems: " + "; ".join(errors)}
        self.kill.reset()
        return 200, {"message": "kill switch cleared"}


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "chaosai-dashboard"

    def log_message(self, *args):                                 # keep the terminal quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _allowed_hosts(self) -> set[str]:
        port = self.server.server_port
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _host_ok(self) -> bool:
        return (self.headers.get("Host") or "") in self._allowed_hosts()

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{h}" for h in self._allowed_hosts()}

    def do_GET(self):
        if not self._host_ok():
            return self._json(403, {"error": "bad Host header"})
        path = self.path.split("?")[0]
        if path == "/":
            return self._send(200, STATIC.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/metrics":
            return self._json(200, self.server.app.metrics())
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._host_ok() or not self._origin_ok():
            return self._json(403, {"error": "cross-origin request refused"})
        if self.headers.get("X-Chaos") != "1" or "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._json(403, {"error": "missing X-Chaos header or JSON content type"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length <= 10_000 else None
        except (ValueError, json.JSONDecodeError):
            body = None
        if not isinstance(body, dict):
            return self._json(400, {"error": "body must be a JSON object"})
        app = self.server.app
        path = self.path.split("?")[0]
        if path == "/api/inject":
            code, out = app.inject(body)
        elif path == "/api/abort":
            code, out = app.abort()
        elif path == "/api/reset":
            code, out = app.reset()
        else:
            code, out = 404, {"error": "not found"}
        self._json(code, out)


class _Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second dashboard bind the same port and share it (two probes, two
    # injectors). Elsewhere it only permits quick restarts, so keep it there.
    allow_reuse_address = os.name != "nt"


def make_server(app: App, port: int) -> ThreadingHTTPServer:
    server = _Server(("127.0.0.1", port), Handler)
    server.app = app
    server.daemon_threads = True
    return server
