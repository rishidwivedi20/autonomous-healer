"""Live dashboard: a local web page that shows every metric moving while a real fault is injected.

  GET  /                 the page (chaosai/static/dashboard.html)
  GET  /api/report       the findings report as JSON (score, history, ranked weaknesses, fixes)
  GET  /report.html|.md|.json   the same report as a download
  GET  /api/metrics      one JSON snapshot: per-second series for both journeys, pipeline gauges,
                         container table, container events, experiment state, kill switch
  POST /api/inject       {target, fault, params?, duration_s?}  start one experiment in the background
  POST /api/agent/start  {max?, use_llm?}  start the injecting agent: it plans, then injects the faults itself, one at a time
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
from .models import Experiment, ExperimentResult, Thresholds
from .observer import Store, events as query_events, gauge_series, percentile
from .safety import SafetyViolation

STATIC = Path(__file__).parent / "static" / "dashboard.html"
GRAPH_JS = Path(__file__).parent / "static" / "graph.js"
REPORT_JS = Path(__file__).parent / "static" / "report.js"
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

    @staticmethod
    def result_dict(r: ExperimentResult) -> dict:
        return {
            "aborted": r.aborted, "abort_reason": r.abort_reason, "recovered": r.recovered,
            "recovery_s": r.recovery_s, "recovery_probe_s": r.recovery_probe_s,
            "recovery_pipeline_s": r.recovery_pipeline_s, "pipeline_assessed": r.pipeline_assessed, "restarts": r.restarts, "rollback_errors": r.rollback_errors,
            "notes": r.notes, "missing_signals": r.missing_signals,
            "baseline": {j: {"n": s.n, "error_rate": s.error_rate, "p95_ms": s.p95_ms} for j, s in r.baseline_by_journey.items()},
            "during": {j: {"n": s.n, "error_rate": s.error_rate, "p95_ms": s.p95_ms} for j, s in r.during_by_journey.items()},
        }

    def begin_external(self, exp: Experiment) -> None:
        """An experiment the injecting agent is about to run: show it exactly like one started with a button."""
        with self.lock:
            self.state = {"running": True, "phase": "baseline", "id": exp.id, "target": exp.target, "fault": exp.fault,
                          "started_at": time.time(), "fault_start": None, "fault_end": None,
                          "duration_s": exp.duration_s, "result": None, "error": None, "by_agent": True}

    def end_external(self, result_file: str | None, error: str | None) -> None:
        result = None
        if result_file:
            try:
                result = self.result_dict(ExperimentResult.model_validate_json(Path(result_file).read_text(encoding="utf-8")))
            except (OSError, ValueError):
                result = None
        with self.lock:
            self.state.update(running=False, phase="idle", result=result, error=error)

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
            r = self.orch.run(exp, baseline_s=self.cfg.dashboard.baseline_s, recovery_timeout_s=self.cfg.experiment.recovery_timeout_s)
            result = self.result_dict(r)
        except (SafetyViolation, InjectionError) as e:
            error = str(e)
        except Exception as e:                                   # never let the thread die silently
            error = f"{type(e).__name__}: {e}"
        with self.lock:
            self.state.update(running=False, phase="idle", result=result, error=error)


class AgentRunner:
    """Runs the injecting agent in a background thread and exposes what it is doing for the page: the plan (with the reason and
    hypothesis for each experiment), which one is running now, how each ended, and the agent's own log. Nothing here decides
    anything: planning, injection, safety and rollback are the same code `chaos agent` uses (`build_fn` assembles them)."""

    LOG_LINES = 40

    def __init__(self, runner: Runner, kill, build_fn):
        self.runner, self.kill, self.build_fn = runner, kill, build_fn
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.state: dict = self._idle()

    @staticmethod
    def _idle() -> dict:
        return {"available": True, "running": False, "phase": "idle", "plan": [], "current": None, "log": [],
                "planner": None, "planner_model": None, "planner_note": None, "stop_reason": None, "error": None,
                "started_at": None, "ended_at": None}

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def _log(self, msg: str) -> None:
        with self.lock:
            self.state["log"] = (self.state["log"] + [{"t": time.time(), "msg": msg}])[-self.LOG_LINES:]

    def start(self, use_llm: bool, max_n: int) -> None:
        if self.kill.engaged:
            raise SafetyViolation("kill switch is engaged; press Reset first")
        if self.runner.snapshot()["running"]:
            raise SafetyViolation("an experiment is already running")
        with self.lock:
            if self.state["running"]:
                raise SafetyViolation("the agent is already running")
            self.state = {**self._idle(), "running": True, "phase": "planning", "started_at": time.time()}
        self.thread = threading.Thread(target=self._run, args=(use_llm, max_n), daemon=True, name="injecting-agent")
        self.thread.start()

    def _update(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)

    def _progress(self, event: str, i: int, exp: Experiment, entry) -> None:
        with self.lock:
            row = self.state["plan"][i] if i < len(self.state["plan"]) else None
            if event == "start":
                self.state.update(phase="injecting", current=i)
                if row:
                    row["status"] = "running"
            elif row and entry is not None:
                row.update(status="skipped" if event == "skipped" else ("error" if event == "error" else ("aborted" if entry.aborted else "done")),
                           aborted=entry.aborted, abort_reason=entry.abort_reason, recovered=entry.recovered,
                           recovery_s=entry.recovery_s, detail=entry.detail[:300])
        if event == "start":
            self.runner.begin_external(exp)
        else:
            self.runner.end_external(entry.result_file if entry is not None and event == "done" else None,
                                     entry.detail if entry is not None and event in ("skipped", "error") else None)

    def _run(self, use_llm: bool, max_n: int) -> None:
        try:
            self._log("asking the planner which faults to inject" + (" (AI planner, heuristic fallback)" if use_llm else " (heuristic planner)"))
            built = self.build_fn(use_llm, max_n, self._log, self._progress)
            plan = built.plan
            rows = [{"n": n, "id": e.id, "target": e.target, "fault": e.fault, "duration_s": e.duration_s,
                     "reason": p.reason, "hypothesis": p.hypothesis, "status": "pending"}
                    for n, (e, p) in enumerate(zip(built.experiments[:max_n], plan.experiments), 1)]
            self._update(plan=rows, planner=plan.planner, planner_model=plan.model, planner_note=plan.fallback_reason)
            self._log(f"plan ready: {len(rows)} experiment(s) by the {plan.planner} planner" + (f" ({plan.model})" if plan.model else ""))
            run = built.agent.run(built.experiments, planner=plan.planner)
            if built.save:
                built.save(run)
            done = [e for e in run.entries if e.status == "done"]
            self._update(phase="stopped" if run.status == "stopped" else "done", stop_reason=run.stop_reason, current=None,
                         summary={"run": len(done), "aborted": sum(e.aborted for e in done), "recovered": sum(bool(e.recovered) for e in done)})
            self._log("agent finished" if run.status != "stopped" else f"agent stopped: {run.stop_reason}")
        except Exception as e:                                       # never let the thread die silently
            self._update(phase="error", error=f"{type(e).__name__}: {e}"[:400])
            self._log(f"agent failed: {type(e).__name__}: {e}"[:300])
        finally:
            with self.lock:
                self.state.update(running=False, ended_at=time.time())
            if self.runner.snapshot()["running"]:                       # an experiment that was cut short must not look live
                self.runner.end_external(None, "the agent stopped before this experiment finished")


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
    def __init__(self, cfg: Config, store: Store, injector, kill, orch, containers_fn, graph_fn=None, report_fn=None, agent_fn=None):
        self.cfg, self.store, self.injector, self.kill = cfg, store, injector, kill
        self.runner = Runner(orch, cfg)
        self.containers_fn = containers_fn
        self.graph_fn = graph_fn
        self.report_fn = report_fn
        self.agent = AgentRunner(self.runner, kill, agent_fn) if agent_fn else None

    def report(self) -> tuple[int, dict]:
        """The findings report (rule-based analysis of the stored experiments) for the score/history/findings panels."""
        if self.report_fn is None:
            return 503, {"error": "the findings report is not available"}
        try:
            return 200, self.report_fn()
        except Exception as e:
            return 503, {"error": f"could not build the report: {type(e).__name__}: {e}"}

    def report_download(self, kind: str) -> tuple[int, bytes, str, str]:
        """(status, body, content type, file name) for /report.html|md|json."""
        from . import report as report_mod
        code, rep = self.report()
        if code != 200 or rep.get("empty"):
            msg = rep.get("error", "no experiments yet: run one first")
            return (code if code != 200 else 404), msg.encode(), "text/plain; charset=utf-8", ""
        if kind == "html":
            return 200, report_mod.to_html(rep).encode(), "text/html; charset=utf-8", "chaosai-findings.html"
        if kind == "md":
            return 200, report_mod.to_markdown(rep).encode(), "text/markdown; charset=utf-8", "chaosai-findings.md"
        return 200, json.dumps(rep, indent=2).encode(), "application/json", "chaosai-findings.json"

    def graph(self) -> tuple[int, dict]:
        """The dependency map (same data as results/graph.json) for the graph panel."""
        if self.graph_fn is None:
            return 503, {"error": "no dependency graph is available (run `chaos discover`)"}
        try:
            return 200, self.graph_fn()
        except Exception as e:
            return 503, {"error": f"could not load the dependency graph: {e}"}

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
            "agent": self.agent.snapshot() if self.agent else {"available": False},
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
        guard = getattr(self.runner.orch, "guard", None)
        if guard is not None and target not in guard.managed_targets:       # refuse up front; the guard would refuse it later anyway
            return 400, {"error": f"{target!r} is not a managed service (label chaosai.managed=true); allowed: {', '.join(sorted(guard.managed_targets))}"}
        if self.kill.engaged:
            return 409, {"error": "kill switch is engaged; press Reset first"}
        if self.agent and self.agent.snapshot()["running"]:
            return 409, {"error": "the injecting agent is running; wait for it or press the kill switch"}
        params = {**self.cfg.faults[fault].params, **(body.get("params") or {})}
        duration = int(body.get("duration_s") or self.cfg.dashboard.duration_s)
        try:
            exp_id = self.runner.start(target, fault, params, duration)
        except SafetyViolation as e:
            return 409, {"error": str(e)}
        return 202, {"message": f"started {fault} on {target}", "id": exp_id}

    def start_agent(self, body: dict) -> tuple[int, dict]:
        if self.agent is None:
            return 404, {"error": "this dashboard was started without the injecting agent"}
        use_llm, max_n = body.get("use_llm", True), body.get("max", 2)
        if not isinstance(use_llm, bool) or isinstance(max_n, bool) or not isinstance(max_n, int):
            return 400, {"error": "use_llm must be true/false and max an integer"}
        limit = max(1, self.cfg.agent.max_experiments)
        if not 1 <= max_n <= limit:
            return 400, {"error": f"max must be between 1 and {limit}"}
        try:
            self.agent.start(use_llm, max_n)
        except SafetyViolation as e:
            return 409, {"error": str(e)}
        return 202, {"message": f"the injecting agent started: it will plan and inject up to {max_n} fault(s) by itself"}

    def abort(self) -> tuple[int, dict]:
        self.kill.engage("dashboard abort")
        try:
            errors = self.injector.restore_all()
        except Exception as e:
            return 500, {"error": f"kill switch is set, but restore failed: {e}"}
        return (500, {"error": "restore problems: " + "; ".join(errors)}) if errors else (200, {"message": "kill switch engaged; all faults undone"})

    def reset(self) -> tuple[int, dict]:
        if self.agent and self.agent.snapshot()["running"]:
            return 409, {"error": "the injecting agent is still running; use the kill switch to stop it"}
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

    def _send(self, code: int, body: bytes, ctype: str, filename: str = "") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
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
        if path == "/graph.js":
            return self._send(200, GRAPH_JS.read_bytes(), "application/javascript; charset=utf-8")
        if path == "/report.js":
            return self._send(200, REPORT_JS.read_bytes(), "application/javascript; charset=utf-8")
        if path in ("/report.html", "/report.md", "/report.json"):
            code, body, ctype, name = self.server.app.report_download(path.rsplit(".", 1)[1])
            return self._send(code, body, ctype, name)
        if path == "/api/report":
            return self._json(*self.server.app.report())
        if path == "/api/metrics":
            return self._json(200, self.server.app.metrics())
        if path == "/api/graph":
            return self._json(*self.server.app.graph())
        self._json(404, {"error": "not found"})

    def do_POST(self):
        # read the (bounded) body before answering anything, even a refusal: replying to a request whose body is still
        # in flight can reset the client's connection (seen on Windows) instead of delivering the 403
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        in_bounds = 0 <= length <= 10_000
        raw = self.rfile.read(length) if in_bounds else b""
        if not in_bounds:
            self.close_connection = True
        if not self._host_ok() or not self._origin_ok():
            return self._json(403, {"error": "cross-origin request refused"})
        if self.headers.get("X-Chaos") != "1" or "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._json(403, {"error": "missing X-Chaos header or JSON content type"})
        try:
            body = json.loads(raw or b"{}") if in_bounds else None
        except (ValueError, json.JSONDecodeError):
            body = None
        if not isinstance(body, dict):
            return self._json(400, {"error": "body must be a JSON object"})
        app = self.server.app
        path = self.path.split("?")[0]
        if path == "/api/inject":
            code, out = app.inject(body)
        elif path == "/api/agent/start":
            code, out = app.start_agent(body)
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
