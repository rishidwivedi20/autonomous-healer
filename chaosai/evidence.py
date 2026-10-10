"""Read-only evidence layer: everything the analyzer is allowed to cite.

Each public method is one agent tool. They read only what the experiments actually stored (the
results/*.json files, results/graph.json, results/baseline.json and the SQLite probe/gauge/event
tables) and return small JSON-able dicts in consistent units (ms, %, seconds relative to the moment the
fault was applied). Nothing here estimates or invents a value: a signal that was not recorded comes back
as null, and `quality` lists known weaknesses of an experiment's data.

The analyzer is only allowed to cite numbers that appear in these tool outputs (see analysis.py), so any
figure worth citing must be computed here, deterministically, from stored data.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from statistics import median

from .models import ExperimentResult, Graph
from .observer import Store, percentile, restart_counts

WINDOWS = ("around", "baseline", "during", "recovery")
JOURNEYS = ("vote", "result")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


DENSE_KEYS = {"per_second", "rows"}           # time series: their `t_s` row labels are indices, not findings


def numbers_in(obj, ignore_ids: set[str] | None = None) -> set[float]:
    """Every number a claim may cite from a JSON-like structure: values, including ones embedded in strings
    ('16m', 'x5.0'). NOT counted: dict keys (`p95_ms` would make 95 citable forever), the `t_s` labels of dense
    time-series rows (they would make almost every small integer citable), experiment ids, 'pid N'. This test
    ('the value appears in what the tools returned') is necessary, not sufficient, so it is kept as tight as it
    can be; evidence is still only as good as the model's reading of the numbers."""
    out: set[float] = set()

    def walk(x, dense=False):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            out.add(float(x))
        elif isinstance(x, str):
            s = x
            for i in ignore_ids or ():
                s = s.replace(i, " ")
            s = re.sub(r"pid\s*\d+", " ", s)
            s = re.sub(r"\bp\d{1,3}\b", " ", s, flags=re.I)   # percentile labels (p50, p95) are names, not values
            out.update(float(m) for m in _NUM.findall(s))
        elif isinstance(x, dict):
            for k, v in x.items():
                if dense and k == "t_s":
                    continue
                walk(v, dense or k in DENSE_KEYS)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v, dense)

    walk(obj)
    return out


def _pct(rate):
    return None if rate is None else round(rate * 100, 1)


def _r(x, n=1):
    return None if x is None else round(float(x), n)


def _jstats(by_journey: dict) -> dict:
    return {j: {"n": s.n, "error_pct": _pct(s.error_rate), "p95_ms": _r(s.p95_ms)} for j, s in by_journey.items()}


class Evidence:
    def __init__(self, results_dir: Path | str, store: Store | None = None):
        self.dir = Path(results_dir)
        db = self.dir / "chaosai.db"
        self.store = store if store is not None else (Store(db) if db.exists() else None)
        self._files: dict[str, Path] = {}
        self._exps: dict[str, ExperimentResult] = {}
        for f in sorted(self.dir.glob("*.json")):
            try:
                r = ExperimentResult.model_validate_json(f.read_text(encoding="utf-8"))
            except ValueError:
                continue                                    # graph.json, baseline.json, plan.json, ...
            if not r.dry_run:
                self._exps[r.experiment.id] = r
                self._files[r.experiment.id] = f

    # ------------------------------------------------------------ helpers
    def ids(self) -> list[str]:
        return list(self._exps)

    def results(self) -> dict[str, ExperimentResult]:
        return dict(self._exps)

    def restarts_of(self, exp_id: str) -> dict:
        return self._restarts(self._exps[exp_id])

    def _need(self, exp_id: str) -> ExperimentResult | dict:
        r = self._exps.get(exp_id)
        return r if r else {"error": f"unknown experiment_id {exp_id!r}", "known_ids": self.ids()}

    @staticmethod
    def _fault_len(r: ExperimentResult) -> int:
        return max(1, len(r.per_second))

    def _bounds(self, r: ExperimentResult) -> tuple[float, float]:
        """The stretch of time that belongs to THIS experiment: from 30 s before the fault (but never before
        the previous experiment finished) to the end of its own recovery measurement. Without this a window
        runs into the next experiment and its crash loop gets blamed on the wrong fault."""
        t0 = r.started_at
        prev_end = max((o.ended_at for o in self._exps.values() if o is not r and o.ended_at <= t0), default=0.0)
        next_start = min((o.started_at for o in self._exps.values() if o.started_at > t0), default=float("inf"))
        # the end is the next experiment's baseline window (or 90 s after the fault), NOT this experiment's own
        # recovery cut-off: a backlog can take longer to drain than the recovery timeout allowed for
        return max(t0 - 30, prev_end), min(t0 + self._fault_len(r) + 90, next_start - 20)

    def _window(self, r: ExperimentResult, name: str) -> tuple[float, float]:
        t0, lf = r.started_at, self._fault_len(r)
        a, b = {"baseline": (-30, 0), "during": (0, lf), "recovery": (lf, lf + 60), "around": (-10, lf + 40)}[name]
        lo, hi = self._bounds(r)
        return max(t0 + a, lo), min(t0 + b, hi)

    def fingerprint(self, ids: list[str]) -> str:
        """Digest of every input the analysis depends on, for cache keys."""
        h = hashlib.sha256()
        for i in sorted(ids):
            h.update(self._files[i].read_bytes())
        for name in ("graph.json", "baseline.json"):
            p = self.dir / name
            h.update(p.read_bytes() if p.exists() else b"-")
        return h.hexdigest()[:24]

    # ------------------------------------------------------------ tools
    def list_experiments(self) -> dict:
        return {"experiments": [
            {"id": i, "target": r.experiment.target, "fault": r.experiment.fault, "aborted": r.aborted,
             "recovered": r.recovered, "data_quality_issues": len(self._quality(r))}
            for i, r in self._exps.items()]}

    def _restarts(self, r: ExperimentResult) -> dict:
        """Automatic restarts, recomputed from the stored events so results written before the reset-aware
        counting (negative counts after a `stop`) are corrected; falls back to the stored value."""
        if self.store is None:
            return r.restarts
        return restart_counts(self.store, r.started_at, r.ended_at)

    def _quality(self, r: ExperimentResult) -> list[str]:
        q = []
        if not r.during_by_journey:
            q.append("no per-journey stats (recorded by an older build); only combined numbers exist")
        if r.rollback_errors:
            q.append("rollback problems were reported: " + "; ".join(r.rollback_errors)[:200])
        if r.abort_reason == "kill switch":
            q.append("ended early by the kill switch; recovery was not measured")
        if not r.pipeline_assessed:
            q.append("recovery judged by the probe alone (no pipeline gauges were usable, or an older build)")
        if r.missing_signals:
            q.append("missing signals: " + "; ".join(r.missing_signals))
        return q

    def get_experiment(self, experiment_id: str) -> dict:
        r = self._need(experiment_id)
        if isinstance(r, dict):
            return r
        e = r.experiment
        return {
            "id": e.id, "target": e.target, "fault": e.fault, "params": e.params, "planned_duration_s": e.duration_s,
            "hypothesis": e.hypothesis or None, "fault_active_s": self._fault_len(r),
            "aborted": r.aborted, "abort_reason": r.abort_reason,
            "recovered": r.recovered, "recovery_s": _r(r.recovery_s),
            "recovery_probe_s": _r(r.recovery_probe_s), "recovery_pipeline_s": _r(r.recovery_pipeline_s),
            "pipeline_assessed": r.pipeline_assessed,
            "restarts": self._restarts(r), "rollback_errors": r.rollback_errors, "notes": r.notes,
            "baseline_by_journey": _jstats(r.baseline_by_journey), "during_by_journey": _jstats(r.during_by_journey),
            "overall": {"baseline": {"n": r.baseline.n, "error_pct": _pct(r.baseline.error_rate), "p95_ms": _r(r.baseline.p95_ms)},
                        "during": {"n": r.during.n, "error_pct": _pct(r.during.error_rate), "p95_ms": _r(r.during.p95_ms)}},
            "quality": self._quality(r),
        }

    def get_metrics(self, experiment_id: str, window: str = "around", journey: str | None = None) -> dict:
        r = self._need(experiment_id)
        if isinstance(r, dict):
            return r
        if window not in WINDOWS:
            return {"error": f"window must be one of {WINDOWS}"}
        if journey is not None and journey not in JOURNEYS:
            return {"error": f"journey must be one of {JOURNEYS}"}
        if self.store is None:
            return {"error": "no probe database (results/chaosai.db) is available"}
        a, b = self._window(r, window)
        t0 = r.started_at
        kinds = (journey,) if journey else JOURNEYS
        per: dict[str, dict[int, list]] = {k: {} for k in kinds}
        for t, kind, ok, lat in self.store.query("SELECT t, kind, ok, latency_ms FROM probes WHERE t >= ? AND t < ?", (a, b)):
            if kind in per:
                per[kind].setdefault(int((t - t0) // 1), []).append((ok, lat))
        rows, summary = [], {}
        secs = sorted({s for k in kinds for s in per[k]})
        for s in secs:
            row = {"t_s": s}
            for k in kinds:
                b_ = per[k].get(s)
                if b_:
                    lats = sorted(l for o, l in b_ if o)
                    row[k] = {"n": len(b_), "error_pct": round(100 * sum(1 for o, _ in b_ if not o) / len(b_), 1),
                              "p95_ms": _r(percentile(lats, 0.95)) if lats else None}
            rows.append(row)
        for k in kinds:
            errs = [row[k]["error_pct"] for row in rows if k in row]
            p95 = [row[k]["p95_ms"] for row in rows if k in row and row[k]["p95_ms"] is not None]
            summary[k] = {"seconds_observed": len(errs), "seconds_with_errors": sum(1 for x in errs if x > 0),
                          "max_error_pct": max(errs) if errs else None, "max_p95_ms": max(p95) if p95 else None,
                          "median_p95_ms": _r(median(p95)) if p95 else None}
        return {"experiment_id": experiment_id, "window": window, "t_s_means": "seconds since the fault was applied",
                "fault_active_s": self._fault_len(r), "summary": summary, "per_second": rows}

    def get_gauges(self, experiment_id: str) -> dict:
        r = self._need(experiment_id)
        if isinstance(r, dict):
            return r
        if self.store is None:
            return {"error": "no database (results/chaosai.db) is available"}
        t0, lf = r.started_at, self._fault_len(r)
        lo, hi = self._bounds(r)
        raw = self.store.query("SELECT t, name, value FROM gauges WHERE t >= ? AND t <= ? ORDER BY t", (lo, hi))
        q = [(round(t - t0, 1), v) for t, n, v in raw if n == "redis_queue_len"]
        d = [(round(t - t0, 1), v) for t, n, v in raw if n == "db_votes"]
        if not q:
            return {"experiment_id": experiment_id, "error": "no gauge readings were stored for this experiment (watcher not running?)"}

        def rate(a, b):
            s = [(t, v) for t, v in d if a <= t < b and v is not None]
            return round((s[-1][1] - s[0][1]) / (s[-1][0] - s[0][0]), 1) if len(s) >= 2 and s[-1][0] > s[0][0] else None

        base_q = [v for t, v in q if t < 0 and v is not None]
        peak_t, peak = max(((t, v) for t, v in q if t >= 0 and v is not None), key=lambda x: x[1], default=(None, None))
        base_max = max(base_q) if base_q else None
        drained = None
        if peak_t is not None and base_max is not None:
            drained = next((t for t, v in q if t > peak_t and v is not None and v <= base_max + 2), None)
        after = rate(lf, lf + 90)
        # the worker's real throughput: votes reaching Postgres per second while the backlog drains
        draining = rate(peak_t, drained) if peak_t is not None and drained is not None else (
            rate(peak_t, lf + 90) if peak_t is not None and peak and peak > (base_max or 0) + 2 else None)
        step = max(1, len(q) // 60)
        rows = [{"t_s": q[i][0], "redis_queue_len": q[i][1],
                 "db_votes": next((v for t, v in d if abs(t - q[i][0]) < 1.5), None)} for i in range(0, len(q), step)]
        return {
            "experiment_id": experiment_id, "t_s_means": "seconds since the fault was applied", "fault_active_s": lf,
            "summary": {
                "baseline_queue_max": base_max, "peak_queue": peak, "peak_queue_at_s": peak_t,
                "queue_back_to_baseline_at_s": drained,
                "votes_per_s_before": rate(-30, 0), "votes_per_s_during": rate(0, lf), "votes_per_s_after": after,
                "votes_per_s_while_draining": draining,
                "peak_backlog_lag_s": round(peak / draining, 1) if peak and draining and draining > 0 else None,
                "db_unreadable_readings": sum(1 for t, v in d if v is None and t >= 0),
                "queue_unreadable_readings": sum(1 for t, v in q if v is None and t >= 0),
                "gauge_readings": len(q)},
            "rows": rows}

    def get_container_events(self, experiment_id: str, service: str | None = None) -> dict:
        r = self._need(experiment_id)
        if isinstance(r, dict):
            return r
        if self.store is None:
            return {"error": "no database (results/chaosai.db) is available"}
        t0, lf = r.started_at, self._fault_len(r)
        out = []
        lo, hi = self._bounds(r)
        for t, svc, kind, val in self.store.query(
                "SELECT t, service, kind, value FROM events WHERE t >= ? AND t <= ? ORDER BY t", (max(t0 - 2, lo), hi)):
            if service and svc != service:
                continue
            prev = self.store.query("SELECT value FROM events WHERE service=? AND kind=? AND t < ? ORDER BY t DESC LIMIT 1", (svc, kind, t))
            if not prev or prev[0][0] == val:
                continue                                    # the watcher's start-up snapshot is not a state change
            out.append({"t_s": round(t - t0, 1), "service": svc, "kind": kind, "value": val})
        return {"experiment_id": experiment_id, "t_s_means": "seconds since the fault was applied",
                "restarts": self._restarts(r), "events": out}

    def get_graph(self) -> dict:
        p = self.dir / "graph.json"
        if not p.exists():
            return {"error": "results/graph.json does not exist; run `chaos discover`"}
        g = Graph.model_validate(json.loads(p.read_text(encoding="utf-8")))
        return {"critical_path": g.critical_path, "services": {
            n: {"replicas": s.replicas, "has_healthcheck": s.healthcheck, "restart_policy": s.restart_policy,
                "has_resource_limits": s.has_limits, "stateful": s.stateful, "depends_on": s.depends_on,
                "fan_in": s.fan_in, "fan_out": s.fan_out, "on_critical_path": s.on_critical_path} for n, s in g.services.items()}}

    def get_baseline(self) -> dict:
        p = self.dir / "baseline.json"
        if not p.exists():
            return {"error": "results/baseline.json does not exist; run `chaos baseline`"}
        b = json.loads(p.read_text(encoding="utf-8"))

        def conv(s):
            return {"n": s["n"], "error_pct": _pct(s["error_rate"]), "p50_ms": _r(s["p50_ms"]), "p95_ms": _r(s["p95_ms"])}
        return {"seconds": b["seconds"], "overall": conv(b["overall"]), "by_journey": {k: conv(v) for k, v in b["by_kind"].items()}}

    # ------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict | None) -> dict:
        args = args or {}
        fn = {"list_experiments": self.list_experiments, "get_experiment": self.get_experiment,
              "get_metrics": self.get_metrics, "get_gauges": self.get_gauges,
              "get_container_events": self.get_container_events, "get_graph": self.get_graph,
              "get_baseline": self.get_baseline}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return fn(**args)
        except TypeError as e:                              # wrong/missing arguments from the model
            return {"error": f"bad arguments for {name}: {e}"}


# Tool definitions sent to the model (the schema mirrors Evidence's method signatures).
_ID = {"experiment_id": {"type": "string", "description": "an id from list_experiments"}}
TOOLS = [
    {"name": "list_experiments", "description": "List every stored experiment (id, target, fault, aborted, recovered, data_quality_issues).",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "get_experiment", "description": "Full stored record of one experiment: params, hypothesis, aborted/recovered, recovery (overall, probe-only, pipeline-only), container restarts, rollback errors, baseline vs during stats per user journey, and data-quality caveats.",
     "input_schema": {"type": "object", "properties": _ID, "required": ["experiment_id"], "additionalProperties": False}},
    {"name": "get_metrics", "description": "Per-second probe metrics (n, error_pct, p95_ms for the vote and result journeys) around one experiment, with a summary. Times are seconds since the fault was applied.",
     "input_schema": {"type": "object", "properties": {**_ID,
                      "window": {"type": "string", "enum": list(WINDOWS), "description": "default 'around' = 10 s before to 40 s after the fault"},
                      "journey": {"type": "string", "enum": list(JOURNEYS)}},
                      "required": ["experiment_id"], "additionalProperties": False}},
    {"name": "get_gauges", "description": "Pipeline gauges read inside the containers: redis_queue_len (votes waiting for the worker) and db_votes (rows in Postgres), with a summary: baseline max, peak queue and when, when it drained, votes/s before/during/after, backlog lag.",
     "input_schema": {"type": "object", "properties": _ID, "required": ["experiment_id"], "additionalProperties": False}},
    {"name": "get_container_events", "description": "Container state/health/restart-count changes around one experiment (watcher start-up snapshots excluded).",
     "input_schema": {"type": "object", "properties": {**_ID, "service": {"type": "string"}}, "required": ["experiment_id"], "additionalProperties": False}},
    {"name": "get_graph", "description": "The service dependency map: per service replicas, healthcheck, restart policy, resource limits, stateful, dependencies, fan-in/out, critical path.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "get_baseline", "description": "The steady-state baseline measured before any fault (error %, p50/p95 ms per journey).",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
]
