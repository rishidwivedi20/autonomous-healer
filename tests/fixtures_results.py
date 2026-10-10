"""Builds a small, internally consistent results folder for the analyzer tests (unit-test data only)."""
import json
from pathlib import Path

from chaosai.models import Experiment, ExperimentResult, Graph, SecondStat, Service, Stats
from chaosai.observer import Store

T_SPIKE = 1_000_000.0
T_KILL = T_SPIKE + 400.0
SPIKE_ID, KILL_ID = "e-spike", "e-kill"


def _stats(n, err, p95):
    return Stats(n=n, error_rate=err, p50_ms=p95 / 3, p95_ms=p95)


def make_results(tmp_path: Path) -> Path:
    d = Path(tmp_path) / "results"
    d.mkdir()
    store = Store(d / "chaosai.db")

    # ---- probes: 5 req/s per journey around both experiments
    probes = [(t0 + s + k * 0.2, kind, 1, 200, lat, None)
              for t0 in (T_SPIKE, T_KILL) for s in range(-30, 111) for k in range(5)
              for kind, lat in (("vote", 15.0), ("result", 7.0))]
    store.executemany("INSERT INTO probes VALUES (?,?,?,?,?,?)", probes)

    # ---- gauges, one reading per second
    gauges = []
    for s in range(-30, 101):                                           # spike: queue builds to 300, drains at 9.5/s
        q = 0.0 if s < 0 else (15.0 * s if s <= 20 else max(0.0, 300.0 - 4.5 * (s - 20)))
        votes = 1000 + 5 * s if s < 0 else (1000 + 9.5 * s if s <= 86 else 1000 + 9.5 * 86 + 5 * (s - 86))
        gauges += [(T_SPIKE + s, "redis_queue_len", q), (T_SPIKE + s, "db_votes", float(votes))]
    for s in range(-30, 101):                                           # kill: quiet pipeline
        gauges += [(T_KILL + s, "redis_queue_len", 0.0), (T_KILL + s, "db_votes", float(5000 + 5 * s))]
    store.executemany("INSERT INTO gauges VALUES (?,?,?)", gauges)

    # ---- events
    ev = lambda t, svc, kind, val: store.execute("INSERT INTO events VALUES (?,?,?,?)", (t, svc, kind, val))
    ev(T_KILL - 100, "db", "restart_count", "2")
    ev(T_KILL - 100, "db", "state", "running")
    ev(T_KILL - 100, "db", "health", "healthy")
    ev(T_SPIKE - 100, "vote", "health", "starting")
    ev(T_KILL - 1, "db", "state", "running")                            # watcher start-up snapshot: same value again
    ev(T_KILL + 1, "db", "restart_count", "3")
    ev(T_KILL + 1, "db", "health", "starting")
    ev(T_SPIKE + 3, "vote", "health", "healthy")

    # ---- experiment results
    spike = ExperimentResult(
        experiment=Experiment(id=SPIKE_ID, target="vote", fault="traffic-spike", params={"multiplier": 5}, duration_s=20,
                              hypothesis="vote p95 rises but no errors"),
        started_at=T_SPIKE, ended_at=T_SPIKE + 110, baseline=_stats(40, 0.0, 19.0), during=_stats(500, 0.0, 54.0),
        baseline_by_journey={"vote": _stats(20, 0.0, 19.0), "result": _stats(20, 0.0, 8.0)},
        during_by_journey={"vote": _stats(250, 0.0, 54.0), "result": _stats(250, 0.0, 33.0)},
        per_second=[SecondStat(t=float(i), n=50, error_rate=0.0, p50_ms=20.0, p95_ms=54.0) for i in range(20)],
        recovered=False, recovery_probe_s=0.0, pipeline_assessed=True, notes=["traffic x5.0"])
    kill = ExperimentResult(
        experiment=Experiment(id=KILL_ID, target="db", fault="kill", duration_s=20),
        started_at=T_KILL, ended_at=T_KILL + 40, baseline=_stats(40, 0.0, 18.0), during=_stats(100, 0.02, 20.0),
        baseline_by_journey={"vote": _stats(20, 0.0, 18.0), "result": _stats(20, 0.0, 8.0)},
        during_by_journey={"vote": _stats(50, 0.0, 20.0), "result": _stats(50, 0.04, 9.0)},
        per_second=[SecondStat(t=float(i), n=10, error_rate=0.0, p50_ms=7.0, p95_ms=20.0) for i in range(20)],
        recovered=True, recovery_s=6.0, recovery_probe_s=1.0, recovery_pipeline_s=6.0, pipeline_assessed=True,
        restarts={"db": 1, "result": 1, "worker": 1}, notes=["crashed db (SIGKILL; restart policy applies)"])
    for r in (spike, kill):
        (d / f"{r.experiment.id}.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")

    # ---- graph + baseline
    svc = lambda name, **kw: Service(name=name, **{"replicas": 1, "restart_policy": "unless-stopped", **kw})
    graph = Graph(services={
        "vote": svc("vote", healthcheck=True, depends_on=["redis"], entrypoint=True, on_critical_path=True),
        "redis": svc("redis", healthcheck=True, stateful=True, fan_in=2, on_critical_path=True),
        "worker": svc("worker", depends_on=["redis", "db"], fan_out=2, on_critical_path=True),
        "db": svc("db", healthcheck=True, stateful=True, fan_in=2, on_critical_path=True),
        "result": svc("result", healthcheck=True, depends_on=["db"], entrypoint=True, on_critical_path=True)},
        critical_path=["vote", "redis", "worker", "db", "result"])
    (d / "graph.json").write_text(graph.model_dump_json(), encoding="utf-8")
    (d / "baseline.json").write_text(json.dumps({
        "seconds": 120, "overall": {"n": 240, "error_rate": 0.0, "p50_ms": 7.8, "p95_ms": 16.6},
        "by_kind": {"vote": {"n": 120, "error_rate": 0.0, "p50_ms": 15.0, "p95_ms": 19.9},
                    "result": {"n": 120, "error_rate": 0.0, "p50_ms": 6.0, "p95_ms": 7.9}}}), encoding="utf-8")
    store.close()
    return d
