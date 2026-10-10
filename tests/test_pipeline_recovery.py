"""Recovery must not be declared while the data pipeline is still broken (the probe alone cannot see that)."""
import pytest

from chaosai.models import Experiment, Stats
from chaosai.observer import Store
from chaosai.orchestrator import (Orchestrator, PipelineBaseline, find_recovery, pipeline_baseline, pipeline_ok)
from chaosai.safety import Guard, KillSwitch

T0 = 1000.0


def gauges(store, rows):
    for t, q, v in rows:
        store.execute("INSERT INTO gauges VALUES (?,?,?)", (t, "redis_queue_len", q))
        store.execute("INSERT INTO gauges VALUES (?,?,?)", (t, "db_votes", v))


def healthy_rows(a, b, v0=100.0, rate=5.0, q=0.0):
    return [(float(t), q, v0 + rate * (t - a)) for t in range(int(a), int(b))]


# ---------------------------------------------------------------- baseline
def test_pipeline_baseline_from_gauges(tmp_path):
    s = Store(tmp_path / "t.db")
    gauges(s, [(t, q, 100 + 5 * (t - 0)) for t, q in [(0, 0), (1, 2), (2, 1), (3, 0), (4, 1)]])
    b = pipeline_baseline(s, 0, 10)
    assert b.queue_max == 2 and b.votes_rate == pytest.approx(5.0)


def test_pipeline_baseline_is_none_without_usable_gauges(tmp_path):
    s = Store(tmp_path / "t.db")
    assert pipeline_baseline(s, 0, 10) is None                      # no readings at all
    gauges(s, [(0, 0, None), (1, 0, None), (2, 0, None)])
    assert pipeline_baseline(s, 0, 10) is None                      # db unreadable: cannot judge
    s2 = Store(tmp_path / "t2.db")
    gauges(s2, [(0, 0, 100)])
    assert pipeline_baseline(s2, 0, 10) is None                     # one reading gives no rate


def test_baseline_vote_rate_never_negative(tmp_path):
    s = Store(tmp_path / "t.db")
    gauges(s, [(0, 0, 500), (4, 0, 100)])
    assert pipeline_baseline(s, 0, 10).votes_rate == 0.0


# ---------------------------------------------------------------- pipeline_ok
BASE = PipelineBaseline(queue_max=2.0, votes_rate=5.0)


def window_rows(tmp_path, rows):
    s = Store(tmp_path / "w.db")
    gauges(s, rows)
    return s


def test_healthy_window_is_ok(tmp_path):
    assert pipeline_ok(window_rows(tmp_path, healthy_rows(0, 6)), 0, 5, BASE)


def test_queue_piling_up_is_not_ok(tmp_path):
    rows = healthy_rows(0, 6)
    rows[2] = (2.0, 9.0, rows[2][2])
    assert not pipeline_ok(window_rows(tmp_path, rows), 0, 5, BASE)


def test_queue_slightly_above_baseline_within_slack_is_ok(tmp_path):
    rows = healthy_rows(0, 6, q=4.0)                                 # 2*1.1 + 2 slack = 4.2
    assert pipeline_ok(window_rows(tmp_path, rows), 0, 5, BASE)


def test_unreadable_gauge_is_not_ok(tmp_path):
    rows = healthy_rows(0, 6)
    rows[3] = (3.0, 0.0, None)
    assert not pipeline_ok(window_rows(tmp_path, rows), 0, 5, BASE)


def test_postgres_not_ingesting_is_not_ok_even_with_an_empty_queue(tmp_path):
    rows = [(float(t), 0.0, 200.0) for t in range(6)]                # votes flat
    assert not pipeline_ok(window_rows(tmp_path, rows), 0, 5, BASE)


def test_half_speed_ingest_is_the_boundary(tmp_path):
    slow = [(float(t), 0.0, 100 + 2.0 * t) for t in range(6)]        # 2/s < 50% of 5/s
    assert not pipeline_ok(window_rows(tmp_path, slow), 0, 5, BASE)
    fine = [(float(t), 0.0, 100 + 3.0 * t) for t in range(6)]        # 3/s >= 2.5/s
    s2 = Store(tmp_path / "w2.db")
    gauges(s2, fine)
    assert pipeline_ok(s2, 0, 5, BASE)


def test_no_readings_yet_is_not_recovery(tmp_path):
    assert not pipeline_ok(Store(tmp_path / "e.db"), 0, 5, BASE)


def test_zero_baseline_rate_skips_the_growth_check(tmp_path):
    flat = [(float(t), 0.0, 50.0) for t in range(6)]
    assert pipeline_ok(window_rows(tmp_path, flat), 0, 5, PipelineBaseline(queue_max=0, votes_rate=0.0))


# ---------------------------------------------------------------- joint scan
def probe_rows(store, a, b, ok=True):
    for t in range(int(a), int(b)):
        for kind in ("vote", "result"):
            store.execute("INSERT INTO probes VALUES (?,?,?,?,?,?)", (t + 0.5, kind, int(ok), 200, 10.0, None))


def test_both_must_hold_in_the_same_window(tmp_path):
    s = Store(tmp_path / "j.db")
    probe_rows(s, 0, 40)                                              # probe healthy throughout
    gauges(s, [(float(t), 30.0, 500.0) for t in range(0, 12)] + healthy_rows(12, 40, v0=500))   # pipeline broken until t=12
    base = Stats(n=50, error_rate=0.0, p50_ms=10, p95_ms=10)
    assert find_recovery(s, 0, 40, base) == 0.0                                                  # probe alone: instantly "recovered"
    assert find_recovery(s, 0, 40, base, pipeline=BASE) == 12.0                                  # truth: 12 s
    assert find_recovery(s, 0, 40, base, pipeline=BASE, probe=False) == 12.0


# ---------------------------------------------------------------- orchestrator end to end (simulated clock)
class Sim:
    def __init__(self, store, probe_ok, gauge_fn):
        self.t, self.store, self.probe_ok, self.gauge_fn = T0, store, probe_ok, gauge_fn
        self._next, self._votes = T0, 100.0

    def clock(self):
        return self.t

    def sleep(self, s):
        target = self.t + s
        while self.t < target - 1e-9:
            self.t = min(self.t + 0.25, target)
            while self._next <= self.t + 1e-9:
                for kind in ("vote", "result"):
                    ok = self.probe_ok(self._next)
                    self.store.execute("INSERT INTO probes VALUES (?,?,?,?,?,?)",
                                       (self._next, kind, int(ok), 200 if ok else 500, 10.0, None))
                g = self.gauge_fn(self._next) if self.gauge_fn else None
                if g is not None:
                    q, progress = g
                    if progress:
                        self._votes += 5
                    self.store.execute("INSERT INTO gauges VALUES (?,?,?)", (self._next, "redis_queue_len", q))
                    self.store.execute("INSERT INTO gauges VALUES (?,?,?)",
                                       (self._next, "db_votes", None if progress is None else self._votes))
                self._next += 1


class Inj:
    def __init__(self, state):
        self.state = state

    class H:
        actions = ["fake"]

    def inject(self, *a, **k):
        self.state["fault"], self.state["t_inject"] = True, self.state["clock"]()
        return self.H()

    def rollback(self, h):
        self.state["fault"], self.state["t_rollback"] = False, self.state["clock"]()
        return []


def build(tmp_path, gauge_fn):
    store, state = Store(tmp_path / "o.db"), {}
    sim = Sim(store, lambda t: True, (lambda t: gauge_fn(t, state)) if gauge_fn else None)
    state["clock"] = sim.clock
    orch = Orchestrator(store, Inj(state), Guard({"db"}, KillSwitch(tmp_path / "KILL")), tmp_path / "res",
                        clock=sim.clock, sleep=sim.sleep)
    return orch, state


def exp(fault="pause"):
    return Experiment(id="e1", target="db", fault=fault, duration_s=10)


def test_pipeline_stall_after_the_fault_delays_recovery_even_though_the_probe_is_fine(tmp_path):
    """The real db-pause case: vote requests keep succeeding, but the queue backs up and postgres stops
    ingesting until well after the fault is lifted."""
    def gauge(t, st):
        broken = st.get("fault") or t < st.get("t_rollback", 1e18) + 8
        return (25.0, False) if broken and st.get("t_inject") and t >= st["t_inject"] else (0.0, True)
    orch, state = build(tmp_path, gauge)
    r = orch.run(exp(), baseline_s=10, recovery_timeout_s=60)
    assert r.pipeline_assessed and r.recovered
    assert r.recovery_probe_s == 0.0                                  # the probe alone would have said "0 s"
    assert 7 <= r.recovery_pipeline_s <= 10
    assert r.recovery_s == r.recovery_pipeline_s                      # the combined verdict follows the laggard


def test_unreadable_gauges_during_the_fault_mean_not_recovered_until_readable(tmp_path):
    def gauge(t, st):
        if st.get("fault"):
            return (0.0, None)                                        # psql hangs while db is paused
        return (0.0, True)
    orch, _ = build(tmp_path, gauge)
    r = orch.run(exp(), baseline_s=10, recovery_timeout_s=60)
    assert r.recovered and r.recovery_s >= 1                          # a window overlapping the unreadable period never counts


def test_no_gauges_falls_back_to_probe_only_and_says_so(tmp_path):
    orch, _ = build(tmp_path, None)
    r = orch.run(exp(), baseline_s=10, recovery_timeout_s=60)
    assert r.recovered and not r.pipeline_assessed and r.recovery_pipeline_s is None
    assert any("pipeline recovery not assessed" in n for n in r.notes)


def test_no_impact_anywhere_still_reports_zero(tmp_path):
    orch, _ = build(tmp_path, lambda t, st: (0.0, True))
    r = orch.run(exp("kill"), baseline_s=10, recovery_timeout_s=60)
    assert r.recovered and r.recovery_s == 0 and r.recovery_pipeline_s == 0 and r.recovery_probe_s == 0


def test_pipeline_that_never_recovers_is_reported_not_faked(tmp_path):
    def gauge(t, st):
        return (30.0, False) if st.get("t_inject") and t >= st["t_inject"] else (0.0, True)
    orch, _ = build(tmp_path, gauge)
    r = orch.run(exp(), baseline_s=10, recovery_timeout_s=20)
    assert not r.recovered and r.recovery_s is None and r.recovery_pipeline_s is None and r.recovery_probe_s == 0.0
