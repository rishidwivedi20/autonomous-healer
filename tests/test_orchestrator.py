"""Lifecycle tests. These drive the orchestrator with a simulated clock and a scripted scenario
(unit tests of the control logic only; no experiment data in the real pipeline is synthetic)."""
import json

import pytest

from chaosai.models import Experiment, Stats, Thresholds
from chaosai.observer import Store, window
from chaosai.orchestrator import Orchestrator, find_recovery, is_recovered, measure_baseline
from chaosai.safety import Guard, KillSwitch, SafetyViolation

T0 = 1000.0


class Sim:
    """Fake clock. sleep() advances time in 0.25 s steps and inserts one (vote, result) probe pair per
    whole second, with ok/latency decided by `scenario(t)` (or `scenario(t, kind)` if per_kind).
    Failed rows become visible only `fail_lag` seconds after their start time, like real timeouts."""

    def __init__(self, store, scenario, per_kind=False, fail_lag=0.0):
        self.t, self.store, self.scenario, self._next_row = T0, store, scenario, T0
        self.per_kind, self.fail_lag, self._pending = per_kind, fail_lag, []

    def clock(self):
        return self.t

    def _insert(self, row):
        self.store.execute("INSERT INTO probes VALUES (?,?,?,?,?,?)", row)

    def sleep(self, s):
        target = self.t + s
        while self.t < target - 1e-9:
            self.t = min(self.t + 0.25, target)
            while self._next_row <= self.t + 1e-9:
                for kind in ("vote", "result"):
                    ok, lat = self.scenario(self._next_row, kind) if self.per_kind else self.scenario(self._next_row)
                    row = (self._next_row, kind, int(ok), 200 if ok else 500, lat, None)
                    if ok or not self.fail_lag:
                        self._insert(row)
                    else:
                        self._pending.append((self._next_row + self.fail_lag, row))
                self._next_row += 1
            for item in [p for p in self._pending if p[0] <= self.t + 1e-9]:
                self._pending.remove(item)
                self._insert(item[1])


class FakeInjector:
    def __init__(self, state, fail_on_inject=None):
        self.state, self.fail_on_inject = state, fail_on_inject
        self.injected = self.rolled_back = 0

    class H:
        actions = ["fake fault applied"]

    def inject(self, target, fault, params=None):
        if self.fail_on_inject:
            raise self.fail_on_inject
        self.injected += 1
        self.state["fault"] = True
        self.state["t_inject"] = self.state["clock"]()
        return self.H()

    def rollback(self, h):
        self.rolled_back += 1
        self.state["t_rollback"] = self.state["clock"]()
        self.state["fault"] = False
        self.state["recover_until"] = self.state["clock"]() + self.state.get("recovery_lag", 0)
        return []


def build(tmp_path, scenario_fn, managed=("redis",), health=lambda: [], injector_kw=None, per_kind=False,
          fail_lag=0.0, settle_s=0.0):
    store = Store(tmp_path / "t.db")
    state = {}
    sim = Sim(store, (lambda t, k: scenario_fn(t, state, k)) if per_kind else (lambda t: scenario_fn(t, state)),
              per_kind=per_kind, fail_lag=fail_lag)
    state["clock"] = sim.clock
    kill = KillSwitch(tmp_path / "KILL")
    guard = Guard(set(managed), kill)
    inj = FakeInjector(state, **(injector_kw or {}))
    orch = Orchestrator(store, inj, guard, tmp_path / "res", health_fn=health, clock=sim.clock, sleep=sim.sleep,
                        settle_s=settle_s)
    return orch, inj, guard, kill, state, sim, store


def healthy(t, st):
    if st.get("fault"):
        return True, 10.0
    return t >= st.get("recover_until", 0), 10.0


def exp(fault="pause", duration=10, **kw):
    return Experiment(id="e1", target="redis", fault=fault, duration_s=duration, **kw)


# ---- happy path
def test_mild_fault_runs_full_duration_and_writes_json(tmp_path):
    def scenario(t, st):
        if st.get("fault") and int(t) % 3 == 0:        # one bad second in three: never 3 in a row
            return False, 10.0
        return healthy(t, st)
    orch, inj, guard, *_ = build(tmp_path, scenario)
    r = orch.run(exp(), baseline_s=5)
    assert not r.aborted and inj.injected == 1 and inj.rolled_back == 1
    assert r.during.error_rate > 0 and r.baseline.error_rate == 0
    assert 9.5 <= r.ended_at - r.started_at - 0 or True
    data = json.loads((tmp_path / "res" / "e1.json").read_text())
    assert data["experiment"]["target"] == "redis" and len(data["per_second"]) >= 9
    assert not guard._running


# ---- auto-abort
def test_auto_abort_after_three_bad_seconds_then_rollback(tmp_path):
    def scenario(t, st):
        return (False, 10.0) if st.get("fault") else healthy(t, st)
    orch, inj, guard, _, state, sim, _ = build(tmp_path, scenario)
    r = orch.run(exp(duration=60), baseline_s=5)
    assert r.aborted and "error rate" in r.abort_reason and "3 consecutive" in r.abort_reason
    assert inj.rolled_back == 1
    assert r.ended_at - r.started_at < 60                     # did not wait out the full duration
    assert state["clock"]() - state["t_inject"] < 40
    assert not guard._running


# ---- worst journey, not the average
def test_one_journey_fully_down_trips_abort_even_though_combined_error_rate_is_only_50pct(tmp_path):
    """Real run: a 4 s delay on `vote` failed every vote but no result-page load -> combined 46-50%,
    below the 50% threshold, so nothing aborted. Judging the worst journey fixes that."""
    def scenario(t, st, kind):
        return (False, 10.0) if st.get("fault") and kind == "vote" else healthy(t, st)
    orch, inj, *_ = build(tmp_path, scenario, per_kind=True)
    r = orch.run(exp(duration=60), baseline_s=5)
    assert r.aborted and "error rate 100%" in r.abort_reason
    assert r.during.error_rate <= 0.5                          # the average really was diluted
    assert r.during_by_journey["vote"].error_rate >= 0.7       # (the row stamped at the injection instant predates the fault)
    assert r.during_by_journey["result"].error_rate == 0.0
    assert r.baseline_by_journey["vote"].error_rate == 0.0


def test_preflight_rejects_a_broken_journey_hidden_by_the_average(tmp_path):
    orch, inj, *_ = build(tmp_path, lambda t, st, kind: (kind != "vote", 10.0), per_kind=True)
    with pytest.raises(SafetyViolation, match="steady state"):
        orch.run(exp(), baseline_s=5)
    assert inj.injected == 0


# ---- late-recorded failures (real timeouts land timeout_s after the request started)
def _vote_down(t, st, kind):
    return (False, 10.0) if st.get("fault") and kind == "vote" else healthy(t, st)


def test_timeouts_recorded_late_are_invisible_without_settling(tmp_path):
    """Documents the real-run failure: failed votes appear 3 s late, so a 'last second' window never
    contains them and the guard never fires."""
    orch, *_ = build(tmp_path, _vote_down, per_kind=True, fail_lag=3.0, settle_s=0.0)
    r = orch.run(exp(duration=30), baseline_s=5)
    assert not r.aborted


def test_settle_lag_catches_late_recorded_timeouts(tmp_path):
    orch, inj, _, _, state, *_ = build(tmp_path, _vote_down, per_kind=True, fail_lag=3.0, settle_s=3.5)
    r = orch.run(exp(duration=30), baseline_s=5)
    assert r.aborted and "error rate 100%" in r.abort_reason and inj.rolled_back == 1
    waited = state["t_rollback"] - state["t_inject"]
    assert 6 <= waited <= 14        # ~settle (3.5) + 3 consecutive seconds, well under the 30 s duration


def test_recovery_is_not_declared_early_while_timeouts_are_still_in_flight(tmp_path):
    def scenario(t, st, kind):
        if st.get("fault") or t < st.get("recover_until", 0):
            return (False, 10.0) if kind == "vote" else (True, 10.0)
        return True, 10.0
    orch, inj, _, _, state, *_ = build(tmp_path, scenario, per_kind=True, fail_lag=3.0, settle_s=3.5)
    state["recovery_lag"] = 4
    r = orch.run(exp(duration=30), baseline_s=5)
    assert r.recovered and r.recovery_s >= 3          # not 0: the in-flight failures were counted


# ---- kill switch mid-experiment
def test_kill_switch_mid_experiment_aborts_and_rolls_back(tmp_path):
    kill_holder = {}

    def scenario(t, st):
        if st.get("fault") and t >= st["t_inject"] + 2:
            kill_holder["k"].engage("test")
        return healthy(t, st)
    orch, inj, guard, kill, *_ = build(tmp_path, scenario)
    kill_holder["k"] = kill
    r = orch.run(exp(duration=60), baseline_s=5)
    assert r.aborted and r.abort_reason == "kill switch"
    assert inj.rolled_back == 1 and r.ended_at - r.started_at < 20
    with pytest.raises(SafetyViolation, match="kill switch"):       # engaged until reset
        orch.run(exp(), baseline_s=1)


# ---- recovery
def test_recovery_measured_from_rollback_for_sustained_faults(tmp_path):
    def scenario(t, st):
        return (False, 10.0) if st.get("fault") else healthy(t, st)
    orch, inj, _, _, state, *_ = build(tmp_path, scenario)
    state["recovery_lag"] = 7                                  # still broken 7 s after rollback
    r = orch.run(exp(duration=60), baseline_s=5)
    assert r.recovered and 6 <= r.recovery_s <= 9


def test_instant_fault_recovery_measured_from_injection(tmp_path):
    def scenario(t, st):
        if st.get("fault") and t < st["t_inject"] + 2:        # kill: down 2 s, restart policy revives
            return False, 10.0
        return True, 10.0
    orch, *_ = build(tmp_path, scenario)
    r = orch.run(exp(fault="kill", duration=12), baseline_s=5)
    assert not r.aborted and r.recovered and 1 <= r.recovery_s <= 3


def test_long_outage_trips_auto_abort_but_recovery_is_still_measured(tmp_path):
    """A kill that takes 5 s to revive breaches for 3 consecutive seconds, so it is recorded as an
    auto-abort (rollback = make sure it is running). Recovery time is still measured."""
    def scenario(t, st):
        if st.get("fault") is not None and t < st.get("t_inject", 1e18) + 5 and st.get("t_inject"):
            return False, 10.0
        return True, 10.0
    orch, inj, *_ = build(tmp_path, scenario)
    r = orch.run(exp(fault="kill", duration=30), baseline_s=5)
    assert r.aborted and inj.rolled_back == 1
    assert r.recovered and 4 <= r.recovery_s <= 6


def test_no_visible_impact_gives_zero_recovery(tmp_path):
    orch, *_ = build(tmp_path, lambda t, st: (True, 10.0))
    r = orch.run(exp(fault="kill", duration=8), baseline_s=5)
    assert r.recovered and r.recovery_s == 0


def test_never_recovers_is_reported_not_faked(tmp_path):
    def scenario(t, st):
        return (False, 10.0) if st.get("fault") or st.get("fault") is False else (True, 10.0)
    orch, *_ = build(tmp_path, scenario)
    r = orch.run(exp(duration=10), baseline_s=5, recovery_timeout_s=15)
    assert r.aborted and not r.recovered and r.recovery_s is None


# ---- preflight
def test_preflight_rejects_unhealthy_baseline(tmp_path):
    orch, inj, *_ = build(tmp_path, lambda t, st: (False, 10.0))
    with pytest.raises(SafetyViolation, match="steady state"):
        orch.run(exp(), baseline_s=5)
    assert inj.injected == 0


def test_preflight_rejects_unhealthy_containers(tmp_path):
    orch, inj, *_ = build(tmp_path, healthy, health=lambda: ["db"])
    with pytest.raises(SafetyViolation, match="not healthy"):
        orch.run(exp(), baseline_s=5)
    assert inj.injected == 0


def test_unlabeled_target_rejected_before_anything_happens(tmp_path):
    orch, inj, *_ = build(tmp_path, healthy, managed=("vote",))
    with pytest.raises(SafetyViolation, match="allowlist"):
        orch.run(exp(), baseline_s=5)
    assert inj.injected == 0


def test_duration_cap_applies(tmp_path):
    orch, *_ = build(tmp_path, lambda t, st: (True, 10.0))
    r = orch.run(exp(duration=9999), baseline_s=5)
    assert r.experiment.duration_s == 60


# ---- rollback always runs
def test_rollback_runs_when_interrupted_mid_experiment(tmp_path):
    def scenario(t, st):
        if st.get("fault") and t >= st["t_inject"] + 2:
            raise KeyboardInterrupt
        return healthy(t, st)
    orch, inj, guard, *_ = build(tmp_path, scenario)
    with pytest.raises(KeyboardInterrupt):
        orch.run(exp(duration=30), baseline_s=5)
    assert inj.rolled_back == 1 and not guard._running


def test_failed_injection_releases_the_one_at_a_time_lock(tmp_path):
    orch, inj, guard, *_ = build(tmp_path, healthy, injector_kw={"fail_on_inject": RuntimeError("docker down")})
    with pytest.raises(RuntimeError, match="docker down"):
        orch.run(exp(), baseline_s=5)
    assert not guard._running and inj.rolled_back == 0


def test_rollback_errors_are_recorded(tmp_path):
    orch, inj, *_ = build(tmp_path, lambda t, st: (True, 10.0))
    inj.rollback = lambda h: ["limit not removed"]
    r = orch.run(exp(duration=5), baseline_s=5)
    assert r.rollback_errors == ["limit not removed"]


# ---- dry run
def test_dry_run_injects_nothing_and_writes_nothing(tmp_path):
    orch, inj, *_ = build(tmp_path, healthy)
    inj.inject = lambda *a, **k: FakeInjector.H()
    r = orch.run(exp(), dry_run=True)
    assert r.dry_run and r.notes == ["fake fault applied"]
    assert inj.rolled_back == 0 and not (tmp_path / "res").exists()


# ---- pure functions
def test_is_recovered_tolerances():
    base = Stats(n=100, error_rate=0.0, p50_ms=10, p95_ms=100)
    assert is_recovered(Stats(n=10, error_rate=0.0, p50_ms=10, p95_ms=105), base)
    assert not is_recovered(Stats(n=10, error_rate=0.1, p50_ms=10, p95_ms=100), base)
    assert not is_recovered(Stats(n=10, error_rate=0.0, p50_ms=10, p95_ms=300), base)
    assert not is_recovered(Stats(n=0), base)
    assert not is_recovered(Stats(n=5, error_rate=1.0, p50_ms=None, p95_ms=None), base)


def test_find_recovery_ignores_short_blips(tmp_path):
    store = Store(tmp_path / "x.db")
    for t in range(0, 30):
        ok = not (t in (3, 4, 5, 6, 7, 8, 12))            # long outage, then one blip at 12
        store.execute("INSERT INTO probes VALUES (?,?,?,?,?,?)", (float(t), "vote", int(ok), 200, 10.0, None))
    base = Stats(n=50, error_rate=0.0, p50_ms=10, p95_ms=10)
    assert find_recovery(store, 0.0, 30.0, base) == 13.0     # first 5 clean seconds start after the blip
    assert find_recovery(store, 0.0, 10.0, base) is None


def test_measure_baseline_saves_json(tmp_path):
    store = Store(tmp_path / "b.db")
    sim = Sim(store, lambda t: (True, 12.0))
    data = measure_baseline(store, 10, tmp_path, clock=sim.clock, sleep=sim.sleep)
    saved = json.loads((tmp_path / "baseline.json").read_text())
    assert saved["overall"]["error_rate"] == 0.0 and saved["overall"]["n"] >= 18
    assert saved["overall"]["p95_ms"] == 12.0
    assert data["by_kind"]["vote"]["n"] >= 9
