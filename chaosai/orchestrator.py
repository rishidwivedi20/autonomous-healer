"""Experiment lifecycle: preflight -> baseline -> inject -> monitor each second -> rollback
-> measure recovery -> write results/<id>.json.

The probe and watcher must already be running (the CLI starts them). Time comes from
injected clock/sleep functions so the lifecycle is unit-testable without waiting.

Abort, preflight and recovery judge the WORST user journey (vote vs result), not the average.

Failures are recorded when a request COMPLETES (a timeout lands `timeout_s` after it started) but
are stamped with its start time. Looking at "the last second" therefore never sees an ongoing
outage. All judgements use a window lagged by `settle_s` (probe timeout + margin) so every request
in it has finished. Cost: abort fires about settle_s + breach_seconds after the fault begins.

Recovery definition: "recovered" = a hold window (5 s) in which error rate is within 10% of
baseline and p95 is within 10% of baseline plus a 50 ms slack (a 10% band on a ~10 ms local
latency is below measurement jitter; the slack is stated here so it is not a hidden fudge).
recovery_s is measured from when the fault ended: the injection moment for instantaneous
faults (kill), the rollback moment for sustained ones. 0 means no visible impact.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .models import Experiment, ExperimentResult, Stats
from .observer import (Store, by_journey, events, missing_signals, per_second, restart_counts, window,
                       worst_window)
from .safety import Guard, SafetyViolation

HOLD_S = 5
TOLERANCE = 0.10
LATENCY_SLACK_MS = 50.0
INSTANT_FAULTS = {"kill"}
GAUGES = ("redis_queue_len", "db_votes")


def is_recovered(s: Stats, baseline: Stats, tol: float = TOLERANCE) -> bool:
    if not s.n or s.error_rate is None or s.p95_ms is None:     # no data, or every request failed
        return False
    if s.error_rate > (baseline.error_rate or 0.0) * (1 + tol):
        return False
    base_p95 = baseline.p95_ms or 0.0
    return s.p95_ms <= base_p95 * (1 + tol) + LATENCY_SLACK_MS


def find_recovery(store: Store, start: float, now: float, baseline: Stats,
                  hold: int = HOLD_S, tol: float = TOLERANCE) -> float | None:
    """Earliest offset (seconds from `start`) from which `hold` consecutive seconds are within
    tolerance of baseline, or None if that has not happened yet."""
    for i in range(int(now - start) - hold + 1):
        if is_recovered(worst_window(store, start + i, start + i + hold), baseline, tol):
            return float(i)
    return None


class Orchestrator:
    def __init__(self, store: Store, injector, guard: Guard, results_dir: Path | str,
                 health_fn: Callable[[], list[str]] = lambda: [],
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 tick_s: float = 1.0, settle_s: float = 0.0):
        self.settle_s = settle_s
        self.store, self.injector, self.guard = store, injector, guard
        self.results_dir = Path(results_dir)
        self.health_fn, self.clock, self.sleep, self.tick_s = health_fn, clock, sleep, tick_s

    # ------------------------------------------------------------ waiting helpers
    def _wait_until(self, t: float) -> bool:
        """Sleep until t in short chunks so the kill switch is noticed quickly. True if killed."""
        while self.clock() < t:
            if self.guard.kill.engaged:
                return True
            self.sleep(min(0.25, max(0.0, t - self.clock())))
        return self.guard.kill.engaged

    # ------------------------------------------------------------ the lifecycle
    def run(self, exp: Experiment, baseline_s: float = 15, recovery_timeout_s: float = 120,
            dry_run: bool = False) -> ExperimentResult:
        exp = self.guard.validate(exp)             # allowlist, duration cap, thresholds only tighten

        if dry_run:
            handle = self.injector.inject(exp.target, exp.fault, exp.params)   # dry-run injector: no effects
            now = self.clock()
            return ExperimentResult(experiment=exp, started_at=now, ended_at=now, dry_run=True,
                                    baseline=Stats(), during=Stats(), notes=handle.actions)

        # ---- preflight + baseline
        t_start = self.clock()
        if self._wait_until(t_start + baseline_s):
            raise SafetyViolation("kill switch engaged during baseline")
        t_base_end = self.clock()
        baseline = window(self.store, t_start, t_base_end)
        # steady state and recovery are judged on the worst journey; reported baseline stays overall
        baseline_worst = worst_window(self.store, t_start, t_base_end)
        self.guard.preflight(baseline_worst, self.health_fn())

        # ---- inject + monitor; rollback in finally, whatever happens
        self.guard.begin()
        handle = None
        aborted, abort_reason, rollback_errors = False, None, []
        t_inject = t_end = self.clock()
        try:
            t_inject = self.clock()
            handle = self.injector.inject(exp.target, exp.fault, exp.params)
            deadline = t_inject + exp.duration_s
            next_tick = t_inject + self.tick_s
            while self.clock() < deadline:
                self._wait_until(next_tick)
                now = self.clock() - self.settle_s          # only requests that have had time to finish
                if self.guard.should_abort(worst_window(self.store, now - 1.0, now), exp.thresholds):
                    aborted, abort_reason = True, self.guard.abort_reason
                    break
                next_tick += self.tick_s
        finally:
            t_end = self.clock()
            if handle is not None:
                rollback_errors = self.injector.rollback(handle)
            self.guard.end()

        # ---- recovery
        t_fault_end = t_inject if exp.fault in INSTANT_FAULTS else t_end
        recovery = None
        while True:
            now = self.clock()
            recovery = find_recovery(self.store, t_fault_end, now - self.settle_s, baseline_worst)
            if recovery is not None or now - t_fault_end >= recovery_timeout_s or self.guard.kill.engaged:
                break
            self.sleep(self.tick_s)

        # ---- result
        now = self.clock()
        result = ExperimentResult(
            experiment=exp, started_at=t_inject, ended_at=now,
            baseline=baseline,
            during=window(self.store, t_inject, t_end),
            baseline_by_journey=by_journey(self.store, t_start, t_base_end),
            during_by_journey=by_journey(self.store, t_inject, t_end),
            per_second=per_second(self.store, t_inject, t_end),
            recovery_s=recovery, recovered=recovery is not None,
            aborted=aborted, abort_reason=abort_reason,
            container_events=events(self.store, t_start, now),
            restarts=restart_counts(self.store, t_inject, now),
            rollback_errors=rollback_errors,
            missing_signals=missing_signals(self.store, t_start, now, GAUGES),
            notes=(handle.actions if handle else []) + (
                ["recovery not measured: kill switch engaged"] if recovery is None and self.guard.kill.engaged else []),
        )
        self.write(result)
        return result

    def write(self, result: ExperimentResult) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / f"{result.experiment.id}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return path


# ---------------------------------------------------------------- standalone baseline
def measure_baseline(store: Store, seconds: float, results_dir: Path | str,
                     health_fn: Callable[[], list[str]] = lambda: [],
                     clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> dict:
    """Wait `seconds`, then save p50/p95/error rate (overall and per probe kind) to baseline.json."""
    t0 = clock()
    while clock() < t0 + seconds:
        sleep(min(0.5, max(0.0, t0 + seconds - clock())))
    t1 = clock()
    data = {
        "started_at": t0, "ended_at": t1, "seconds": seconds,
        "overall": window(store, t0, t1).model_dump(),
        "by_kind": {k: window(store, t0, t1, k).model_dump() for k in ("vote", "result")},
        "unhealthy_containers": health_fn(),
        "missing_signals": missing_signals(store, t0, t1, GAUGES),
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "baseline.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
