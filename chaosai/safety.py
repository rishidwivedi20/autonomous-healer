"""Guardrails. Written before any injection code; every experiment passes through here.

Pure logic (no Docker import) so it is fully unit-testable. The orchestrator wires
it to the injector and wraps every injection in try/finally.
"""
from __future__ import annotations

import signal
import threading
from pathlib import Path

from .models import Experiment, Stats, Thresholds

LABEL = "chaosai.managed"
MAX_DURATION_S = 60
MAX_THRESHOLDS = Thresholds(error_rate=0.5, p95_ms=3000.0, breach_seconds=3)


class SafetyViolation(Exception):
    pass


def is_managed(labels: dict | None) -> bool:
    return (labels or {}).get(LABEL) == "true"


class KillSwitch:
    """File-based so `chaos abort` from another terminal stops a running experiment."""

    def __init__(self, path: Path):
        self.path = path
        self._event = threading.Event()

    @property
    def engaged(self) -> bool:
        return self._event.is_set() or self.path.exists()

    def engage(self, why: str = "manual") -> None:
        self._event.set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(why, encoding="utf-8")

    def reset(self) -> None:
        self._event.clear()
        self.path.unlink(missing_ok=True)

    def install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            self.engage(f"signal {signum}")
        signal.signal(signal.SIGINT, handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handler)


class Guard:
    def __init__(self, managed_targets: set[str], kill: KillSwitch, max_duration_s: int = MAX_DURATION_S):
        self.max_duration_s = max(1, min(max_duration_s, MAX_DURATION_S))   # config may only tighten the hard cap
        self.managed_targets = managed_targets   # service names whose container has the label
        self.kill = kill
        self._running = False
        self._breaches = 0
        self.abort_reason: str | None = None

    # ------------------------------------------------------------ before
    def validate(self, exp: Experiment) -> Experiment:
        """Reject unsafe experiments and clamp the rest to hard limits."""
        if exp.target not in self.managed_targets:
            raise SafetyViolation(f"{exp.target!r} is not labeled {LABEL}=true (allowlist)")
        if exp.duration_s < 1:
            raise SafetyViolation("duration must be >= 1s")
        t = exp.thresholds
        # Planners may only tighten thresholds, never loosen them.
        clamped = Thresholds(
            error_rate=min(t.error_rate, MAX_THRESHOLDS.error_rate),
            p95_ms=min(t.p95_ms, MAX_THRESHOLDS.p95_ms),
            breach_seconds=max(1, min(t.breach_seconds, MAX_THRESHOLDS.breach_seconds)),
        )
        return exp.model_copy(update={
            "duration_s": min(exp.duration_s, self.max_duration_s),
            "thresholds": clamped,
        })

    def preflight(self, baseline: Stats, unhealthy: list[str], max_error_rate: float = 0.05) -> None:
        if self.kill.engaged:
            raise SafetyViolation("kill switch engaged (run `chaos reset` to clear)")
        if baseline.n == 0 or baseline.error_rate is None:
            raise SafetyViolation("no baseline traffic; is the app reachable?")
        if baseline.error_rate > max_error_rate:
            raise SafetyViolation(f"not in steady state (error rate {baseline.error_rate:.0%})")
        if unhealthy:
            raise SafetyViolation(f"containers not healthy: {', '.join(unhealthy)}")

    def begin(self) -> None:
        """One experiment at a time."""
        if self._running:
            raise SafetyViolation("another experiment is already running")
        self._running = True
        self._breaches = 0
        self.abort_reason = None

    def end(self) -> None:
        self._running = False

    # ------------------------------------------------------------ during
    def should_abort(self, stats: Stats, t: Thresholds) -> bool:
        """Call once per second with that second's stats."""
        if self.kill.engaged:
            self.abort_reason = "kill switch"
            return True
        reasons = []
        if stats.n and stats.error_rate is not None and stats.error_rate > t.error_rate:
            reasons.append(f"error rate {stats.error_rate:.0%} > {t.error_rate:.0%}")
        if stats.n and stats.p95_ms is not None and stats.p95_ms > t.p95_ms:
            reasons.append(f"p95 {stats.p95_ms:.0f}ms > {t.p95_ms:.0f}ms")
        self._breaches = self._breaches + 1 if reasons else 0
        if self._breaches >= t.breach_seconds:
            self.abort_reason = f"{'; '.join(reasons)} for {self._breaches} consecutive s"
            return True
        return False
