"""The injector agent: introduces the planned faults one at a time, autonomously and safely.

It does not decide *what* is broken (the planner does) and does not diagnose (the Day 2 analyzer
does). Its job is to turn a plan into real, observed faults without ever leaving the system worse
off than it found it:

  * before each fault it waits (up to agent.health_wait_s) for every container to be healthy,
    and refuses to inject into a system that is not;
  * each experiment goes through Orchestrator.run, i.e. the safety guard, the per-second abort
    rules and the try/finally rollback;
  * it stops the whole run on: kill switch, rollback problems, an injector/Docker error,
    unhealthy containers, or too many consecutive refusals;
  * it pauses agent.cooldown_s between experiments so one fault's aftermath cannot contaminate
    the next baseline.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from .config import AgentCfg
from .injector import InjectionError
from .models import Experiment
from .safety import Guard, SafetyViolation


class AgentEntry(BaseModel):
    id: str
    target: str
    fault: str
    status: str                          # done | skipped | error | planned (dry run)
    detail: str = ""
    aborted: bool = False
    abort_reason: Optional[str] = None
    recovered: Optional[bool] = None
    recovery_s: Optional[float] = None
    rollback_errors: list[str] = Field(default_factory=list)
    result_file: Optional[str] = None


class AgentRun(BaseModel):
    started_at: float
    ended_at: float = 0.0
    dry_run: bool = False
    planner: str = ""
    status: str = "completed"            # completed | stopped
    stop_reason: Optional[str] = None
    entries: list[AgentEntry] = Field(default_factory=list)
    not_run: list[str] = Field(default_factory=list)      # ids that were planned but never started


class InjectorAgent:
    def __init__(self, orchestrator, guard: Guard, health_fn: Callable[[], list[str]], cfg: AgentCfg,
                 baseline_s: int, recovery_timeout_s: int,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = lambda m: None,
                 progress: Callable[[str, int, Experiment, Optional["AgentEntry"]], None] = lambda *a: None):
        """`progress(event, index, experiment, entry)` is called when an experiment starts ("start") and when it ends
        ("done", "skipped" or "error"); the dashboard uses it to show what the agent is doing right now."""
        self.orch, self.guard, self.health_fn, self.cfg = orchestrator, guard, health_fn, cfg
        self.baseline_s, self.recovery_timeout_s = baseline_s, recovery_timeout_s
        self.clock, self.sleep, self.log, self.progress = clock, sleep, log, progress

    # ------------------------------------------------------------ helpers
    def _pause(self, seconds: float) -> bool:
        """Sleep in short chunks. True if the kill switch was engaged meanwhile."""
        end = self.clock() + seconds
        while self.clock() < end:
            if self.guard.kill.engaged:
                return True
            self.sleep(min(0.5, max(0.0, end - self.clock())))
        return self.guard.kill.engaged

    def _wait_healthy(self) -> list[str]:
        """Unhealthy containers remaining after waiting up to health_wait_s (empty = all healthy)."""
        deadline = self.clock() + self.cfg.health_wait_s
        bad = self.health_fn()
        while bad and self.clock() < deadline and not self.guard.kill.engaged:
            self.sleep(1.0)
            bad = self.health_fn()
        return bad

    # ------------------------------------------------------------ the run
    def run(self, experiments: list[Experiment], dry_run: bool = False, planner: str = "") -> AgentRun:
        todo = experiments[: self.cfg.max_experiments]
        run = AgentRun(started_at=self.clock(), dry_run=dry_run, planner=planner,
                       not_run=[e.id for e in experiments[self.cfg.max_experiments:]])
        skips = 0

        def stop(reason: str, remaining: list[Experiment]) -> AgentRun:
            run.status, run.stop_reason = "stopped", reason
            run.not_run = [e.id for e in remaining] + run.not_run
            run.ended_at = self.clock()
            self.log(f"agent stopped: {reason}")
            return run

        for i, exp in enumerate(todo):
            rest = todo[i:]
            if self.guard.kill.engaged:
                return stop("kill switch engaged", rest)

            if not dry_run:
                bad = self._wait_healthy()
                if bad:
                    return stop(f"containers not healthy after {self.cfg.health_wait_s:g}s: {', '.join(bad)}", rest)

            self.log(f"[{i + 1}/{len(todo)}] injecting {exp.fault} into {exp.target} ({exp.duration_s}s)")
            self.progress("start", i, exp, None)
            try:
                result = self.orch.run(exp, baseline_s=self.baseline_s,
                                       recovery_timeout_s=self.recovery_timeout_s, dry_run=dry_run)
            except SafetyViolation as e:
                if self.guard.kill.engaged:
                    return stop("kill switch engaged", rest)
                skips += 1
                run.entries.append(AgentEntry(id=exp.id, target=exp.target, fault=exp.fault,
                                              status="skipped", detail=str(e)))
                self.progress("skipped", i, exp, run.entries[-1])
                self.log(f"    refused by safety guard: {e}")
                if skips >= self.cfg.max_consecutive_skips:
                    return stop(f"{skips} experiments in a row refused by the safety guard (last: {e})", rest[1:])
                continue
            except InjectionError as e:
                run.entries.append(AgentEntry(id=exp.id, target=exp.target, fault=exp.fault,
                                              status="error", detail=str(e)))
                self.progress("error", i, exp, run.entries[-1])
                return stop(f"injector error: {e}", rest[1:])

            skips = 0
            entry = AgentEntry(
                id=exp.id, target=exp.target, fault=exp.fault, status="planned" if dry_run else "done",
                detail="; ".join(result.notes), aborted=result.aborted, abort_reason=result.abort_reason,
                recovered=None if dry_run else result.recovered, recovery_s=result.recovery_s,
                rollback_errors=result.rollback_errors,
                result_file=None if dry_run else str(Path(self.orch.results_dir) / f"{exp.id}.json"))
            run.entries.append(entry)
            self.progress("planned" if dry_run else "done", i, exp, entry)
            if dry_run:
                self.log(f"    {result.notes[0] if result.notes else 'planned'}")
                continue

            self.log("    " + ("ABORTED (" + str(result.abort_reason) + "); " if result.aborted else "")
                     + ("recovered in %.0fs" % result.recovery_s if result.recovered else "recovery not confirmed")
                     + (f"; worst journey during fault: "
                        f"{max((s.error_rate or 0 for s in result.during_by_journey.values()), default=0):.0%} errors"))

            if result.rollback_errors:
                return stop(f"rollback problems after {exp.id}: {result.rollback_errors} (run `chaos abort`)", rest[1:])
            if result.abort_reason == "kill switch":
                return stop("kill switch engaged", rest[1:])
            if result.aborted and self.cfg.stop_on_abort:
                return stop(f"{exp.id} was auto-aborted and agent.stop_on_abort is true", rest[1:])

            if exp is not todo[-1] and self._pause(self.cfg.cooldown_s):
                return stop("kill switch engaged", rest[1:])

        run.ended_at = self.clock()
        return run
