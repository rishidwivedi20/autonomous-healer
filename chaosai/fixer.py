"""The fixing agent: apply one vetted fix to the sandbox, prove it with the same experiment, and put things back.

It is the third agent of the system, after the injecting agent (agent.py: breaks things safely) and the finding agent
(finder.py: explains what broke). The loop for one fix:

    before   run the verification experiment on the system as it is, and measure it
    check    does the problem the fix addresses really show up in that measurement? (if not, change nothing)
    apply    write the undo to disk FIRST, then change the sandbox; wait until it is healthy
    after    run the very same experiment again, and measure it
    judge    compare resilience scores and the fix's own metric; only measured numbers can say "improved"
    revert   put the sandbox back exactly as found, unless the fix was measured to help AND keep was asked for

Safety rules, none of which depend on any AI:
  * it can only choose from CATALOG, a hard-coded list of reversible fixes; a model (or a person) picks an id, never a command;
  * `docker compose` is run as an argument list (no shell), and only for services labelled chaosai.managed=true, with a
    replica count inside fixed bounds;
  * the undo is written to results/active_fixes.json before the change, so `chaos reset` can still undo it if this process dies;
  * the one-experiment-at-a-time lock is held while the sandbox is being changed, and the kill switch is checked at every step;
  * the verification experiments go through the same Orchestrator (safety guard, abort rules, rollback) as any other, and their
    results are stored apart from the experiment history (results/fix-runs/<id>/), so a "fixed" run never replaces the real
    history that the resilience score is computed from;
  * the revert runs in a `finally`: an error or the kill switch cannot leave the change in place.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, Optional

from pydantic import BaseModel, Field

from .analysis import Analysis, Weakness
from .config import FixerCfg
from .models import Experiment, ExperimentResult, Graph, Thresholds
from .safety import SafetyViolation
from .scoring import score_experiment

MAX_REPLICAS = 5


class FixError(Exception):
    pass


# ---------------------------------------------------------------- the catalogue
@dataclass(frozen=True)
class Recipe:
    """One vetted, reversible fix. Everything the agent may do is spelled out here, in data."""
    id: str
    title: str
    service: str                       # the one managed service it changes
    replicas: int                      # the replica count it sets (compose --scale); undone by restoring the original count
    why: str
    change: str                        # what a person would put in the compose file
    keywords: tuple[str, ...]          # a weakness matches if it names the service and any of these words
    verify: dict                       # the experiment that proves it: target, fault, duration_s and votes_per_s (the spike's size)
    primary: str                       # the metric (see METRICS) judged for this fix; lower is better
    problem_metric: str                # measured in the BEFORE run to confirm the problem exists...
    problem_min: float                 # ...at least this large; otherwise there is nothing to fix

    def matches(self, w: Weakness) -> bool:
        text = f"{w.title} {w.description}".lower()
        return self.service in w.services and any(k in text for k in self.keywords)

    def problem_present(self, metrics: dict) -> bool:
        v = metrics.get(self.problem_metric)
        return v is not None and v >= self.problem_min


CATALOG: dict[str, Recipe] = {r.id: r for r in (
    Recipe(
        id="scale-worker", title="Run three worker replicas", service="worker", replicas=3,
        why="Workers pop votes from one shared redis list, so replicas share the load: a backlog should drain about three "
            "times as fast, until postgres becomes the limit.",
        change="docker compose up -d --no-deps --scale worker=3 worker   (the same as worker: deploy: replicas: 3 in the compose file)",
        keywords=("throughput", "capacity", "backlog", "scal", "replica", "queue", "drain"),
        verify={"target": "vote", "fault": "traffic-spike", "votes_per_s": 15, "duration_s": 20},     # one worker stores ~9.5 votes/s, three ~28
        primary="peak_queue", problem_metric="peak_queue", problem_min=20.0),
)}


# ---------------------------------------------------------------- measuring and judging
METRICS: dict[str, tuple[str, str, bool]] = {          # key: (label, unit, lower is better)
    "peak_queue": ("redis queue peak", "votes", True),
    "backlog_lag_s": ("backlog lag at the peak", "s", True),
    "queue_back_to_baseline_at_s": ("queue back to baseline at", "s", True),
    "drain_rate": ("votes stored per second while draining", "votes/s", False),
    "recovery_s": ("recovery time (users and pipeline)", "s", True),
    "recovery_pipeline_s": ("pipeline recovery time", "s", True),
    "error_pct": ("worst journey error rate", "%", True),
    "vote_p95_ms": ("vote p95 latency during the fault", "ms", True),
    "result_p95_ms": ("result p95 latency during the fault", "ms", True),
    "collateral_restarts": ("other services restarted", "", True),
}


class Measured(BaseModel):
    experiment_id: str
    result_file: Optional[str] = None
    score: Optional[float] = None
    band: Optional[str] = None
    components: dict[str, Optional[float]] = Field(default_factory=dict)     # 0..1 per scoring component
    metrics: dict[str, Optional[float]] = Field(default_factory=dict)
    aborted: bool = False
    abort_reason: Optional[str] = None
    recovered: bool = False
    restarts: dict[str, int] = Field(default_factory=dict)


def measure(result: ExperimentResult, graph: Optional[Graph], gauges: Optional[dict], result_file: Optional[str] = None) -> Measured:
    sc = score_experiment(result, graph)
    during = {j: s for j, s in result.during_by_journey.items() if s.n}
    errs = [s.error_rate for s in during.values() if s.error_rate is not None]
    g = (gauges or {}).get("summary") or {}
    m = {
        "peak_queue": g.get("peak_queue"), "backlog_lag_s": g.get("peak_backlog_lag_s"),
        "queue_back_to_baseline_at_s": g.get("queue_back_to_baseline_at_s"), "drain_rate": g.get("votes_per_s_while_draining"),
        "recovery_s": result.recovery_s if result.recovered else None, "recovery_pipeline_s": result.recovery_pipeline_s,
        "error_pct": round(100 * max(errs), 1) if errs else None,
        "vote_p95_ms": during["vote"].p95_ms if "vote" in during else None,
        "result_p95_ms": during["result"].p95_ms if "result" in during else None,
        "collateral_restarts": float(sum(1 for s, n in result.restarts.items() if n and s != result.experiment.target)),
    }
    return Measured(experiment_id=result.experiment.id, result_file=result_file, score=sc.score, band=sc.band,
                    components={c.name: c.score for c in sc.components}, metrics=m, aborted=result.aborted,
                    abort_reason=result.abort_reason, recovered=result.recovered, restarts=dict(result.restarts))


def compare_rows(before: Measured, after: Measured) -> list[dict]:
    """One row per metric both runs have a value for (or either has): before, after, relative change, and whether that is better."""
    rows = []
    for key, (label, unit, lower_better) in METRICS.items():
        b, a = before.metrics.get(key), after.metrics.get(key)
        if b is None and a is None:
            continue
        change = round(100 * (a - b) / b, 1) if b not in (None, 0) and a is not None else (0.0 if a is not None and a == b else None)
        if change is None or abs(change) < 5:
            verdict = "same" if change is not None else "n/a"
        else:
            verdict = "better" if (change < 0) == lower_better else "worse"
        rows.append({"key": key, "label": label, "unit": unit, "before": b, "after": a, "change_pct": change, "verdict": verdict})
    return rows


def judge(before: Measured, after: Measured, primary: str, min_score_gain: float, min_metric_gain: float) -> tuple[str, str]:
    """('improved' | 'worse' | 'inconclusive', why). Improvement needs BOTH a score gain and a gain in the fix's own metric,
    and no component of the score may have dropped noticeably; anything less is not claimed."""
    if after.aborted and not before.aborted:
        return "worse", "the verification experiment was auto-aborted after the change, but not before it"
    ds = after.score - before.score if before.score is not None and after.score is not None else None
    pb, pa = before.metrics.get(primary), after.metrics.get(primary)
    gain = (pb - pa) / pb if pb and pa is not None and pb > 0 else None
    label = METRICS[primary][0]
    if (ds is not None and ds <= -min_score_gain) or (gain is not None and gain <= -min_metric_gain):
        return "worse", f"the score moved {ds:+.1f} points and {label} changed {-100 * gain:+.0f}%" if ds is not None and gain is not None else "the measurements got worse"
    dropped = [n for n, b in before.components.items() if b is not None and after.components.get(n) is not None and after.components[n] < b - 0.15]
    if ds is not None and ds >= min_score_gain and gain is not None and gain >= min_metric_gain and not dropped:
        return "improved", f"the score rose {ds:+.1f} points and {label} improved {100 * gain:.0f}%"
    why = []
    if ds is None or ds < min_score_gain:
        why.append("the score did not rise by %.1f points" % min_score_gain + (f" (it moved {ds:+.1f})" if ds is not None else ""))
    if gain is None or gain < min_metric_gain:
        why.append(f"{label} did not improve by {100 * min_metric_gain:.0f}%" + (f" (it moved {100 * gain:+.0f}%)" if gain is not None else " (not measurable)"))
    if dropped:
        why.append("the " + ", ".join(dropped) + " component of the score dropped")
    return "inconclusive", "; ".join(why) + ": the fix is not claimed to help (one run each; repeat to be sure)"


def claim(rows: list[dict], before: Measured, after: Measured) -> str:
    def fmt(v, unit):
        return "n/a" if v is None else (f"{v:g}" + (f" {unit}" if unit else ""))
    parts = [f"{r['label']} {fmt(r['before'], r['unit'])} -> {fmt(r['after'], r['unit'])}" for r in rows
             if r["key"] in ("peak_queue", "recovery_pipeline_s", "recovery_s", "error_pct", "vote_p95_ms")][:4]
    score = f"resilience score {before.score:.1f} -> {after.score:.1f}" if before.score is not None and after.score is not None else "score n/a"
    return "; ".join([score] + parts)


class FixRun(BaseModel):
    id: str
    started_at: float
    ended_at: float = 0.0
    status: str = "error"          # improved | inconclusive | worse | skipped | stopped | error | planned
    detail: str = ""
    recipe_id: str
    title: str
    service: str
    change: str
    why: str
    addresses: str = ""            # the weakness (from the findings) that led to this fix
    experiment: dict = Field(default_factory=dict)
    original_replicas: Optional[int] = None
    applied_replicas: Optional[int] = None
    before: Optional[Measured] = None
    after: Optional[Measured] = None
    comparison: list[dict] = Field(default_factory=list)
    claim: str = ""
    kept: bool = False
    reverted: bool = False
    revert_errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- docker compose, safely
def find_docker() -> str:
    env = os.environ.get("CHAOSAI_DOCKER")
    if env and Path(env).exists():
        return env
    found = shutil.which("docker")
    if found:
        return found
    if os.name == "nt":
        for base in (os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramFiles", "")):
            for sub in ("Programs/DockerDesktop/resources/bin", "Docker/Docker/resources/bin"):
                p = Path(base) / sub / "docker.exe" if base else None
                if p and p.exists():
                    return str(p)
    raise FixError("the docker command was not found: add Docker Desktop's bin folder to PATH, or set CHAOSAI_DOCKER")


class ComposeRunner:
    """The only way the fixing agent changes the sandbox. It cannot run anything but `docker compose ... up --scale`."""

    def __init__(self, compose_file: Path | str, managed: Collection[str], run: Callable = subprocess.run, timeout_s: float = 240.0,
                 docker: Optional[str] = None):
        # absolute, because docker compose runs from the file's folder: a relative path would then be resolved against that folder
        self.compose_file, self.managed, self._run, self.timeout_s, self._docker = Path(compose_file).resolve(), managed, run, timeout_s, docker

    def command(self, service: str, replicas: int) -> list[str]:
        if service not in self.managed:
            raise FixError(f"{service!r} is not a managed service (label chaosai.managed=true); refusing to change it")
        if isinstance(replicas, bool) or not isinstance(replicas, int) or not 1 <= replicas <= MAX_REPLICAS:
            raise FixError(f"replicas must be an integer from 1 to {MAX_REPLICAS}, not {replicas!r}")
        return [self._docker or find_docker(), "compose", "-f", str(self.compose_file), "up", "-d", "--no-deps",
                "--scale", f"{service}={replicas}", service]

    def scale(self, service: str, replicas: int) -> list[str]:
        cmd = self.command(service, replicas)
        try:
            p = self._run(cmd, capture_output=True, text=True, timeout=self.timeout_s, cwd=str(self.compose_file.parent))
        except subprocess.TimeoutExpired:
            raise FixError(f"docker compose did not finish within {self.timeout_s:g}s") from None
        except OSError as e:
            raise FixError(f"could not run docker compose: {e}") from None
        if p.returncode != 0:
            raise FixError("docker compose failed: " + ((p.stderr or p.stdout or "").strip()[-400:] or f"exit code {p.returncode}"))
        return cmd


# ---------------------------------------------------------------- the agent
class FixingAgent:
    def __init__(self, orch_factory: Callable[[Path], object], compose: ComposeRunner, replicas_fn: Callable[[str], int],
                 health_fn: Callable[[], list[str]], guard, graph: Optional[Graph], cfg: FixerCfg, results_dir: Path | str,
                 baseline_s: float, recovery_timeout_s: float, thresholds: Thresholds, probe_interval_s: float = 1.0,
                 gauges_fn: Callable[[Path, str], Optional[dict]] = lambda d, i: None,
                 state_path: Path | str | None = None, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = lambda m: None):
        self.orch_factory, self.compose, self.replicas_fn, self.health_fn = orch_factory, compose, replicas_fn, health_fn
        self.guard, self.graph, self.cfg = guard, graph, cfg
        self.results_dir = Path(results_dir)
        self.baseline_s, self.recovery_timeout_s, self.thresholds = baseline_s, recovery_timeout_s, thresholds
        self.probe_interval_s = probe_interval_s
        self.gauges_fn, self.clock, self.sleep, self.log = gauges_fn, clock, sleep, log
        self.state_path = Path(state_path) if state_path else self.results_dir / "active_fixes.json"

    # ------------------------------------------------------------ choosing
    @staticmethod
    def select(analysis: Analysis) -> list[tuple[Recipe, Weakness]]:
        """Recipes that fit the findings, most severe weakness first (the analysis lists weaknesses in that order)."""
        out, seen = [], set()
        for w in analysis.weaknesses:
            for r in CATALOG.values():
                if r.id not in seen and r.matches(w):
                    out.append((r, w))
                    seen.add(r.id)
        return out

    # ------------------------------------------------------------ state: the undo is written BEFORE the change
    def _state(self) -> dict:
        try:
            s = json.loads(self.state_path.read_text(encoding="utf-8"))
            return s if isinstance(s, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if state:
            self.state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        elif self.state_path.exists():
            self.state_path.unlink()

    def pending(self, include_kept: bool = False) -> dict:
        return {k: v for k, v in self._state().items() if include_kept or not v.get("kept")}

    # ------------------------------------------------------------ small helpers
    @contextmanager
    def _exclusive(self):
        """Hold the one-experiment-at-a-time lock while the sandbox is being changed."""
        lock = getattr(self.guard, "lock", None)
        if getattr(self.guard, "_running", False):
            raise SafetyViolation("an experiment is running; not changing the sandbox now")
        if lock is not None and not lock.acquire():
            raise SafetyViolation(f"an experiment is running in another process ({lock.holder()}); not changing the sandbox now")
        try:
            yield
        finally:
            if lock is not None:
                lock.release()

    def _check_kill(self) -> None:
        if self.guard.kill.engaged:
            raise SafetyViolation("kill switch engaged")

    def _wait(self, cond: Callable[[], bool], seconds: float, killable: bool = True) -> bool:
        """Poll `cond` until it holds or `seconds` pass. The kill switch is watched, except during cleanup, which must finish."""
        end = self.clock() + seconds
        while True:
            if killable:
                self._check_kill()
            if cond():
                return True
            if self.clock() >= end:
                return False
            self.sleep(1.0)

    def _healthy(self) -> list[str]:
        return self.health_fn()

    def _scale(self, service: str, replicas: int) -> None:
        with self._exclusive():
            self.log(f"    {' '.join(self.compose.scale(service, replicas)[-5:])}")

    def _settle(self, service: str, replicas: int, killable: bool = True) -> None:
        ok = self._wait(lambda: self.replicas_fn(service) == replicas and not self._healthy(), self.cfg.health_wait_s, killable)
        if not ok:
            raise FixError(f"{service} did not reach {replicas} healthy replica(s) within {self.cfg.health_wait_s:g}s "
                           f"(has {self.replicas_fn(service)}; unhealthy: {', '.join(self._healthy()) or 'none'})")

    def _spike_multiplier(self, recipe: Recipe) -> int:
        """The verification spike is specified as a vote rate, so it means the same whatever the probe's own rate is:
        the command line's probe sends 1 request/s per journey, the dashboard's 5, and the spike multiplies that."""
        return max(1, round(recipe.verify["votes_per_s"] * self.probe_interval_s))

    def _experiment(self, recipe: Recipe, fix_id: str, label: str, fix_dir: Path) -> tuple[ExperimentResult, Measured]:
        self._check_kill()
        spec = recipe.verify
        exp = Experiment(id=f"{fix_id}-{label}-{spec['target']}-{spec['fault']}", target=spec["target"], fault=spec["fault"],
                         params={"multiplier": self._spike_multiplier(recipe)}, duration_s=spec["duration_s"], thresholds=self.thresholds,
                         hypothesis=f"{recipe.title}: the {label} run of the verification experiment")
        self.log(f"    verification experiment ({label}): {spec['fault']} on {spec['target']}")
        orch = self.orch_factory(fix_dir)
        res = orch.run(exp, baseline_s=self.baseline_s, recovery_timeout_s=self.recovery_timeout_s)
        if res.rollback_errors:
            raise FixError(f"rollback problems after the {label} run: {res.rollback_errors} (run `chaos abort`)")
        file = str(fix_dir / f"{exp.id}.json")
        return res, measure(res, self.graph, self.gauges_fn(fix_dir, exp.id), file)

    # ------------------------------------------------------------ the loop
    def run(self, recipe_id: str, weakness: str = "", keep: Optional[bool] = None, dry_run: bool = False) -> FixRun:
        """The whole loop for one fix; the record is saved to results/fixes/ once the change has been reverted (or kept)."""
        run = self._run(recipe_id, weakness, keep, dry_run)
        if run.status != "planned":
            self.save(run)
        return run

    def _run(self, recipe_id: str, weakness: str, keep: Optional[bool], dry_run: bool) -> FixRun:
        recipe = CATALOG.get(recipe_id)
        if recipe is None:
            raise FixError(f"unknown fix {recipe_id!r}; the catalogue has: {', '.join(CATALOG)}")
        if recipe.service not in self.guard.managed_targets:
            raise SafetyViolation(f"{recipe.service!r} is not labeled chaosai.managed=true (allowlist)")
        keep = self.cfg.keep if keep is None else keep
        fix_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.clock())) + "-" + recipe.id
        run = FixRun(id=fix_id, started_at=self.clock(), recipe_id=recipe.id, title=recipe.title, service=recipe.service,
                     change=recipe.change, why=recipe.why, addresses=weakness,
                     experiment={**recipe.verify, "params": {"multiplier": self._spike_multiplier(recipe)}},
                     applied_replicas=recipe.replicas)
        if dry_run:
            run.status, run.detail = "planned", "dry run: nothing was changed"
            run.notes = [f"1. run {recipe.verify['fault']} on {recipe.verify['target']} at {recipe.verify['votes_per_s']} votes/s for {recipe.verify['duration_s']}s and measure it (before)",
                         f"2. confirm {recipe.problem_metric} >= {recipe.problem_min:g}; otherwise stop with no change",
                         f"3. record the undo, then set {recipe.service} to {recipe.replicas} replicas",
                         "4. run the same experiment again (after) and compare the resilience scores",
                         "5. put the original replica count back" + (" unless the fix improved things and keep is on" if keep else "")]
            return run
        fix_dir = self.results_dir / "fix-runs" / fix_id
        original: Optional[int] = None
        applied = False
        verdict = None
        try:
            self._check_kill()
            self._wait(lambda: not self._healthy(), self.cfg.health_wait_s)
            bad = self._healthy()
            if bad:
                return self._finish(run, "stopped", f"containers not healthy after {self.cfg.health_wait_s:g}s: {', '.join(bad)}")
            original = self.replicas_fn(recipe.service)
            run.original_replicas = original
            if original >= recipe.replicas:
                return self._finish(run, "skipped", f"{recipe.service} already runs {original} replica(s); nothing to change")

            res_b, run.before = self._experiment(recipe, fix_id, "before", fix_dir)
            if not recipe.problem_present(run.before.metrics):
                v = run.before.metrics.get(recipe.problem_metric)
                return self._finish(run, "skipped", f"the problem was not reproduced: {METRICS[recipe.problem_metric][0]} was "
                                    f"{'n/a' if v is None else f'{v:g}'} (needs at least {recipe.problem_min:g}), so no change was made")
            self._check_kill()
            self._wait(lambda: False, self.cfg.cooldown_s)                        # let the first run's aftermath die down
            self._check_kill()

            self._write_state({**self._state(), recipe.id: {"service": recipe.service, "original_replicas": original,
                                                            "target_replicas": recipe.replicas, "fix_id": fix_id,
                                                            "applied_at": self.clock(), "kept": False}})   # write-ahead undo
            applied = True
            self._scale(recipe.service, recipe.replicas)
            self._settle(recipe.service, recipe.replicas)
            self._wait(lambda: False, self.cfg.warmup_s)
            self._check_kill()

            res_a, run.after = self._experiment(recipe, fix_id, "after", fix_dir)
            run.comparison = compare_rows(run.before, run.after)
            verdict, why = judge(run.before, run.after, recipe.primary, self.cfg.min_score_gain, self.cfg.min_metric_gain)
            run.claim = claim(run.comparison, run.before, run.after)
            return self._finish(run, verdict, why)
        except SafetyViolation as e:
            return self._finish(run, "stopped", str(e))
        except Exception as e:                                                       # never leave a half-finished change behind silently
            return self._finish(run, "error", f"{type(e).__name__}: {e}")
        finally:
            if applied:
                self._after_run(run, recipe, original, keep and verdict == "improved")

    def _finish(self, run: FixRun, status: str, detail: str) -> FixRun:
        run.status, run.detail = status, detail
        return run

    def _after_run(self, run: FixRun, recipe: Recipe, original: Optional[int], keep_it: bool) -> None:
        """Runs in `finally`: revert the change (or keep it, if asked and measured to help). Never raises."""
        if keep_it:
            state = self._state()
            if recipe.id in state:
                state[recipe.id]["kept"] = True
                self._write_state(state)
            run.kept = True
            run.notes.append(f"kept: {recipe.service} stays at {recipe.replicas} replicas (undo with `chaos fix --revert`)")
            return
        errors = self._revert_one(recipe.id, recipe.service, original if original is not None else 1)
        run.reverted, run.revert_errors = not errors, errors
        if errors:
            run.notes.append("REVERT FAILED: run `chaos reset` or `chaos fix --revert`")

    # ------------------------------------------------------------ revert
    def _revert_one(self, recipe_id: str, service: str, replicas: int) -> list[str]:
        errors = []
        try:
            self._scale(service, replicas)
            self._settle(service, replicas, killable=False)
        except Exception as e:
            errors.append(f"revert of {recipe_id}: {type(e).__name__}: {e}")
        if not errors:
            state = self._state()
            state.pop(recipe_id, None)
            self._write_state(state)
        return errors

    def revert_pending(self, include_kept: bool = False) -> list[str]:
        """Undo every fix recorded in the state file (used by `chaos reset` / `chaos abort` / `chaos fix --revert`)."""
        errors = []
        for rid, entry in self.pending(include_kept).items():
            errors += self._revert_one(rid, entry["service"], int(entry.get("original_replicas", 1)))
        return errors

    def save(self, run: FixRun) -> Path:
        run.ended_at = self.clock()
        out = self.results_dir / "fixes" / f"{run.id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return out
