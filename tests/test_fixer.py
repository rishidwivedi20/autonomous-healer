"""The fixing agent, with a fake Docker, orchestrator and clock (unit tests of the control logic and the judging;
no measurement in the real pipeline is synthetic)."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from chaosai import fixer
from chaosai.analysis import Analysis, Weakness
from chaosai.config import FixerCfg
from chaosai.fixer import (CATALOG, ComposeRunner, FixError, FixingAgent, FixRun, Measured, compare_rows, find_docker, judge, measure)
from chaosai.models import Thresholds
from chaosai.safety import KillSwitch, SafetyViolation
from tests.test_scoring import S, graph, result

SCALE = CATALOG["scale-worker"]
MANAGED = {"vote", "redis", "worker", "db", "result"}


class World:
    """Everything the agent talks to, recording the order of events."""

    def __init__(self, tmp_path, before=None, after=None, gauges_before=None, gauges_after=None, replicas=1, health=None):
        self.tmp, self.events, self.t = tmp_path, [], 1_000.0
        self.replicas = {"worker": replicas}
        self.health = health or []
        self.scripts = {"before": before or {"recovery_s": 60.0}, "after": after or {"recovery_s": 6.0}}
        self.gauges = {"before": gauges_before or {"peak_queue": 309.0, "peak_backlog_lag_s": 32.5, "votes_per_s_while_draining": 9.5},
                       "after": gauges_after or {"peak_queue": 40.0, "peak_backlog_lag_s": 4.0, "votes_per_s_while_draining": 28.0}}
        self.raise_in = {}                  # label -> exception raised by the verification run
        self.scale_error = None             # raised by compose.scale (set to None to heal)
        self.on_scale = None
        self.kill = KillSwitch(tmp_path / "KILL")
        self.guard = SimpleNamespace(kill=self.kill, managed_targets=set(MANAGED), lock=None, _running=False)
        self.state_path = tmp_path / "active_fixes.json"

    # collaborators
    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def replicas_fn(self, service):
        return self.replicas[service]

    def orch_factory(self, fix_dir):
        world = self

        class Orch:
            def run(self, exp, baseline_s, recovery_timeout_s):
                label = "before" if "-before-" in exp.id else "after"
                world.events.append(("run", label))
                if label == "after":
                    assert world.state_path.exists(), "the undo must be on disk before the change is tested"
                if label in world.raise_in:
                    raise world.raise_in[label]
                r = result(exp.id, target=exp.target, fault=exp.fault, **world.scripts[label]).model_copy(update={"experiment": exp})   # as the real orchestrator stores it
                fix_dir.mkdir(parents=True, exist_ok=True)
                (fix_dir / f"{exp.id}.json").write_text(r.model_dump_json(), encoding="utf-8")
                return r
        return Orch()

    def gauges_fn(self, fix_dir, exp_id):
        label = "before" if "-before-" in exp_id else "after"
        return {"summary": self.gauges[label]}

    class Compose:
        def __init__(self, world):
            self.world = world

        def scale(self, service, replicas):
            w = self.world
            if w.scale_error:
                raise w.scale_error
            w.events.append(("scale", service, replicas))
            w.replicas[service] = replicas
            if w.on_scale:
                w.on_scale()
            return ["docker", "compose", "up", "-d", "--no-deps", "--scale", f"{service}={replicas}", service]

    def agent(self, probe_interval_s=1.0, **cfg):
        return FixingAgent(self.orch_factory, self.Compose(self), self.replicas_fn, lambda: self.health, self.guard, graph(),
                           FixerCfg(cooldown_s=0, warmup_s=0, health_wait_s=5, **cfg), self.tmp, baseline_s=1, recovery_timeout_s=1,
                           thresholds=Thresholds(), probe_interval_s=probe_interval_s, gauges_fn=self.gauges_fn, state_path=self.state_path,
                           clock=self.clock, sleep=self.sleep)


@pytest.fixture
def w(tmp_path):
    return World(tmp_path)


def run(w, **kw):
    return w.agent().run("scale-worker", weakness="Single worker caps throughput", **kw)


# ---------------------------------------------------------------- the happy path
def test_a_fix_that_helps_is_measured_applied_verified_and_reverted(w):
    r = run(w)
    assert r.status == "improved" and r.reverted and not r.kept and r.revert_errors == []
    assert w.events == [("run", "before"), ("scale", "worker", 3), ("run", "after"), ("scale", "worker", 1)]
    assert r.before.metrics["peak_queue"] == 309.0 and r.after.metrics["peak_queue"] == 40.0
    assert r.after.score > r.before.score + 3 and "peak" in r.claim and "resilience score" in r.claim
    assert not w.state_path.exists() and w.replicas["worker"] == 1                       # the sandbox is as it was found
    assert r.original_replicas == 1 and r.applied_replicas == 3 and r.addresses == "Single worker caps throughput"


def test_the_record_is_saved_and_the_verification_runs_stay_out_of_the_experiment_history(w):
    r = run(w)
    saved = json.loads((w.tmp / "fixes" / f"{r.id}.json").read_text(encoding="utf-8"))
    assert FixRun.model_validate(saved).status == "improved" and saved["ended_at"] >= saved["started_at"]
    fix_dir = w.tmp / "fix-runs" / r.id
    assert len(list(fix_dir.glob("*.json"))) == 2                                           # before and after, in their own folder
    assert [p for p in w.tmp.glob("*.json") if p.name != "active_fixes.json"] == []         # nothing in results/*.json
    assert r.before.result_file.startswith(str(fix_dir)) and r.after.result_file.startswith(str(fix_dir))


def test_the_same_experiment_is_run_before_and_after(w):
    run(w)
    ids = [json.loads(p.read_text(encoding="utf-8"))["experiment"] for p in sorted((w.tmp / "fix-runs").glob("*/*.json"))]
    assert {(e["target"], e["fault"], json.dumps(e["params"]), e["duration_s"]) for e in ids} == {("vote", "traffic-spike", '{"multiplier": 15}', 20)}


# ---------------------------------------------------------------- the verification spike is a vote rate, so it means the same whatever the probe rate is
@pytest.mark.parametrize("interval,multiplier", [(1.0, 15), (0.2, 3), (0.05, 1), (2.0, 30)])
def test_the_spike_multiplier_follows_the_probe_rate(tmp_path, interval, multiplier):
    w = World(tmp_path)
    r = w.agent(probe_interval_s=interval).run("scale-worker")
    assert r.experiment["params"] == {"multiplier": multiplier} and r.experiment["votes_per_s"] == 15
    saved = [json.loads(p.read_text(encoding="utf-8"))["experiment"]["params"] for p in (tmp_path / "fix-runs").glob("*/*.json")]
    assert saved == [{"multiplier": multiplier}] * 2                                       # the same size before and after


def test_the_dry_run_states_the_vote_rate(w):
    r = w.agent().run("scale-worker", dry_run=True)
    assert "15 votes/s for 20s" in r.notes[0] and r.experiment["params"] == {"multiplier": 15}


# ---------------------------------------------------------------- when nothing should change
def test_it_changes_nothing_when_the_problem_does_not_reproduce(tmp_path):
    w = World(tmp_path, gauges_before={"peak_queue": 5.0})
    r = run(w)
    assert r.status == "skipped" and "not reproduced" in r.detail and "needs at least 20" in r.detail
    assert w.events == [("run", "before")] and not r.reverted and r.after is None and not w.state_path.exists()


def test_it_changes_nothing_when_the_fix_is_already_in_place(tmp_path):
    w = World(tmp_path, replicas=3)
    r = run(w)
    assert r.status == "skipped" and "already runs 3" in r.detail and w.events == []


def test_it_will_not_start_on_an_unhealthy_sandbox(tmp_path):
    w = World(tmp_path, health=["db"])
    r = run(w)
    assert r.status == "stopped" and "not healthy" in r.detail and "db" in r.detail and w.events == []


def test_a_refused_verification_run_stops_before_any_change(w):
    w.raise_in["before"] = SafetyViolation("not in steady state (error rate 100%)")
    r = run(w)
    assert r.status == "stopped" and "steady state" in r.detail and w.events == [("run", "before")] and not w.state_path.exists()


def test_rollback_problems_in_the_before_run_stop_everything(tmp_path):
    w = World(tmp_path)
    orig = w.orch_factory

    def bad(fix_dir):
        o = orig(fix_dir)
        inner = o.run
        o.run = lambda *a, **k: inner(*a, **k).model_copy(update={"rollback_errors": ["limit not restored"]})
        return o
    w.orch_factory = bad
    r = w.agent().run("scale-worker")
    assert r.status == "error" and "rollback problems" in r.detail and not any(e[0] == "scale" for e in w.events)


def test_dry_run_plans_without_touching_anything(w):
    r = w.agent().run("scale-worker", dry_run=True)
    assert r.status == "planned" and len(r.notes) == 5 and w.events == [] and not (w.tmp / "fixes").exists()


# ---------------------------------------------------------------- verdicts other than improved
def test_a_fix_that_makes_things_worse_is_reported_as_worse_and_reverted(tmp_path):
    w = World(tmp_path, before={"recovery_s": 6.0}, after={"recovery_s": 60.0}, gauges_before={"peak_queue": 100.0}, gauges_after={"peak_queue": 300.0})
    r = w.agent().run("scale-worker")
    assert r.status == "worse" and r.reverted and w.replicas["worker"] == 1


def test_a_small_change_is_inconclusive_and_not_claimed(tmp_path):
    w = World(tmp_path, before={"recovery_s": 30.0}, after={"recovery_s": 29.0}, gauges_before={"peak_queue": 100.0}, gauges_after={"peak_queue": 95.0})
    r = w.agent().run("scale-worker")
    assert r.status == "inconclusive" and "not claimed" in r.detail and r.reverted


def test_an_improvement_in_the_metric_without_a_score_gain_is_not_claimed(tmp_path):
    w = World(tmp_path, before={"recovery_s": 3.0}, after={"recovery_s": 3.0}, gauges_before={"peak_queue": 100.0}, gauges_after={"peak_queue": 30.0})
    r = w.agent().run("scale-worker")
    assert r.status == "inconclusive" and "score did not rise" in r.detail


# ---------------------------------------------------------------- keeping a fix
def test_keep_leaves_an_improving_fix_in_place_and_records_it(w):
    r = run(w, keep=True)
    assert r.status == "improved" and r.kept and not r.reverted
    assert w.events[-1] == ("run", "after") and w.replicas["worker"] == 3
    agent = w.agent()
    assert agent.pending() == {} and "scale-worker" in agent.pending(include_kept=True)          # reset leaves it alone
    assert agent.revert_pending() == [] and w.replicas["worker"] == 3
    assert agent.revert_pending(include_kept=True) == [] and w.replicas["worker"] == 1 and not w.state_path.exists()


def test_keep_is_ignored_when_the_fix_did_not_help(tmp_path):
    w = World(tmp_path, before={"recovery_s": 30.0}, after={"recovery_s": 30.0})
    r = w.agent().run("scale-worker", keep=True)
    assert r.status == "inconclusive" and r.reverted and not r.kept and w.replicas["worker"] == 1


# ---------------------------------------------------------------- failures never leave the change behind
def test_an_error_during_the_after_run_still_reverts(w):
    w.raise_in["after"] = RuntimeError("probe died")
    r = run(w)
    assert r.status == "error" and "probe died" in r.detail and r.reverted
    assert w.events[-1] == ("scale", "worker", 1) and not w.state_path.exists()


def test_the_kill_switch_stops_the_run_and_the_revert_still_happens(w):
    w.on_scale = w.kill.engage
    r = run(w)
    assert r.status == "stopped" and "kill switch" in r.detail
    assert r.reverted and w.replicas["worker"] == 1                                          # cleanup is not blocked by the kill switch


def test_a_failed_revert_is_reported_and_the_undo_stays_on_disk_for_chaos_reset(w):
    def break_revert():
        w.scale_error = FixError("docker compose failed: boom")        # the scale-down will fail
    w.on_scale = break_revert
    r = run(w)
    assert not r.reverted and r.revert_errors and "boom" in r.revert_errors[0] and any("REVERT FAILED" in n for n in r.notes)
    assert w.state_path.exists() and w.replicas["worker"] == 3
    w.scale_error = None
    assert w.agent().revert_pending() == [] and w.replicas["worker"] == 1 and not w.state_path.exists()


def test_the_undo_is_written_before_the_change(w):
    seen = {}
    w.on_scale = lambda: seen.update(json.loads(w.state_path.read_text(encoding="utf-8")))
    run(w)
    assert seen["scale-worker"] == {"service": "worker", "original_replicas": 1, "target_replicas": 3, "fix_id": seen["scale-worker"]["fix_id"],
                                    "applied_at": seen["scale-worker"]["applied_at"], "kept": False}


def test_a_busy_lock_blocks_the_change_and_nothing_is_scaled(tmp_path):
    w = World(tmp_path)
    w.guard.lock = SimpleNamespace(acquire=lambda: False, holder=lambda: "pid 99", release=lambda: None)
    r = run(w)
    assert r.status == "stopped" and "another process" in r.detail and not any(e[0] == "scale" for e in w.events)


def test_the_lock_is_held_only_while_the_sandbox_changes_and_released_afterwards(tmp_path):
    w = World(tmp_path)
    log = []
    w.guard.lock = SimpleNamespace(acquire=lambda: log.append("acquire") or True, holder=lambda: "me", release=lambda: log.append("release"))
    run(w)
    assert log == ["acquire", "release", "acquire", "release"]                               # one for the change, one for the revert


def test_an_experiment_already_running_in_this_process_blocks_a_change(tmp_path):
    w = World(tmp_path)
    w.guard._running = True
    r = run(w)
    assert r.status == "stopped" and "experiment is running" in r.detail


# ---------------------------------------------------------------- the allowlist and the catalogue
def test_unknown_fixes_and_unmanaged_services_are_refused(w):
    with pytest.raises(FixError, match="unknown fix 'rm-rf'"):
        w.agent().run("rm-rf")
    w.guard.managed_targets = {"vote"}
    with pytest.raises(SafetyViolation, match="allowlist"):
        w.agent().run("scale-worker")


def weakness(title, services, description="", severity="high"):
    return Weakness(title=title, severity=severity, services=services, description=description, evidence=[])


def analysis(*ws):
    return Analysis(summary="s", experiments=[], weaknesses=list(ws), fixes=[])


def test_a_throughput_weakness_on_the_worker_selects_the_scale_fix():
    sel = FixingAgent.select(analysis(weakness("Single worker caps pipeline throughput; any burst or stall builds a backlog", ["worker", "redis"])))
    assert [(r.id, w_.title[:13]) for r, w_ in sel] == [("scale-worker", "Single worker")]


def test_llm_style_titles_match_too_and_unrelated_weaknesses_do_not():
    assert FixingAgent.select(analysis(weakness("Worker capacity & latency are not resilient", ["worker"])))
    assert FixingAgent.select(analysis(weakness("Redis is a single point of failure", ["redis"]))) == []
    assert FixingAgent.select(analysis(weakness("Queue backlog under load", ["redis"]))) == []          # names no worker: not this fix
    assert FixingAgent.select(analysis(weakness("The worker has no healthcheck", ["worker"]))) == []


def test_each_recipe_is_picked_once_for_the_most_severe_matching_weakness():
    sel = FixingAgent.select(analysis(weakness("Worker throughput cap", ["worker"], severity="critical"), weakness("Worker backlog again", ["worker"])))
    assert len(sel) == 1 and sel[0][1].title == "Worker throughput cap"


def test_every_recipe_is_internally_consistent():
    for r in CATALOG.values():
        assert r.service in MANAGED and 1 < r.replicas <= fixer.MAX_REPLICAS
        assert r.primary in fixer.METRICS and r.problem_metric in fixer.METRICS and set(r.verify) == {"target", "fault", "votes_per_s", "duration_s"} and r.verify["votes_per_s"] > 0
        assert r.verify["duration_s"] <= 60 and r.verify["target"] in MANAGED


# ---------------------------------------------------------------- judging and measuring
def m(score, comps=None, aborted=False, **metrics):
    return Measured(experiment_id="x", score=score, components=comps or {}, metrics=metrics, aborted=aborted)


def test_judge_requires_both_a_score_gain_and_a_gain_in_the_fixs_own_metric():
    assert judge(m(60, peak_queue=300), m(90, peak_queue=100), "peak_queue", 3, 0.25)[0] == "improved"
    assert judge(m(60, peak_queue=300), m(61, peak_queue=100), "peak_queue", 3, 0.25)[0] == "inconclusive"      # metric only
    assert judge(m(60, peak_queue=300), m(90, peak_queue=290), "peak_queue", 3, 0.25)[0] == "inconclusive"      # score only
    assert judge(m(None, peak_queue=300), m(90, peak_queue=100), "peak_queue", 3, 0.25)[0] == "inconclusive"    # no score to compare
    assert judge(m(60, peak_queue=None), m(90, peak_queue=100), "peak_queue", 3, 0.25)[0] == "inconclusive"     # metric not measured


def test_judge_calls_a_regression_worse_and_an_abort_after_the_change_worse():
    assert judge(m(90, peak_queue=100), m(60, peak_queue=300), "peak_queue", 3, 0.25)[0] == "worse"
    assert judge(m(90, peak_queue=100), m(91, peak_queue=200), "peak_queue", 3, 0.25)[0] == "worse"             # metric alone doubled
    v, why = judge(m(90, peak_queue=100), m(95, peak_queue=50, aborted=True), "peak_queue", 3, 0.25)
    assert v == "worse" and "auto-aborted" in why
    assert judge(m(90, aborted=True, peak_queue=100), m(95, aborted=True, peak_queue=50), "peak_queue", 3, 0.25)[0] == "improved"


def test_a_big_drop_in_any_component_blocks_an_improved_verdict():
    before, after = m(60, {"availability": 1.0, "recovery": 0.0}, peak_queue=300), m(75, {"availability": 0.7, "recovery": 1.0}, peak_queue=100)
    v, why = judge(before, after, "peak_queue", 3, 0.25)
    assert v == "inconclusive" and "availability component" in why


def test_compare_rows_know_which_direction_is_better():
    rows = {r["key"]: r for r in compare_rows(m(1, peak_queue=300, drain_rate=9.5, error_pct=0.0, recovery_s=60.0),
                                              m(1, peak_queue=40, drain_rate=28.0, error_pct=0.0, recovery_s=62.0))}
    assert rows["peak_queue"]["verdict"] == "better" and rows["peak_queue"]["change_pct"] == pytest.approx(-86.7)
    assert rows["drain_rate"]["verdict"] == "better"                                       # higher is better here
    assert rows["error_pct"]["verdict"] == "same" and rows["recovery_s"]["verdict"] == "same"   # within 5%
    worse = {r["key"]: r for r in compare_rows(m(1, drain_rate=28.0), m(1, drain_rate=9.0))}
    assert worse["drain_rate"]["verdict"] == "worse"


def test_measure_extracts_the_metrics_from_a_result_and_its_gauges():
    r = result("e", target="vote", fault="traffic-spike", vote=S(err=0.2, p95=300.0), recovery_s=12.0, restarts={"vote": 1, "worker": 1})
    r.recovery_pipeline_s = 11.0
    got = measure(r, graph(), {"summary": {"peak_queue": 309.0, "peak_backlog_lag_s": 32.5, "votes_per_s_while_draining": 9.5,
                                          "queue_back_to_baseline_at_s": 80.0}}, "f.json")
    assert got.metrics["peak_queue"] == 309.0 and got.metrics["drain_rate"] == 9.5 and got.metrics["error_pct"] == 20.0
    assert got.metrics["vote_p95_ms"] == 300.0 and got.metrics["recovery_s"] == 12.0 and got.metrics["recovery_pipeline_s"] == 11.0
    assert got.metrics["collateral_restarts"] == 1.0 and got.score is not None and got.result_file == "f.json"
    assert measure(r, None, None).metrics["peak_queue"] is None                           # no gauges: missing, not zero


# ---------------------------------------------------------------- docker compose, safely
def runner_recording(rc=0, stderr="", stdout="", exc=None):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        if exc:
            raise exc
        return SimpleNamespace(returncode=rc, stderr=stderr, stdout=stdout)
    return run, calls


def test_compose_runs_exactly_one_fixed_command_without_a_shell(tmp_path):
    run_, calls = runner_recording()
    c = ComposeRunner(tmp_path / "docker-compose.yml", MANAGED, run=run_, timeout_s=33, docker="docker")
    cmd = c.scale("worker", 3)
    assert cmd == ["docker", "compose", "-f", str(tmp_path / "docker-compose.yml"), "up", "-d", "--no-deps", "--scale", "worker=3", "worker"]
    argv, kw = calls[0]
    assert argv == cmd and kw["cwd"] == str(tmp_path) and kw["timeout"] == 33 and "shell" not in kw and kw["capture_output"] and kw["text"]


@pytest.mark.parametrize("service,replicas,why", [("nginx", 2, "not a managed service"), ("worker; rm -rf /", 2, "not a managed service"),
                                                 ("worker", 0, "from 1 to"), ("worker", 6, "from 1 to"), ("worker", True, "from 1 to"),
                                                 ("worker", "3", "from 1 to"), ("worker", 2.5, "from 1 to")])
def test_compose_refuses_unmanaged_services_and_replica_counts_outside_the_bounds(tmp_path, service, replicas, why):
    run_, calls = runner_recording()
    with pytest.raises(FixError, match=why):
        ComposeRunner(tmp_path / "c.yml", MANAGED, run=run_, docker="docker").scale(service, replicas)
    assert calls == []


def test_compose_failures_surface_the_error_text_not_a_traceback(tmp_path):
    run_, _ = runner_recording(rc=1, stderr="no such service: worker\n")
    with pytest.raises(FixError, match="docker compose failed: no such service"):
        ComposeRunner(tmp_path / "c.yml", MANAGED, run=run_, docker="docker").scale("worker", 2)
    run_, _ = runner_recording(exc=subprocess.TimeoutExpired("docker", 5))
    with pytest.raises(FixError, match="did not finish within"):
        ComposeRunner(tmp_path / "c.yml", MANAGED, run=run_, docker="docker", timeout_s=5).scale("worker", 2)
    run_, _ = runner_recording(exc=FileNotFoundError("docker"))
    with pytest.raises(FixError, match="could not run docker compose"):
        ComposeRunner(tmp_path / "c.yml", MANAGED, run=run_, docker="docker").scale("worker", 2)


def test_find_docker_prefers_the_override_then_the_path_and_fails_clearly(tmp_path, monkeypatch):
    exe = tmp_path / "mydocker"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setenv("CHAOSAI_DOCKER", str(exe))
    assert find_docker() == str(exe)
    monkeypatch.delenv("CHAOSAI_DOCKER")
    monkeypatch.setattr(fixer.shutil, "which", lambda name: "/usr/bin/docker")
    assert find_docker() == "/usr/bin/docker"
    monkeypatch.setattr(fixer.shutil, "which", lambda name: None)
    monkeypatch.setattr(fixer, "os", SimpleNamespace(name="posix", environ={}))
    with pytest.raises(FixError, match="docker command was not found"):
        find_docker()


def test_a_relative_compose_path_is_made_absolute_so_it_still_resolves_from_the_composes_own_folder(tmp_path, monkeypatch):
    """Seen live: with `deploy/docker-compose.yml` and cwd set to `deploy/`, docker looked for deploy/deploy/docker-compose.yml."""
    (tmp_path / "deploy").mkdir()
    monkeypatch.chdir(tmp_path)
    run_, calls = runner_recording()
    cmd = ComposeRunner("deploy/docker-compose.yml", MANAGED, run=run_, docker="docker").scale("worker", 3)
    f = cmd[cmd.index("-f") + 1]
    assert Path(f).is_absolute() and Path(f) == (tmp_path / "deploy" / "docker-compose.yml").resolve()
    assert calls[0][1]["cwd"] == str((tmp_path / "deploy").resolve())
