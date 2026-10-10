from types import SimpleNamespace

import pytest

from chaosai.agent import InjectorAgent
from chaosai.config import AgentCfg
from chaosai.injector import InjectionError
from chaosai.models import Experiment
from chaosai.safety import Guard, KillSwitch, SafetyViolation


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def exps(*names):
    return [Experiment(id=f"e{i}", target=t, fault=f, duration_s=10) for i, (t, f) in enumerate(names, 1)]


def result(aborted=False, reason=None, recovered=True, rollback=None, notes=("applied",)):
    return SimpleNamespace(aborted=aborted, abort_reason=reason, recovered=recovered, recovery_s=2.0,
                           rollback_errors=rollback or [], notes=list(notes),
                           during_by_journey={"vote": SimpleNamespace(error_rate=0.5)})


class FakeOrch:
    def __init__(self, script):
        self.script, self.calls, self.results_dir = list(script), [], "res"

    def run(self, exp, baseline_s, recovery_timeout_s, dry_run=False):
        self.calls.append((exp.id, baseline_s, recovery_timeout_s, dry_run))
        nxt = self.script.pop(0) if self.script else result()
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def make(tmp_path, script=(), health=lambda: [], cfg=None, clock=None):
    clock = clock or Clock()
    kill = KillSwitch(tmp_path / "KILL")
    guard = Guard({"redis", "db", "worker"}, kill)
    orch = FakeOrch(script)
    log = []
    agent = InjectorAgent(orch, guard, health, cfg or AgentCfg(cooldown_s=5, health_wait_s=10), baseline_s=15,
                          recovery_timeout_s=120, clock=clock.now, sleep=clock.sleep, log=log.append)
    return agent, orch, kill, clock, log


E3 = exps(("db", "kill"), ("redis", "pause"), ("worker", "kill"))


def test_runs_every_planned_experiment_in_order(tmp_path):
    agent, orch, *_ = make(tmp_path)
    run = agent.run(E3)
    assert run.status == "completed" and run.stop_reason is None
    assert [c[0] for c in orch.calls] == ["e1", "e2", "e3"]
    assert [e.status for e in run.entries] == ["done"] * 3 and run.entries[0].result_file.endswith("e1.json")
    assert all(c[1:3] == (15, 120) for c in orch.calls)                    # parameters flow through


def test_cooldown_between_experiments_but_not_after_the_last(tmp_path):
    agent, _, _, clock, _ = make(tmp_path)
    agent.run(E3)
    assert clock.t == pytest.approx(10.0, abs=1.0)                         # two 5 s cooldowns, none after the third


def test_max_experiments_caps_the_run_and_reports_the_rest(tmp_path):
    agent, orch, *_ = make(tmp_path, cfg=AgentCfg(max_experiments=2, cooldown_s=0))
    run = agent.run(E3)
    assert len(orch.calls) == 2 and run.not_run == ["e3"] and run.status == "completed"


def test_stops_on_rollback_problems_and_runs_nothing_more(tmp_path):
    agent, orch, *_ = make(tmp_path, script=[result(), result(rollback=["limit not removed"])])
    run = agent.run(E3)
    assert run.status == "stopped" and "rollback problems" in run.stop_reason and "chaos abort" in run.stop_reason
    assert len(orch.calls) == 2 and run.not_run == ["e3"]


def test_stops_when_the_kill_switch_ends_an_experiment(tmp_path):
    agent, orch, *_ = make(tmp_path, script=[result(aborted=True, reason="kill switch")])
    run = agent.run(E3)
    assert run.status == "stopped" and run.stop_reason == "kill switch engaged" and len(orch.calls) == 1
    assert run.not_run == ["e2", "e3"]


def test_kill_switch_engaged_before_start_runs_nothing(tmp_path):
    agent, orch, kill, *_ = make(tmp_path)
    kill.engage("test")
    run = agent.run(E3)
    assert orch.calls == [] and run.stop_reason == "kill switch engaged" and run.not_run == ["e1", "e2", "e3"]


def test_kill_switch_during_cooldown_stops_before_the_next_fault(tmp_path):
    clock = Clock()
    agent, orch, kill, _, _ = make(tmp_path, clock=clock, cfg=AgentCfg(cooldown_s=30))
    real_sleep = clock.sleep
    clock.sleep = lambda s: (real_sleep(s), kill.engage("during cooldown") if clock.t > 3 else None)[0]
    agent.sleep = clock.sleep
    run = agent.run(E3)
    assert len(orch.calls) == 1 and run.stop_reason == "kill switch engaged"


def test_waits_for_containers_to_become_healthy_then_proceeds(tmp_path):
    states = iter([["redis"], ["redis"], []] + [[]] * 10)
    agent, orch, *_ = make(tmp_path, health=lambda: next(states))
    run = agent.run(E3[:1])
    assert run.status == "completed" and len(orch.calls) == 1


def test_refuses_to_inject_into_an_unhealthy_system(tmp_path):
    agent, orch, *_ = make(tmp_path, health=lambda: ["db"])
    run = agent.run(E3)
    assert orch.calls == [] and run.status == "stopped" and "not healthy" in run.stop_reason and "db" in run.stop_reason


def test_a_refused_experiment_is_skipped_and_the_agent_continues(tmp_path):
    agent, orch, *_ = make(tmp_path, script=[SafetyViolation("'ghost' is not labeled"), result()],
                           cfg=AgentCfg(cooldown_s=0))
    run = agent.run(E3[:2])
    assert [e.status for e in run.entries] == ["skipped", "done"] and "not labeled" in run.entries[0].detail
    assert run.status == "completed"


def test_consecutive_refusals_stop_the_run(tmp_path):
    bad = SafetyViolation("not in steady state")
    agent, orch, *_ = make(tmp_path, script=[bad, bad, result()], cfg=AgentCfg(cooldown_s=0, max_consecutive_skips=2))
    run = agent.run(E3)
    assert run.status == "stopped" and "refused by the safety guard" in run.stop_reason and len(orch.calls) == 2
    assert run.not_run == ["e3"]


def test_a_success_resets_the_refusal_streak(tmp_path):
    bad = SafetyViolation("flaky")
    agent, orch, *_ = make(tmp_path, script=[bad, result(), bad], cfg=AgentCfg(cooldown_s=0, max_consecutive_skips=2))
    run = agent.run(E3)
    assert run.status == "completed" and [e.status for e in run.entries] == ["skipped", "done", "skipped"]


def test_injector_error_stops_the_run(tmp_path):
    agent, orch, *_ = make(tmp_path, script=[InjectionError("docker down")])
    run = agent.run(E3)
    assert run.status == "stopped" and "injector error" in run.stop_reason and run.entries[0].status == "error"


def test_auto_abort_is_recorded_and_continues_by_default(tmp_path):
    agent, orch, *_ = make(tmp_path, script=[result(aborted=True, reason="error rate 100% > 50% for 3 consecutive s")],
                           cfg=AgentCfg(cooldown_s=0))
    run = agent.run(E3[:2])
    assert run.status == "completed" and run.entries[0].aborted and len(orch.calls) == 2


def test_stop_on_abort_option(tmp_path):
    agent, orch, *_ = make(tmp_path, script=[result(aborted=True, reason="x")], cfg=AgentCfg(cooldown_s=0, stop_on_abort=True))
    run = agent.run(E3)
    assert run.status == "stopped" and "stop_on_abort" in run.stop_reason and len(orch.calls) == 1


def test_dry_run_checks_everything_but_never_waits_or_injects(tmp_path):
    agent, orch, kill, clock, _ = make(tmp_path, health=lambda: ["db"], script=[result(notes=("[dry-run] would apply kill",))] * 3)
    run = agent.run(E3, dry_run=True)
    assert run.dry_run and run.status == "completed" and all(c[3] for c in orch.calls)
    assert [e.status for e in run.entries] == ["planned"] * 3 and clock.t == 0     # no health wait, no cooldown


def test_dry_run_still_reports_guard_rejections(tmp_path):
    agent, orch, *_ = make(tmp_path, script=[SafetyViolation("not labeled")])
    run = agent.run(exps(("ghost", "kill")), dry_run=True)
    assert run.entries[0].status == "skipped"
