"""The dashboard's autonomous-agent panel: the agent starts from a button, and the page can see what it is doing."""
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from chaosai import config
from chaosai.agent import InjectorAgent
from chaosai.config import AgentCfg
from chaosai.dashboard import App, make_server
from chaosai.models import Experiment, Plan, PlannedExperiment
from chaosai.observer import Store
from chaosai.safety import Guard, KillSwitch
from tests.test_agent import FakeOrch as AgentOrch, result
from tests.test_dashboard import FakeInjector, call, wait_for


class QuickOrch(AgentOrch):
    """The agent tests' orchestrator, with the injector the dashboard's Runner wraps and a run() that can be held."""

    def __init__(self, script=(), gate=None):
        super().__init__(script)
        self.injector, self.gate = FakeInjector(), gate

    def run(self, exp, baseline_s, recovery_timeout_s, dry_run=False):
        if self.gate:
            assert self.gate.wait(5)
        return super().run(exp, baseline_s, recovery_timeout_s, dry_run)


def planned(i, target, fault):
    return PlannedExperiment(rank=i, target=target, fault=fault, duration_s=10, hypothesis=f"h{i}", reason=f"why {i}", risk_score=1.0)


def make(tmp_path, gate=None, script=(), fail_build=None):
    cfg = config.load(Path("none.yaml"))
    kill = KillSwitch(tmp_path / "KILL")
    orch = QuickOrch(script, gate)
    saved = []

    def build(use_llm, max_n, log, progress):
        if fail_build:
            raise fail_build
        pairs = [("db", "kill"), ("redis", "pause"), ("worker", "kill")][:max_n]
        exps = [Experiment(id=f"e{i}", target=t, fault=f, duration_s=10) for i, (t, f) in enumerate(pairs, 1)]
        plan = Plan(planner="llm" if use_llm else "heuristic", created_at=1.0, model="m-1" if use_llm else None,
                    experiments=[planned(i, e.target, e.fault) for i, e in enumerate(exps, 1)])
        agent = InjectorAgent(orch, Guard({"redis", "db", "worker"}, kill), lambda: [], AgentCfg(cooldown_s=0, health_wait_s=1),
                              baseline_s=1, recovery_timeout_s=1, sleep=lambda s: None, log=log, progress=progress)
        return SimpleNamespace(plan=plan, experiments=exps, agent=agent, save=saved.append)

    app = App(cfg, Store(tmp_path / "t.db"), orch.injector, kill, orch, lambda: {}, agent_fn=build)
    return app, orch, kill, saved


def finished(app):
    return wait_for(lambda: not app.agent.snapshot()["running"])


def test_the_agent_plans_runs_every_experiment_and_the_page_state_shows_each_step(tmp_path):
    app, orch, kill, saved = make(tmp_path)
    code, out = app.start_agent({"max": 2, "use_llm": True})
    assert code == 202 and "by itself" in out["message"]
    assert finished(app)
    a = app.metrics()["agent"]
    assert a["phase"] == "done" and a["planner"] == "llm" and a["planner_model"] == "m-1" and a["error"] is None
    assert [(r["fault"], r["target"], r["status"]) for r in a["plan"]] == [("kill", "db", "done"), ("pause", "redis", "done")]
    assert a["plan"][0]["reason"] == "why 1" and a["plan"][0]["hypothesis"] == "h1" and a["plan"][0]["recovered"] is True
    assert a["summary"] == {"run": 2, "aborted": 0, "recovered": 2} and a["current"] is None and a["log"]
    assert [c[0] for c in orch.calls] == ["e1", "e2"] and len(saved) == 1 and saved[0].status == "completed"
    assert app.runner.snapshot()["running"] is False                    # the shared experiment state is released at the end


def test_while_an_experiment_runs_the_page_sees_which_one_and_that_the_agent_started_it(tmp_path):
    gate = threading.Event()
    app, orch, kill, _ = make(tmp_path, gate=gate)
    app.start_agent({"max": 2, "use_llm": False})
    assert wait_for(lambda: app.runner.snapshot()["running"])
    m = app.metrics()
    assert m["agent"]["phase"] == "injecting" and m["agent"]["current"] == 0 and m["agent"]["plan"][0]["status"] == "running"
    assert m["agent"]["plan"][1]["status"] == "pending" and m["agent"]["planner"] == "heuristic"
    assert m["experiment"]["by_agent"] is True and (m["experiment"]["target"], m["experiment"]["fault"]) == ("db", "kill")
    gate.set()
    assert finished(app)


def test_an_auto_aborted_experiment_is_marked_as_such(tmp_path):
    app, *_ = make(tmp_path, script=[result(aborted=True, reason="error rate 100% > 50%")])
    app.start_agent({"max": 1, "use_llm": True})
    assert finished(app)
    row = app.metrics()["agent"]["plan"][0]
    assert row["status"] == "aborted" and "error rate" in row["abort_reason"]
    assert app.metrics()["agent"]["summary"]["aborted"] == 1


def test_manual_injection_and_reset_are_refused_while_the_agent_runs(tmp_path):
    gate = threading.Event()
    app, orch, kill, _ = make(tmp_path, gate=gate)
    app.start_agent({"max": 1, "use_llm": True})
    assert wait_for(lambda: app.runner.snapshot()["running"])
    assert app.inject({"target": "db", "fault": "kill"})[0] == 409
    assert app.reset()[0] == 409
    assert app.start_agent({"max": 1, "use_llm": True})[0] == 409
    gate.set()
    assert finished(app)


def test_the_kill_switch_stops_the_agent_and_blocks_a_new_start(tmp_path):
    gate = threading.Event()
    app, orch, kill, _ = make(tmp_path, gate=gate)
    app.start_agent({"max": 3, "use_llm": True})
    assert wait_for(lambda: app.runner.snapshot()["running"])
    assert app.abort()[0] == 200
    gate.set()
    assert finished(app)
    a = app.metrics()["agent"]
    assert a["phase"] == "stopped" and "kill switch" in a["stop_reason"]
    assert app.start_agent({"max": 1, "use_llm": True})[0] == 409


def test_a_planner_failure_is_shown_not_swallowed(tmp_path):
    app, *_ = make(tmp_path, fail_build=RuntimeError("no docker"))
    app.start_agent({"max": 1, "use_llm": True})
    assert finished(app)
    a = app.metrics()["agent"]
    assert a["phase"] == "error" and "no docker" in a["error"] and a["running"] is False
    assert app.runner.snapshot()["running"] is False


@pytest.mark.parametrize("body", [{"max": 0}, {"max": 99}, {"max": "2"}, {"max": True}, {"use_llm": "yes"}])
def test_start_validates_its_input(tmp_path, body):
    app, *_ = make(tmp_path)
    assert app.start_agent(body)[0] == 400 and app.agent.snapshot()["phase"] == "idle"


def test_a_dashboard_without_the_agent_says_so(tmp_path):
    cfg = config.load(Path("none.yaml"))
    orch = QuickOrch()
    app = App(cfg, Store(tmp_path / "t.db"), orch.injector, KillSwitch(tmp_path / "KILL"), orch, lambda: {})
    assert app.metrics()["agent"] == {"available": False} and app.start_agent({})[0] == 404


def test_progress_events_come_in_order_and_do_not_change_how_the_agent_runs(tmp_path):
    from tests.test_agent import E3, make as make_agent
    events = []
    agent, orch, kill, clock, log = make_agent(tmp_path, [result(), result(aborted=True, reason="x"), result()])
    agent.progress = lambda ev, i, exp, entry: events.append((ev, i, exp.id, entry.status if entry else None))
    run = agent.run(E3)
    assert events == [("start", 0, "e1", None), ("done", 0, "e1", "done"), ("start", 1, "e2", None), ("done", 1, "e2", "done"),
                      ("start", 2, "e3", None), ("done", 2, "e3", "done")]
    assert run.status == "completed" and len(run.entries) == 3


# ---------------------------------------------------------------- over HTTP
@pytest.fixture
def served(tmp_path):
    app, orch, kill, _ = make(tmp_path)
    server = make_server(app, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SimpleNamespace(app=app, port=server.server_port)
    server.shutdown()
    server.server_close()


def test_the_start_endpoint_works_and_is_protected_like_the_others(served):
    assert call(served.port, "POST", "/api/agent/start", {"max": 1, "use_llm": False}, headers={"X-Chaos": "0"})[0] == 403
    assert call(served.port, "POST", "/api/agent/start", {"max": 1}, headers={"Origin": "http://evil.example"})[0] == 403
    assert call(served.port, "POST", "/api/agent/start", {"max": 1}, host="evil.example")[0] == 403
    code, out = call(served.port, "POST", "/api/agent/start", {"max": 1, "use_llm": False})
    assert code == 202 and finished(served.app)
    code, m = call(served.port, "GET", "/api/metrics")
    assert code == 200 and m["agent"]["phase"] == "done" and m["agent"]["plan"][0]["status"] == "done"


def test_the_page_has_the_agent_panel_wired_to_the_endpoint(served):
    code, html = call(served.port, "GET", "/")
    assert code == 200 and 'id="agentPanel"' in html and "/api/agent/start" in html and "renderAgent(d)" in html
    assert "Autonomous injecting agent" in html and 'id="agentStart"' in html
