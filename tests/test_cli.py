import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chaosai import cli

COMPOSE = Path(__file__).resolve().parent.parent / "deploy" / "docker-compose.yml"
runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(cli, "COMPOSE", COMPOSE)
    (tmp_path / "results").mkdir()
    return tmp_path / "results"


class FakeInjector:
    def __init__(self, errors=None):
        self.errors, self.calls = errors or [], 0

    def restore_all(self):
        self.calls += 1
        return self.errors


def test_dry_run_prints_plan_and_touches_nothing(isolated):
    r = runner.invoke(cli.app, ["run", "redis", "kill", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "DRY RUN" in r.output and "would apply kill to redis" in r.output
    assert list(isolated.glob("*.json")) == []


def test_dry_run_shows_clamped_duration_and_thresholds(isolated):
    r = runner.invoke(cli.app, ["run", "redis", "network-delay", "--dry-run", "--duration", "600",
                                "-p", "latency_ms=2000", "--error-rate", "0.9"])
    assert r.exit_code == 0
    assert "duration 60s" in r.output and "error_rate>50%" in r.output and "'latency_ms': 2000" in r.output


def test_dry_run_rejects_unlabeled_target(isolated):
    r = runner.invoke(cli.app, ["run", "ghost", "kill", "--dry-run"])
    assert r.exit_code == 2 and "allowlist" in r.output


def test_unknown_fault_is_a_clean_error_not_a_traceback(isolated):
    r = runner.invoke(cli.app, ["run", "redis", "explode", "--dry-run"])
    assert r.exit_code == 2 and "unknown fault" in r.output and "Traceback" not in r.output


def test_run_without_docker_fails_cleanly_and_creates_nothing(isolated, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no docker")
    monkeypatch.setattr(cli, "make_injector", boom)
    r = runner.invoke(cli.app, ["run", "redis", "kill"])
    assert r.exit_code == 2 and "cannot reach Docker" in r.output
    assert not (isolated / "chaosai.db").exists()


def test_bad_param_format(isolated):
    r = runner.invoke(cli.app, ["run", "redis", "kill", "--dry-run", "-p", "oops"])
    assert r.exit_code != 0


def test_discover_writes_graph(isolated):
    r = runner.invoke(cli.app, ["discover", "--compose", str(COMPOSE)])
    assert r.exit_code == 0, r.output
    assert "vote --> redis" in r.output and (isolated / "graph.json").exists()


def test_abort_engages_kill_switch_and_restores(isolated, monkeypatch):
    inj = FakeInjector()
    monkeypatch.setattr(cli, "make_injector", lambda *a, **k: inj)
    r = runner.invoke(cli.app, ["abort"])
    assert r.exit_code == 0 and (isolated / "KILL").exists() and inj.calls == 1


def test_abort_without_docker_still_sets_kill_switch(isolated, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no docker")
    monkeypatch.setattr(cli, "make_injector", boom)
    r = runner.invoke(cli.app, ["abort"])
    assert r.exit_code == 1 and (isolated / "KILL").exists() and "could not reach Docker" in r.output


def test_reset_clears_kill_switch(isolated, monkeypatch):
    (isolated / "KILL").write_text("x")
    monkeypatch.setattr(cli, "make_injector", lambda *a, **k: FakeInjector())
    r = runner.invoke(cli.app, ["reset"])
    assert r.exit_code == 0 and not (isolated / "KILL").exists()


def test_reset_refuses_when_restore_fails(isolated, monkeypatch):
    (isolated / "KILL").write_text("x")
    monkeypatch.setattr(cli, "make_injector", lambda *a, **k: FakeInjector(errors=["redis: stuck"]))
    r = runner.invoke(cli.app, ["reset"])
    assert r.exit_code == 1 and (isolated / "KILL").exists() and "redis: stuck" in r.output


def test_parse_params_types():
    assert cli.parse_params(["latency_ms=800", "cpus=0.25", "mem=32m", "flag=true"]) == {
        "latency_ms": 800, "cpus": 0.25, "mem": "32m", "flag": True}


# ---- plan
def test_plan_offline_dry_run_prints_ranking_and_writes_nothing(isolated):
    r = runner.invoke(cli.app, ["plan", "--offline", "--dry-run", "--top", "4"])
    assert r.exit_code == 0, r.output
    assert "planner: heuristic" in r.output and "1. db / kill" in r.output and "hypothesis:" in r.output
    assert "4. " in r.output and "5. " not in r.output
    assert not (isolated / "plan.json").exists()


def test_plan_writes_plan_json_and_reports_llm_fallback(isolated, monkeypatch):
    import anthropic

    def no_creds(*a, **k):
        raise TypeError("Could not resolve authentication method")   # as the real SDK does without credentials
    monkeypatch.setattr(anthropic, "Anthropic", no_creds)
    r = runner.invoke(cli.app, ["plan", "--top", "3"])
    assert r.exit_code == 0, r.output
    assert "fell back to heuristic" in r.output and "ANTHROPIC_API_KEY" in r.output
    saved = json.loads((isolated / "plan.json").read_text())
    assert saved["planner"] == "heuristic" and len(saved["experiments"]) == 3


def test_plan_counts_past_experiments(isolated):
    from chaosai.models import Experiment, ExperimentResult, Stats
    res = ExperimentResult(experiment=Experiment(id="old", target="db", fault="kill"), started_at=1, ended_at=2,
                           baseline=Stats(), during=Stats())
    (isolated / "old.json").write_text(res.model_dump_json())
    r = runner.invoke(cli.app, ["plan", "--offline", "--dry-run"])
    assert "1 past experiment(s) considered" in r.output
    assert "1. db / kill" not in r.output          # already tested once, loses its untested bonus


# ---- agent
def test_agent_dry_run_plans_validates_and_injects_nothing(isolated):
    r = runner.invoke(cli.app, ["agent", "--dry-run", "--offline", "--top", "3"])
    assert r.exit_code == 0, r.output
    assert "DRY RUN: the agent would inject 3 fault(s)" in r.output
    assert r.output.count("[dry-run] would apply") == 3
    assert list(isolated.glob("*.json")) == [] or all(p.name == "graph.json" for p in isolated.glob("*.json"))
    assert not (isolated / "agent-runs").exists() and not (isolated / "KILL").exists()


def test_agent_max_flag_caps_the_run(isolated):
    r = runner.invoke(cli.app, ["agent", "--dry-run", "--offline", "--top", "4", "--max", "2"])
    assert "would inject 2 fault(s)" in r.output and r.output.count("[dry-run] would apply") == 2


def test_agent_without_docker_refuses_before_injecting(isolated, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no docker")
    monkeypatch.setattr(cli, "make_injector", boom)
    r = runner.invoke(cli.app, ["agent", "--offline", "--yes", "--top", "2"])
    assert r.exit_code == 2 and "cannot reach Docker" in r.output and not (isolated / "chaosai.db").exists()


def test_agent_asks_for_confirmation_and_cancels_cleanly(isolated):
    r = runner.invoke(cli.app, ["agent", "--offline", "--top", "2"], input="n\n")
    assert r.exit_code == 1 and "Continue?" in r.output and "cancelled; nothing injected" in r.output


# ---- report / autopilot
def _with_results(tmp_path, monkeypatch):
    from tests.fixtures_results import make_results
    (tmp_path / "fx").mkdir()
    d = make_results(tmp_path / "fx")
    monkeypatch.setattr(cli, "RESULTS", d)
    return d


def test_report_writes_findings_and_prints_the_score(tmp_path, monkeypatch):
    d = _with_results(tmp_path, monkeypatch)
    r = runner.invoke(cli.app, ["report", "--offline"])
    assert r.exit_code == 0, r.output
    assert "RESILIENCE SCORE" in r.output and "2 experiment(s), analysis by deterministic rules" in r.output
    for name in ("findings.md", "findings.html", "findings.json", "analysis.json"):
        assert (d / name).exists(), name


def test_report_last_limits_the_scope(tmp_path, monkeypatch):
    d = _with_results(tmp_path, monkeypatch)
    r = runner.invoke(cli.app, ["report", "--offline", "--last", "1"])
    assert r.exit_code == 0 and "1 experiment(s)" in r.output


def test_report_with_nothing_stored_is_a_clean_error(isolated):
    r = runner.invoke(cli.app, ["report", "--offline"])
    assert r.exit_code == 1 and "no stored experiments" in r.output and not (isolated / "findings.md").exists()


def test_autopilot_dry_run_plans_and_injects_nothing(isolated):
    r = runner.invoke(cli.app, ["autopilot", "--dry-run", "--offline", "--top", "2"])
    assert r.exit_code == 0, r.output
    assert "DRY RUN: the agent would inject 2 fault(s)" in r.output
    assert not (isolated / "agent-runs").exists() and not (isolated / "findings.md").exists()


def test_autopilot_asks_for_confirmation_and_cancels_cleanly(isolated):
    r = runner.invoke(cli.app, ["autopilot", "--offline", "--top", "2"], input="n\n")
    assert r.exit_code == 1 and "AUTOPILOT" in r.output and "cancelled; nothing injected" in r.output


def _fake_agent(monkeypatch, entries, status="completed", seen=None):
    from chaosai.agent import AgentEntry, AgentRun

    def fake(c, exps, acfg, planner_name, stamp):
        if seen is not None:
            seen.append(([e.id for e in exps], planner_name))
        return AgentRun(started_at=1.0, ended_at=2.0, status=status, stop_reason="boom" if status == "stopped" else None,
                        entries=[AgentEntry(id=i, target="x", fault="y", status=s) for i, s in entries])
    monkeypatch.setattr(cli, "execute_agent", fake)


def test_autopilot_chains_agent_then_report_with_no_manual_step(tmp_path, monkeypatch):
    from tests.fixtures_results import KILL_ID, SPIKE_ID
    d = _with_results(tmp_path, monkeypatch)
    seen = []
    _fake_agent(monkeypatch, [(SPIKE_ID, "done"), (KILL_ID, "done")], seen=seen)
    r = runner.invoke(cli.app, ["autopilot", "--offline", "--yes", "--top", "2"])
    assert r.exit_code == 0, r.output
    assert len(seen) == 1 and len(seen[0][0]) == 2 and seen[0][1] == "heuristic"          # it planned and handed the plan to the agent
    assert "analyzing, scoring and writing the report" in r.output and "RESILIENCE SCORE" in r.output
    assert (d / "findings.md").exists() and (d / "findings.html").exists()


def test_autopilot_reports_only_what_the_agent_actually_ran(tmp_path, monkeypatch):
    from tests.fixtures_results import KILL_ID, SPIKE_ID
    d = _with_results(tmp_path, monkeypatch)
    _fake_agent(monkeypatch, [(SPIKE_ID, "done"), (KILL_ID, "skipped")])
    r = runner.invoke(cli.app, ["autopilot", "--offline", "--yes", "--top", "2"])
    assert r.exit_code == 0 and "1 experiment(s)" in r.output
    assert [e["id"] for e in json.loads((d / "findings.json").read_text(encoding="utf-8"))["experiments"]] == [SPIKE_ID]


def test_autopilot_still_reports_when_the_agent_stopped_early_but_exits_non_zero(tmp_path, monkeypatch):
    from tests.fixtures_results import SPIKE_ID
    d = _with_results(tmp_path, monkeypatch)
    _fake_agent(monkeypatch, [(SPIKE_ID, "done")], status="stopped")
    r = runner.invoke(cli.app, ["autopilot", "--offline", "--yes", "--top", "2"])
    assert r.exit_code == 1 and (d / "findings.md").exists()


def test_autopilot_with_no_completed_experiment_does_not_write_a_report(tmp_path, monkeypatch):
    d = _with_results(tmp_path, monkeypatch)
    _fake_agent(monkeypatch, [("e1", "skipped")], status="stopped")
    r = runner.invoke(cli.app, ["autopilot", "--offline", "--yes", "--top", "2"])
    assert r.exit_code == 1 and "nothing to report" in r.output and not (d / "findings.md").exists()
