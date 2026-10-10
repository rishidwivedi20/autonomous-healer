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
