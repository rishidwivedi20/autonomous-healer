"""`chaos fix` and `chaos pipeline` (the fixing agent and all three agents together), with the Docker-facing parts faked."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from chaosai import cli, fixer
from chaosai.agent import AgentEntry, AgentRun
from chaosai.fixer import FixRun, Measured, compare_rows
from tests.fixtures_results import KILL_ID, SPIKE_ID, make_results

COMPOSE = Path(__file__).resolve().parent.parent / "deploy" / "docker-compose.yml"
runner = CliRunner()


@pytest.fixture
def results(tmp_path, monkeypatch):
    """A results folder holding two stored experiments (a traffic spike that backs the queue up, and a db kill)."""
    (tmp_path / "w").mkdir()
    d = make_results(tmp_path / "w")
    monkeypatch.setattr(cli, "RESULTS", d)
    monkeypatch.setattr(cli, "COMPOSE", COMPOSE)
    return d


def improved_run(id_="20261011-130000-scale-worker", status="improved", reverted=True, kept=False, revert_errors=None):
    before = Measured(experiment_id="b", score=60.0, metrics={"peak_queue": 309.0, "recovery_pipeline_s": 80.0})
    after = Measured(experiment_id="a", score=96.0, metrics={"peak_queue": 40.0, "recovery_pipeline_s": 12.0})
    return FixRun(id=id_, started_at=1.0, ended_at=2.0, status=status, detail="the score rose +36.0 points and the redis queue peak improved 87%",
                  recipe_id="scale-worker", title="Run three worker replicas", service="worker", change="docker compose up -d --scale worker=3 worker",
                  why="share the load", addresses="Single worker caps pipeline throughput", before=before, after=after, comparison=compare_rows(before, after),
                  claim="resilience score 60.0 -> 96.0; redis queue peak 309 votes -> 40 votes", reverted=reverted, kept=kept, revert_errors=revert_errors or [])


class FakeFixer:
    """Stands in for the real fixing agent: records what it was asked and saves a record like the real one does."""
    def __init__(self, results_dir, run=None, raises=None):
        self.results_dir, self.calls, self.result, self.raises = results_dir, [], run or improved_run(), raises

    def run(self, recipe, weakness="", keep=None, dry_run=False):
        self.calls.append((recipe, weakness, keep))
        if self.raises:
            raise self.raises
        d = self.results_dir / "fixes"
        d.mkdir(exist_ok=True)
        (d / f"{self.result.id}.json").write_text(self.result.model_dump_json(), encoding="utf-8")
        return self.result


def fake_live(monkeypatch, fixer_obj):
    stopped = []
    monkeypatch.setattr(cli, "start_live", lambda c, kill, **kw: SimpleNamespace(stop=lambda: stopped.append(1), injector=SimpleNamespace(restore_all=lambda: [])))
    monkeypatch.setattr(cli, "build_fixing_agent", lambda c, live, graph, compose: fixer_obj)
    return stopped


# ---------------------------------------------------------------- chaos fix
def test_fix_list_shows_the_catalogue(results):
    r = runner.invoke(cli.app, ["fix", "--list"])
    assert r.exit_code == 0 and "scale-worker: Run three worker replicas" in r.output and "--scale worker=3" in r.output
    assert "traffic-spike on vote" in r.output and "redis queue peak" in r.output


def test_fix_dry_run_prints_the_plan_and_changes_nothing(results):
    r = runner.invoke(cli.app, ["fix", "--recipe", "scale-worker", "--dry-run"])
    assert r.exit_code == 0 and "DRY RUN of the fixing agent" in r.output and "nothing was changed" in r.output
    assert "record the undo, then set worker to 3 replicas" in r.output and "put the original replica count back" in r.output
    assert not (results / "fixes").exists() and not (results / "active_fixes.json").exists()


def test_fix_without_a_recipe_picks_one_from_the_findings(results):
    r = runner.invoke(cli.app, ["fix", "--offline", "--dry-run"])
    assert r.exit_code == 0 and "the findings call for: Run three worker replicas" in r.output
    assert "addresses: Single worker caps pipeline throughput" in r.output


def test_fix_with_no_matching_finding_says_so_and_changes_nothing(results):
    (results / f"{SPIKE_ID}.json").unlink()                      # without the spike there is no backlog, so no worker weakness
    r = runner.invoke(cli.app, ["fix", "--offline", "--dry-run"])
    assert r.exit_code == 1 and "no vetted fix matches the findings" in r.output


def test_fix_with_nothing_stored_is_a_clear_error(tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(cli, "RESULTS", tmp_path / "empty")
    monkeypatch.setattr(cli, "COMPOSE", COMPOSE)
    r = runner.invoke(cli.app, ["fix", "--offline", "--dry-run"])
    assert r.exit_code == 1 and "no stored experiments" in r.output


def test_an_unknown_recipe_is_refused_before_anything_happens(results):
    r = runner.invoke(cli.app, ["fix", "--recipe", "rm-rf", "--yes"])
    assert r.exit_code != 0 and "unknown fix 'rm-rf'" in r.output and "scale-worker" in r.output


def test_fix_asks_for_confirmation_and_cancels_cleanly(results, monkeypatch):
    f = FakeFixer(results)
    fake_live(monkeypatch, f)
    r = runner.invoke(cli.app, ["fix", "--recipe", "scale-worker"], input="n\n")
    assert r.exit_code == 1 and "cancelled; nothing changed" in r.output and f.calls == []
    assert "twice (before and after)" in r.output and "puts the original back" in r.output


def test_fix_runs_the_agent_and_prints_the_before_and_after_table(results, monkeypatch):
    f = FakeFixer(results)
    stopped = fake_live(monkeypatch, f)
    r = runner.invoke(cli.app, ["fix", "--recipe", "scale-worker", "--yes"])
    assert r.exit_code == 0, r.output
    assert f.calls == [("scale-worker", "", None)] and stopped == [1]                 # the live probe and watcher are always stopped
    assert "FIX scale-worker: IMPROVED" in r.output and "redis queue peak (votes)" in r.output and "309" in r.output and "-87%" in r.output
    assert "resilience score 60.0 -> 96.0" in r.output and "sandbox: reverted to the original" in r.output and "saved" in r.output


def test_keep_is_passed_on_and_a_kept_fix_says_how_to_undo_it(results, monkeypatch):
    f = FakeFixer(results, improved_run(kept=True, reverted=False))
    fake_live(monkeypatch, f)
    r = runner.invoke(cli.app, ["fix", "--recipe", "scale-worker", "--keep", "--yes"])
    assert r.exit_code == 0 and f.calls[0][2] is True and "kept as changed (undo with `chaos fix --revert`)" in r.output


@pytest.mark.parametrize("run,code", [(improved_run(status="inconclusive"), 0), (improved_run(status="worse"), 0), (improved_run(status="skipped"), 0),
                                      (improved_run(status="stopped"), 1), (improved_run(status="error"), 1),
                                      (improved_run(reverted=False, revert_errors=["revert of scale-worker: boom"]), 1)])
def test_exit_codes_reflect_whether_the_agent_finished_cleanly(results, monkeypatch, run, code):
    fake_live(monkeypatch, FakeFixer(results, run))
    r = runner.invoke(cli.app, ["fix", "--recipe", "scale-worker", "--yes"])
    assert r.exit_code == code, r.output
    assert ("NOT reverted" in r.output) == bool(run.revert_errors)


def test_a_refusal_from_the_agent_is_a_clean_error(results, monkeypatch):
    fake_live(monkeypatch, FakeFixer(results, raises=cli.SafetyViolation("an experiment is running")))
    r = runner.invoke(cli.app, ["fix", "--recipe", "scale-worker", "--yes"])
    assert r.exit_code == 1 and "an experiment is running" in r.output


def test_fix_revert_with_nothing_recorded_is_a_no_op(results):
    r = runner.invoke(cli.app, ["fix", "--revert"])
    assert r.exit_code == 0 and "every recorded fix is reverted" in r.output


def test_fix_revert_reports_problems(results, monkeypatch):
    monkeypatch.setattr(cli, "revert_pending_fixes", lambda c, include_kept=False: ["revert of scale-worker: boom"] if include_kept else [])
    r = runner.invoke(cli.app, ["fix", "--revert"])
    assert r.exit_code == 1 and "boom" in r.output


# ---------------------------------------------------------------- reset / abort undo recorded fixes
def test_reset_refuses_to_clear_the_kill_switch_when_a_fix_could_not_be_undone(results, monkeypatch):
    monkeypatch.setattr(cli, "make_injector", lambda *a, **k: SimpleNamespace(restore_all=lambda: []))
    monkeypatch.setattr(cli, "revert_pending_fixes", lambda c, include_kept=False: ["revert of scale-worker: docker compose failed"])
    cli.kill_switch().engage("test")
    r = runner.invoke(cli.app, ["reset"])
    assert r.exit_code == 1 and "docker compose failed" in r.output and cli.kill_switch().engaged


def test_reset_and_abort_ask_to_revert_pending_fixes_but_never_the_kept_ones(results, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "make_injector", lambda *a, **k: SimpleNamespace(restore_all=lambda: []))
    monkeypatch.setattr(cli, "revert_pending_fixes", lambda c, include_kept=False: seen.append(include_kept) or [])
    assert runner.invoke(cli.app, ["reset"]).exit_code == 0 and runner.invoke(cli.app, ["abort"]).exit_code == 0
    assert seen == [False, False]


def test_reverting_with_no_state_file_never_touches_docker(results, monkeypatch):
    monkeypatch.setattr(cli, "make_injector", lambda *a, **k: pytest.fail("Docker must not be needed when nothing is recorded"))
    assert cli.revert_pending_fixes(cli.cfg(), include_kept=True) == []


def test_reverting_without_docker_reports_it_instead_of_crashing(results, monkeypatch):
    (results / "active_fixes.json").write_text(json.dumps({"scale-worker": {"service": "worker", "original_replicas": 1}}), encoding="utf-8")

    def boom(*a, **k):
        raise RuntimeError("no docker")
    monkeypatch.setattr(cli, "make_injector", boom)
    errors = cli.revert_pending_fixes(cli.cfg())
    assert errors and "could not reach Docker" in errors[0]


# ---------------------------------------------------------------- chaos pipeline: all three agents
def fake_injector(monkeypatch, ids, status="completed", stop_reason=None):
    calls = []

    def execute(c, exps, acfg, planner_name, stamp):
        calls.append(([e.id for e in exps], planner_name))
        return AgentRun(started_at=1.0, ended_at=2.0, status=status, stop_reason=stop_reason,
                        entries=[AgentEntry(id=i, target="x", fault="y", status="done", recovered=True) for i in ids])
    monkeypatch.setattr(cli, "execute_agent", execute)
    return calls


def test_pipeline_dry_run_plans_all_the_agents_and_changes_nothing(results):
    r = runner.invoke(cli.app, ["pipeline", "--dry-run", "--offline", "--top", "2"])
    assert r.exit_code == 0, r.output
    assert "AGENT 1/3: INJECTING (plan)" in r.output and "AGENT 3/3: FIXING (plan)" in r.output and "nothing was changed" in r.output
    assert "DRY RUN: the agent would inject 2 fault(s)" in r.output and not (results / "fixes").exists()


def test_pipeline_dry_run_without_the_fixing_agent(results):
    r = runner.invoke(cli.app, ["pipeline", "--dry-run", "--offline", "--top", "2", "--no-fix"])
    assert r.exit_code == 0 and "AGENT 1/3" in r.output and "AGENT 3/3" not in r.output


def test_pipeline_asks_for_confirmation_and_cancels_cleanly(results, monkeypatch):
    calls = fake_injector(monkeypatch, [KILL_ID])
    r = runner.invoke(cli.app, ["pipeline", "--offline", "--top", "2"], input="n\n")
    assert r.exit_code == 1 and "PIPELINE" in r.output and "cancelled; nothing injected" in r.output and calls == []


def test_the_three_agents_run_in_order_and_the_report_includes_the_validated_fix(results, monkeypatch):
    inj = fake_injector(monkeypatch, [SPIKE_ID, KILL_ID])
    f = FakeFixer(results)
    stopped = fake_live(monkeypatch, f)
    r = runner.invoke(cli.app, ["pipeline", "--offline", "--yes", "--top", "2"])
    assert r.exit_code == 0, r.output
    order = [r.output.index(m) for m in ("AGENT 1/3: INJECTING", "AGENT 2/3: FINDING", "AGENT 3/3: FIXING", "updating the report with the validated fix", "== SUMMARY ==")]
    assert order == sorted(order)
    assert len(inj) == 1 and f.calls[0][0] == "scale-worker" and f.calls[0][1].startswith("Single worker") and stopped == [1]
    md = (results / "findings.md").read_text(encoding="utf-8")
    assert "## Fix validation: before and after" in md and "Run three worker replicas: measured improvement" in md
    assert "injecting: 2 experiment(s) run" in r.output and "finding:" in r.output and "resilience score" in r.output
    assert "fixing:    scale-worker -> improved: resilience score 60.0 -> 96.0" in r.output and "findings.html" in r.output


def test_a_weakness_the_run_did_not_exercise_is_found_in_the_stored_history(results, monkeypatch):
    fake_injector(monkeypatch, [KILL_ID])                          # this run only killed the db: no backlog evidence in it
    f = FakeFixer(results)
    fake_live(monkeypatch, f)
    r = runner.invoke(cli.app, ["pipeline", "--offline", "--yes", "--top", "2"])
    assert r.exit_code == 0 and "found in the stored history: this run's experiments did not exercise it" in r.output and f.calls


def test_no_fix_stops_after_the_finding_agent(results, monkeypatch):
    fake_injector(monkeypatch, [SPIKE_ID])
    f = FakeFixer(results)
    fake_live(monkeypatch, f)
    r = runner.invoke(cli.app, ["pipeline", "--offline", "--yes", "--no-fix"])
    assert r.exit_code == 0 and "AGENT 2/3" in r.output and "AGENT 3/3" not in r.output and "fixing:    skipped" in r.output and f.calls == []


def test_keep_fix_is_passed_to_the_fixing_agent(results, monkeypatch):
    fake_injector(monkeypatch, [SPIKE_ID])
    f = FakeFixer(results)
    fake_live(monkeypatch, f)
    assert runner.invoke(cli.app, ["pipeline", "--offline", "--yes", "--keep-fix"]).exit_code == 0 and f.calls[0][2] is True


def test_a_stopped_injecting_agent_means_no_fix_and_a_failing_exit_code(results, monkeypatch):
    fake_injector(monkeypatch, [SPIKE_ID], status="stopped", stop_reason="kill switch engaged")
    f = FakeFixer(results)
    fake_live(monkeypatch, f)
    r = runner.invoke(cli.app, ["pipeline", "--offline", "--yes"])
    assert r.exit_code == 1 and f.calls == [] and "stopped: kill switch engaged" in r.output and "AGENT 3/3" not in r.output


def test_no_experiments_means_nothing_to_analyze(results, monkeypatch):
    fake_injector(monkeypatch, [])
    r = runner.invoke(cli.app, ["pipeline", "--offline", "--yes"])
    assert r.exit_code == 1 and "nothing to analyze" in r.output


def test_a_fixing_agent_that_cannot_run_does_not_undo_the_findings(results, monkeypatch):
    fake_injector(monkeypatch, [SPIKE_ID])
    fake_live(monkeypatch, FakeFixer(results, raises=fixer.FixError("docker command was not found")))
    r = runner.invoke(cli.app, ["pipeline", "--offline", "--yes"])
    assert r.exit_code == 0 and "the fixing agent could not run: docker command was not found" in r.output
    assert "fixing:    skipped" in r.output and (results / "findings.html").exists()


def test_a_fix_that_stopped_or_failed_to_revert_fails_the_pipeline(results, monkeypatch):
    fake_injector(monkeypatch, [SPIKE_ID])
    fake_live(monkeypatch, FakeFixer(results, improved_run(status="error", reverted=False, revert_errors=["boom"])))
    r = runner.invoke(cli.app, ["pipeline", "--offline", "--yes"])
    assert r.exit_code == 1 and "NOT reverted: boom" in r.output
