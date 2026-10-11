"""The fixing agent's default job: rank recommended changes from the findings and change nothing."""
import json

from chaosai.analysis import Analysis, Evid, ExpAnalysis, Fix, Weakness
from chaosai.recommend import recommend, to_markdown, write


def ev(claim="queue 300", eid="e1"):
    return [Evid(experiment_id=eid, claim=claim)]


def weakness(title, sev, services):
    return Weakness(title=title, severity=sev, services=services, description=title, evidence=ev())


def fix(title, addresses, change="x: y", rationale="because"):
    return Fix(title=title, change=change, rationale=rationale, addresses=addresses, evidence=ev())


def analysis(weaknesses, fixes):
    return Analysis(summary="s", experiments=[ExpAnalysis(experiment_id="e1", root_cause="c")], weaknesses=weaknesses, fixes=fixes,
                    generated_by="llm")


def test_changes_are_ranked_by_the_most_severe_weakness_they_address(tmp_path):
    a = analysis([weakness("Crit", "critical", ["db"]), weakness("Low one", "low", ["vote"]), weakness("Med", "medium", ["redis"])],
                 [fix("Low fix", ["Low one"]), fix("Both", ["Med", "Crit"]), fix("Med fix", ["Med"])])
    rec = recommend(a, tmp_path)
    assert [r.title for r in rec.items] == ["Both", "Med fix", "Low fix"] and [r.rank for r in rec.items] == [1, 2, 3]
    assert rec.items[0].severity == "critical" and rec.items[0].services == ["redis", "db"]
    assert rec.changes_made is False and rec.analysis_by == "llm"


def test_a_vetted_fix_is_offered_for_measuring_only_when_the_fix_itself_names_it(tmp_path):
    a = analysis([weakness("Backlog on the worker queue", "high", ["worker", "redis"])],
                 [fix("Run more worker replicas", ["Backlog on the worker queue"], "worker:\n  deploy:\n    replicas: 3"),
                  fix("Use append-only persistence", ["Backlog on the worker queue"], "redis-server --appendonly yes", "AOF bounds loss of the queue")])
    rec = recommend(a, tmp_path)
    by = {r.title: r for r in rec.items}
    assert by["Run more worker replicas"].can_be_validated_by == "scale-worker"
    assert by["Use append-only persistence"].can_be_validated_by is None             # same weakness, different fix: no false link


def test_a_stored_measurement_is_attached_but_a_skipped_or_stopped_run_is_not(tmp_path):
    fixes = tmp_path / "fixes"
    fixes.mkdir()
    (fixes / "20260101-000001-scale-worker.json").write_text(json.dumps(
        {"id": "20260101-000001-scale-worker", "recipe_id": "scale-worker", "status": "improved", "claim": "queue peak 112 votes -> 12 votes", "detail": ""}))
    (fixes / "20260101-000002-scale-worker.json").write_text(json.dumps(
        {"id": "20260101-000002-scale-worker", "recipe_id": "scale-worker", "status": "stopped", "claim": "", "detail": "kill switch"}))
    a = analysis([weakness("Single worker", "high", ["worker"])], [fix("Run more worker replicas", ["Single worker"])])
    item = recommend(a, tmp_path).items[0]
    assert item.measured_status == "improved" and "112 votes -> 12 votes" in item.measured       # the newer 'stopped' run measured nothing
    fresh = recommend(a, tmp_path / "empty").items[0]
    assert fresh.measured is None and fresh.can_be_validated_by == "scale-worker"


def test_how_to_prove_it_names_the_experiments_that_exposed_the_weakness(tmp_path):
    a = analysis([weakness("W", "high", ["db"])], [fix("F", ["W"])])
    assert "re-run e1" in recommend(a, tmp_path).items[0].prove_with


def test_the_files_are_written_and_say_nothing_was_changed(tmp_path):
    a = analysis([weakness("W", "high", ["db"])], [fix("F", ["W"], "services:\n  db:\n    restart: always")])
    rec = recommend(a, tmp_path)
    paths = write(rec, tmp_path)
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["changes_made"] is False
    md = paths["md"].read_text(encoding="utf-8")
    assert "## 1. [HIGH] F" in md and "restart: always" in md and "nothing was changed" in md and md == to_markdown(rec)


def test_no_fixes_means_no_recommendations(tmp_path):
    rec = recommend(analysis([weakness("W", "low", ["db"])], []), tmp_path)
    assert rec.items == [] and "No changes recommended" in to_markdown(rec)
