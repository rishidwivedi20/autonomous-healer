"""The finding agent, and the fix-validation sections of the report (unit-test data only)."""
import json

import pytest

from chaosai import report
from chaosai.analysis import Analysis
from chaosai.config import AnalyzerCfg
from chaosai.finder import FindingAgent, FindingError
from chaosai.fixer import FixRun, Measured
from chaosai.models import Graph
from tests.fixtures_results import KILL_ID, SPIKE_ID, make_results


@pytest.fixture
def env(tmp_path):
    d = make_results(tmp_path)
    graph = Graph.model_validate_json((d / "graph.json").read_text(encoding="utf-8"))
    return d, graph


def agent(env, **kw):
    d, graph = env
    return FindingAgent(AnalyzerCfg(), d, graph, use_llm=False, **kw)


# ---------------------------------------------------------------- the agent
def test_the_agent_analyses_scores_and_writes_everything(env):
    d, _ = env
    found = agent(env).run()
    assert sorted(found.ids) == sorted([SPIKE_ID, KILL_ID]) and found.analysis.generated_by == "rules" and found.analysis.weaknesses
    assert found.report["score"]["score"] is not None and set(found.paths) == {"json", "md", "html"}
    assert all(p.exists() for p in found.paths.values()) and (d / "analysis.json").exists()
    assert Analysis.model_validate_json((d / "analysis.json").read_text(encoding="utf-8")).summary == found.analysis.summary


def test_the_scope_limits_what_is_analysed(env):
    found = agent(env).run([KILL_ID, "no-such-experiment"])
    assert found.ids == [KILL_ID] and [e["id"] for e in found.report["experiments"]] == [KILL_ID]


def test_nothing_to_analyse_is_a_clear_error_and_writes_nothing(tmp_path):
    (tmp_path / "results").mkdir()
    with pytest.raises(FindingError, match="no stored experiments"):
        FindingAgent(AnalyzerCfg(), tmp_path / "results", None, use_llm=False).run()
    assert list((tmp_path / "results").iterdir()) == []


def test_an_analysis_that_is_supplied_is_used_not_recomputed(env):
    first = agent(env).run()
    custom = first.analysis.model_copy(update={"summary": "a summary only the caller could have written"})
    again = agent(env).run(analysis=custom)
    assert again.report["analysis"]["summary"] == "a summary only the caller could have written"


def test_the_history_analysis_is_always_the_deterministic_one(env):
    a = FindingAgent(AnalyzerCfg(provider="gemini", llm_model="gemini-3.5-flash"), env[0], env[1], use_llm=True).rules_over_history()
    assert a.generated_by == "rules" and "LLM disabled" in a.fallback_reason          # no key, no network, no model is ever called for it


# ---------------------------------------------------------------- fix validations in the report
def fix_run(tmp_path, id_="20261011-120000-scale-worker", status="improved", **kw):
    before = Measured(experiment_id="b", score=60.0, metrics={"peak_queue": 309.0, "recovery_pipeline_s": 80.0})
    after = Measured(experiment_id="a", score=96.0, metrics={"peak_queue": 40.0, "recovery_pipeline_s": 12.0})
    from chaosai.fixer import compare_rows
    run = FixRun(id=id_, started_at=1.0, ended_at=2.0, status=status, detail="the score rose +36.0 points and the redis queue peak improved 87%",
                 recipe_id="scale-worker", title="Run three worker replicas", service="worker", change="docker compose up -d --scale worker=3 worker",
                 why="share the load", addresses="Single worker caps throughput", before=before, after=after,
                 comparison=compare_rows(before, after), claim="resilience score 60.0 -> 96.0; redis queue peak 309 votes -> 40 votes", reverted=True, **kw)
    d = tmp_path / "fixes"
    d.mkdir(exist_ok=True)
    (d / f"{id_}.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return run


def test_validated_fixes_appear_in_the_markdown_html_and_json_reports(env, tmp_path):
    d, _ = env
    fix_run(d)
    found = agent(env).run()
    assert found.report["fix_validations"][0]["status"] == "improved"
    md = found.paths["md"].read_text(encoding="utf-8")
    assert "## Fix validation: before and after" in md and "Run three worker replicas: measured improvement" in md
    assert "| redis queue peak (votes) |" in md and "309" in md and "resilience score 60.0 -> 96.0" in md and "Sandbox afterwards: reverted to the original" in md
    html = found.paths["html"].read_text(encoding="utf-8")
    assert "Fix validation: before and after" in html and "measured improvement" in html and "redis queue peak" in html
    assert json.loads(found.paths["json"].read_text(encoding="utf-8"))["fix_validations"][0]["recipe_id"] == "scale-worker"


def test_without_validated_fixes_the_section_is_absent(env):
    found = agent(env).run()
    assert found.report["fix_validations"] == []
    assert "Fix validation" not in found.paths["md"].read_text(encoding="utf-8") and "Fix validation" not in found.paths["html"].read_text(encoding="utf-8")


def test_each_status_is_worded_honestly(env):
    d, _ = env
    fix_run(d, "20261011-120001-scale-worker", "inconclusive")
    fix_run(d, "20261011-120002-scale-worker", "worse")
    md = agent(env).run().paths["md"].read_text(encoding="utf-8")
    assert "no measurable improvement (not claimed)" in md and "made things worse" in md


def test_a_kept_fix_is_reported_as_kept(env):
    d, _ = env
    fix_run(d, kept=True).model_copy(update={"reverted": False})
    p = d / "fixes" / "20261011-120000-scale-worker.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data.update(kept=True, reverted=False)
    p.write_text(json.dumps(data), encoding="utf-8")
    assert "Sandbox afterwards: kept as changed" in agent(env).run().paths["md"].read_text(encoding="utf-8")


def test_hostile_text_in_a_fix_record_is_escaped(env):
    d, _ = env
    run = fix_run(d)
    p = d / "fixes" / f"{run.id}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data.update(title="<img src=x onerror=alert(1)>", detail="<script>x</script>", claim="</td><script>y</script>")
    p.write_text(json.dumps(data), encoding="utf-8")
    html = agent(env).run().paths["html"].read_text(encoding="utf-8")
    assert "<img src=x" not in html and "<script>x</script>" not in html and "<script>y</script>" not in html and "&lt;img src=x" in html


def test_load_fixes_is_newest_first_limited_and_skips_unreadable_files(tmp_path):
    for i in range(7):
        fix_run(tmp_path, f"2026101{i}-100000-scale-worker")
    (tmp_path / "fixes" / "29999999-corrupt.json").write_text("{not json", encoding="utf-8")
    got = report.load_fixes(tmp_path, limit=5)
    assert len(got) == 5 and got[0]["id"] == "20261016-100000-scale-worker" and all("status" in g for g in got)
    assert report.load_fixes(tmp_path / "nowhere") == []


def test_the_dashboards_cached_report_refreshes_when_a_fix_result_appears(env):
    d, graph = env
    live = report.Live(d, None, lambda: graph.model_dump())
    first = live.get()
    assert first["fix_validations"] == [] and live.get() is first and live.builds == 1
    fix_run(d)
    second = live.get()
    assert live.builds == 2 and second["fix_validations"][0]["status"] == "improved"
