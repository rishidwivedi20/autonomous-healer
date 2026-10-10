"""Findings report: built from stored results only, rendered to JSON / Markdown / HTML (unit-test data only)."""
import json

import pytest

from chaosai import analyzer as analyzer_mod, config as config_mod, report
from chaosai.analysis import Analysis, Evid, ExpAnalysis, Fix, Weakness
from chaosai.evidence import Evidence
from chaosai.models import Graph
from tests.fixtures_results import KILL_ID, SPIKE_ID, make_results


@pytest.fixture
def env(tmp_path):
    d = make_results(tmp_path)
    ev = Evidence(d)
    graph = Graph.model_validate_json((d / "graph.json").read_text(encoding="utf-8"))
    analysis = analyzer_mod.analyze(ev, ev.ids(), config_mod.load().analyzer, use_llm=False, cache_dir=d / "llm_cache")
    return d, ev, graph, analysis


def hostile_analysis(ids):
    evid = [Evid(experiment_id=ids[0], claim="<script>alert(1)</script> | pipe")]
    return Analysis(summary="s <b>", generated_by="rules",
                    experiments=[ExpAnalysis(experiment_id=i, root_cause="r|c", evidence=evid) for i in ids],
                    weaknesses=[Weakness(title="<img src=x onerror=alert(1)>", severity="high", services=["a|b"], description="d", evidence=evid)],
                    fixes=[Fix(title="f", change="</pre><script>x</script>", rationale="why", addresses=["<img src=x onerror=alert(1)>"], evidence=evid)])


def test_build_has_the_score_experiments_weaknesses_and_fixes(env):
    d, ev, graph, analysis = env
    rep = report.build(ev, analysis, graph, now=0)
    assert rep["score"]["score"] is not None and rep["score"]["n_experiments"] == 2
    assert {e["id"] for e in rep["experiments"]} == {SPIKE_ID, KILL_ID}
    assert rep["weaknesses"] and rep["weaknesses"][0]["n"] == 1 and rep["fixes"]
    assert rep["analysis"]["by"] == "rules" and rep["limitations"]
    kill = next(e for e in rep["experiments"] if e["id"] == KILL_ID)
    assert kill["restarts"] == {"db": 1, "result": 1, "worker": 1} and kill["recovery_s"] == 6.0
    assert kill["worst_journey"] == "result" and kill["worst_error_rate"] == pytest.approx(0.04)
    assert {c["name"] for c in kill["components"]} == {"availability", "latency", "recovery", "blast_radius"}


def test_the_score_in_the_report_is_the_scoring_modules_score(env):
    from chaosai.scoring import overall, score_experiment
    d, ev, graph, analysis = env
    expect = overall([score_experiment(ev.results()[i], graph, ev.restarts_of(i)) for i in ev.ids()])
    assert report.build(ev, analysis, graph)["score"]["score"] == expect.score


def test_scope_limits_the_report_to_those_experiments(env):
    d, ev, graph, analysis = env
    rep = report.build(ev, analysis, graph, scope=[KILL_ID])
    assert [e["id"] for e in rep["experiments"]] == [KILL_ID] and rep["score"]["n_experiments"] == 1


def test_verification_reruns_the_worst_scoring_cited_experiments_with_a_real_command(env):
    d, ev, graph, analysis = env
    rep = report.build(ev, analysis, graph)
    scores = {e["id"]: e["score"] for e in rep["experiments"]}
    for f in rep["fixes"]:
        v = f["verification"]
        assert len(v) <= report.MAX_VERIFY
        assert [x["score_before"] for x in v] == sorted(x["score_before"] for x in v)
        assert all(x["command"].startswith("chaos run ") and scores[x["experiment_id"]] == x["score_before"] for x in v)
    cmds = [x["command"] for f in rep["fixes"] for x in f["verification"]]
    assert "chaos run vote traffic-spike -p multiplier=5" in cmds or "chaos run db kill" in cmds


def test_verification_lists_one_entry_per_distinct_experiment():
    rows = {i: {"id": i, "target": "db", "fault": "kill", "params": {}, "score": s, "started_at": t} for i, s, t in (("a", 90.0, 2.0), ("b", 70.0, 1.0))}
    v = report._verification(["a", "b", "a"], rows)
    assert [x["experiment_id"] for x in v] == ["a"]           # a started later than b, so a is the current state of that experiment


def test_markdown_has_every_section_and_the_numbers(env):
    d, ev, graph, analysis = env
    md = report.to_markdown(report.build(ev, analysis, graph, now=0))
    for h in ("# Resilience findings report", "## Resilience score", "## Summary", "## Ranked weaknesses", "## Recommended fixes",
              "## Experiments", "## Per-experiment detail", "## Limitations", "Verify by re-running:"):
        assert h in md, h
    assert f"`{KILL_ID}`" in md and "| availability | 40 |" in md and "deterministic rules" in md


def test_html_is_a_complete_page_with_the_same_content(env):
    d, ev, graph, analysis = env
    page = report.to_html(report.build(ev, analysis, graph, now=0))
    assert page.startswith("<!doctype html>") and "<title>Resilience findings</title>" in page and page.endswith("</main>")
    assert KILL_ID in page and "Ranked weaknesses" in page and "prefers-color-scheme:dark" in page


def test_untrusted_text_is_escaped_in_html_and_markdown_tables(env):
    d, ev, graph, _ = env
    rep = report.build(ev, hostile_analysis(ev.ids()), graph)
    page = report.to_html(rep)
    assert "<script>" not in page and "<img src=x" not in page and "</pre><script>" not in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page and "&lt;/pre&gt;&lt;script&gt;" in page
    md = report.to_markdown(rep)
    assert "| a|b" not in md                                                    # pipes would break a table row


def test_missing_score_is_reported_as_missing_not_zero(env):
    d, ev, graph, analysis = env
    rep = report.build(ev, analysis, graph)
    rep["score"].update(score=None, band=None, by_component={})
    assert "No score" in report.to_markdown(rep) and "No score" in report.to_html(rep)


def test_write_creates_three_consistent_files(env, tmp_path):
    d, ev, graph, analysis = env
    out = tmp_path / "out"
    out.mkdir()
    rep = report.build(ev, analysis, graph)
    paths = report.write(out, rep)
    assert set(paths) == {"json", "md", "html"} and all(p.exists() and p.stat().st_size > 500 for p in paths.values())
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["score"]["score"] == rep["score"]["score"]


def test_the_report_adds_no_numbers_the_stored_results_do_not_contain(env):
    """Every experiment row figure comes from the stored result; spot-check the ones a reader would quote."""
    d, ev, graph, analysis = env
    rep = report.build(ev, analysis, graph)
    spike = next(e for e in rep["experiments"] if e["id"] == SPIKE_ID)
    r = ev.results()[SPIKE_ID]
    assert spike["p95_ms"]["vote"] == [r.baseline_by_journey["vote"].p95_ms, r.during_by_journey["vote"].p95_ms]
    assert spike["duration_s"] == r.experiment.duration_s and spike["recovered"] is r.recovered
