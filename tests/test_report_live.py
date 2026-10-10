"""report.Live: the cached report the dashboard serves."""
import json

from chaosai import report
from tests.fixtures_results import KILL_ID, make_results


def test_live_builds_once_and_rebuilds_only_when_an_experiment_result_changes(tmp_path):
    d = make_results(tmp_path)
    live = report.Live(d, None, None)
    first = live.get()
    assert first["score"]["n_runs"] == 2 and live.builds == 1
    assert live.get() is first and live.builds == 1                                   # nothing changed: served from the cache
    (d / "graph.json").write_text((d / "graph.json").read_text(encoding="utf-8") + " ", encoding="utf-8")
    (d / "analysis.json").write_text("{}", encoding="utf-8")
    assert live.get() is first and live.builds == 1                                   # graph/analysis files are not experiments
    p = d / f"{KILL_ID}.json"
    p.write_text(p.read_text(encoding="utf-8").replace('"recovery_s": 6.0', '"recovery_s": 9.0'), encoding="utf-8")
    second = live.get()
    assert live.builds == 2 and second is not first
    assert next(e for e in second["experiments"] if e["id"] == KILL_ID)["recovery_s"] == 9.0


def test_live_picks_up_a_newly_finished_experiment(tmp_path):
    d = make_results(tmp_path)
    live = report.Live(d, None, None)
    assert len(live.get()["experiments"]) == 2
    src = (d / f"{KILL_ID}.json").read_text(encoding="utf-8")
    (d / "e-new.json").write_text(src.replace(KILL_ID, "e-new"), encoding="utf-8")
    assert len(live.get()["experiments"]) == 3 and live.builds == 2


def test_live_with_no_experiments_is_empty_and_does_not_fail(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    assert report.Live(d, None, None).get() == {"empty": True}


def test_live_uses_the_graph_for_blast_radius_and_never_calls_a_model(tmp_path):
    d = make_results(tmp_path)
    graph = json.loads((d / "graph.json").read_text(encoding="utf-8"))
    rep = report.Live(d, None, lambda: graph).get()
    kill = next(e for e in rep["experiments"] if e["id"] == KILL_ID)
    blast = next(c for c in kill["components"] if c["name"] == "blast_radius")
    assert "other service(s) restarted" in blast["basis"] and rep["analysis"]["by"] == "rules"
    assert rep["dependency_graph"]["critical_path"] == graph["critical_path"]


def test_live_survives_a_result_file_that_is_half_written(tmp_path):
    d = make_results(tmp_path)
    live = report.Live(d, None, None)
    live.get()
    (d / "e-partial.json").write_text('{"experiment": {"id": "e-par', encoding="utf-8")
    assert len(live.get()["experiments"]) == 2                                         # skipped, as Evidence skips unparsable files
