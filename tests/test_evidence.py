import pytest

from chaosai.analysis import Analysis, Evid, ExpAnalysis, Fix, Recorder, Seen, Weakness, claim_numbers, matches, validate
from chaosai.evidence import Evidence, numbers_in
from tests.fixtures_results import KILL_ID, SPIKE_ID, make_results


@pytest.fixture
def ev(tmp_path):
    return Evidence(make_results(tmp_path))


# ---------------------------------------------------------------- the tools
def test_experiments_listed_and_described_in_consistent_units(ev):
    assert ev.ids() == [KILL_ID, SPIKE_ID] or set(ev.ids()) == {KILL_ID, SPIKE_ID}
    e = ev.get_experiment(SPIKE_ID)
    assert e["target"] == "vote" and e["params"] == {"multiplier": 5} and e["hypothesis"] == "vote p95 rises but no errors"
    assert e["during_by_journey"]["vote"] == {"n": 250, "error_pct": 0.0, "p95_ms": 54.0}
    assert e["fault_active_s"] == 20 and e["recovered"] is False
    k = ev.get_experiment(KILL_ID)
    assert k["hypothesis"] is None and k["restarts"] == {"db": 1, "result": 1, "worker": 1}
    assert k["during_by_journey"]["result"]["error_pct"] == 4.0                     # a rate stored as 0.04 is reported as a percentage


def test_metrics_windows_and_summary(ev):
    m = ev.get_metrics(SPIKE_ID, "during", "vote")
    assert m["window"] == "during" and set(m["summary"]) == {"vote"}
    assert [r["t_s"] for r in m["per_second"]][:3] == [0, 1, 2] and len(m["per_second"]) == 20
    assert m["summary"]["vote"]["max_error_pct"] == 0.0 and m["summary"]["vote"]["median_p95_ms"] == 15.0
    around = ev.get_metrics(SPIKE_ID)
    assert around["per_second"][0]["t_s"] == -10 and set(around["summary"]) == {"vote", "result"}


def test_gauge_summary_matches_what_was_stored(ev):
    s = ev.get_gauges(SPIKE_ID)["summary"]
    assert s["baseline_queue_max"] == 0.0 and s["peak_queue"] == 300.0 and s["peak_queue_at_s"] == 20.0
    assert s["votes_per_s_before"] == 5.0 and s["votes_per_s_while_draining"] == pytest.approx(9.5, abs=0.2)
    assert 85 <= s["queue_back_to_baseline_at_s"] <= 90
    assert s["peak_backlog_lag_s"] == pytest.approx(300 / s["votes_per_s_while_draining"], abs=0.2)
    assert s["db_unreadable_readings"] == 0


def test_quiet_pipeline_has_no_backlog_numbers(ev):
    s = ev.get_gauges(KILL_ID)["summary"]
    assert s["peak_queue"] == 0.0 and s["votes_per_s_while_draining"] is None and s["peak_backlog_lag_s"] is None


def test_one_experiments_window_never_runs_into_the_next(ev):
    """Regression for a real bug: a 90 s window swallowed the next experiment's crash loop."""
    spike_rows = ev.get_gauges(SPIKE_ID)["rows"]
    assert max(r["t_s"] for r in spike_rows) <= 100
    assert ev._bounds(ev._exps[SPIKE_ID])[1] < ev._exps[KILL_ID].started_at - 20 + 1
    assert ev.get_container_events(SPIKE_ID)["events"] == [{"t_s": 3.0, "service": "vote", "kind": "health", "value": "healthy"}]


def test_container_events_exclude_the_watchers_startup_snapshot(ev):
    got = ev.get_container_events(KILL_ID, "db")["events"]
    assert got == [{"t_s": 1.0, "service": "db", "kind": "restart_count", "value": "3"},
                   {"t_s": 1.0, "service": "db", "kind": "health", "value": "starting"}]


def test_graph_and_baseline(ev):
    g = ev.get_graph()
    assert g["services"]["worker"] == {"replicas": 1, "has_healthcheck": False, "restart_policy": "unless-stopped",
                                       "has_resource_limits": False, "stateful": False, "depends_on": ["redis", "db"],
                                       "fan_in": 0, "fan_out": 2, "on_critical_path": True}
    assert ev.get_baseline()["by_journey"]["vote"] == {"n": 120, "error_pct": 0.0, "p50_ms": 15.0, "p95_ms": 19.9}


def test_bad_requests_return_errors_not_exceptions(ev):
    assert "unknown experiment_id" in ev.call("get_experiment", {"experiment_id": "nope"})["error"]
    assert "window must be" in ev.call("get_metrics", {"experiment_id": SPIKE_ID, "window": "forever"})["error"]
    assert "journey must be" in ev.call("get_metrics", {"experiment_id": SPIKE_ID, "journey": "cart"})["error"]
    assert "unknown tool" in ev.call("delete_everything", {})["error"]
    assert "bad arguments" in ev.call("get_experiment", {})["error"]
    assert "bad arguments" in ev.call("get_experiment", {"experiment_id": "x", "extra": 1})["error"]


def test_missing_files_are_reported_not_invented(tmp_path):
    d = make_results(tmp_path)
    (d / "graph.json").unlink()
    (d / "baseline.json").unlink()
    e = Evidence(d)
    assert "chaos discover" in e.get_graph()["error"] and "chaos baseline" in e.get_baseline()["error"]


def test_fingerprint_changes_when_any_input_changes(tmp_path):
    d = make_results(tmp_path)
    a = Evidence(d).fingerprint([SPIKE_ID])
    assert Evidence(d).fingerprint([SPIKE_ID]) == a
    (d / "baseline.json").write_text("{}")
    assert Evidence(d).fingerprint([SPIKE_ID]) != a
    b = Evidence(d).fingerprint([SPIKE_ID])
    p = d / f"{SPIKE_ID}.json"
    p.write_text(p.read_text().replace('"recovered": false', '"recovered": true'))
    assert Evidence(d).fingerprint([SPIKE_ID]) != b


def test_dry_runs_and_non_experiment_files_are_ignored(tmp_path):
    d = make_results(tmp_path)
    (d / "plan.json").write_text("[]")
    (d / "junk.json").write_text("{not json")
    assert set(Evidence(d).ids()) == {KILL_ID, SPIKE_ID}


# ---------------------------------------------------------------- number extraction
def test_numbers_in_ignores_experiment_ids_and_pids():
    nums = numbers_in({"id": "20261010-133949-redis-kill-6347", "notes": ["crashed redis (SIGKILL pid 9516)", "cpu limit 0.1"],
                       "p95": 54.0, "flag": True}, ignore_ids={"20261010-133949-redis-kill-6347"})
    assert nums == {54.0, 0.1}


def test_numbers_are_found_inside_strings_like_16m():
    assert numbers_in({"params": {"mem": "16m"}}) == {16.0}


def test_claim_numbers_strip_ids_and_pids():
    assert claim_numbers("e-spike queue 309 votes at pid 12 and 1.2 s", ["e-spike"]) == [309.0, 1.2]


def test_matches_tolerates_rounding_but_not_invention():
    pool = {93.4, 9.5, 309.0}
    assert matches(93.0, pool) and matches(9.5, pool) and matches(309.0, pool) and matches(94.0, pool)
    assert not matches(120.0, pool) and not matches(10.5, pool) and not matches(40.0, pool)


# ---------------------------------------------------------------- the validator
def good_analysis(**over):
    base = dict(
        summary="s",
        experiments=[ExpAnalysis(experiment_id=SPIKE_ID, hypothesis_held=True, root_cause="burst",
                                 evidence=[Evid(experiment_id=SPIKE_ID, claim="queue peaked at 300 votes")]),
                     ExpAnalysis(experiment_id=KILL_ID, hypothesis_held=None, root_cause="db crash",
                                 evidence=[Evid(experiment_id=KILL_ID, claim="restarts db x1, result x1, worker x1")])],
        weaknesses=[Weakness(title="W1", severity="high", services=["worker"], description="d",
                             evidence=[Evid(experiment_id=SPIKE_ID, claim="the worker drained at 9.5 votes/s")]),
                    Weakness(title="W2", severity="low", services=["db"], description="d",
                             evidence=[Evid(experiment_id=KILL_ID, claim="db restarted 1 time")])],
        fixes=[Fix(title="F1", change="worker:\n  deploy:\n    replicas: 3", rationale="r", addresses=["W1"],
                   evidence=[Evid(experiment_id=SPIKE_ID, claim="queue peaked at 300 votes")])])
    base.update(over)
    return Analysis(**base)


@pytest.fixture
def seen(ev):
    rec = Recorder(ev)
    for i in (SPIKE_ID, KILL_ID):
        rec.call("get_experiment", {"experiment_id": i})
        rec.call("get_gauges", {"experiment_id": i})
    rec.call("get_graph", {})
    return rec.seen


HYP = {SPIKE_ID: True, KILL_ID: False}
IDS = [SPIKE_ID, KILL_ID]


def test_a_well_formed_analysis_is_accepted(seen):
    assert validate(good_analysis(), seen, IDS, HYP) == []


def test_a_number_no_tool_returned_is_rejected_with_the_number_named(seen):
    a = good_analysis()
    a.weaknesses[0].evidence[0].claim = "the worker drained at 77 votes/s"
    problems = validate(a, seen, IDS, HYP)
    assert len(problems) == 1 and "[77.0]" in problems[0] and "no tool returned" in problems[0]


def test_numbers_are_checked_per_experiment(seen):
    a = good_analysis()
    a.weaknesses[1].evidence[0].claim = "db peaked at 300 votes"          # 300 exists, but only for the spike experiment
    assert any("[300.0]" in p for p in validate(a, seen, IDS, HYP))


def test_global_tool_numbers_may_be_cited_for_any_experiment(seen):
    a = good_analysis()
    a.weaknesses[0].evidence[0].claim = "worker runs 1 replica(s) and fan-out is 2"      # from get_graph
    assert validate(a, seen, IDS, HYP) == []


def test_unknown_experiment_and_never_queried_experiment(ev):
    rec = Recorder(ev)
    rec.call("get_experiment", {"experiment_id": SPIKE_ID})
    a = good_analysis()
    a.weaknesses[0].evidence[0].experiment_id = "made-up"
    problems = validate(a, rec.seen, IDS, HYP)
    assert any("unknown experiment_id 'made-up'" in p for p in problems)
    assert any(f"never called a data tool for it" in p and KILL_ID in p for p in problems)


def test_coverage_ranking_hypothesis_and_references(seen):
    a = good_analysis(experiments=good_analysis().experiments[:1])
    assert any(f"missing an analysis for {KILL_ID}" in p for p in validate(a, seen, IDS, HYP))

    a = good_analysis(weaknesses=list(reversed(good_analysis().weaknesses)))
    assert any("ranked most severe first" in p for p in validate(a, seen, IDS, HYP))

    a = good_analysis()
    a.experiments[1].hypothesis_held = True                               # the kill experiment has no stored hypothesis
    assert any("hypothesis_held must be null" in p for p in validate(a, seen, IDS, HYP))

    a = good_analysis()
    a.fixes[0].addresses = ["nonexistent weakness"]
    assert any("not one of the weakness titles" in p for p in validate(a, seen, IDS, HYP))


def test_every_weakness_and_fix_needs_evidence_and_a_concrete_change(seen):
    a = good_analysis()
    a.weaknesses[0].evidence = []
    a.fixes[0].evidence = []
    a.fixes[0].change = "  "
    problems = " | ".join(validate(a, seen, IDS, HYP))
    assert "weakness 'W1': cite at least one" in problems and "fix 'F1': cite at least one" in problems and "`change` must be" in problems


def test_trivial_numbers_zero_and_one_are_allowed(seen):
    a = good_analysis()
    a.weaknesses[1].evidence[0].claim = "0 errors and 1 restart"
    assert validate(a, seen, IDS, HYP) == []


def test_percentile_labels_are_names_not_cited_values(seen):
    a = good_analysis()
    a.weaknesses[0].evidence[0].claim = "vote p95 reached 54 ms and p50 stayed low"      # 54 is a returned value; p95 / p50 are labels
    assert claim_numbers(a.weaknesses[0].evidence[0].claim, IDS) == [54.0]
    assert validate(a, seen, IDS, HYP) == [] or all("[95.0]" not in p and "[50.0]" not in p for p in validate(a, seen, IDS, HYP))
    assert 95.0 not in numbers_in({"hypothesis": "vote p95 rises"})
