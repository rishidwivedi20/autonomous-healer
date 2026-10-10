"""Scoring tests on hand-made results (unit-test data only; the pipeline never fabricates measurements)."""
import pytest

from chaosai.models import Experiment, ExperimentResult, Graph, Service, Stats
from chaosai.scoring import WEIGHTS, band, overall, score_experiment


def S(n=100, err=0.0, p95=20.0):
    return Stats(n=n, error_rate=err, p50_ms=p95 / 2, p95_ms=p95)


def graph():
    sv = lambda n, **k: Service(name=n, **k)
    return Graph(services={"vote": sv("vote", entrypoint=True, depends_on=["redis"]), "redis": sv("redis", stateful=True, fan_in=2),
                           "worker": sv("worker", depends_on=["redis", "db"]), "db": sv("db", stateful=True, fan_in=2),
                           "result": sv("result", entrypoint=True, depends_on=["db"])},
                 critical_path=["vote", "redis", "worker", "db", "result"])


def result(id="e1", target="redis", fault="kill", t=1000.0, vote=None, res=None, base_vote=None, base_res=None,
           recovered=True, recovery_s=3.0, restarts=None, pipeline=True):
    vote, res = vote or S(), res or S(p95=8.0)
    base_vote, base_res = base_vote or S(p95=20.0), base_res or S(p95=8.0)
    return ExperimentResult(
        experiment=Experiment(id=id, target=target, fault=fault), started_at=t, ended_at=t + 60,
        baseline=base_vote, during=vote, baseline_by_journey={"vote": base_vote, "result": base_res},
        during_by_journey={"vote": vote, "result": res}, recovered=recovered,
        recovery_s=recovery_s if recovered else None, pipeline_assessed=pipeline, restarts=restarts or {})


def comp(sc, name):
    return next(c for c in sc.components if c.name == name)


def test_weights_sum_to_100():
    assert sum(WEIGHTS.values()) == 100


def test_a_fault_with_no_user_impact_and_fast_recovery_scores_100():
    sc = score_experiment(result(), graph())
    assert sc.score == 100.0 and sc.band == "good" and sc.missing == []


def test_availability_is_the_worst_journeys_error_uplift():
    sc = score_experiment(result(vote=S(err=0.4), res=S(err=0.1)), graph())
    assert comp(sc, "availability").score == pytest.approx(0.6)
    assert "'vote'" in comp(sc, "availability").basis and "40%" in comp(sc, "availability").basis


def test_baseline_errors_are_not_blamed_on_the_fault():
    sc = score_experiment(result(vote=S(err=0.3), base_vote=S(err=0.3, p95=20.0)), graph())
    assert comp(sc, "availability").score == 1.0


def test_latency_is_full_marks_up_to_1_5x_and_zero_at_10x():
    ok = score_experiment(result(vote=S(p95=20.0 * 1.5 + 25.0)), graph())            # (p95+50)/(base+50) = 1.5 exactly
    assert comp(ok, "latency").score == pytest.approx(1.0)
    bad = score_experiment(result(vote=S(p95=20.0 * 10 + 450.0)), graph())            # ratio 10
    assert comp(bad, "latency").score == pytest.approx(0.0)
    mid = score_experiment(result(vote=S(p95=20.0 * 5.75 + 50 * 4.75)), graph())      # ratio 5.75, halfway
    assert comp(mid, "latency").score == pytest.approx(0.5, abs=0.01)


def test_latency_slack_stops_tiny_baselines_from_exploding_the_ratio():
    sc = score_experiment(result(vote=S(p95=40.0), base_vote=S(p95=5.0)), graph())     # 8x raw, but only +35 ms
    assert comp(sc, "latency").score > 0.95                                              # without the slack it would be ~0.24


def test_latency_is_not_scored_when_nothing_succeeded():
    r = result(vote=Stats(n=100, error_rate=1.0, p50_ms=None, p95_ms=None))
    sc = score_experiment(r, graph())
    names = [c.name for c in sc.components if c.score is None]
    assert names == []                                    # 'result' journey still has a p95, so latency is scored from it
    r2 = result(vote=Stats(n=100, error_rate=1.0), res=Stats(n=100, error_rate=1.0))
    sc2 = score_experiment(r2, graph())
    assert comp(sc2, "latency").score is None
    assert any(m.startswith("latency") for m in sc2.missing)
    assert sc2.score is not None                          # the rest is rescaled, not zeroed


def test_missing_components_rescale_the_remaining_weights():
    r = result(vote=Stats(n=100, error_rate=0.5), res=Stats(n=100, error_rate=0.5))   # no p95 anywhere
    sc = score_experiment(r, graph())
    av, rec, bl = comp(sc, "availability").score, comp(sc, "recovery").score, comp(sc, "blast_radius").score
    expected = 100 * (40 * av + 25 * rec + 15 * bl) / 80
    assert sc.score == pytest.approx(expected, abs=0.05)


def test_recovery_curve_and_non_recovery():
    f = lambda s: comp(score_experiment(result(recovery_s=s), graph()), "recovery").score
    assert f(2) == 1.0 and f(5) == 1.0
    assert f(32.5) == pytest.approx(0.5)
    assert f(60) == 0.0 and f(120) == 0.0
    nr = comp(score_experiment(result(recovered=False), graph()), "recovery")
    assert nr.score == 0.0 and "did not recover" in nr.basis


def test_recovery_notes_when_the_pipeline_could_not_be_assessed():
    c = comp(score_experiment(result(pipeline=False), graph()), "recovery")
    assert "users only" in c.basis


def test_blast_radius_counts_collateral_restarts_but_not_the_target():
    clean = comp(score_experiment(result(restarts={"redis": 3}), graph()), "blast_radius")
    assert clean.score == 1.0                                                    # the target restarting is the fault itself
    cascade = comp(score_experiment(result(target="db", restarts={"db": 1, "worker": 1, "result": 1}), graph()), "blast_radius")
    assert cascade.score == pytest.approx(0.75)                                  # (1 - 2/4 restarts + 1 - 0/2 journeys) / 2
    assert "worker" in cascade.basis and "result" in cascade.basis


def test_blast_radius_counts_degraded_journeys():
    c = comp(score_experiment(result(vote=S(err=0.5)), graph()), "blast_radius")
    assert c.score == pytest.approx(0.75)                                        # no restarts (1.0) and 1 of 2 journeys hurt (0.5)


def test_restart_override_wins_over_the_stored_value():
    r = result(target="db", restarts={"worker": 1})
    assert comp(score_experiment(r, graph(), restarts={}), "blast_radius").score == 1.0


def test_without_a_graph_only_journeys_count_for_blast_radius():
    c = comp(score_experiment(result(vote=S(err=0.5))), "blast_radius")
    assert c.score == pytest.approx(0.5)


def test_overall_is_weighted_by_target_criticality():
    g = graph()
    a = score_experiment(result("a", target="db", fault="pause", t=1.0), g)                       # 100, critical (weight 5)
    b = score_experiment(result("b", target="worker", fault="kill", t=2.0, vote=S(err=1.0), res=S(err=1.0)), g)
    o = overall([a, b])
    assert a.weight > b.weight
    assert min(a.score, b.score) < o.score < max(a.score, b.score)
    assert o.score == pytest.approx((a.score * a.weight + b.score * b.weight) / (a.weight + b.weight), abs=0.06)


def test_overall_counts_only_the_latest_run_of_each_distinct_experiment():
    g = graph()
    old = score_experiment(result("old", t=1.0, vote=S(err=1.0), res=S(err=1.0)), g)             # terrible, but superseded
    new = score_experiment(result("new", t=2.0), g)
    other = score_experiment(result("other", target="db", fault="stop", t=3.0), g)
    o = overall([old, new, other])
    assert (o.n_runs, o.n_experiments) == (3, 2)
    assert o.score == pytest.approx((new.score * new.weight + other.score * other.weight) / (new.weight + other.weight), abs=0.06)
    assert len(o.experiments) == 3                                                                # history is still listed


def test_different_params_are_different_experiments():
    g = graph()
    r1, r2 = result("p1", target="vote", fault="traffic-spike", t=1.0), result("p2", target="vote", fault="traffic-spike", t=2.0)
    r1.experiment.params, r2.experiment.params = {"multiplier": 2}, {"multiplier": 5}
    assert overall([score_experiment(r1, g), score_experiment(r2, g)]).n_experiments == 2


def test_overall_with_nothing_to_score_is_none_not_zero():
    o = overall([])
    assert o.score is None and o.band is None and o.n_experiments == 0
    no_data = result(vote=Stats(), res=Stats(), recovered=False)
    no_data.baseline_by_journey = {}
    sc = score_experiment(no_data)
    assert sc.score is not None                       # recovery=0 is a measurement: "did not recover"


def test_bands():
    assert [band(x) for x in (100, 85, 84.9, 60, 59.9, 0)] == ["good", "good", "fair", "fair", "poor", "poor"]
    assert band(None) is None


def test_by_component_is_reported_for_the_overall_score():
    g = graph()
    o = overall([score_experiment(result("a", t=1.0, vote=S(err=0.5)), g)])
    assert o.by_component["availability"] == 50.0 and o.by_component["recovery"] == 100.0


def test_a_run_cut_short_by_the_operators_kill_switch_has_no_recovery_score():
    r = result(recovered=False)
    r.aborted, r.abort_reason = True, "kill switch"
    sc = score_experiment(r, graph())
    assert comp(sc, "recovery").score is None and any(m.startswith("recovery") for m in sc.missing)
    assert sc.score is not None                       # availability, latency and blast radius still count, rescaled


def test_an_auto_abort_that_never_recovered_still_scores_recovery_zero():
    r = result(recovered=False)
    r.aborted, r.abort_reason = True, "error rate 100% > 50% for 3 consecutive s"
    assert comp(score_experiment(r, graph()), "recovery").score == 0.0
