"""Rule W8 (network trouble that makes users wait) on stored-style results. Unit-test data only."""
import pytest

from chaosai.analysis import Recorder, validate
from chaosai.evidence import Evidence
from chaosai.models import Experiment, ExperimentResult, SecondStat, Stats
from chaosai.observer import Store
from chaosai.rules import analyse
from tests.fixtures_results import KILL_ID, SPIKE_ID, make_results

NET_ID, T_NET = "e-net", 1_002_000.0
TITLE = "Network trouble on a dependency makes users wait"


def S(n, err, p95):
    return Stats(n=n, error_rate=err, p50_ms=p95 / 3, p95_ms=p95)


def add_net(d, target="redis", fault="network-delay", lat=1800.0, ok=True):
    """A network experiment whose vote journey takes `lat` ms (and optionally fails) while the fault is active."""
    store = Store(d / "chaosai.db")
    rows = []
    for s in range(-30, 111):
        for k in range(5):
            t = T_NET + s + k * 0.2
            active = 0 <= s < 20
            rows.append((t, "vote", int(ok or not active), 200 if (ok or not active) else 500, lat if active else 20.0, None))
            rows.append((t, "result", 1, 200, 7.0, None))
    store.executemany("INSERT INTO probes VALUES (?,?,?,?,?,?)", rows)
    store.close()
    err = 0.0 if ok else 1.0
    r = ExperimentResult(
        experiment=Experiment(id=NET_ID, target=target, fault=fault, params={"latency_ms": 500}, duration_s=20),
        started_at=T_NET, ended_at=T_NET + 100, baseline=S(40, 0.0, 25.0), during=S(200, err / 2, lat),
        baseline_by_journey={"vote": S(20, 0.0, 25.0), "result": S(20, 0.0, 8.0)},
        during_by_journey={"vote": S(100, err, lat), "result": S(100, 0.0, 8.0)},
        per_second=[SecondStat(t=float(i), n=10, error_rate=err / 2, p50_ms=lat / 3, p95_ms=lat) for i in range(20)],
        recovered=True, recovery_s=2.0, recovery_probe_s=1.0, pipeline_assessed=False, notes=["network delay"])
    (d / f"{NET_ID}.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")


IDS = [SPIKE_ID, KILL_ID, NET_ID]
HYP = {SPIKE_ID: True, KILL_ID: False, NET_ID: False}


def run(tmp_path, **kw):
    d = make_results(tmp_path)
    add_net(d, **kw)
    ev = Evidence(d)
    rec = Recorder(ev)
    return analyse(rec, IDS), rec


def w8(a):
    return next((w for w in a.weaknesses if w.title.startswith(TITLE)), None)


def test_a_dependency_delay_that_makes_users_wait_is_reported_with_the_stored_numbers(tmp_path):
    a, rec = run(tmp_path)
    w = w8(a)
    assert w and w.severity == "high" and "redis" in w.services and "vote" in w.services
    claim = w.evidence[0].claim
    assert w.evidence[0].experiment_id == NET_ID
    assert "1800" in claim and "25 ms" in claim and "network-delay on redis" in claim
    fix = next(f for f in a.fixes if f.addresses == [w.title])
    assert "socket_timeout" in fix.change and fix.evidence == w.evidence


def test_the_new_rule_passes_the_same_validator_the_llm_must_pass(tmp_path):
    a, rec = run(tmp_path)
    assert w8(a) and validate(a, rec.seen, IDS, HYP) == []


def test_a_mild_slowdown_is_not_reported(tmp_path):
    a, _ = run(tmp_path, lat=60.0)                         # (60+50)/(25+50) = 1.5x
    assert w8(a) is None


def test_a_moderate_slowdown_is_medium_severity(tmp_path):
    a, _ = run(tmp_path, lat=400.0)                        # (400+50)/(25+50) = 6x: reported, but under the 20x bar
    assert w8(a) and w8(a).severity == "medium"


def test_a_journey_that_went_down_is_left_to_the_outage_rule(tmp_path):
    a, rec = run(tmp_path, target="vote", ok=False)
    assert w8(a) is None
    assert any(w.title.startswith("A degraded network on vote") for w in a.weaknesses)
    assert validate(a, rec.seen, IDS, HYP) == []


def test_other_faults_never_trigger_it(tmp_path):
    a, _ = run(tmp_path, fault="cpu-limit")
    assert w8(a) is None
