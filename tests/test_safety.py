import pytest

from chaosai.models import Experiment, Stats, Thresholds
from chaosai.safety import MAX_DURATION_S, Guard, KillSwitch, SafetyViolation, is_managed


@pytest.fixture
def guard(tmp_path):
    return Guard({"redis", "worker"}, KillSwitch(tmp_path / "KILL"))


def exp(**kw):
    base = dict(id="e1", target="redis", fault="kill", duration_s=30)
    return Experiment(**{**base, **kw})


def bad(rate=0.9, p95=100.0):
    return Stats(n=5, error_rate=rate, p50_ms=10, p95_ms=p95)


def good():
    return Stats(n=5, error_rate=0.0, p50_ms=10, p95_ms=50)


# ---- allowlist
def test_label_allowlist():
    assert is_managed({"chaosai.managed": "true"})
    assert not is_managed({"chaosai.managed": "false"})
    assert not is_managed({})
    assert not is_managed(None)


def test_unlabeled_target_rejected(guard):
    with pytest.raises(SafetyViolation, match="allowlist"):
        guard.validate(exp(target="some-other-container"))


def test_managed_target_accepted(guard):
    assert guard.validate(exp()).target == "redis"


# ---- duration cap
def test_duration_capped(guard):
    assert guard.validate(exp(duration_s=600)).duration_s == MAX_DURATION_S == 60


def test_short_duration_untouched(guard):
    assert guard.validate(exp(duration_s=10)).duration_s == 10


def test_zero_duration_rejected(guard):
    with pytest.raises(SafetyViolation):
        guard.validate(exp(duration_s=0))


# ---- thresholds can only tighten
def test_loose_thresholds_clamped(guard):
    out = guard.validate(exp(thresholds=Thresholds(error_rate=1.0, p95_ms=99999, breach_seconds=60)))
    assert out.thresholds.error_rate == 0.5
    assert out.thresholds.p95_ms == 3000
    assert out.thresholds.breach_seconds == 3


def test_tight_thresholds_kept(guard):
    out = guard.validate(exp(thresholds=Thresholds(error_rate=0.2, p95_ms=500, breach_seconds=1)))
    assert (out.thresholds.error_rate, out.thresholds.p95_ms, out.thresholds.breach_seconds) == (0.2, 500, 1)


# ---- abort logic: 3 consecutive seconds
def test_abort_after_three_consecutive_breaches(guard):
    t = Thresholds()
    assert not guard.should_abort(bad(), t)
    assert not guard.should_abort(bad(), t)
    assert guard.should_abort(bad(), t)
    assert "error rate" in guard.abort_reason


def test_breach_streak_resets_on_good_second(guard):
    t = Thresholds()
    guard.should_abort(bad(), t)
    guard.should_abort(bad(), t)
    assert not guard.should_abort(good(), t)
    assert not guard.should_abort(bad(), t)
    assert not guard.should_abort(bad(), t)
    assert guard.should_abort(bad(), t)


def test_latency_breach_aborts(guard):
    t = Thresholds()
    slow = Stats(n=5, error_rate=0.0, p50_ms=100, p95_ms=9000)
    assert not guard.should_abort(slow, t)
    assert not guard.should_abort(slow, t)
    assert guard.should_abort(slow, t)
    assert "p95" in guard.abort_reason


def test_empty_seconds_do_not_count_as_breach(guard):
    t = Thresholds()
    empty = Stats(n=0)
    for _ in range(10):
        assert not guard.should_abort(empty, t)


def test_exact_threshold_is_not_a_breach(guard):
    t = Thresholds(error_rate=0.5, p95_ms=3000)
    edge = Stats(n=5, error_rate=0.5, p50_ms=1, p95_ms=3000)
    for _ in range(5):
        assert not guard.should_abort(edge, t)


# ---- kill switch
def test_kill_switch_aborts_immediately(guard):
    guard.kill.engage("test")
    assert guard.should_abort(good(), Thresholds())
    assert guard.abort_reason == "kill switch"


def test_kill_file_from_another_process_is_seen(tmp_path):
    kill_path = tmp_path / "KILL"
    g = Guard({"redis"}, KillSwitch(kill_path))
    kill_path.write_text("abort from cli")   # simulates `chaos abort`
    assert g.should_abort(good(), Thresholds())


def test_kill_switch_blocks_preflight(guard):
    guard.kill.engage()
    with pytest.raises(SafetyViolation, match="kill switch"):
        guard.preflight(good(), [])


def test_reset_clears_kill_switch(guard):
    guard.kill.engage()
    guard.kill.reset()
    assert not guard.kill.engaged


# ---- one at a time
def test_only_one_experiment_at_a_time(guard):
    guard.begin()
    with pytest.raises(SafetyViolation, match="already running"):
        guard.begin()
    guard.end()
    guard.begin()


# ---- preflight
def test_preflight_ok(guard):
    guard.preflight(good(), [])


def test_preflight_rejects_no_traffic(guard):
    with pytest.raises(SafetyViolation, match="no baseline"):
        guard.preflight(Stats(), [])


def test_preflight_rejects_unhealthy_baseline(guard):
    with pytest.raises(SafetyViolation, match="steady state"):
        guard.preflight(bad(rate=0.3), [])


def test_preflight_rejects_unhealthy_containers(guard):
    with pytest.raises(SafetyViolation, match="not healthy"):
        guard.preflight(good(), ["db"])


def test_preflight_refusal_can_carry_a_hint_about_what_the_probe_saw(guard):
    bad = Stats(n=50, error_rate=1.0, p50_ms=None, p95_ms=None)
    with pytest.raises(SafetyViolation, match=r"steady state \(error rate 100%\); probe errors: 9x PoolTimeout"):
        guard.preflight(bad, [], hint="; probe errors: 9x PoolTimeout")
