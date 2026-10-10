from pathlib import Path

import pytest
from typer.testing import CliRunner

from chaosai import cli, config
from chaosai.models import Thresholds
from chaosai.safety import MAX_DURATION_S, MAX_THRESHOLDS

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
runner = CliRunner()


def write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


def test_defaults_equal_the_hard_coded_values_the_code_used_before():
    c = config.load(Path("does-not-exist.yaml"))
    assert (c.probe.interval_s, c.probe.timeout_s) == (1.0, 3.0)
    assert (c.safety.max_duration_s, c.safety.abort_error_rate, c.safety.abort_p95_ms, c.safety.breach_seconds) == (
        MAX_DURATION_S, MAX_THRESHOLDS.error_rate, MAX_THRESHOLDS.p95_ms, MAX_THRESHOLDS.breach_seconds)
    assert (c.experiment.baseline_s, c.experiment.recovery_hold_s, c.experiment.recovery_tolerance,
            c.experiment.recovery_latency_slack_ms) == (15, 5, 0.10, 50.0)
    assert c.faults["network-delay"].params == {"latency_ms": 500} and c.faults["pause"].duration_s == 20


def test_repo_config_yaml_matches_the_defaults():
    """config.yaml documents the defaults; if it drifts from them, behaviour silently changes."""
    assert config.load(REPO_CONFIG).model_dump() == config.load(Path("does-not-exist.yaml")).model_dump()


def test_partial_file_overrides_only_what_it_names(tmp_path):
    c = config.load(write(tmp_path, "probe: {timeout_s: 2.0}\nfaults:\n  kill: {duration_s: 10}\n"))
    assert c.probe.timeout_s == 2.0 and c.probe.interval_s == 1.0
    assert c.faults["kill"].duration_s == 10
    assert c.faults["stop"].duration_s == 30 and "latency_ms" in c.faults["network-delay"].params   # untouched faults keep defaults


@pytest.mark.parametrize("text", [
    "safety: {max_duration_s: 61}",
    "safety: {abort_error_rate: 0.9}",
    "safety: {abort_p95_ms: 9000}",
    "safety: {breach_seconds: 10}",
    "faults: {kill: {duration_s: 120}}",
])
def test_safety_limits_cannot_be_loosened(tmp_path, text):
    with pytest.raises(ValueError, match="hard limit|looser"):
        config.load(write(tmp_path, text))


def test_safety_limits_can_be_tightened(tmp_path):
    c = config.load(write(tmp_path, "safety: {max_duration_s: 20, abort_error_rate: 0.2, abort_p95_ms: 800, breach_seconds: 2}"))
    assert (c.safety.max_duration_s, c.safety.abort_error_rate, c.safety.breach_seconds) == (20, 0.2, 2)


@pytest.mark.parametrize("text, why", [
    ("safety: {abort_eror_rate: 0.2}", "abort_eror_rate"),         # typo
    ("bogus: 1", "bogus"),
    ("faults: {explode: {duration_s: 5}}", "unknown fault"),
    ("probe: {timeout_s: 0}", "timeout_s"),
    ("planner: {llm_effort: turbo}", "llm_effort"),
])
def test_typos_and_nonsense_fail_loudly(tmp_path, text, why):
    with pytest.raises(ValueError, match=why):
        config.load(write(tmp_path, text))


def test_env_var_selects_the_config_file(tmp_path, monkeypatch):
    p = write(tmp_path, "experiment: {baseline_s: 99}")
    monkeypatch.setenv("CHAOSAI_CONFIG", str(p))
    assert config.load().experiment.baseline_s == 99


# ---------------------------------------------------------------- the CLI honours the parameters
@pytest.fixture
def cfg_env(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(cli, "COMPOSE", REPO_CONFIG.parent / "deploy" / "docker-compose.yml")
    (tmp_path / "results").mkdir()

    def use(text):
        monkeypatch.setenv("CHAOSAI_CONFIG", str(write(tmp_path, text)))
    return use


def test_config_command_prints_effective_values(cfg_env):
    cfg_env("probe: {timeout_s: 2.5}")
    r = runner.invoke(cli.app, ["config"])
    assert r.exit_code == 0 and "timeout_s: 2.5" in r.output and "vote_url: http://localhost:8080/" in r.output


def test_invalid_config_is_a_clean_error(cfg_env):
    cfg_env("safety: {abort_error_rate: 0.99}")
    r = runner.invoke(cli.app, ["config"])
    assert r.exit_code == 2 and "invalid config" in r.output and "Traceback" not in r.output


def test_run_defaults_come_from_faults_config(cfg_env):
    cfg_env("faults:\n  network-delay: {duration_s: 12, params: {latency_ms: 777}}\n")
    r = runner.invoke(cli.app, ["run", "vote", "network-delay", "--dry-run"])
    assert r.exit_code == 0 and "duration 12s" in r.output and "'latency_ms': 777" in r.output


def test_flags_override_config_params(cfg_env):
    cfg_env("faults:\n  network-delay: {duration_s: 12, params: {latency_ms: 777}}\n")
    r = runner.invoke(cli.app, ["run", "vote", "network-delay", "--dry-run", "--duration", "15", "-p", "latency_ms=900"])
    assert "duration 15s" in r.output and "'latency_ms': 900" in r.output


def test_safety_config_tightens_the_guard(cfg_env):
    cfg_env("safety: {max_duration_s: 20, abort_error_rate: 0.2, abort_p95_ms: 800, breach_seconds: 2}")
    r = runner.invoke(cli.app, ["run", "vote", "pause", "--dry-run", "--duration", "60"])
    assert "duration 20s" in r.output and "error_rate>20%" in r.output and "p95>800ms" in r.output and "for 2s" in r.output


def test_plan_respects_planner_config(cfg_env):
    cfg_env("planner: {top: 2, use_llm: false}")
    r = runner.invoke(cli.app, ["plan", "--dry-run"])
    assert "2. " in r.output and "3. " not in r.output and "planner.use_llm: false" in r.output


def test_plan_uses_fault_defaults_from_config(cfg_env):
    cfg_env("faults:\n  pause: {duration_s: 9}\nplanner: {use_llm: false}\n")
    r = runner.invoke(cli.app, ["plan", "--dry-run", "--top", "6"])
    assert "duration 9s" in r.output
