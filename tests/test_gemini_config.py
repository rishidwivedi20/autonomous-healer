"""Provider settings, the .env prefix, and `chaos llm-check` for Gemini."""
import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from chaosai import cli, config, gemini
from chaosai.envfile import load_env

runner = CliRunner()
KEY = "AIzaSyFAKE-llm-check-key-0123456789"


# ---- config
def test_a_claude_model_with_the_gemini_provider_is_rejected_at_load_time():
    with pytest.raises(ValidationError, match="provider is gemini but llm_model is 'claude-opus-5-5'"):
        config.AnalyzerCfg(provider="gemini")
    with pytest.raises(ValidationError, match="provider is gemini"):
        config.PlannerCfg(provider="gemini", llm_model="claude-opus-5-5")


def test_a_gemini_model_with_the_anthropic_provider_is_rejected_at_load_time():
    with pytest.raises(ValidationError, match="provider is anthropic but llm_model is 'gemini-3.5-flash'"):
        config.AnalyzerCfg(provider="anthropic", llm_model="gemini-3.5-flash")


def test_matching_provider_and_model_load_and_the_default_stays_anthropic():
    assert config.AnalyzerCfg().provider == "anthropic" and config.PlannerCfg().provider == "anthropic"
    assert config.AnalyzerCfg(provider="gemini", llm_model="gemini-3.5-flash").provider == "gemini"


def test_an_unknown_provider_is_an_error():
    with pytest.raises(ValidationError):
        config.AnalyzerCfg(provider="openai", llm_model="gpt")


def test_the_repo_config_yaml_loads_and_every_model_matches_its_provider():
    """Loading validates this already (a mismatch is a load-time error); this pins that the shipped file stays loadable."""
    c = config.load(Path(__file__).parent.parent / "config.yaml")
    for section in (c.planner, c.analyzer):
        if section.provider == "pool":
            assert section.members and all(m.provider in ("anthropic", "gemini", "mistral", "openrouter") for m in section.members)
        else:
            assert (section.provider == "gemini") == section.llm_model.startswith("gemini")


# ---- .env
def test_the_gemini_key_is_loaded_from_env_but_other_vendors_names_are_not(tmp_path, monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    p = tmp_path / ".env"
    p.write_text(f"GEMINI_API_KEY={KEY}\nGOOGLE_API_KEY=nope\nOPENAI_API_KEY=nope\n", encoding="utf-8")
    try:
        assert load_env(p) == ["GEMINI_API_KEY"]
        assert os.environ["GEMINI_API_KEY"] == KEY and "GOOGLE_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ
    finally:
        os.environ.pop("GEMINI_API_KEY", None)


# ---- llm-check
@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_llm_check_without_a_gemini_key_says_where_to_get_one(no_key):
    r = runner.invoke(cli.app, ["llm-check", "--provider", "gemini"])
    assert r.exit_code == 1 and "aistudio.google.com/apikey" in r.output and "GEMINI_API_KEY" in r.output


def test_llm_check_no_send_masks_the_key_and_makes_no_request(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(gemini, "GeminiClient", lambda *a, **k: pytest.fail("no request expected with --no-send"))
    r = runner.invoke(cli.app, ["llm-check", "--provider", "gemini", "--no-send"])
    assert r.exit_code == 0 and "key found" in r.output and KEY not in r.output


def test_llm_check_lists_models_and_sends_one_request(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("GEMINI_API_KEY", KEY)

    class Fake:
        def __init__(self):
            self.messages = self

        def list_models(self):
            return ["gemini-a", "gemini-b"]

        def create(self, **kw):
            assert kw["model"] == "gemini-b"
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn",
                                   usage=SimpleNamespace(input_tokens=8, output_tokens=2))
    monkeypatch.setattr(gemini, "GeminiClient", Fake)
    r = runner.invoke(cli.app, ["llm-check", "--provider", "gemini", "--models", "--model", "gemini-b"])
    assert r.exit_code == 0 and "2 model(s)" in r.output and "gemini-a" in r.output and "OK: gemini-b answered 'ok'" in r.output
    assert KEY not in r.output


@pytest.mark.parametrize("status,hint", [(404, "try --models"), (400, "invalid or the model name is wrong"), (429, "rate limited per model")])
def test_llm_check_turns_api_errors_into_actionable_hints(monkeypatch, status, hint):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)

    class Fake:
        def __init__(self):
            self.messages = self

        def create(self, **kw):
            raise gemini.GeminiError("boom", status)
    monkeypatch.setattr(gemini, "GeminiClient", Fake)
    r = runner.invoke(cli.app, ["llm-check", "--provider", "gemini"])
    assert r.exit_code == 1 and "Gemini error: boom" in r.output and hint in r.output and KEY not in r.output


def test_llm_check_rejects_an_unknown_provider():
    assert runner.invoke(cli.app, ["llm-check", "--provider", "openai"]).exit_code == 2
