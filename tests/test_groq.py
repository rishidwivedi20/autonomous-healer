"""The Groq adapter, offline: scripted HTTP responses stand in for the API. The shared translation is covered in
test_mistral.py and test_openrouter.py; this checks what is specific to Groq (endpoint, key, config, CLI)."""
import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from chaosai import cli, config
from chaosai.config import AnalyzerCfg, PoolMember
from chaosai.envfile import load_env
from chaosai.groq import GroqClient, GroqError
from chaosai.llmclient import make_client
from chaosai.pool import make_member_client
from tests.test_mistral import chat

KEY = "gsk_test-KEY-should-never-leak-0123456789"
runner = CliRunner()


def make(*script):
    reqs, queue = [], list(script)

    def handler(request):
        reqs.append(request)
        item = queue.pop(0)
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)
    return GroqClient(key=KEY, transport=httpx.MockTransport(handler), sleep=lambda s: None), reqs


def test_it_posts_to_groq_with_the_bearer_key_and_returns_the_answer():
    c, reqs = make(chat("ok"))
    resp = c.messages.create(model="openai/gpt-oss-120b", max_tokens=64, messages=[{"role": "user", "content": "hi"}])
    assert str(reqs[0].url) == "https://api.groq.com/openai/v1/chat/completions"
    assert reqs[0].headers["authorization"] == f"Bearer {KEY}"
    assert resp.content[0].text == "ok" and resp.stop_reason == "end_turn"


def test_a_long_rate_limit_wait_fails_at_once_naming_groq_and_not_the_key():
    c, reqs = make(httpx.Response(429, headers={"retry-after": "3000"}, json={"error": {"message": "daily tokens used"}}))
    with pytest.raises(GroqError) as e:
        c.messages.create(model="openai/gpt-oss-120b", max_tokens=8, messages=[{"role": "user", "content": "hi"}])
    assert len(reqs) == 1 and e.value.status == 429 and e.value.retry_after_s == 3000 and KEY not in str(e.value)


def test_a_missing_key_is_reported_by_name(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GroqError, match="GROQ_API_KEY"):
        GroqClient()


def test_the_factories_and_the_pool_build_a_groq_client(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    assert isinstance(make_client("groq"), GroqClient)
    assert isinstance(make_member_client(PoolMember(provider="groq", model="openai/gpt-oss-120b")), GroqClient)


def test_slash_model_ids_are_valid_for_groq_but_a_gemini_name_is_not():
    AnalyzerCfg(provider="groq", llm_model="openai/gpt-oss-120b")
    AnalyzerCfg(provider="groq", llm_model="llama-3.3-70b-versatile")
    with pytest.raises(ValidationError, match="gemini model"):
        AnalyzerCfg(provider="groq", llm_model="gemini-3-flash-preview")


def test_a_groq_key_in_dotenv_is_loaded(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text(f"GROQ_API_KEY={KEY}\n", encoding="utf-8")
    load_env(f)
    import os
    assert os.environ["GROQ_API_KEY"] == KEY
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_llm_check_without_a_key_says_where_to_get_one(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = runner.invoke(cli.app, ["llm-check", "--provider", "groq"])
    assert r.exit_code == 1 and "console.groq.com/keys" in r.output
