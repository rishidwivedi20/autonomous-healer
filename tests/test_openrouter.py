"""The OpenRouter adapter, offline: scripted HTTP responses stand in for the API. Also the config rules for model ids."""
import json
import os
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from chaosai import analyzer, cli, config, openrouter, planner
from chaosai.config import AnalyzerCfg, PlannerCfg
from chaosai.envfile import load_env
from chaosai.evidence import Evidence
from chaosai.llmclient import make_client
from chaosai.mistral import MistralClient
from chaosai.openrouter import OpenRouterClient, OpenRouterError
from tests.fixtures_results import make_results
from tests.test_gemini import IDS, graph_of, payload, read_calls
from tests.test_mistral import chat, tc

KEY = "sk-or-v1-test-KEY-should-never-leak-0123456789"
M = "openai/gpt-oss-120b:free"
runner = CliRunner()


def make(*script, cls=OpenRouterClient, **kw):
    reqs, slept, queue = [], [], list(script)

    def handler(request):
        reqs.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)
    c = cls(key=KEY, transport=httpx.MockTransport(handler), sleep=slept.append, **kw)
    return c, reqs, slept


def body(req):
    return json.loads(req.content)


def ask(c, **kw):
    return c.beta.messages.create(model=kw.pop("model", M), max_tokens=100, messages=kw.pop("messages", [{"role": "user", "content": "x"}]), **kw)


# ---------------------------------------------------------------- request
def test_request_goes_to_openrouter_with_a_bearer_key_and_app_headers():
    c, reqs, _ = make(chat("ok"))
    ask(c)
    r = reqs[0]
    assert str(r.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert r.headers["authorization"] == f"Bearer {KEY}" and KEY not in str(r.url)
    assert r.headers["x-title"] == "chaos-ai" and r.headers["http-referer"].startswith("https://github.com/")


def test_requests_only_go_to_upstream_providers_that_support_tools_and_schemas():
    c, reqs, _ = make(chat("ok"))
    ask(c, tools=[{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}])
    b = body(reqs[0])
    assert b["provider"] == {"require_parameters": True} and b["model"] == M and b["tool_choice"] == "auto"
    assert b["tools"][0]["function"]["name"] == "t"


def test_the_mistral_client_does_not_send_openrouter_routing_fields():
    c, reqs, _ = make(chat("ok"), cls=MistralClient)
    ask(c, model="mistral-small-latest")
    assert "provider" not in body(reqs[0]) and "x-title" not in reqs[0].headers


def test_missing_key_names_the_service_and_the_variable(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OpenRouterError, match="no OpenRouter credentials found .*OPENROUTER_API_KEY"):
        OpenRouterClient()


# ---------------------------------------------------------------- reasoning models
def reasoning_message():
    msg = chat("", calls=[tc("get_graph", {}, "idAAAAAA1")])
    msg["choices"][0]["message"]["reasoning"] = "I should read the graph first."
    msg["choices"][0]["message"]["reasoning_details"] = [{"type": "reasoning.text", "text": "step", "signature": "sig-1"}]
    return msg


def test_reasoning_fields_are_sent_back_unchanged_on_the_next_turn():
    c, reqs, _ = make(reasoning_message(), chat("done"))
    first = ask(c)
    ask(c, messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": first.content},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "idAAAAAA1", "content": "{}"}]}])
    replayed = body(reqs[1])["messages"][1]
    assert replayed["reasoning"] == "I should read the graph first."
    assert replayed["reasoning_details"] == [{"type": "reasoning.text", "text": "step", "signature": "sig-1"}]
    assert replayed["tool_calls"][0]["id"] == "idAAAAAA1"


def test_other_providers_do_not_replay_reasoning_fields():
    c, reqs, _ = make(reasoning_message(), chat("done"), cls=MistralClient)
    first = ask(c, model="mistral-small-latest")
    ask(c, model="mistral-small-latest", messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": first.content},
                                                  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "idAAAAAA1", "content": "{}"}]}])
    assert "reasoning" not in body(reqs[1])["messages"][1]


# ---------------------------------------------------------------- errors and limits
def test_nested_error_bodies_are_readable_and_not_retried_when_final():
    c, reqs, slept = make(httpx.Response(402, json={"error": {"code": 402, "message": "Insufficient credits. Add more at openrouter.ai"}}))
    with pytest.raises(OpenRouterError, match="HTTP 402: Insufficient credits") as e:
        ask(c)
    assert len(reqs) == 1 and slept == [] and e.value.status == 402 and KEY not in str(e.value)


def test_an_error_returned_with_http_200_is_raised_with_its_code_and_message():
    c, _, _ = make({"error": {"code": 429, "message": "upstream provider is rate limited"}})
    with pytest.raises(OpenRouterError, match="OpenRouter error 429: upstream provider is rate limited") as e:
        ask(c)
    assert e.value.status == 429


def test_a_daily_free_limit_fails_at_once_and_says_when_it_resets(monkeypatch):
    monkeypatch.setattr(openrouter, "time", SimpleNamespace(time=lambda: 1_000_000.0))
    reset_ms = str(int((1_000_000.0 + 3 * 3600) * 1000))
    daily = httpx.Response(429, headers={"x-ratelimit-reset": reset_ms},
                           json={"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day"}})
    c, reqs, slept = make(daily)
    with pytest.raises(OpenRouterError) as e:
        ask(c)
    assert slept == [] and len(reqs) == 1 and e.value.status == 429
    assert "daily free-model request limit reached" in str(e.value) and "3.0 h" in str(e.value) and e.value.retry_after_s == pytest.approx(3 * 3600)


def test_a_daily_limit_without_a_reset_header_still_fails_at_once():
    c, reqs, slept = make(httpx.Response(429, json={"error": {"message": "free-models-per-day exceeded"}}))
    with pytest.raises(OpenRouterError, match="daily free-model request limit reached") as e:
        ask(c)
    assert slept == [] and len(reqs) == 1 and e.value.retry_after_s == 6 * 3600


def test_a_per_minute_limit_is_retried_with_retry_after():
    c, reqs, slept = make(httpx.Response(429, headers={"retry-after": "4"}, json={"error": {"message": "Rate limit exceeded: free-models-per-min"}}), chat("ok"))
    assert ask(c).content[0].text == "ok" and slept == [4.0] and len(reqs) == 2


def test_list_models_keeps_only_models_that_accept_tools():
    c, reqs, _ = make({"data": [{"id": "openai/gpt-oss-120b:free", "supported_parameters": ["tools", "max_tokens"]},
                                {"id": "openai/gpt-oss-120b", "supported_parameters": ["tools"]},
                                {"id": "vendor/no-tools", "supported_parameters": ["max_tokens"]},
                                {"id": "vendor/unknown"}]})
    assert c.list_models() == ["openai/gpt-oss-120b", "openai/gpt-oss-120b:free"]
    assert reqs[0].url.path == "/api/v1/models"


def test_make_client_picks_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    assert isinstance(make_client("openrouter"), OpenRouterClient)


# ---------------------------------------------------------------- the real loops through the adapter
OCFG = AnalyzerCfg(provider="openrouter", llm_model=M)


def read_call_msg():
    return chat(calls=[tc(n, a, f"call{i:05d}") for i, (n, a) in enumerate(read_calls())])


def test_the_analyzer_loop_runs_through_openrouter_and_passes_validation(tmp_path):
    ev = Evidence(make_results(tmp_path))
    c, reqs, _ = make(read_call_msg(), chat(calls=[tc("submit_analysis", payload(ev), "submit001")]))
    a = analyzer.analyze(ev, IDS, OCFG, client=c)
    assert a.generated_by == "llm" and a.model == M and a.weaknesses and a.fixes and any("0 rejected" in n for n in a.notes)
    assert all(body(r)["provider"] == {"require_parameters": True} for r in reqs)


def test_a_fabricated_number_is_rejected_through_openrouter_and_the_model_can_fix_it(tmp_path):
    ev = Evidence(make_results(tmp_path))
    good = payload(ev)
    bad = json.loads(json.dumps(good))
    bad["weaknesses"][0]["evidence"][0]["claim"] += " and exactly 98765 votes were lost"
    c, reqs, _ = make(read_call_msg(), chat(calls=[tc("submit_analysis", bad, "submit001")]), chat(calls=[tc("submit_analysis", good, "submit002")]))
    a = analyzer.analyze(ev, IDS, OCFG, client=c)
    assert a.generated_by == "llm" and any("1 rejected" in n for n in a.notes)
    assert "REJECTED" in body(reqs[2])["messages"][-1]["content"]


def test_a_daily_limit_falls_back_to_the_rules_and_says_why(tmp_path):
    ev = Evidence(make_results(tmp_path))
    a = analyzer.analyze(ev, IDS, OCFG, client=make(httpx.Response(429, json={"error": {"message": "free-models-per-day"}}))[0])
    assert a.generated_by == "rules" and "daily free-model request limit reached" in a.fallback_reason and a.weaknesses


def test_the_planner_runs_through_openrouter(tmp_path):
    g = graph_of(tmp_path)
    cands = planner.candidates(g, [], None)
    pick = {"experiments": [{"candidate_id": cands[0].id, "duration_s": 20, "hypothesis": "h1", "reason": "r1"}]}
    c, reqs, _ = make(chat(json.dumps(pick)))
    plan = planner.llm_plan(g, [], n=3, client=c, cfg=PlannerCfg(provider="openrouter", llm_model=M))
    assert plan.planner == "llm" and plan.model == M and body(reqs[0])["response_format"]["type"] == "json_schema"


# ---------------------------------------------------------------- config and .env
def test_openrouter_provider_accepts_vendor_slash_model_ids():
    assert config.AnalyzerCfg(provider="openrouter", llm_model="openai/gpt-oss-120b").llm_model == "openai/gpt-oss-120b"
    assert config.PlannerCfg(provider="openrouter", llm_model="mistralai/mistral-small-3.2-24b-instruct:free").provider == "openrouter"


def test_openrouter_rejects_bare_model_names_and_other_vendors_names():
    with pytest.raises(ValidationError, match="OpenRouter model ids look like vendor/model"):
        config.AnalyzerCfg(provider="openrouter", llm_model="my-model")
    with pytest.raises(ValidationError, match="which is a anthropic model"):
        config.AnalyzerCfg(provider="openrouter", llm_model="claude-opus-5-5")


@pytest.mark.parametrize("provider", ["gemini", "mistral", "anthropic"])
def test_a_slash_id_belongs_to_openrouter_not_to_the_direct_providers(provider):
    with pytest.raises(ValidationError, match=f"provider is {provider} but llm_model is 'openai/gpt-oss-120b', which is a openrouter model"):
        config.AnalyzerCfg(provider=provider, llm_model="openai/gpt-oss-120b")


def test_gemini_still_accepts_the_models_prefix_form():
    assert config.AnalyzerCfg(provider="gemini", llm_model="models/gemini-3.5-flash").provider == "gemini"


def test_the_openrouter_key_is_loaded_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text(f"OPENROUTER_API_KEY={KEY}\n", encoding="utf-8")
    try:
        assert load_env(p) == ["OPENROUTER_API_KEY"] and os.environ["OPENROUTER_API_KEY"] == KEY
    finally:
        os.environ.pop("OPENROUTER_API_KEY", None)


# ---------------------------------------------------------------- llm-check
def test_llm_check_without_an_openrouter_key_says_where_to_get_one():
    r = runner.invoke(cli.app, ["llm-check", "--provider", "openrouter"])
    assert r.exit_code == 1 and "openrouter.ai/keys" in r.output and "OPENROUTER_API_KEY" in r.output


def test_llm_check_no_send_masks_the_key_and_makes_no_request(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr(openrouter, "OpenRouterClient", lambda *a, **k: pytest.fail("no request expected with --no-send"))
    r = runner.invoke(cli.app, ["llm-check", "--provider", "openrouter", "--no-send"])
    assert r.exit_code == 0 and "key found" in r.output and KEY not in r.output


def fake_client(models=None, error=None):
    class Fake:
        def __init__(self):
            self.messages = self

        def list_models(self):
            return models or []

        def create(self, **kw):
            if error:
                raise error
            assert kw["model"] == "openai/gpt-oss-120b:free"
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn",
                                   usage=SimpleNamespace(input_tokens=8, output_tokens=2))
    return Fake


def test_llm_check_lists_filtered_models_and_sends_one_request(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr(openrouter, "OpenRouterClient", fake_client(["openai/gpt-oss-120b", "openai/gpt-oss-120b:free", "vendor/other"]))
    r = runner.invoke(cli.app, ["llm-check", "--provider", "openrouter", "--models", "--match", "GPT-OSS"])
    assert r.exit_code == 0 and "2 model(s) that support tools matching 'GPT-OSS'" in r.output and "vendor/other" not in r.output
    assert "OK: openai/gpt-oss-120b:free answered 'ok'" in r.output and KEY not in r.output


@pytest.mark.parametrize("status,hint", [(401, "key was rejected"), (402, "add some on openrouter.ai"), (404, "try --models --match"), (429, "per-minute and daily limits")])
def test_llm_check_turns_api_errors_into_actionable_hints(monkeypatch, status, hint):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr(openrouter, "OpenRouterClient", fake_client(error=OpenRouterError("boom", status)))
    r = runner.invoke(cli.app, ["llm-check", "--provider", "openrouter"])
    assert r.exit_code == 1 and "OpenRouter error: boom" in r.output and hint in r.output and KEY not in r.output
