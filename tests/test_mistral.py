"""The Mistral adapter, offline: scripted HTTP responses stand in for the API (unit tests of the translation and the
control flow; no experiment data is synthetic). Also drives the real analyzer and planner loops through it."""
import json
import os

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from chaosai import analyzer, cli, config, mistral, planner
from chaosai.config import AnalyzerCfg, PlannerCfg
from chaosai.envfile import load_env
from chaosai.evidence import Evidence
from chaosai.llmclient import make_client
from chaosai.mistral import MistralClient, MistralError
from tests.fixtures_results import make_results
from tests.test_gemini import IDS, graph_of, payload, read_calls

KEY = "MISTRALtest-KEY-should-never-leak-0123456789"
M = "mistral-test-latest"
runner = CliRunner()


def tc(name, args, id_="tcA000001"):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def chat(text="", calls=None, finish=None, content=None):
    msg = {"role": "assistant", "content": content if content is not None else text}
    if calls:
        msg["tool_calls"] = calls
    return {"model": M, "choices": [{"index": 0, "message": msg, "finish_reason": finish or ("tool_calls" if calls else "stop")}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5}}


def make(*script, **kw):
    reqs, slept, queue = [], [], list(script)

    def handler(request):
        reqs.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)
    c = MistralClient(key=KEY, transport=httpx.MockTransport(handler), sleep=slept.append, **kw)
    return c, reqs, slept


def body(req):
    return json.loads(req.content)


def ask(c, **kw):
    return c.beta.messages.create(model=M, max_tokens=kw.pop("max_tokens", 100), messages=kw.pop("messages", [{"role": "user", "content": "x"}]), **kw)


# ---------------------------------------------------------------- request translation
def test_request_goes_to_chat_completions_with_a_bearer_header_never_the_key_in_the_url():
    c, reqs, _ = make(chat("ok"))
    ask(c)
    r = reqs[0]
    assert str(r.url) == "https://api.mistral.ai/v1/chat/completions"
    assert r.headers["authorization"] == f"Bearer {KEY}" and KEY not in str(r.url)


def test_system_tools_and_limits_are_translated():
    schema = {"type": "object", "additionalProperties": False, "required": ["a"],
              "properties": {"a": {"type": "string"}, "held": {"type": ["boolean", "null"]}}}
    c, reqs, _ = make(chat("ok"))
    ask(c, max_tokens=777, system="be careful", tools=[{"name": "t", "description": "d", "input_schema": schema}],
        betas=["x"], fallbacks="default", output_config={"effort": "high"})
    b = body(reqs[0])
    assert b["model"] == M and b["max_tokens"] == 777 and b["tool_choice"] == "auto"
    assert b["messages"] == [{"role": "system", "content": "be careful"}, {"role": "user", "content": "x"}]
    assert b["tools"] == [{"type": "function", "function": {"name": "t", "description": "d", "parameters": schema}}]
    assert "thinking" not in json.dumps(b) and "effort" not in json.dumps(b)             # Anthropic-only settings are not sent


def test_no_tools_means_no_tool_fields():
    c, reqs, _ = make(chat("ok"))
    ask(c)
    assert "tools" not in body(reqs[0]) and "tool_choice" not in body(reqs[0]) and "response_format" not in body(reqs[0])


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["x"], "properties": {"x": {"type": "integer"}}}
FMT = {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}


def test_structured_output_uses_a_strict_json_schema_response_format():
    c, reqs, _ = make(chat('{"x": 1}'))
    ask(c, output_config=FMT)
    assert body(reqs[0])["response_format"] == {"type": "json_schema", "json_schema": {"name": "answer", "schema": SCHEMA, "strict": True}}


def test_a_model_that_rejects_json_schema_is_retried_in_json_object_mode_with_the_schema_in_the_prompt():
    reject = httpx.Response(400, json={"message": "response_format json_schema is not supported by this model"})
    c, reqs, _ = make(reject, chat('{"x": 1}'))
    r = ask(c, system="plan things", output_config=FMT)
    assert r.content[0].text == '{"x": 1}' and len(reqs) == 2
    second = body(reqs[1])
    assert second["response_format"] == {"type": "json_object"}
    assert second["messages"][0]["role"] == "system" and "plan things" in second["messages"][0]["content"]
    assert '"x"' in second["messages"][0]["content"] and "JSON Schema" in second["messages"][0]["content"]


def test_an_unrelated_400_is_not_retried_in_json_object_mode():
    c, reqs, _ = make(httpx.Response(400, json={"message": "model not found"}))
    with pytest.raises(MistralError, match="model not found"):
        ask(c, output_config=FMT)
    assert len(reqs) == 1


# ---------------------------------------------------------------- response translation
def test_text_response_and_usage_mapping():
    c, _, _ = make(chat("hello"))
    r = ask(c)
    assert [b.type for b in r.content] == ["text"] and r.content[0].text == "hello" and r.stop_reason == "end_turn"
    assert (r.usage.input_tokens, r.usage.output_tokens) == (11, 5)


def test_chunked_content_keeps_the_text_and_drops_thinking():
    c, _, _ = make(chat(content=[{"type": "thinking", "thinking": [{"type": "text", "text": "private"}]}, {"type": "text", "text": "answer"}]))
    assert [b.text for b in ask(c).content] == ["answer"]


def test_tool_calls_become_tool_use_blocks_keeping_the_ids_mistral_issued():
    c, _, _ = make(chat(calls=[tc("get_graph", {}, "idAAAAAA1"), tc("get_experiment", {"experiment_id": "e1"}, "idBBBBBB2")]))
    r = ask(c)
    assert r.stop_reason == "tool_use" and [b.type for b in r.content] == ["tool_use", "tool_use"]
    assert [(b.id, b.name) for b in r.content] == [("idAAAAAA1", "get_graph"), ("idBBBBBB2", "get_experiment")]
    assert r.content[1].input == {"experiment_id": "e1"}


def test_text_before_tool_calls_is_kept_and_arguments_given_as_an_object_are_tolerated():
    call = {"id": "idAAAAAA1", "type": "function", "function": {"name": "t", "arguments": {"a": 1}}}
    c, _, _ = make(chat("let me look", calls=[call]))
    r = ask(c)
    assert [b.type for b in r.content] == ["text", "tool_use"] and r.content[1].input == {"a": 1}


def test_malformed_tool_arguments_are_a_retryable_turn_not_a_crash():
    bad = {"id": "idAAAAAA1", "type": "function", "function": {"name": "t", "arguments": "{not json"}}
    c, _, _ = make(chat(calls=[bad]))
    r = ask(c)
    assert r.stop_reason == "end_turn" and [b.type for b in r.content] == ["text"] and "malformed" in r.content[0].text


def test_finish_reasons_map_to_anthropic_stop_reasons():
    for finish, expect in (("stop", "end_turn"), ("length", "max_tokens"), ("model_length", "max_tokens")):
        c, _, _ = make(chat("x", finish=finish))
        assert ask(c).stop_reason == expect
    c, _, _ = make(chat("x", finish="error"))
    with pytest.raises(MistralError, match="reported an error"):
        ask(c)


def test_no_choices_is_an_error_and_an_empty_answer_still_yields_a_replayable_block():
    c, _, _ = make({"choices": []})
    with pytest.raises(MistralError, match="no choices"):
        ask(c)
    c, _, _ = make(chat(""))
    assert [b.text for b in ask(c).content] == [""]


# ---------------------------------------------------------------- replay: assistant messages and tool results
def test_the_models_own_message_is_replayed_unchanged_and_tool_results_follow_as_tool_messages():
    c, reqs, _ = make(chat("looking", calls=[tc("get_graph", {}, "idAAAAAA1"), tc("get_baseline", {}, "idBBBBBB2")]), chat("done"))
    first = ask(c)
    ids = [b.id for b in first.content if b.type == "tool_use"]
    ask(c, messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": first.content},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": ids[0], "content": json.dumps({"services": 5})},
                                                  {"type": "tool_result", "tool_use_id": ids[1], "content": "REJECTED: fix it", "is_error": True}]}])
    msgs = body(reqs[1])["messages"]
    assert msgs[0] == {"role": "user", "content": "q"}
    assert msgs[1] == {"role": "assistant", "content": "looking", "tool_calls": [tc("get_graph", {}, "idAAAAAA1"), tc("get_baseline", {}, "idBBBBBB2")]}
    assert msgs[2] == {"role": "tool", "tool_call_id": "idAAAAAA1", "name": "get_graph", "content": '{"services": 5}'}
    assert msgs[3] == {"role": "tool", "tool_call_id": "idBBBBBB2", "name": "get_baseline", "content": "REJECTED: fix it"}
    assert len(msgs) == 4                                                               # one assistant message, not one per block


def test_replay_keeps_the_models_exact_argument_text_and_reasoning_chunks_rather_than_rebuilding_them():
    compact = {"id": "idAAAAAA1", "type": "function", "function": {"name": "t", "arguments": '{"a":1,"b":[2,3]}'}}     # json.dumps would add spaces
    thinking = {"type": "thinking", "thinking": [{"type": "text", "text": "private reasoning"}]}
    c, reqs, _ = make(chat(content=[thinking, {"type": "text", "text": "calling"}], calls=[compact]), chat("done"))
    first = ask(c)
    ask(c, messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": first.content},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "idAAAAAA1", "content": "{}"}]}])
    replayed = body(reqs[1])["messages"][1]
    assert replayed["tool_calls"][0]["function"]["arguments"] == '{"a":1,"b":[2,3]}'
    assert replayed["content"] == [thinking, {"type": "text", "text": "calling"}]


def test_an_assistant_message_built_from_blocks_is_translated_when_there_is_nothing_to_replay():
    c, reqs, _ = make(chat("ok"))
    ask(c, messages=[{"role": "user", "content": "q"},
                     {"role": "assistant", "content": [{"type": "text", "text": "hi"}, {"type": "tool_use", "id": "idAAAAAA1", "name": "t", "input": {"a": 1}}]},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "idAAAAAA1", "content": {"ok": True}}]},
                     {"role": "assistant", "content": "plain"}, {"role": "user", "content": "next"}])
    msgs = body(reqs[0])["messages"]
    assert msgs[1] == {"role": "assistant", "content": "hi", "tool_calls": [tc("t", {"a": 1}, "idAAAAAA1")]}
    assert msgs[2]["role"] == "tool" and msgs[2]["content"] == '{"ok": true}'
    assert msgs[3] == {"role": "assistant", "content": "plain"} and msgs[4] == {"role": "user", "content": "next"}


# ---------------------------------------------------------------- credentials, retries, rate limits
def test_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(MistralError, match="MISTRAL_API_KEY"):
        MistralClient()


def test_rate_limit_is_retried_after_the_servers_retry_after():
    limited = httpx.Response(429, headers={"retry-after": "3"}, json={"message": "Rate limit exceeded"})
    c, reqs, slept = make(limited, chat("ok"))
    assert ask(c).content[0].text == "ok" and slept == [3.0] and len(reqs) == 2


def test_server_errors_back_off_exponentially_and_give_up_after_max_retries():
    c, reqs, slept = make(*[httpx.Response(503, json={"message": "overloaded"})] * 3, max_retries=2)
    with pytest.raises(MistralError, match="HTTP 503: overloaded") as e:
        ask(c)
    assert len(reqs) == 3 and slept == [2.0, 4.0] and e.value.status == 503


def test_a_very_long_retry_after_fails_immediately_with_a_readable_message():
    c, reqs, slept = make(httpx.Response(429, headers={"retry-after": "7200"}, json={"message": "monthly limit reached"}))
    with pytest.raises(MistralError) as e:
        ask(c)
    assert slept == [] and len(reqs) == 1 and e.value.status == 429
    assert "120 min" in str(e.value) and "monthly limit reached" in str(e.value)


def test_a_plan_with_a_zero_request_limit_fails_immediately_and_says_how_to_fix_it():
    zero = httpx.Response(429, headers={"x-ratelimit-limit-req-minute": "0", "x-ratelimit-remaining-req-minute": "0"},
                          json={"object": "error", "message": "Rate limit exceeded", "type": "rate_limited", "code": "1300"})
    c, reqs, slept = make(zero)
    with pytest.raises(MistralError) as e:
        ask(c)
    assert slept == [] and len(reqs) == 1 and e.value.status == 429                     # retrying a limit of zero can never work
    assert "0 requests per minute" in str(e.value) and "Experiment" in str(e.value) and KEY not in str(e.value)


def test_a_normal_per_minute_limit_is_still_retried():
    limited = httpx.Response(429, headers={"x-ratelimit-limit-req-minute": "60", "x-ratelimit-remaining-req-minute": "0"}, json={"message": "Rate limit exceeded"})
    c, reqs, slept = make(limited, chat("ok"))
    assert ask(c).content[0].text == "ok" and slept == [2.0] and len(reqs) == 2


def test_client_errors_are_not_retried_and_never_contain_the_key():
    c, reqs, slept = make(httpx.Response(401, json={"message": "Unauthorized", "detail": "bad key"}))
    with pytest.raises(MistralError) as e:
        ask(c)
    assert len(reqs) == 1 and slept == [] and e.value.status == 401 and KEY not in str(e.value)


def test_validation_errors_with_a_detail_list_are_readable():
    c, _, _ = make(httpx.Response(422, json={"detail": [{"loc": ["body", "model"], "msg": "field required"}]}))
    with pytest.raises(MistralError, match="field required"):
        ask(c)


def test_network_failures_are_retried_then_reported_without_a_url_or_key():
    c, reqs, slept = make(*[httpx.ConnectError("boom")] * 3, max_retries=2)
    with pytest.raises(MistralError, match="cannot reach the Mistral API") as e:
        ask(c)
    assert len(reqs) == 3 and len(slept) == 2 and KEY not in str(e.value)


def test_list_models_keeps_chat_models_with_function_calling():
    c, reqs, _ = make({"data": [
        {"id": "mistral-large-latest", "capabilities": {"completion_chat": True, "function_calling": True}},
        {"id": "mistral-large-2512", "capabilities": {"completion_chat": True, "function_calling": True}},
        {"id": "mistral-large-latest", "capabilities": {"completion_chat": True, "function_calling": True}},
        {"id": "mistral-embed", "capabilities": {"completion_chat": False, "function_calling": False}},
        {"id": "pixtral-nofn", "capabilities": {"completion_chat": True, "function_calling": False}}]})
    assert c.list_models() == ["mistral-large-2512", "mistral-large-latest"]
    assert reqs[0].headers["authorization"] == f"Bearer {KEY}"


def test_make_client_picks_the_provider(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", KEY)
    assert isinstance(make_client("mistral"), MistralClient)


# ---------------------------------------------------------------- the real analyzer loop through the adapter
MCFG = AnalyzerCfg(provider="mistral", llm_model=M)


def read_call_msg():
    return chat(calls=[tc(n, a, f"call{i:05d}") for i, (n, a) in enumerate(read_calls())])


def test_the_analyzer_loop_runs_through_the_mistral_adapter_and_passes_validation(tmp_path):
    ev = Evidence(make_results(tmp_path))
    c, reqs, _ = make(read_call_msg(), chat(calls=[tc("submit_analysis", payload(ev), "submit001")]))
    a = analyzer.analyze(ev, IDS, MCFG, client=c)
    assert a.generated_by == "llm" and a.model == M and a.weaknesses and a.fixes and any("0 rejected" in n for n in a.notes)
    second = body(reqs[1])["messages"]
    tool_msgs = [m for m in second if m["role"] == "tool"]
    assert len(tool_msgs) == len(read_calls()) and [m["name"] for m in tool_msgs][:2] == ["get_graph", "get_experiment"]
    assert tool_msgs[0]["tool_call_id"] == "call00000"


def test_a_fabricated_number_is_rejected_through_the_adapter_and_the_model_can_fix_it(tmp_path):
    ev = Evidence(make_results(tmp_path))
    good = payload(ev)
    bad = json.loads(json.dumps(good))
    bad["weaknesses"][0]["evidence"][0]["claim"] += " and exactly 98765 votes were lost"
    c, reqs, _ = make(read_call_msg(), chat(calls=[tc("submit_analysis", bad, "submit001")]), chat(calls=[tc("submit_analysis", good, "submit002")]))
    a = analyzer.analyze(ev, IDS, MCFG, client=c)
    assert a.generated_by == "llm" and any("1 rejected" in n for n in a.notes)
    feedback = body(reqs[2])["messages"][-1]
    assert feedback["role"] == "tool" and feedback["name"] == "submit_analysis" and "REJECTED" in feedback["content"] and "98765" in feedback["content"]


def test_a_rate_limit_falls_back_to_the_rules_and_says_why(tmp_path):
    ev = Evidence(make_results(tmp_path))
    limited = httpx.Response(429, headers={"retry-after": "7200"}, json={"message": "monthly limit"})
    a = analyzer.analyze(ev, IDS, MCFG, client=make(limited)[0])
    assert a.generated_by == "rules" and "rate limit exceeded" in a.fallback_reason and a.weaknesses


def test_the_mistral_analysis_is_cached_and_the_cache_is_separate_per_provider(tmp_path):
    ev = Evidence(make_results(tmp_path))
    c, reqs, _ = make(read_call_msg(), chat(calls=[tc("submit_analysis", payload(ev), "submit001")]))
    analyzer.analyze(ev, IDS, MCFG, client=c, cache_dir=tmp_path / "cache")
    n = len(reqs)
    again = analyzer.analyze(ev, IDS, MCFG, client=make()[0], cache_dir=tmp_path / "cache")
    assert again.generated_by == "llm" and len(reqs) == n and any("served from cache" in x for x in again.notes)
    other = AnalyzerCfg(provider="gemini", llm_model="gemini-3.5-flash")
    fp = ev.fingerprint(IDS)
    assert analyzer._cache_path(tmp_path, MCFG, IDS, fp) != analyzer._cache_path(tmp_path, other, IDS, fp)


# ---------------------------------------------------------------- the planner through the adapter
def test_the_planner_runs_through_the_mistral_adapter(tmp_path):
    g = graph_of(tmp_path)
    cands = planner.candidates(g, [], None)
    pick = {"experiments": [{"candidate_id": cands[0].id, "duration_s": 20, "hypothesis": "h1", "reason": "r1"},
                            {"candidate_id": cands[1].id, "duration_s": 25, "hypothesis": "h2", "reason": "r2"}]}
    c, reqs, _ = make(chat(json.dumps(pick)))
    plan = planner.llm_plan(g, [], n=3, client=c, cfg=PlannerCfg(provider="mistral", llm_model=M))
    assert plan.planner == "llm" and plan.model == M and [e.target for e in plan.experiments] == [cands[0].target, cands[1].target]
    assert body(reqs[0])["response_format"]["type"] == "json_schema"


def test_the_planner_still_rejects_an_invented_candidate_and_falls_back(tmp_path):
    g = graph_of(tmp_path)
    c, _, _ = make(chat(json.dumps({"experiments": [{"candidate_id": 9999, "duration_s": 20, "hypothesis": "h", "reason": "r"}]})))
    plan = planner.make_plan(g, [], n=3, client=c, cfg=PlannerCfg(provider="mistral", llm_model=M))
    assert plan.planner == "heuristic" and "unknown candidate_id" in plan.fallback_reason


def test_a_mistral_rate_limit_makes_the_planner_fall_back_to_the_heuristic(tmp_path):
    g = graph_of(tmp_path)
    limited = httpx.Response(429, headers={"retry-after": "7200"}, json={"message": "monthly limit"})
    plan = planner.make_plan(g, [], n=3, client=make(limited)[0], cfg=PlannerCfg(provider="mistral", llm_model=M))
    assert plan.planner == "heuristic" and "rate limit exceeded" in plan.fallback_reason and plan.experiments


# ---------------------------------------------------------------- config and .env
def test_mistral_provider_and_model_load_together():
    assert config.AnalyzerCfg(provider="mistral", llm_model="mistral-small-latest").provider == "mistral"
    for name in ("codestral-latest", "ministral-8b-latest", "magistral-medium-latest", "open-mistral-nemo", "pixtral-large-latest"):
        assert config.PlannerCfg(provider="mistral", llm_model=name).llm_model == name


@pytest.mark.parametrize("provider,model,vendor", [("mistral", "claude-opus-5-5", "anthropic"), ("mistral", "gemini-3.5-flash", "gemini"),
                                                   ("anthropic", "mistral-large-latest", "mistral"), ("gemini", "mistral-large-latest", "mistral"),
                                                   ("gemini", "claude-opus-5-5", "anthropic"), ("anthropic", "gemini-3.5-flash", "gemini")])
def test_a_model_from_another_vendor_is_rejected_at_load_time_and_names_the_vendor(provider, model, vendor):
    with pytest.raises(ValidationError, match=f"provider is {provider} but llm_model is '{model}', which is a {vendor} model"):
        config.AnalyzerCfg(provider=provider, llm_model=model)


def test_an_unrecognised_model_name_is_allowed_for_any_provider():
    assert config.AnalyzerCfg(provider="mistral", llm_model="my-finetune-7b").llm_model == "my-finetune-7b"


def test_the_mistral_key_is_loaded_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text(f"MISTRAL_API_KEY={KEY}\n", encoding="utf-8")
    try:
        assert load_env(p) == ["MISTRAL_API_KEY"] and os.environ["MISTRAL_API_KEY"] == KEY
    finally:
        os.environ.pop("MISTRAL_API_KEY", None)


# ---------------------------------------------------------------- llm-check
def test_llm_check_without_a_mistral_key_says_where_to_get_one():
    r = runner.invoke(cli.app, ["llm-check", "--provider", "mistral"])
    assert r.exit_code == 1 and "console.mistral.ai/api-keys" in r.output and "MISTRAL_API_KEY" in r.output


def test_llm_check_no_send_masks_the_key_and_makes_no_request(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", KEY)
    monkeypatch.setattr(mistral, "MistralClient", lambda *a, **k: pytest.fail("no request expected with --no-send"))
    r = runner.invoke(cli.app, ["llm-check", "--provider", "mistral", "--no-send"])
    assert r.exit_code == 0 and "key found" in r.output and KEY not in r.output


def test_llm_check_lists_models_and_sends_one_request(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("MISTRAL_API_KEY", KEY)

    class Fake:
        def __init__(self):
            self.messages = self

        def list_models(self):
            return ["mistral-a", "mistral-b"]

        def create(self, **kw):
            assert kw["model"] == "mistral-b"
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn",
                                   usage=SimpleNamespace(input_tokens=8, output_tokens=2))
    monkeypatch.setattr(mistral, "MistralClient", Fake)
    r = runner.invoke(cli.app, ["llm-check", "--provider", "mistral", "--models", "--model", "mistral-b"])
    assert r.exit_code == 0 and "2 chat model(s)" in r.output and "mistral-a" in r.output and "OK: mistral-b answered 'ok'" in r.output
    assert KEY not in r.output


@pytest.mark.parametrize("status,hint", [(401, "key was rejected"), (404, "try --models"), (429, "Experiment tier has low limits")])
def test_llm_check_turns_api_errors_into_actionable_hints(monkeypatch, status, hint):
    monkeypatch.setenv("MISTRAL_API_KEY", KEY)

    class Fake:
        def __init__(self):
            self.messages = self

        def create(self, **kw):
            raise MistralError("boom", status)
    monkeypatch.setattr(mistral, "MistralClient", Fake)
    r = runner.invoke(cli.app, ["llm-check", "--provider", "mistral"])
    assert r.exit_code == 1 and "Mistral error: boom" in r.output and hint in r.output and KEY not in r.output


def test_llm_check_does_not_tell_you_to_wait_when_the_plan_allows_zero_requests(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", KEY)

    class Fake:
        def __init__(self):
            self.messages = self

        def create(self, **kw):
            raise MistralError("this key's plan allows 0 requests per minute, so no call can succeed: activate the free Experiment plan", 429)
    monkeypatch.setattr(mistral, "MistralClient", Fake)
    r = runner.invoke(cli.app, ["llm-check", "--provider", "mistral"])
    assert r.exit_code == 1 and "activate the free Experiment plan" in r.output and "wait and retry" not in r.output
