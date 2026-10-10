"""The Gemini adapter, offline: scripted HTTP responses stand in for the API (unit tests of the translation and the
control flow; no experiment data is synthetic). Also drives the real analyzer and planner loops through it."""
import json

import httpx
import pytest

from chaosai import analyzer, planner
from chaosai.analysis import Recorder
from chaosai.config import AnalyzerCfg, PlannerCfg
from chaosai.evidence import Evidence
from chaosai.gemini import GeminiClient, GeminiError, to_gemini_schema
from chaosai.llmclient import make_client
from chaosai.models import Graph
from chaosai.rules import analyse
from tests.fixtures_results import KILL_ID, SPIKE_ID, make_results

KEY = "AIzaSyTEST-KEY-should-never-leak-0123456789"
M = "gemini-test-flash"


def text_resp(text, finish="STOP", **extra):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 5, "thoughtsTokenCount": 7}, **extra}


def call_resp(*calls, sig=None):
    parts = [{"functionCall": {"name": n, "args": a}} for n, a in calls]
    if sig:
        parts[0]["thoughtSignature"] = sig
    return {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 3}}


def make(*script, **kw):
    reqs, slept, queue = [], [], list(script)

    def handler(request):
        reqs.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)
    c = GeminiClient(key=KEY, transport=httpx.MockTransport(handler), sleep=slept.append, **kw)
    return c, reqs, slept


def body(req):
    return json.loads(req.content)


# ---------------------------------------------------------------- request translation
def test_request_goes_to_generate_content_with_the_key_in_a_header_never_the_url():
    c, reqs, _ = make(text_resp("ok"))
    c.beta.messages.create(model="models/" + M, max_tokens=500, messages=[{"role": "user", "content": "hi"}])
    r = reqs[0]
    assert str(r.url) == f"https://generativelanguage.googleapis.com/v1beta/models/{M}:generateContent"
    assert r.headers["x-goog-api-key"] == KEY and KEY not in str(r.url)


def test_system_tools_effort_and_limits_are_translated():
    tools = [{"name": "t", "description": "d", "input_schema": {
        "type": "object", "additionalProperties": False, "required": ["a"],
        "properties": {"a": {"type": "string"}, "held": {"type": ["boolean", "null"], "description": "x"},
                       "xs": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"n": {"type": "integer"}}}}}}}]
    c, reqs, _ = make(text_resp("ok"))
    c.beta.messages.create(model=M, max_tokens=777, system="be careful", tools=tools, betas=["x"], fallbacks="default",
                           messages=[{"role": "user", "content": "go"}], output_config={"effort": "medium"})
    b = body(reqs[0])
    assert b["systemInstruction"] == {"parts": [{"text": "be careful"}]}
    assert b["contents"] == [{"role": "user", "parts": [{"text": "go"}]}]
    assert b["generationConfig"] == {"maxOutputTokens": 777, "thinkingConfig": {"thinkingLevel": "medium"}}
    assert b["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
    decl = b["tools"][0]["functionDeclarations"][0]
    assert decl["name"] == "t" and decl["description"] == "d"
    p = decl["parameters"]
    assert "additionalProperties" not in json.dumps(p)
    assert p["properties"]["held"] == {"type": "boolean", "nullable": True, "description": "x"}
    assert p["required"] == ["a"] and p["properties"]["xs"]["items"]["properties"]["n"] == {"type": "integer"}


def test_effort_levels_map_to_thinking_levels():
    levels = {}
    for effort in ("low", "high", "xhigh", "max"):
        c, reqs, _ = make(text_resp("ok"))
        c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}], output_config={"effort": effort})
        levels[effort] = body(reqs[0])["generationConfig"]["thinkingConfig"]["thinkingLevel"]
    assert levels == {"low": "low", "high": "high", "xhigh": "high", "max": "high"}


def test_a_property_named_like_a_schema_keyword_is_not_filtered_out():
    s = to_gemini_schema({"type": "object", "properties": {"description": {"type": "string"}, "required": {"type": "boolean"},
                                                           "type": {"type": "string"}, "additionalProperties": {"type": "integer"}}})
    assert set(s["properties"]) == {"description", "required", "type", "additionalProperties"}


def test_structured_output_uses_response_schema_and_json_mime_type():
    schema = {"type": "object", "additionalProperties": False, "required": ["x"], "properties": {"x": {"type": "integer"}}}
    c, reqs, _ = make(text_resp('{"x": 1}'))
    c.create(model=M, max_tokens=100, messages=[{"role": "user", "content": "x"}],
             output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
    g = body(reqs[0])["generationConfig"]
    assert g["responseMimeType"] == "application/json"
    assert g["responseSchema"] == {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}}


# ---------------------------------------------------------------- response translation
def test_text_response_and_usage_mapping():
    c, _, _ = make(text_resp("hello"))
    r = c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}])
    assert [b.type for b in r.content] == ["text"] and r.content[0].text == "hello" and r.stop_reason == "end_turn"
    assert (r.usage.input_tokens, r.usage.output_tokens) == (11, 12)               # thinking tokens count as output


def test_function_calls_become_tool_use_blocks_with_unique_ids():
    c, _, _ = make(call_resp(("get_graph", {}), ("get_experiment", {"experiment_id": "e1"})))
    r = c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}])
    assert r.stop_reason == "tool_use" and [b.type for b in r.content] == ["tool_use", "tool_use"]
    assert [b.name for b in r.content] == ["get_graph", "get_experiment"] and r.content[1].input == {"experiment_id": "e1"}
    assert len({b.id for b in r.content}) == 2


def test_tool_round_trip_replays_the_models_own_parts_including_the_thought_signature():
    c, reqs, _ = make(call_resp(("get_graph", {}), sig="SIG-abc"), text_resp("done"))
    first = c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "q"}])
    use = first.content[0]
    c.create(model=M, max_tokens=10, messages=[
        {"role": "user", "content": "q"}, {"role": "assistant", "content": first.content},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": json.dumps({"services": 5})}]}])
    contents = body(reqs[1])["contents"]
    assert contents[1] == {"role": "model", "parts": [{"functionCall": {"name": "get_graph", "args": {}}, "thoughtSignature": "SIG-abc"}]}
    assert contents[2] == {"role": "user", "parts": [{"functionResponse": {"name": "get_graph", "response": {"services": 5}}}]}


def test_tool_errors_and_non_object_results_are_wrapped_as_objects():
    c, reqs, _ = make(call_resp(("a", {}), ("b", {})), text_resp("x"))
    first = c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "q"}])
    ids = [b.id for b in first.content]
    c.create(model=M, max_tokens=10, messages=[
        {"role": "user", "content": "q"}, {"role": "assistant", "content": first.content},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": ids[0], "content": "REJECTED: fix it", "is_error": True},
                                      {"type": "tool_result", "tool_use_id": ids[1], "content": "[1, 2]"}]}])
    parts = body(reqs[1])["contents"][2]["parts"]
    assert parts[0]["functionResponse"] == {"name": "a", "response": {"error": "REJECTED: fix it"}}
    assert parts[1]["functionResponse"] == {"name": "b", "response": {"result": [1, 2]}}


def test_finish_reasons_map_to_anthropic_stop_reasons():
    cases = [(text_resp("x", finish="SAFETY"), "refusal"), (text_resp("x", finish="MAX_TOKENS"), "max_tokens"),
             (text_resp("x", finish="STOP"), "end_turn"), ({"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}, "refusal")]
    for resp, expect in cases:
        c, _, _ = make(resp)
        assert c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}]).stop_reason == expect


def test_thinking_budget_exhaustion_is_reported_as_max_tokens_not_an_empty_answer():
    c, _, _ = make({"candidates": [{"content": {}, "finishReason": "MAX_TOKENS"}], "usageMetadata": {"promptTokenCount": 8, "thoughtsTokenCount": 60}})
    r = c.create(model=M, max_tokens=64, messages=[{"role": "user", "content": "x"}])
    assert r.stop_reason == "max_tokens" and r.content == [] and r.usage.output_tokens == 60     # the loop raises on max_tokens before any replay


def test_a_malformed_function_call_is_a_retryable_turn_and_thought_parts_are_dropped():
    bad = {"candidates": [{"content": {}, "finishReason": "MALFORMED_FUNCTION_CALL"}]}
    c, _, _ = make(bad, {"candidates": [{"content": {"parts": [{"text": "private", "thought": True}, {"text": "answer"}]}, "finishReason": "STOP"}]})
    r1 = c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}])
    assert r1.stop_reason == "end_turn" and "malformed" in r1.content[0].text
    r2 = c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}])
    assert [b.text for b in r2.content] == ["answer"]


def test_no_candidates_and_no_block_reason_is_an_error():
    c, _, _ = make({"candidates": []})
    with pytest.raises(GeminiError, match="no candidates"):
        c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}])


# ---------------------------------------------------------------- credentials, retries, quota
def test_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        GeminiClient()


def test_rate_limit_is_retried_after_the_servers_suggested_delay():
    limited = httpx.Response(429, headers={"retry-after": "3"}, json={"error": {"status": "RESOURCE_EXHAUSTED", "message": "slow down"}})
    c, reqs, slept = make(limited, text_resp("ok"))
    assert c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}]).content[0].text == "ok"
    assert slept == [3.0] and len(reqs) == 2


def test_retry_info_in_the_body_is_honoured_and_server_errors_back_off_exponentially():
    info = httpx.Response(429, text='{"error": {"details": [{"retryDelay": "12s"}]}}')
    c, _, slept = make(info, httpx.Response(503, json={"error": {"message": "busy"}}), text_resp("ok"))
    c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}])
    assert slept == [12.5, 4.0]


def test_persistent_server_errors_give_up_after_max_retries():
    c, reqs, slept = make(*[httpx.Response(503, json={"error": {"status": "UNAVAILABLE", "message": "high demand"}})] * 3, max_retries=2)
    with pytest.raises(GeminiError, match="503 UNAVAILABLE: high demand") as e:
        c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}])
    assert len(reqs) == 3 and len(slept) == 2 and e.value.status == 503


def test_a_spent_daily_quota_fails_immediately_with_a_readable_message():
    quota = httpx.Response(429, text=json.dumps({"error": {"status": "RESOURCE_EXHAUSTED", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20",
                         "quotaDimensions": {"model": "gemini-x"}}]}, {"retryDelay": "27366s"}]}}))
    c, reqs, slept = make(quota)
    with pytest.raises(GeminiError) as e:
        c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}])
    assert slept == [] and len(reqs) == 1 and e.value.status == 429
    msg = str(e.value)
    assert "PerDayPerProjectPerModel-FreeTier" in msg and "limit 20" in msg and "gemini-x" in msg and "7.6 h" in msg


def test_client_errors_are_not_retried_and_never_contain_the_key():
    c, reqs, slept = make(httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT", "message": "API key not valid"}}))
    with pytest.raises(GeminiError) as e:
        c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}])
    assert len(reqs) == 1 and slept == [] and e.value.status == 400 and KEY not in str(e.value)


def test_a_model_that_rejects_thinking_config_is_retried_without_it():
    c, reqs, _ = make(httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT", "message": "Unknown name thinkingLevel"}}), text_resp("ok"))
    r = c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}], output_config={"effort": "low"})
    assert r.content[0].text == "ok" and "thinkingConfig" in body(reqs[0])["generationConfig"] and "thinkingConfig" not in body(reqs[1])["generationConfig"]


def test_network_failures_are_retried_then_reported_without_a_url_or_key():
    c, reqs, slept = make(*[httpx.ConnectError("boom")] * 3, max_retries=2)
    with pytest.raises(GeminiError, match="cannot reach the Gemini API") as e:
        c.create(model=M, max_tokens=10, messages=[{"role": "user", "content": "x"}])
    assert len(reqs) == 3 and len(slept) == 2 and KEY not in str(e.value)


def test_list_models_keeps_only_generate_content_models():
    c, _, _ = make({"models": [{"name": "models/b", "supportedGenerationMethods": ["generateContent"]},
                               {"name": "models/a", "supportedGenerationMethods": ["generateContent", "countTokens"]},
                               {"name": "models/emb", "supportedGenerationMethods": ["embedContent"]}]})
    assert c.list_models() == ["a", "b"]


def test_make_client_picks_the_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    assert isinstance(make_client("gemini"), GeminiClient)


# ---------------------------------------------------------------- the real analyzer loop through the adapter
IDS = [SPIKE_ID, KILL_ID]
GCFG = AnalyzerCfg(provider="gemini", llm_model=M)


def payload(ev):
    a = analyse(Recorder(ev), IDS)
    return a.model_dump(exclude={"generated_by", "model", "fallback_reason", "notes"})


def read_calls():
    calls = [("get_graph", {})]
    for i in IDS:
        calls += [("get_experiment", {"experiment_id": i}), ("get_gauges", {"experiment_id": i}),
                  ("get_metrics", {"experiment_id": i, "window": "during"})]
    return calls


def test_the_analyzer_loop_runs_through_the_gemini_adapter_and_passes_validation(tmp_path):
    ev = Evidence(make_results(tmp_path))
    c, reqs, _ = make(call_resp(*read_calls(), sig="S1"), call_resp(("submit_analysis", payload(ev))))
    a = analyzer.analyze(ev, IDS, GCFG, client=c)
    assert a.generated_by == "llm" and a.model == M and a.weaknesses and a.fixes
    assert any("0 rejected" in n for n in a.notes)
    second = body(reqs[1])["contents"]
    assert second[1]["parts"][0]["thoughtSignature"] == "S1"                           # replayed verbatim
    answers = second[2]["parts"]
    assert len(answers) == len(read_calls()) and all("functionResponse" in p for p in answers)
    assert [p["functionResponse"]["name"] for p in answers][:2] == ["get_graph", "get_experiment"]


def test_a_fabricated_number_is_rejected_through_the_adapter_and_the_model_can_fix_it(tmp_path):
    ev = Evidence(make_results(tmp_path))
    good = payload(ev)
    bad = json.loads(json.dumps(good))
    bad["weaknesses"][0]["evidence"][0]["claim"] += " and exactly 98765 votes were lost"
    c, reqs, _ = make(call_resp(*read_calls()), call_resp(("submit_analysis", bad)), call_resp(("submit_analysis", good)))
    a = analyzer.analyze(ev, IDS, GCFG, client=c)
    assert a.generated_by == "llm" and any("1 rejected" in n for n in a.notes)
    feedback = body(reqs[2])["contents"][-1]["parts"][0]["functionResponse"]
    assert feedback["name"] == "submit_analysis" and "REJECTED" in feedback["response"]["error"] and "98765" in feedback["response"]["error"]


def test_a_quota_error_falls_back_to_the_rules_and_says_why(tmp_path):
    ev = Evidence(make_results(tmp_path))
    quota = httpx.Response(429, text='{"error": {"details": [{"quotaId": "PerDay", "quotaValue": "20"}, {"retryDelay": "3600s"}]}}')
    a = analyzer.analyze(ev, IDS, GCFG, client=make(quota)[0])
    assert a.generated_by == "rules" and "quota exhausted" in a.fallback_reason and a.weaknesses


def test_the_gemini_analysis_is_cached_and_replayed_without_another_call(tmp_path):
    ev = Evidence(make_results(tmp_path))
    c, reqs, _ = make(call_resp(*read_calls()), call_resp(("submit_analysis", payload(ev))))
    analyzer.analyze(ev, IDS, GCFG, client=c, cache_dir=tmp_path / "cache")
    n = len(reqs)
    again = analyzer.analyze(ev, IDS, GCFG, client=make()[0], cache_dir=tmp_path / "cache")      # an empty script would fail on any call
    assert again.generated_by == "llm" and len(reqs) == n and any("served from cache" in x for x in again.notes)


def test_the_cache_is_separate_per_provider(tmp_path):
    ev = Evidence(make_results(tmp_path))
    other = AnalyzerCfg(provider="anthropic", llm_model="claude-opus-5-5")
    assert analyzer._cache_path(tmp_path, GCFG, IDS, ev.fingerprint(IDS)) != analyzer._cache_path(tmp_path, other, IDS, ev.fingerprint(IDS))


# ---------------------------------------------------------------- the planner through the adapter
def graph_of(tmp_path):
    return Graph.model_validate_json((make_results(tmp_path) / "graph.json").read_text(encoding="utf-8"))


def test_the_planner_runs_through_the_gemini_adapter(tmp_path):
    g = graph_of(tmp_path)
    cands = planner.candidates(g, [], None)
    pick = {"experiments": [{"candidate_id": cands[0].id, "duration_s": 20, "hypothesis": "h1", "reason": "r1"},
                            {"candidate_id": cands[1].id, "duration_s": 25, "hypothesis": "h2", "reason": "r2"}]}
    c, reqs, _ = make(text_resp(json.dumps(pick)))
    plan = planner.llm_plan(g, [], n=3, client=c, cfg=PlannerCfg(provider="gemini", llm_model=M))
    assert plan.planner == "llm" and plan.model == M and [e.target for e in plan.experiments] == [cands[0].target, cands[1].target]
    gen = body(reqs[0])["generationConfig"]
    assert gen["responseMimeType"] == "application/json" and "additionalProperties" not in json.dumps(gen["responseSchema"])


def test_the_planner_still_rejects_a_candidate_gemini_invented_and_falls_back(tmp_path):
    g = graph_of(tmp_path)
    c, _, _ = make(text_resp(json.dumps({"experiments": [{"candidate_id": 9999, "duration_s": 20, "hypothesis": "h", "reason": "r"}]})))
    plan = planner.make_plan(g, [], n=3, client=c, cfg=PlannerCfg(provider="gemini", llm_model=M))
    assert plan.planner == "heuristic" and "unknown candidate_id" in plan.fallback_reason


def test_a_gemini_quota_error_makes_the_planner_fall_back_to_the_heuristic(tmp_path):
    g = graph_of(tmp_path)
    quota = httpx.Response(429, text='{"error": {"details": [{"quotaId": "PerDay"}, {"retryDelay": "3600s"}]}}')
    plan = planner.make_plan(g, [], n=3, client=make(quota)[0], cfg=PlannerCfg(provider="gemini", llm_model=M))
    assert plan.planner == "heuristic" and "quota exhausted" in plan.fallback_reason and plan.experiments
