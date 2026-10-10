import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from chaosai import analyzer, cli
from chaosai.analysis import Analysis, Recorder, validate
from chaosai.analyzer import AnalyzerError, analyze, run_agent
from chaosai.config import AnalyzerCfg
from chaosai.evidence import TOOLS, Evidence
from chaosai.rules import analyse
from tests.fixtures_results import KILL_ID, SPIKE_ID, make_results

IDS = [SPIKE_ID, KILL_ID]
HYP = {SPIKE_ID: True, KILL_ID: False}
CFG = AnalyzerCfg()


@pytest.fixture
def ev(tmp_path):
    return Evidence(make_results(tmp_path))


# ---------------------------------------------------------------- a scripted stand-in for the model
def tu(name, inp, id_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def resp(*blocks, stop="tool_use"):
    return SimpleNamespace(stop_reason=stop, content=list(blocks), usage=SimpleNamespace(input_tokens=100, output_tokens=50))


class Scripted:
    def __init__(self, *script):
        self.script, self.calls = list(script), []
        outer = self

        class Messages:
            def create(self, **kw):
                outer.calls.append({**kw, "messages": list(kw["messages"])})      # snapshot: the loop keeps appending
                nxt = outer.script.pop(0)
                if isinstance(nxt, Exception):
                    raise nxt
                return nxt
        self.beta = SimpleNamespace(messages=Messages())


def data_calls(prefix="t"):
    """The tool calls a careful analyst would make (the same ones the rule-based analyzer makes)."""
    blocks = [tu("get_graph", {}, f"{prefix}-graph")]
    for n, i in enumerate(IDS):
        blocks += [tu("get_experiment", {"experiment_id": i}, f"{prefix}-e{n}"),
                   tu("get_gauges", {"experiment_id": i}, f"{prefix}-g{n}"),
                   tu("get_metrics", {"experiment_id": i, "window": "during"}, f"{prefix}-m{n}")]
    return resp(*blocks)


def payload(ev):
    rec = Recorder(ev)
    a = analyse(rec, IDS)
    return a.model_dump(exclude={"generated_by", "model", "fallback_reason", "notes"})


def submit(p, id_="sub"):
    return resp(tu("submit_analysis", p, id_))


# ---------------------------------------------------------------- the rule-based analyzer
def test_rules_find_the_patterns_the_data_shows_and_pass_the_validator(ev):
    rec = Recorder(ev)
    a = analyse(rec, IDS)
    titles = [w.title for w in a.weaknesses]
    assert any(t.startswith("Single worker caps") for t in titles)
    assert any(t.startswith("Dependents exit") for t in titles)
    assert any("no healthcheck" in t for t in titles)
    assert not any("invisible" in t for t in titles)                      # pipeline recovery 6 s vs probe 1 s: below the 10 s bar
    assert validate(a, rec.seen, IDS, HYP) == []
    assert a.generated_by == "rules" and all(f.addresses and f.evidence and f.change.strip() for f in a.fixes)
    order = ["critical", "high", "medium", "low"]
    assert [order.index(w.severity) for w in a.weaknesses] == sorted(order.index(w.severity) for w in a.weaknesses)


def test_rule_claims_quote_stored_values(ev):
    a = analyse(Recorder(ev), IDS)
    w = next(w for w in a.weaknesses if w.title.startswith("Single worker"))
    first = w.evidence[0].claim
    assert "300" in first and "9.5" in first and "31.6" in first          # peak queue, drain rate, backlog lag: all from get_gauges
    spike = next(e for e in a.experiments if e.experiment_id == SPIKE_ID)
    assert spike.hypothesis_held is None and "vote" in spike.blast_radius and "worker" in spike.blast_radius
    kill = next(e for e in a.experiments if e.experiment_id == KILL_ID)
    assert set(kill.blast_radius) >= {"db", "result", "worker"}


def test_headline_evidence_prefers_clean_experiments(tmp_path):
    d = make_results(tmp_path)
    p = d / f"{KILL_ID}.json"
    p.write_text(p.read_text().replace('"rollback_errors": []', '"rollback_errors": ["limit not restored"]'))
    ev = Evidence(d)
    a = analyse(Recorder(ev), IDS)
    w = next(w for w in a.weaknesses if w.title.startswith("Dependents exit"))
    assert w.evidence[0].experiment_id == KILL_ID                          # only candidate, so still used: fall-through works
    q = ev.get_experiment(KILL_ID)["quality"]
    assert any("rollback problems" in x for x in q)


# ---------------------------------------------------------------- the agent loop
def test_happy_path_tools_then_valid_submission(ev):
    client = Scripted(data_calls(), submit(payload(ev)))
    a, stats = run_agent(ev, IDS, client, CFG)
    assert a.generated_by == "llm" and a.model == "claude-opus-5-5" and len(a.weaknesses) >= 3
    assert stats["turns"] == 2 and stats["tool_calls"] == 7 and stats["rejected"] == 0
    # the tool results were fed back, all in ONE user message, matching the tool_use ids
    second = client.calls[1]["messages"]
    assert second[-1]["role"] == "user" and len(second[-1]["content"]) == 7        # all results in ONE user message
    results = second[-1]["content"]
    assert {r["tool_use_id"] for r in results} == {"t-graph", "t-e0", "t-g0", "t-m0", "t-e1", "t-g1", "t-m1"}
    assert all(r["type"] == "tool_result" and json.loads(r["content"]) for r in results)


def test_assistant_content_is_passed_back_unchanged_including_thinking_blocks(ev):
    thinking = SimpleNamespace(type="thinking", thinking="", signature="sig")
    first = resp(thinking, *data_calls().content)
    client = Scripted(first, submit(payload(ev)))
    run_agent(ev, IDS, client, CFG)
    assistant = client.calls[1]["messages"][1]
    assert assistant["role"] == "assistant" and assistant["content"] is first.content and thinking in assistant["content"]


def test_a_fabricated_number_is_rejected_with_the_reason_and_the_model_can_fix_it(ev):
    bad = payload(ev)
    bad["weaknesses"][0]["evidence"][0]["claim"] = "the worker drained at 77 votes/s"
    client = Scripted(data_calls(), submit(bad, "s1"), submit(payload(ev), "s2"))
    a, stats = run_agent(ev, IDS, client, CFG)
    assert stats["rejected"] == 1 and a.generated_by == "llm"
    rejection = client.calls[2]["messages"][-1]["content"][0]
    assert rejection["tool_use_id"] == "s1" and rejection["is_error"] is True
    assert "REJECTED" in rejection["content"] and "77.0" in rejection["content"] and "no tool returned" in rejection["content"]


def test_submitting_without_ever_reading_the_data_is_rejected(ev):
    client = Scripted(submit(payload(ev), "early"), data_calls(), submit(payload(ev), "late"))
    a, stats = run_agent(ev, IDS, client, CFG)
    assert stats["rejected"] == 1
    assert "never called a data tool" in client.calls[1]["messages"][-1]["content"][0]["content"]


def test_schema_violations_are_sent_back_not_raised(ev):
    client = Scripted(data_calls(), submit({"summary": "x"}, "bad"), submit(payload(ev), "good"))
    a, stats = run_agent(ev, IDS, client, CFG)
    assert stats["rejected"] == 1
    assert "does not match the schema" in client.calls[2]["messages"][-1]["content"][0]["content"]


def test_unknown_tool_and_bad_arguments_come_back_as_errors_the_model_can_read(ev):
    client = Scripted(resp(tu("rm_rf", {}, "x1"), tu("get_experiment", {"experiment_id": "nope"}, "x2")), data_calls(), submit(payload(ev)))
    run_agent(ev, IDS, client, CFG)
    out = {r["tool_use_id"]: json.loads(r["content"]) for r in client.calls[1]["messages"][-1]["content"]}
    assert "unknown tool" in out["x1"]["error"] and "unknown experiment_id" in out["x2"]["error"]


def test_a_turn_that_ends_without_submitting_is_nudged_then_allowed_to_finish(ev):
    text = SimpleNamespace(type="text", text="Here is my analysis in prose...")
    client = Scripted(data_calls(), resp(text, stop="end_turn"), submit(payload(ev)))
    a, stats = run_agent(ev, IDS, client, CFG)
    assert stats["rejected"] == 1 and "calling submit_analysis" in client.calls[2]["messages"][-1]["content"]


def test_giving_up_after_too_many_rejections(ev):
    bad = payload(ev)
    bad["summary"] = ""
    client = Scripted(data_calls(), *[submit(bad, f"s{i}") for i in range(10)])
    with pytest.raises(AnalyzerError, match="no valid analysis after 4 rejections"):
        run_agent(ev, IDS, client, AnalyzerCfg(max_rejects=3))


def test_turn_limit(ev):
    client = Scripted(*[data_calls(f"r{i}") for i in range(10)])
    with pytest.raises(AnalyzerError, match="within 3 turns"):
        run_agent(ev, IDS, client, AnalyzerCfg(max_turns=3))


@pytest.mark.parametrize("stop, why", [("refusal", "refused"), ("max_tokens", "cut off")])
def test_refusal_and_truncation_raise(ev, stop, why):
    with pytest.raises(AnalyzerError, match=why):
        run_agent(ev, IDS, Scripted(resp(stop=stop)), CFG)


def test_request_matches_the_opus_5_5_rules(ev):
    client = Scripted(data_calls(), submit(payload(ev)))
    run_agent(ev, IDS, client, CFG)
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["fallbacks"] == "default"
    assert kw["output_config"] == {"effort": "medium"}
    names = [t["name"] for t in kw["tools"]]
    assert names == [t["name"] for t in TOOLS] + ["submit_analysis"] and len(set(names)) == len(names)
    for banned in ("tool_choice", "temperature", "top_p", "top_k", "thinking", "budget_tokens", "prefill"):
        assert banned not in kw                                            # forced tool_choice 400s on Opus 5.5; sampling params are removed
    assert [m["role"] for m in kw["messages"]] == ["user"] and SPIKE_ID in kw["messages"][0]["content"]
    assert "submit_analysis" in kw["system"] and "EVIDENCE RULE" in kw["system"]
    for t in kw["tools"]:
        assert t["input_schema"]["type"] == "object" and t["input_schema"]["additionalProperties"] is False


# ---------------------------------------------------------------- analyze(): fallback and cache
def test_offline_never_touches_the_client(ev):
    client = Scripted()
    a = analyze(ev, IDS, CFG, use_llm=False, client=client)
    assert a.generated_by == "rules" and "offline" in a.fallback_reason and client.calls == []


def test_config_can_disable_the_llm(ev):
    a = analyze(ev, IDS, AnalyzerCfg(use_llm=False), client=Scripted())
    assert a.generated_by == "rules"


def test_llm_success_is_returned_with_usage_notes(ev):
    a = analyze(ev, IDS, CFG, client=Scripted(data_calls(), submit(payload(ev))))
    assert a.generated_by == "llm" and a.fallback_reason is None
    assert any("2 turn(s), 7 data-tool call(s), 0 rejected" in n and "200 input" not in n for n in a.notes) or any("tokens" in n for n in a.notes)


@pytest.mark.parametrize("script, why", [
    ([resp(stop="refusal")], "refused"),
    ([resp(stop="max_tokens")], "cut off"),
    ([ConnectionError("no wifi")], "ConnectionError"),
])
def test_any_failure_falls_back_to_the_rules_and_says_why(ev, script, why):
    a = analyze(ev, IDS, CFG, client=Scripted(*script))
    assert a.generated_by == "rules" and why in a.fallback_reason and len(a.weaknesses) >= 3


def test_an_invalid_model_never_reaches_the_user_it_falls_back(ev):
    bad = payload(ev)
    bad["summary"] = ""
    a = analyze(ev, IDS, AnalyzerCfg(max_rejects=1), client=Scripted(data_calls(), submit(bad), submit(bad), submit(bad)))
    assert a.generated_by == "rules" and "no valid analysis" in a.fallback_reason


def test_missing_credentials_message_is_readable(ev, monkeypatch):
    import anthropic

    class NoCreds:
        def __init__(self, *a, **k):
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

        @staticmethod
        def _create(**kw):
            raise TypeError('"Could not resolve authentication method. Expected one of api_key, auth_token, or credentials to be set."')
    monkeypatch.setattr(anthropic, "Anthropic", NoCreds)
    a = analyze(ev, IDS, CFG)
    assert a.generated_by == "rules" and "ANTHROPIC_API_KEY" in a.fallback_reason


def test_nothing_to_analyse_is_an_error(ev):
    with pytest.raises(AnalyzerError, match="no experiments"):
        analyze(ev, [], CFG)


def test_validated_answers_are_cached_and_replayed_without_the_api(ev, tmp_path):
    cache = tmp_path / "cache"
    first = analyze(ev, IDS, CFG, client=Scripted(data_calls(), submit(payload(ev))), cache_dir=cache)
    assert first.generated_by == "llm" and list(cache.glob("analysis-*.json"))
    broken = Scripted(ConnectionError("offline"))
    again = analyze(ev, IDS, CFG, client=broken, cache_dir=cache)
    assert broken.calls == [] and again.generated_by == "llm"
    assert [w.title for w in again.weaknesses] == [w.title for w in first.weaknesses]
    assert any("served from cache" in n for n in again.notes)


def test_changed_inputs_invalidate_the_cache(tmp_path):
    d = make_results(tmp_path)
    ev1 = Evidence(d)
    cache = tmp_path / "cache"
    analyze(ev1, IDS, CFG, client=Scripted(data_calls(), submit(payload(ev1))), cache_dir=cache)
    (d / "baseline.json").write_text("{}")
    ev2 = Evidence(d)
    c2 = Scripted(data_calls(), submit(payload(ev2)))
    analyze(ev2, IDS, CFG, client=c2, cache_dir=cache)
    assert len(c2.calls) == 2                                              # new fingerprint: a real call, not the stale cache
    assert len(list(cache.glob("analysis-*.json"))) == 2


def test_failures_and_rule_fallbacks_are_not_cached(ev, tmp_path):
    cache = tmp_path / "cache"
    analyze(ev, IDS, CFG, client=Scripted(resp(stop="refusal")), cache_dir=cache)
    assert not list(cache.glob("analysis-*.json"))


# ---------------------------------------------------------------- the CLI
runner = CliRunner()


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    d = make_results(tmp_path)
    monkeypatch.setattr(cli, "RESULTS", d)
    return d


def test_analyze_offline_prints_ranked_findings_and_writes_the_file(cli_env):
    r = runner.invoke(cli.app, ["analyze", "--offline"])
    assert r.exit_code == 0, r.output
    assert "2 experiment(s) in scope" in r.output and "analysis by: deterministic rules" in r.output
    assert "WEAKNESSES (most severe first)" in r.output and "FIXES" in r.output and "replicas: 3" in r.output
    saved = Analysis.model_validate_json((cli_env / "analysis.json").read_text())
    assert saved.generated_by == "rules" and saved.weaknesses


def test_analyze_dry_run_writes_nothing_and_last_limits_scope(cli_env):
    r = runner.invoke(cli.app, ["analyze", "--offline", "--dry-run", "--last", "1"])
    assert r.exit_code == 0 and "1 experiment(s) in scope" in r.output and not (cli_env / "analysis.json").exists()


def test_analyze_rejects_unknown_ids_cleanly(cli_env):
    r = runner.invoke(cli.app, ["analyze", "--offline", "-i", "nope"])
    assert r.exit_code == 2 and "unknown experiment id" in r.output and "Traceback" not in r.output


def test_analyze_without_experiments_is_a_clean_error(tmp_path, monkeypatch):
    (tmp_path / "res").mkdir()
    monkeypatch.setattr(cli, "RESULTS", tmp_path / "res")
    r = runner.invoke(cli.app, ["analyze", "--offline"])
    assert r.exit_code == 1 and "no experiments to analyse" in r.output


def test_analyze_default_falls_back_with_a_readable_reason_when_there_is_no_key(cli_env, monkeypatch):
    import anthropic

    def no_creds(*a, **k):
        raise TypeError("Could not resolve authentication method")
    monkeypatch.setattr(anthropic, "Anthropic", no_creds)
    r = runner.invoke(cli.app, ["analyze", "--dry-run"])
    assert r.exit_code == 0 and "ANTHROPIC_API_KEY" in r.output and "deterministic rules" in r.output


def test_llm_check_reports_a_missing_key_and_never_prints_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    r = runner.invoke(cli.app, ["llm-check"])
    assert r.exit_code == 1 and "no key found" in r.output


def test_llm_check_with_a_key_masks_it_and_reports_the_answer(monkeypatch):
    import anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-SUPERSECRETVALUE-wxyz")

    class Fake:
        def __init__(self, *a, **k):
            self.messages = SimpleNamespace(create=lambda **kw: SimpleNamespace(
                model="claude-opus-5-5", stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ok")],
                usage=SimpleNamespace(input_tokens=12, output_tokens=3)))
    monkeypatch.setattr(anthropic, "Anthropic", Fake)
    r = runner.invoke(cli.app, ["llm-check"])
    assert r.exit_code == 0 and "sk-ant-...wxyz" in r.output and "SUPERSECRET" not in r.output
    assert "OK: claude-opus-5-5 answered 'ok'" in r.output and "in=12 out=3" in r.output


# ---------------------------------------------------------------- the turn budget grows with the scope
def test_turn_budget_is_flat_up_to_three_experiments_then_grows_and_is_capped():
    from chaosai.analyzer import TURN_CAP, effective_turns
    cfg = AnalyzerCfg(max_turns=16, turns_per_experiment=4)
    assert [effective_turns(cfg, n) for n in (1, 2, 3)] == [16, 16, 16]
    assert effective_turns(cfg, 6) == 28 and effective_turns(cfg, 10) == 44
    assert effective_turns(cfg, 500) == TURN_CAP == 80
    assert effective_turns(AnalyzerCfg(max_turns=16, turns_per_experiment=0), 50) == 16        # the old flat behaviour is still available


def test_the_system_prompt_tells_the_model_to_batch_its_tool_calls():
    assert "several tools in one turn" in analyzer.SYSTEM
