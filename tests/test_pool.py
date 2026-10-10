"""The LLM pool: rotation across runs, sticky conversations, failover, cooldowns, and the real analyzer/planner loops
running through it with the real adapters over scripted HTTP (unit tests; no experiment data is synthetic)."""
import json
import re
from types import SimpleNamespace

import httpx
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from chaosai import analyzer, cli, config, planner, pool as pool_mod
from chaosai.config import AnalyzerCfg, PlannerCfg, PoolMember, llm_label, pool_label
from chaosai.evidence import Evidence
from chaosai.gemini import GeminiClient
from chaosai.llmclient import make_client, model_label
from chaosai.mistral import MistralClient
from chaosai.openrouter import OpenRouterClient
from chaosai.pool import PoolClient, PoolError, make_member_client
from tests.fixtures_results import make_results
from tests.test_gemini import IDS, call_resp, graph_of, payload, read_calls
from tests.test_mistral import chat, tc

runner = CliRunner()
A = PoolMember(provider="mistral", model="m-a")
B = PoolMember(provider="openrouter", model="v/b:free")
C = PoolMember(provider="mistral", model="m-c")
G = PoolMember(provider="gemini", model="g-1")
USER = [{"role": "user", "content": "x"}]
MID = [{"role": "user", "content": "x"}, {"role": "assistant", "content": [SimpleNamespace(type="text", text="t")]},
       {"role": "user", "content": "next"}]


class Failure(Exception):
    def __init__(self, msg="boom", status=None, retry_after_s=None):
        super().__init__(msg)
        self.status, self.retry_after_s = status, retry_after_s


def ok(text="ok"):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                           usage=SimpleNamespace(input_tokens=1, output_tokens=1))


class Fake:
    """A member client: returns or raises what it was scripted with and records every call."""
    def __init__(self, *script):
        self.script, self.calls = list(script), []
        self.beta = SimpleNamespace(messages=self)

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


def make_pool(tmp_path, clock, members, fakes, alternate=False):
    """Pure round-robin by default, so the rotation tests are about rotation; provider alternation has its own tests."""
    return PoolClient(members, state_path=tmp_path / "pool.json", clock=clock, make=lambda m: fakes[pool_label(m)], alternate_providers=alternate)


def state(tmp_path):
    return json.loads((tmp_path / "pool.json").read_text(encoding="utf-8"))


def first(pool, **kw):
    return pool.create(messages=USER, max_tokens=10, **kw)


# ---------------------------------------------------------------- rotation and stickiness
def test_each_new_run_starts_on_the_next_member_even_across_separate_processes(tmp_path, clock):
    fakes = {pool_label(m): Fake(ok(), ok()) for m in (A, B, C)}
    order = []
    for _ in range(4):                                                     # a fresh PoolClient per run, like separate CLI commands
        p = make_pool(tmp_path, clock, [A, B, C], fakes)
        first(p)
        order.append(next(i for i, m in enumerate((A, B, C)) if len(fakes[pool_label(m)].calls) > order.count(i)))
    assert order == [0, 1, 2, 0]


def test_a_conversation_stays_on_its_member_for_every_turn(tmp_path, clock):
    fakes = {pool_label(m): Fake(ok(), ok(), ok()) for m in (A, B)}
    state_file = tmp_path / "pool.json"
    state_file.write_text(json.dumps({"next": 1}), encoding="utf-8")        # so the new run starts on B
    p = make_pool(tmp_path, clock, [A, B], fakes)
    first(p)
    p.create(messages=MID, max_tokens=10)
    p.create(messages=MID + MID, max_tokens=10)
    assert len(fakes[pool_label(B)].calls) == 3 and len(fakes[pool_label(A)].calls) == 0


def test_the_callers_model_is_ignored_and_everything_else_is_forwarded(tmp_path, clock):
    fake = Fake(ok())
    p = make_pool(tmp_path, clock, [A], {pool_label(A): fake})
    p.create(model="claude-opus-5-5", messages=USER, max_tokens=77, system="sys", tools=[{"name": "t"}], output_config={"effort": "low"},
             betas=["b"], fallbacks="default")
    kw = fake.calls[0]
    assert kw["model"] == "m-a" and kw["messages"] == USER and kw["max_tokens"] == 77 and kw["system"] == "sys"
    assert kw["tools"] == [{"name": "t"}] and kw["output_config"] == {"effort": "low"} and kw["betas"] == ["b"]


def test_used_models_name_the_members_that_really_answered(tmp_path, clock):
    p = make_pool(tmp_path, clock, [A, B], {pool_label(A): Fake(ok()), pool_label(B): Fake(ok())})
    assert model_label(p, "claude-opus-5-5") == "claude-opus-5-5"            # nothing answered yet: the configured name
    first(p)
    assert p.used_models == ["mistral:m-a"] and model_label(p, "x") == "mistral:m-a"
    assert model_label(object(), "plain-model") == "plain-model"             # a single client has no such list


# ---------------------------------------------------------------- failover and cooldowns
def test_a_failing_member_hands_the_same_request_to_the_next_and_is_left_alone_until_it_cools_down(tmp_path, clock):
    fakes = {pool_label(A): Fake(Failure("quota", 429), ok()), pool_label(B): Fake(ok(), ok()), pool_label(C): Fake(ok(), ok())}
    calls = lambda m: len(fakes[pool_label(m)].calls)

    def run():
        first(make_pool(tmp_path, clock, [A, B, C], fakes))                  # a fresh PoolClient per run, like separate commands

    run()                                                                    # starts on A, which fails: B answers the same request
    assert (calls(A), calls(B), calls(C)) == (1, 1, 0)
    run()                                                                    # rotation moved past B: C
    assert (calls(A), calls(B), calls(C)) == (1, 1, 1)
    run()                                                                    # rotation points at A, but A is cooling: skipped, B answers
    assert (calls(A), calls(B), calls(C)) == (1, 2, 1)
    run()                                                                    # C
    assert (calls(A), calls(B), calls(C)) == (1, 2, 2)
    clock.t += 200                                                           # past the 120 s default cooldown for a 429
    run()                                                                    # rotation is back at A, which is welcome again
    assert (calls(A), calls(B), calls(C)) == (2, 2, 2)


@pytest.mark.parametrize("exc,seconds", [(Failure("x", 429), 120), (Failure("x", 404), 6 * 3600), (Failure("x", 401), 3600), (Failure("x", 503), 60),
                                         (Failure("x", None), 60), (Failure("x", 429, 7200), 7200), (Failure("x", 429, 10 ** 7), 24 * 3600)])
def test_cooldown_lengths(tmp_path, clock, exc, seconds):
    fakes = {pool_label(A): Fake(exc), pool_label(B): Fake(ok())}
    first(make_pool(tmp_path, clock, [A, B], fakes))
    assert state(tmp_path)["cooldown"][pool_label(A)]["until"] - clock.t == pytest.approx(seconds)


def test_an_unusable_member_is_skipped_with_a_reason_and_the_rest_still_work(tmp_path, clock):
    def make(m):
        if m is A:
            raise PoolError("MISTRAL_API_KEY_2 is not set")
        return Fake(ok())
    p = PoolClient([A, B], state_path=tmp_path / "pool.json", clock=clock, make=make)
    first(p)
    assert p.used_models == ["openrouter:v/b:free"] and "not set" in state(tmp_path)["cooldown"][pool_label(A)]["reason"]


def test_a_defect_in_our_own_code_is_not_masked_by_failover(tmp_path, clock):
    fakes = {pool_label(A): Fake(TypeError("bad kwarg")), pool_label(B): Fake(ok())}
    with pytest.raises(TypeError):
        first(make_pool(tmp_path, clock, [A, B], fakes))
    assert fakes[pool_label(B)].calls == [] and not (tmp_path / "pool.json").exists()       # no cooldown was recorded either


def test_when_every_member_fails_the_error_lists_each_reason(tmp_path, clock):
    fakes = {pool_label(A): Fake(Failure("quota exhausted for m-a", 429)), pool_label(B): Fake(Failure("overloaded", 503))}
    with pytest.raises(PoolError) as e:
        first(make_pool(tmp_path, clock, [A, B], fakes))
    msg = str(e.value)
    assert "no pool member could answer" in msg and "mistral:m-a: quota exhausted" in msg and "openrouter:v/b:free: overloaded" in msg


def test_when_every_member_is_cooling_the_error_says_for_how_long(tmp_path, clock):
    fakes = {pool_label(A): Fake(Failure("q", 429)), pool_label(B): Fake(Failure("o", 503))}
    with pytest.raises(PoolError):
        first(make_pool(tmp_path, clock, [A, B], fakes))
    with pytest.raises(PoolError) as e:
        first(make_pool(tmp_path, clock, [A, B], fakes))                     # no member is even called this time
    assert "unavailable for about 2 more min" in str(e.value) and "unavailable for about 1 more min" in str(e.value)
    assert len(fakes[pool_label(A)].calls) == 1 and len(fakes[pool_label(B)].calls) == 1


# ---------------------------------------------------------------- a run that is already under way
def test_a_run_under_way_fails_over_to_a_member_that_can_take_it_over_and_never_into_gemini(tmp_path, clock):
    fakes = {pool_label(A): Fake(ok(), Failure("overloaded", 503)), pool_label(G): Fake(ok()), pool_label(B): Fake(ok())}
    p = make_pool(tmp_path, clock, [A, G, B], fakes)
    first(p)                                                                 # starts on A
    p.create(messages=MID, max_tokens=10)                                    # A fails mid-run: Gemini is skipped, B takes over
    assert fakes[pool_label(G)].calls == [] and fakes[pool_label(B)].calls[0]["messages"] == MID
    assert p.used_models == ["mistral:m-a", "openrouter:v/b:free"]


def test_a_gemini_run_can_hand_over_to_others_but_a_run_with_only_gemini_left_fails_clearly(tmp_path, clock):
    fakes = {pool_label(G): Fake(ok(), Failure("quota", 429, 3600)), pool_label(B): Fake(ok())}
    p = make_pool(tmp_path, clock, [G, B], fakes)
    first(p)
    p.create(messages=MID, max_tokens=10)
    assert len(fakes[pool_label(B)].calls) == 1                              # Gemini failed mid-run, the other member continued
    only = {pool_label(G): Fake(ok(), Failure("quota", 429, 3600))}
    p2 = PoolClient([G], state_path=tmp_path / "other.json", clock=clock, make=lambda m: only[pool_label(m)])
    first(p2)
    with pytest.raises(PoolError, match="no pool member could answer"):
        p2.create(messages=MID, max_tokens=10)


# ---------------------------------------------------------------- saved state is only a hint
def test_a_corrupt_state_file_is_ignored_and_rewritten(tmp_path, clock):
    (tmp_path / "pool.json").write_text("{not json", encoding="utf-8")
    first(make_pool(tmp_path, clock, [A, B], {pool_label(A): Fake(ok()), pool_label(B): Fake(ok())}))
    assert state(tmp_path)["next"] == 1


def test_an_unwritable_state_path_never_breaks_a_run(tmp_path, clock):
    blocker = tmp_path / "file.txt"
    blocker.write_text("x", encoding="utf-8")
    p = PoolClient([A], state_path=blocker / "sub" / "pool.json", clock=clock, make=lambda m: Fake(ok()))
    assert first(p).content[0].text == "ok"


def test_an_empty_pool_is_an_error():
    with pytest.raises(PoolError, match="no members"):
        PoolClient([])


# ---------------------------------------------------------------- building real member clients
def test_a_member_uses_its_own_key_variable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_2", "second-key")
    c = make_member_client(PoolMember(provider="gemini", model="g-1", key_env="GEMINI_API_KEY_2"))
    assert isinstance(c, GeminiClient) and c._key == "second-key"


def test_a_member_whose_key_variable_is_unset_is_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY_2", raising=False)
    with pytest.raises(PoolError, match="GEMINI_API_KEY_2 is not set"):
        make_member_client(PoolMember(provider="gemini", model="g-1", key_env="GEMINI_API_KEY_2"))


def test_member_clients_per_provider(monkeypatch):
    for var in ("MISTRAL_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, "k")
    assert isinstance(make_member_client(A), MistralClient) and isinstance(make_member_client(B), OpenRouterClient)
    import anthropic
    assert isinstance(make_member_client(PoolMember(provider="anthropic", model="claude-opus-5-5")), anthropic.Anthropic)


def test_an_anthropic_member_without_a_key_is_a_clean_error_not_a_crash(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(PoolError, match="no Anthropic credentials"):
        make_member_client(PoolMember(provider="anthropic", model="claude-opus-5-5"))


def test_make_client_builds_a_pool_from_the_members(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAOSAI_RESULTS", str(tmp_path))
    p = make_client("pool", [A, B])
    assert isinstance(p, PoolClient) and p.state_path == tmp_path / "llm_pool.json" and len(p.members) == 2


# ---------------------------------------------------------------- llm-check support
def test_check_tries_every_member_regardless_of_cooldowns_and_status_shows_them(tmp_path, clock):
    fakes = {pool_label(A): Fake(Failure("quota", 429, 1800)), pool_label(B): Fake(ok("ok"), ok("ok"))}
    p = make_pool(tmp_path, clock, [A, B], fakes)
    first(p)                                                                 # A fails and is put on a 30 min cooldown; B answers
    status = {label: (left, why) for label, left, why in p.status()}
    assert status[pool_label(A)][0] == pytest.approx(1800) and "quota" in status[pool_label(A)][1] and status[pool_label(B)][0] == 0
    fakes[pool_label(A)].script = [ok("ok")]
    results = p.check()
    assert [(label, worked) for label, worked, _ in results] == [(pool_label(A), True), (pool_label(B), True)]
    assert "answered 'ok'" in results[0][2]


# ---------------------------------------------------------------- the real loops through the pool, with the real adapters
def http(*script, cls, **kw):
    queue = list(script)
    reqs = []

    def handler(request):
        reqs.append(request)
        item = queue.pop(0)
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)
    c = cls(key="test-key", transport=httpx.MockTransport(handler), sleep=lambda s: None, **kw)
    c.reqs = reqs
    return c


def read_call_msg(prefix="call"):
    return chat(calls=[tc(n, a, f"{prefix}{i:05d}") for i, (n, a) in enumerate(read_calls())])


def submit_msg(ev, id_="submit001"):
    return chat(calls=[tc("submit_analysis", payload(ev), id_)])


def pool_cfg(*members):
    return AnalyzerCfg(provider="pool", members=list(members))


def test_the_analyzer_fails_over_at_the_start_when_the_first_member_has_no_quota(tmp_path, clock):
    ev = Evidence(make_results(tmp_path))
    quota = httpx.Response(429, text='{"error": {"details": [{"quotaId": "PerDay", "quotaValue": "20"}, {"retryDelay": "3600s"}]}}')
    gem = http(quota, cls=GeminiClient)
    mis = http(read_call_msg(), submit_msg(ev), cls=MistralClient)
    p = PoolClient([G, A], state_path=tmp_path / "pool.json", clock=clock, make=lambda m: gem if m is G else mis)
    a = analyzer.analyze(ev, IDS, pool_cfg(G, A), client=p)
    assert a.generated_by == "llm" and a.model == "mistral:m-a" and a.weaknesses
    assert state(tmp_path)["cooldown"][pool_label(G)]["until"] - clock.t == pytest.approx(3600)       # the API's own retryDelay


def test_the_analyzer_moves_to_another_member_halfway_through_a_run(tmp_path, clock):
    ev = Evidence(make_results(tmp_path))
    mis = http(read_call_msg("mid"), httpx.Response(503, json={"message": "overloaded"}), cls=MistralClient, max_retries=0)
    opr = http(submit_msg(ev), cls=OpenRouterClient)
    p = PoolClient([A, B], state_path=tmp_path / "pool.json", clock=clock, make=lambda m: mis if m is A else opr)
    a = analyzer.analyze(ev, IDS, pool_cfg(A, B), client=p)
    assert a.generated_by == "llm" and a.model == "mistral:m-a, openrouter:v/b:free"
    sent = json.loads(opr.reqs[0].content)["messages"]
    assert [m["role"] for m in sent][:3] == ["system", "user", "assistant"] and sum(m["role"] == "tool" for m in sent) == len(read_calls())
    assert {m["name"] for m in sent if m["role"] == "tool"} == {"get_graph", "get_experiment", "get_gauges", "get_metrics"}   # names from the history
    assert "reasoning" not in sent[2]                                                                   # no foreign raw fields leak across providers


def test_a_gemini_run_handed_to_mistral_gets_tool_call_ids_mistral_accepts(tmp_path, clock):
    ev = Evidence(make_results(tmp_path))
    gem = http(call_resp(*read_calls(), sig="S1"), httpx.Response(503, json={"error": {"message": "busy"}}), cls=GeminiClient, max_retries=0)
    mis = http(submit_msg(ev), cls=MistralClient)
    p = PoolClient([G, A], state_path=tmp_path / "pool.json", clock=clock, make=lambda m: gem if m is G else mis)
    a = analyzer.analyze(ev, IDS, pool_cfg(G, A), client=p)
    assert a.generated_by == "llm" and a.model == "gemini:g-1, mistral:m-a"
    sent = json.loads(mis.reqs[0].content)["messages"]
    call_ids = [c["id"] for c in sent[2]["tool_calls"]]
    assert all(re.fullmatch(r"[A-Za-z0-9]{9}", i) for i in call_ids) and len(set(call_ids)) == len(read_calls())
    assert [m["tool_call_id"] for m in sent if m["role"] == "tool"] == call_ids                         # every result still matches its call


def test_one_providers_raw_reasoning_fields_are_not_replayed_to_another_after_a_handover(tmp_path, clock):
    thinking = chat("", calls=[tc("get_graph", {}, "idAAAAAA1")])
    thinking["choices"][0]["message"]["reasoning"] = "private chain of thought"
    thinking["choices"][0]["message"]["reasoning_details"] = [{"type": "reasoning.text", "text": "step"}]
    opr = http(thinking, httpx.Response(503, json={"error": {"message": "busy"}}), cls=OpenRouterClient, max_retries=0)
    mis = http(chat("done"), cls=MistralClient)
    p = PoolClient([B, A], state_path=tmp_path / "pool.json", clock=clock, make=lambda m: opr if m is B else mis)
    turn1 = p.create(messages=USER, max_tokens=10)
    p.create(messages=[USER[0], {"role": "assistant", "content": turn1.content},
                       {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "idAAAAAA1", "content": "{}"}]}], max_tokens=10)
    sent = json.loads(mis.reqs[0].content)["messages"]
    assert sent[1] == {"role": "assistant", "content": "", "tool_calls": [tc("get_graph", {}, "idAAAAAA1")]}     # rebuilt, no reasoning fields
    assert sent[2] == {"role": "tool", "tool_call_id": "idAAAAAA1", "name": "get_graph", "content": "{}"}


def test_every_member_failing_falls_back_to_the_rules_and_lists_why(tmp_path, clock):
    ev = Evidence(make_results(tmp_path))
    mis = http(httpx.Response(401, json={"message": "Unauthorized"}), cls=MistralClient)
    opr = http(httpx.Response(402, json={"error": {"message": "Insufficient credits"}}), cls=OpenRouterClient)
    p = PoolClient([A, B], state_path=tmp_path / "pool.json", clock=clock, make=lambda m: mis if m is A else opr)
    a = analyzer.analyze(ev, IDS, pool_cfg(A, B), client=p)
    assert a.generated_by == "rules" and "PoolError: no pool member could answer" in a.fallback_reason
    assert "Unauthorized" in a.fallback_reason and "Insufficient credits" in a.fallback_reason and a.weaknesses


def test_a_pooled_analysis_is_cached_and_the_cache_depends_on_the_members(tmp_path, clock):
    ev = Evidence(make_results(tmp_path))
    mis = http(read_call_msg(), submit_msg(ev), cls=MistralClient)
    cfg = pool_cfg(A)
    analyzer.analyze(ev, IDS, cfg, client=PoolClient([A], state_path=tmp_path / "pool.json", clock=clock, make=lambda m: mis), cache_dir=tmp_path / "cache")
    again = analyzer.analyze(ev, IDS, cfg, client=PoolClient([A], state_path=tmp_path / "pool.json", clock=clock, make=lambda m: pytest.fail("cache expected")),
                             cache_dir=tmp_path / "cache")
    assert again.generated_by == "llm" and any("served from cache" in n for n in again.notes)
    fp = ev.fingerprint(IDS)
    assert analyzer._cache_path(tmp_path, pool_cfg(A), IDS, fp) != analyzer._cache_path(tmp_path, pool_cfg(A, B), IDS, fp)


def test_the_planner_fails_over_and_records_the_member_that_answered(tmp_path, clock):
    g = graph_of(tmp_path)
    cands = planner.candidates(g, [], None)
    pick = {"experiments": [{"candidate_id": cands[0].id, "duration_s": 20, "hypothesis": "h", "reason": "r"}]}
    mis = http(httpx.Response(429, headers={"retry-after": "7200"}, json={"message": "limit"}), cls=MistralClient)
    opr = http(chat(json.dumps(pick)), cls=OpenRouterClient)
    p = PoolClient([A, B], state_path=tmp_path / "pool.json", clock=clock, make=lambda m: mis if m is A else opr)
    plan = planner.llm_plan(g, [], n=3, client=p, cfg=PlannerCfg(provider="pool", members=[A, B]))
    assert plan.planner == "llm" and plan.model == "openrouter:v/b:free" and plan.experiments


# ---------------------------------------------------------------- config
def test_a_pool_needs_members_and_other_providers_must_not_have_any():
    with pytest.raises(ValidationError, match="provider is pool but `members` is empty"):
        AnalyzerCfg(provider="pool")
    with pytest.raises(ValidationError, match="only used with provider: pool"):
        AnalyzerCfg(provider="gemini", llm_model="gemini-3.5-flash", members=[G])
    assert PlannerCfg(provider="pool", members=[G, B]).llm_model == "claude-opus-5-5"          # llm_model is simply ignored for a pool


def test_members_are_checked_like_single_providers():
    with pytest.raises(ValidationError, match="provider is gemini but llm_model is 'claude-opus-5-5'"):
        PoolMember(provider="gemini", model="claude-opus-5-5")
    with pytest.raises(ValidationError, match="OpenRouter model ids look like vendor/model"):
        PoolMember(provider="openrouter", model="gpt-oss")
    with pytest.raises(ValidationError):
        PoolMember(provider="pool", model="x")


@pytest.mark.parametrize("name,good", [("GEMINI_API_KEY_2", True), ("OPENROUTER_API_KEY_TEAM", True), ("MISTRAL_KEY", True),
                                       ("PATH", False), ("gemini_api_key", False), ("EVIL_GEMINI_API_KEY", False), ("GEMINI-KEY", False)])
def test_key_variable_names_are_restricted(name, good):
    if good:
        assert PoolMember(provider="gemini", model="g-1", key_env=name).key_env == name
    else:
        with pytest.raises(ValidationError, match="key_env"):
            PoolMember(provider="gemini", model="g-1", key_env=name)


def test_llm_label_identifies_the_members_and_their_keys():
    assert llm_label(AnalyzerCfg(provider="gemini", llm_model="gemini-3.5-flash")) == "gemini:gemini-3.5-flash"
    keyed = PoolMember(provider="gemini", model="g-1", key_env="GEMINI_API_KEY_2")
    assert llm_label(pool_cfg(G, keyed)) == "pool[gemini:g-1,gemini:g-1@GEMINI_API_KEY_2]"


YAML = """
planner: {provider: pool, members: &m [{provider: gemini, model: gemini-3-flash-preview}, {provider: openrouter, model: "openai/gpt-oss-120b:free", key_env: OPENROUTER_API_KEY}]}
analyzer: {provider: pool, members: *m}
"""


def test_one_yaml_list_can_be_shared_by_the_planner_and_analyzer(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(YAML, encoding="utf-8")
    c = config.load(p)
    assert [pool_label(m) for m in c.planner.members] == [pool_label(m) for m in c.analyzer.members]
    assert c.analyzer.members[1].model == "openai/gpt-oss-120b:free" and yaml.safe_load(YAML)["planner"]["provider"] == "pool"


# ---------------------------------------------------------------- llm-check --provider pool
def write_config(tmp_path, monkeypatch, text=YAML):
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    monkeypatch.setenv("CHAOSAI_CONFIG", str(p))
    monkeypatch.setattr(cli, "RESULTS", tmp_path / "results")


def test_llm_check_pool_without_a_pool_explains_how_to_configure_one():
    r = runner.invoke(cli.app, ["llm-check", "--provider", "pool"])
    assert r.exit_code == 1 and "provider: pool" in r.output and "members:" in r.output


def test_llm_check_pool_no_send_lists_members_and_saved_cooldowns_without_any_request(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch)
    (tmp_path / "results").mkdir()
    label = "gemini:gemini-3-flash-preview"
    (tmp_path / "results" / "llm_pool.json").write_text(json.dumps({"cooldown": {label: {"until": 9e12, "reason": "quota exhausted"}}}), encoding="utf-8")
    real = pool_mod.PoolClient
    monkeypatch.setattr(pool_mod, "PoolClient", lambda members, state_path=None: real(
        members, state_path=state_path, make=lambda m: pytest.fail("--no-send must not build a client or send a request")))
    r = runner.invoke(cli.app, ["llm-check", "--provider", "pool", "--no-send"])
    assert r.exit_code == 0 and "2 pool member(s)" in r.output and f"{label}: on cooldown" in r.output and "quota exhausted" in r.output
    assert "openrouter:openai/gpt-oss-120b:free@OPENROUTER_API_KEY: available" in r.output


def test_llm_check_pool_sends_one_request_per_member_and_reports_each(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch)
    fakes = {"gemini:gemini-3-flash-preview": Fake(ok("ok")), "openrouter:openai/gpt-oss-120b:free@OPENROUTER_API_KEY": Fake(Failure("Insufficient credits", 402))}
    real = pool_mod.PoolClient
    monkeypatch.setattr(pool_mod, "PoolClient", lambda members, state_path=None: real(members, state_path=state_path, make=lambda m: fakes[pool_label(m)]))
    r = runner.invoke(cli.app, ["llm-check", "--provider", "pool"])
    assert r.exit_code == 0
    assert "OK   gemini:gemini-3-flash-preview: answered 'ok'" in r.output and "FAIL openrouter:openai/gpt-oss-120b:free@OPENROUTER_API_KEY: Insufficient credits" in r.output


def test_llm_check_pool_fails_when_no_member_answers(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch)
    real = pool_mod.PoolClient
    monkeypatch.setattr(pool_mod, "PoolClient", lambda members, state_path=None: real(members, state_path=state_path, make=lambda m: Fake(Failure("down", 503))))
    r = runner.invoke(cli.app, ["llm-check", "--provider", "pool"])
    assert r.exit_code == 1 and "no pool member answered" in r.output


def test_a_cached_plan_reports_the_member_that_really_produced_it_not_the_configured_default(tmp_path, clock):
    g = graph_of(tmp_path)
    cands = planner.candidates(g, [], None)
    pick = {"experiments": [{"candidate_id": cands[0].id, "duration_s": 20, "hypothesis": "h", "reason": "r"}]}
    opr = http(chat(json.dumps(pick)), cls=OpenRouterClient)
    cfg = PlannerCfg(provider="pool", members=[B], llm_model="claude-opus-5-5")
    p = PoolClient([B], state_path=tmp_path / "pool.json", clock=clock, make=lambda m: opr)
    first_plan = planner.llm_plan(g, [], n=3, client=p, cache_dir=tmp_path / "cache", cfg=cfg)
    again = planner.llm_plan(g, [], n=3, client=None, cache_dir=tmp_path / "cache", cfg=cfg)          # no client: it must come from the cache
    assert first_plan.model == "openrouter:v/b:free" and again.model == "openrouter:v/b:free"
    assert any("served from cache" in n for n in again.notes)


# ---------------------------------------------------------------- consecutive steps use different providers
GM1 = PoolMember(provider="gemini", model="g-1")
GM2 = PoolMember(provider="gemini", model="g-2")


def answered_by(tmp_path, clock, members, fakes, alternate=True):
    """Run one new conversation through a fresh pool (like a separate command) and say which member answered it."""
    p = make_pool(tmp_path, clock, members, fakes, alternate)
    first(p)
    return p.used_models[0]


def provider_sequence(tmp_path, clock, members, fakes, runs, alternate=True):
    return [answered_by(tmp_path, clock, members, fakes, alternate).split(":")[0] for _ in range(runs)]


def test_consecutive_runs_use_different_providers_when_more_than_one_is_available(tmp_path, clock):
    members = [GM1, GM2, B]
    fakes = {pool_label(m): Fake(*[ok()] * 6) for m in members}
    assert provider_sequence(tmp_path, clock, members, fakes, 6) == ["gemini", "openrouter", "gemini", "openrouter", "gemini", "openrouter"]


def test_without_the_preference_two_models_of_one_provider_can_answer_back_to_back(tmp_path, clock):
    members = [GM1, GM2, B]
    fakes = {pool_label(m): Fake(*[ok()] * 6) for m in members}
    assert provider_sequence(tmp_path, clock, members, fakes, 3, alternate=False) == ["gemini", "gemini", "openrouter"]


def test_a_single_provider_keeps_rotating_through_its_own_models(tmp_path, clock):
    members = [GM1, GM2]
    fakes = {pool_label(m): Fake(*[ok()] * 4) for m in members}
    assert [answered_by(tmp_path, clock, members, fakes) for _ in range(4)] == ["gemini:g-1", "gemini:g-2", "gemini:g-1", "gemini:g-2"]


def test_the_preferred_provider_is_skipped_when_it_is_unavailable(tmp_path, clock):
    members = [GM1, B]
    fakes = {pool_label(GM1): Fake(ok(), ok(), ok()), pool_label(B): Fake(Failure("no credits", 402))}
    assert [answered_by(tmp_path, clock, members, fakes) for _ in range(3)] == ["gemini:g-1"] * 3     # OpenRouter failed once and is cooling
    assert len(fakes[pool_label(B)].calls) == 1


def test_the_last_provider_is_saved_so_separate_commands_alternate_too(tmp_path, clock):
    members = [GM1, B]
    fakes = {pool_label(m): Fake(ok(), ok()) for m in members}
    first(make_pool(tmp_path, clock, members, fakes, alternate=True))
    assert state(tmp_path)["last_provider"] == "gemini"
    first(make_pool(tmp_path, clock, members, fakes, alternate=True))
    assert state(tmp_path)["last_provider"] == "openrouter"


def test_a_run_under_way_ignores_the_preference_and_stays_on_its_member(tmp_path, clock):
    members = [GM1, B]
    fakes = {pool_label(GM1): Fake(ok(), ok(), ok()), pool_label(B): Fake(ok())}
    p = make_pool(tmp_path, clock, members, fakes, alternate=True)
    first(p)
    p.create(messages=MID, max_tokens=10)
    p.create(messages=MID + MID, max_tokens=10)
    assert len(fakes[pool_label(GM1)].calls) == 3 and fakes[pool_label(B)].calls == []
