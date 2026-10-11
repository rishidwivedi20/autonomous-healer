import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chaosai import planner
from chaosai.discovery import discover
from chaosai.models import Experiment, ExperimentResult, Stats
from chaosai.planner import (MAX_PER_FAULT, MAX_PER_TARGET, PlannerError, candidates, heuristic_plan, llm_plan,
                             load_history, make_plan, select, to_experiments)

COMPOSE = Path(__file__).resolve().parent.parent / "deploy" / "docker-compose.yml"


@pytest.fixture(scope="module")
def graph():
    return discover(COMPOSE)[0]


def hist(target, fault, aborted=False):
    return {"id": "x", "target": target, "fault": fault, "aborted": aborted, "abort_reason": None,
            "worst_journey_error_rate": 0.0, "recovery_s": 0.0, "recovered": True, "restarts": {}}


# ---------------------------------------------------------------- heuristic
def test_candidates_respect_applicability(graph):
    pairs = {(c.target, c.fault) for c in candidates(graph)}
    assert ("redis", "memory-limit") in pairs and ("db", "memory-limit") in pairs
    assert ("vote", "memory-limit") not in pairs and ("worker", "memory-limit") not in pairs     # stateless
    assert {t for t, f in pairs if f == "traffic-spike"} == {"vote", "result"}                  # user-facing only
    assert ("worker", "network-loss") not in pairs                                              # nothing calls it
    assert ("redis", "network-loss") in pairs


def test_score_is_criticality_times_fragility_times_untested_times_fault_weight(graph):
    c = next(c for c in candidates(graph) if (c.target, c.fault) == ("redis", "kill"))
    f = c.factors
    assert f == {"criticality": 5.0, "fragility": 3.5, "untested": 1.5, "fault_weight": 1.0}
    assert c.score == pytest.approx(5 * 3.5 * 1.5 * 1.0)


def test_most_critical_fragile_stateful_services_rank_first(graph):
    top = [(c.target, c.fault) for c in candidates(graph)[:3]]
    assert top[0] == ("db", "kill")
    assert {t for t, _ in top} <= {"db", "redis"}


def test_ranking_is_deterministic(graph):
    a = [(c.id, c.target, c.fault, c.score) for c in candidates(graph)]
    b = [(c.id, c.target, c.fault, c.score) for c in candidates(graph)]
    assert a == b and [c.id for c in candidates(graph)] == list(range(len(a)))


def test_testing_something_lowers_its_priority(graph):
    before = next(c for c in candidates(graph) if (c.target, c.fault) == ("db", "kill"))
    once = next(c for c in candidates(graph, [hist("db", "kill")]) if (c.target, c.fault) == ("db", "kill"))
    twice = next(c for c in candidates(graph, [hist("db", "kill")] * 2) if (c.target, c.fault) == ("db", "kill"))
    assert before.score > once.score > twice.score
    assert once.factors["untested"] == 1.0 and twice.factors["untested"] == 0.7
    assert "tested 1x before" in once.reason and "never tested" in before.reason


def test_plan_is_diverse(graph):
    plan = heuristic_plan(graph, n=6)
    assert len(plan.experiments) == 6
    targets = [p.target for p in plan.experiments]
    faults = [p.fault for p in plan.experiments]
    assert max(targets.count(t) for t in set(targets)) <= MAX_PER_TARGET
    assert max(faults.count(f) for f in set(faults)) <= MAX_PER_FAULT
    assert [p.rank for p in plan.experiments] == [1, 2, 3, 4, 5, 6]
    assert plan.planner == "heuristic" and plan.fallback_reason is None


def test_plan_never_contains_duplicates_or_unknown_faults(graph):
    plan = heuristic_plan(graph, n=20)
    keys = [(p.target, p.fault) for p in plan.experiments]
    assert len(keys) == len(set(keys))
    assert all(p.fault in planner.FAULT_WEIGHT for p in plan.experiments)
    assert all(5 <= p.duration_s <= 60 for p in plan.experiments)


def test_reasons_cite_only_graph_facts(graph):
    c = next(c for c in candidates(graph) if (c.target, c.fault) == ("worker", "kill"))
    assert "no healthcheck" in c.reason and "single replica" in c.reason and "stateful" not in c.reason
    assert "vote" not in c.reason         # worker has no dependents; nothing invented


def test_to_experiments_are_valid_experiments(graph):
    exps = to_experiments(heuristic_plan(graph, n=3), stamp="T")
    assert all(isinstance(e, Experiment) and e.id.startswith("T-") for e in exps)
    assert exps[0].hypothesis and exps[0].duration_s <= 60


# ---------------------------------------------------------------- history
def test_load_history_reads_real_results_only(tmp_path):
    def result(id_, dry=False):
        return ExperimentResult(experiment=Experiment(id=id_, target="redis", fault="kill"), started_at=1, ended_at=2,
                                dry_run=dry, baseline=Stats(), during=Stats(n=2, error_rate=0.25)).model_dump_json()
    (tmp_path / "a.json").write_text(result("a"))
    (tmp_path / "b.json").write_text(result("b", dry=True))             # dry run: never counts as tested
    (tmp_path / "graph.json").write_text('{"services": {}}')             # not an experiment
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "plan.json").write_text("[]")
    h = load_history(tmp_path)
    assert [x["id"] for x in h] == ["a"] and h[0]["worst_journey_error_rate"] == 0.25


# ---------------------------------------------------------------- LLM path (stub client; not a live API call)
class StubClient:
    def __init__(self, text=None, stop_reason="end_turn", error=None, blocks=None):
        self.calls = []
        outer = self

        class Messages:
            def create(self, **kw):
                outer.calls.append(kw)
                if error:
                    raise error
                content = blocks if blocks is not None else [SimpleNamespace(type="text", text=text)]
                return SimpleNamespace(stop_reason=stop_reason, content=content)
        self.beta = SimpleNamespace(messages=Messages())


def reply(*items):
    return json.dumps({"experiments": [{"candidate_id": i, "duration_s": d, "hypothesis": f"h{i}", "reason": f"r{i}"}
                                       for i, d in items]})


def test_llm_plan_uses_model_ordering_and_text(graph):
    cands = candidates(graph)
    client = StubClient(reply((cands[3].id, 25), (cands[0].id, 40)))
    plan = llm_plan(graph, [], n=6, client=client)
    assert plan.planner == "llm" and plan.model == "claude-opus-5-5" and plan.fallback_reason is None
    assert [(p.target, p.fault) for p in plan.experiments] == [(cands[3].target, cands[3].fault),
                                                               (cands[0].target, cands[0].fault)]
    assert plan.experiments[0].duration_s == 25 and plan.experiments[0].hypothesis == f"h{cands[3].id}"
    assert plan.experiments[0].params == cands[3].params                  # params come from the catalog, not the model
    assert plan.experiments[0].risk_score == cands[3].score


def test_request_shape_matches_opus_5_5_rules(graph):
    cands = candidates(graph)
    client = StubClient(reply((cands[0].id, 30)))
    llm_plan(graph, [hist("redis", "kill")], n=4, client=client)
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["fallbacks"] == "default"
    fmt = kw["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False
    assert kw["output_config"]["effort"] == "medium"
    for banned in ("temperature", "top_p", "top_k", "budget_tokens", "thinking", "prefill"):
        assert banned not in kw                                            # removed/rejected on Opus 5.5
    assert [m["role"] for m in kw["messages"]] == ["user"]                 # no assistant prefill
    body = json.loads(kw["messages"][0]["content"])
    assert body["choose_up_to"] == 4 and body["stored_results"][0]["target"] == "redis"
    assert len(body["candidates"]) <= 14 and "services" in body and body["critical_path"]


@pytest.mark.parametrize("bad, why", [
    ("not json at all", "not the expected JSON"),
    ('{"experiments": []}', "empty plan"),
    ('{"wrong": 1}', "not the expected JSON"),
    (reply((9999, 30)), "unknown candidate_id"),
    (reply((0, 30), (0, 30)), "duplicate"),
    (reply((0, 3)), "outside 5..60"),
    (reply((0, 600)), "outside 5..60"),
    (reply(*[(i, 30) for i in range(7)]), "at most"),
    (json.dumps({"experiments": [{"candidate_id": 0, "duration_s": 30, "hypothesis": " ", "reason": "r"}]}), "empty"),
])
def test_invalid_llm_output_falls_back_with_reason(graph, bad, why):
    plan = make_plan(graph, [], n=6, client=StubClient(bad))
    assert plan.planner == "heuristic" and why in plan.fallback_reason
    assert len(plan.experiments) == 6                                       # a usable plan regardless


def test_llm_cannot_choose_outside_the_candidate_pool(graph):
    far = candidates(graph)[-1].id                                          # exists, but beyond the 14 offered
    assert far >= 14
    with pytest.raises(PlannerError, match="unknown candidate_id"):
        llm_plan(graph, [], client=StubClient(reply((far, 30))))


def test_refusal_truncation_and_missing_text_fall_back(graph):
    for client, why in [(StubClient("x", stop_reason="refusal"), "refused"),
                        (StubClient("x", stop_reason="max_tokens"), "truncated"),
                        (StubClient(blocks=[SimpleNamespace(type="thinking", thinking="")]), "no text block")]:
        plan = make_plan(graph, [], client=client)
        assert plan.planner == "heuristic" and why in plan.fallback_reason


def test_api_errors_fall_back(graph):
    plan = make_plan(graph, [], client=StubClient(error=ConnectionError("no wifi")))
    assert plan.planner == "heuristic" and "ConnectionError" in plan.fallback_reason


def test_missing_credentials_message_is_readable(graph, monkeypatch):
    """The real SDK raises this exact TypeError at request time when no credentials resolve (seen live).
    Simulated here so the test can never make a real, billable API call on a machine that has a login."""
    import anthropic

    class NoCreds:
        def __init__(self, *a, **k):
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

        @staticmethod
        def _create(**kw):
            raise TypeError('"Could not resolve authentication method. Expected one of api_key, auth_token, or credentials to be set."')
    monkeypatch.setattr(anthropic, "Anthropic", NoCreds)
    plan = make_plan(graph, [], n=3)
    assert plan.planner == "heuristic" and "ANTHROPIC_API_KEY" in plan.fallback_reason


def test_offline_flag_never_touches_the_client(graph):
    client = StubClient(reply((0, 30)))
    plan = make_plan(graph, [], use_llm=False, client=client)
    assert plan.planner == "heuristic" and client.calls == [] and "offline" in plan.fallback_reason


# ---------------------------------------------------------------- cache (demo survives no Wi-Fi)
def test_validated_response_is_cached_and_replayed_without_the_api(graph, tmp_path):
    cands = candidates(graph)
    first = StubClient(reply((cands[1].id, 20)))
    p1 = llm_plan(graph, [], client=first, cache_dir=tmp_path)
    assert len(first.calls) == 1 and list(tmp_path.glob("plan-*.json"))
    broken = StubClient(error=ConnectionError("offline"))
    p2 = llm_plan(graph, [], client=broken, cache_dir=tmp_path)
    assert broken.calls == [] and p2.planner == "llm"
    assert [e.hypothesis for e in p2.experiments] == [e.hypothesis for e in p1.experiments]
    assert any("cache" in n for n in p2.notes)


def test_invalid_responses_are_not_cached(graph, tmp_path):
    make_plan(graph, [], client=StubClient("garbage"), cache_dir=tmp_path)
    assert not list(tmp_path.glob("plan-*.json"))


def test_cache_key_changes_with_inputs(graph, tmp_path):
    cands = candidates(graph)
    llm_plan(graph, [], client=StubClient(reply((cands[0].id, 20))), cache_dir=tmp_path)
    c2 = StubClient(reply((cands[0].id, 20)))
    llm_plan(graph, [hist("redis", "kill")], client=c2, cache_dir=tmp_path)    # new history -> new prompt -> new call
    assert len(c2.calls) == 1 and len(list(tmp_path.glob("plan-*.json"))) == 2


# ---------------------------------------------------------------- a pool retries a bad answer on another member
class PoolStub:
    """Answers each new conversation from a script, like a pool whose members take turns; `members` is what marks it a pool."""
    def __init__(self, *answers, members=3):
        self.members, self.calls, self.answers = list(range(members)), 0, list(answers)
        outer = self

        class Messages:
            def create(self, **kw):
                outer.calls += 1
                stop, text = outer.answers.pop(0)
                return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])
        self.beta = SimpleNamespace(messages=Messages())


def test_a_truncated_answer_is_retried_on_the_next_pool_member(graph):
    cands = candidates(graph)
    client = PoolStub(("max_tokens", "{"), ("end_turn", reply((cands[0].id, 30))))
    plan = llm_plan(graph, [], n=6, client=client)
    assert plan.planner == "llm" and client.calls == 2 and any("attempt 1 was unusable" in n and "truncated" in n for n in plan.notes)


def test_every_pool_attempt_failing_falls_back_listing_each_reason(graph):
    client = PoolStub(("max_tokens", "{"), ("end_turn", "not json"), ("end_turn", "{}"))
    plan = make_plan(graph, [], n=6, client=client)
    assert plan.planner == "heuristic" and client.calls == 3 and "truncated" in plan.fallback_reason and "not the expected JSON" in plan.fallback_reason


def test_a_single_provider_client_is_not_retried(graph):
    client = StubClient("{}", stop_reason="max_tokens")
    plan = make_plan(graph, [], n=6, client=client)
    assert plan.planner == "heuristic" and len(client.calls) == 1
