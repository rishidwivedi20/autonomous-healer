"""Batched analysis: each batch of experiments is a fresh small conversation (so no request nears a free tier's per-minute
token cap and a pool can hand every batch to a different member), a failed batch falls back alone, answered batches are cached
so a rerun resumes, and the merge joins the parts without losing the validator's guarantees."""
import json

import pytest

from chaosai import analyzer
from chaosai.analysis import Analysis, Evid, Fix, Recorder, Weakness
from chaosai.analyzer import AnalyzerError, _merge, _shrink, analyze
from chaosai.config import AnalyzerCfg
from chaosai.evidence import Evidence
from chaosai.rules import analyse
from tests.fixtures_results import KILL_ID, SPIKE_ID, make_results
from tests.test_analyzer import Scripted, resp, submit, tu

IDS = [SPIKE_ID, KILL_ID]
CFG = AnalyzerCfg(batch_size=1)


@pytest.fixture
def ev(tmp_path):
    return Evidence(make_results(tmp_path))


def reads(i, prefix):
    return resp(tu("get_graph", {}, f"{prefix}-graph"), tu("get_experiment", {"experiment_id": i}, f"{prefix}-e"),
                tu("get_gauges", {"experiment_id": i}, f"{prefix}-g"), tu("get_metrics", {"experiment_id": i, "window": "during"}, f"{prefix}-m"))


def answer(ev, ids):
    a = analyse(Recorder(ev), ids)
    return submit(a.model_dump(exclude={"generated_by", "model", "fallback_reason", "notes"}))


def test_every_batch_is_a_fresh_conversation_without_list_experiments_and_the_parts_merge(ev):
    client = Scripted(reads(SPIKE_ID, "a"), answer(ev, [SPIKE_ID]), reads(KILL_ID, "b"), answer(ev, [KILL_ID]))
    a = analyze(ev, IDS, CFG, client=client)
    assert len(client.calls) == 4
    assert len(client.calls[0]["messages"]) == 1 and len(client.calls[2]["messages"]) == 1       # batch 2 does not carry batch 1's history
    for call in client.calls:
        assert "list_experiments" not in [t["name"] for t in call["tools"]]
        assert "one batch of a larger set" in call["system"]
    assert a.generated_by == "llm" and [e.experiment_id for e in a.experiments] == IDS
    titles = [w.title.lower() for w in a.weaknesses]
    assert len(titles) == len(set(titles))
    order = ["critical", "high", "medium", "low"]
    assert [order.index(w.severity) for w in a.weaknesses] == sorted(order.index(w.severity) for w in a.weaknesses)
    assert any("2 batch(es)" in n for n in a.notes)


def test_a_batch_the_model_cannot_answer_falls_back_alone_and_is_not_hidden(ev, tmp_path):
    cache = tmp_path / "cache"
    client = Scripted(reads(SPIKE_ID, "a"), answer(ev, [SPIKE_ID]), ConnectionError("offline"))
    a = analyze(ev, IDS, CFG, client=client, cache_dir=cache)
    assert a.generated_by == "llm" and [e.experiment_id for e in a.experiments] == IDS
    assert "1 of 2 batch(es) were analysed by the rule-based analyzer" in a.summary and any("partial fallback" in n for n in a.notes)
    assert len(list(cache.glob("analysis-*.json"))) == 1                  # the answered batch only; the partly rule-based whole is not cached


def test_a_rerun_resumes_from_the_batches_already_answered(ev, tmp_path):
    cache = tmp_path / "cache"
    analyze(ev, IDS, CFG, client=Scripted(reads(SPIKE_ID, "a"), answer(ev, [SPIKE_ID]), ConnectionError("offline")), cache_dir=cache)
    again = Scripted(reads(KILL_ID, "b"), answer(ev, [KILL_ID]))
    a = analyze(ev, IDS, CFG, client=again, cache_dir=cache)
    assert len(again.calls) == 2                                          # only the second batch is asked
    assert a.generated_by == "llm" and "by rules" in " ".join(a.notes) and any("1 from cache" in n for n in a.notes)
    assert "analysed by the rule-based analyzer" not in a.summary
    assert len(list(cache.glob("analysis-*.json"))) == 3                  # two batches and the whole
    served = analyze(ev, IDS, CFG, client=Scripted(ConnectionError("must not be called")), cache_dir=cache)
    assert any("served from cache" in n for n in served.notes)


def test_when_no_batch_gets_an_answer_the_whole_analysis_falls_back_with_the_reasons(ev):
    a = analyze(ev, IDS, CFG, client=Scripted(ConnectionError("down"), ConnectionError("down")))
    assert a.generated_by == "rules" and "no batch got an answer" in a.fallback_reason and "down" in a.fallback_reason


def test_few_enough_experiments_use_the_single_conversation(ev):
    client = Scripted(resp(tu("get_graph", {}, "g")), answer(ev, IDS))
    analyze(ev, IDS, AnalyzerCfg(batch_size=2), client=client)
    assert "list_experiments" in [t["name"] for t in client.calls[0]["tools"]] and "one batch" not in client.calls[0]["system"]


def _w(title, sev, svc, claim, eid=SPIKE_ID):
    return Weakness(title=title, severity=sev, services=svc, description=f"{title} ({sev})", evidence=[Evid(experiment_id=eid, claim=claim)])


def test_merge_joins_same_titled_weaknesses_and_fixes_and_ranks_them():
    p1 = Analysis(summary="one", experiments=[], weaknesses=[_w("Single worker", "medium", ["worker"], "queue 300")], fixes=[
        Fix(title="More workers", change="replicas: 3", rationale="r", addresses=["Single worker"], evidence=[Evid(experiment_id=SPIKE_ID, claim="queue 300")])])
    p2 = Analysis(summary="two", experiments=[], weaknesses=[_w("single  WORKER", "high", ["redis"], "lag 31.6", KILL_ID), _w("No limits", "low", ["db"], "x")], fixes=[
        Fix(title="more workers", change="replicas: 3", rationale="r", addresses=["single  WORKER"], evidence=[Evid(experiment_id=KILL_ID, claim="lag 31.6")])])
    m = _merge([p1, p2], [])
    assert [w.title for w in m.weaknesses] == ["Single worker", "No limits"]
    w = m.weaknesses[0]
    assert w.severity == "high" and w.services == ["worker", "redis"] and len(w.evidence) == 2 and w.description.endswith("(high)")
    assert len(m.fixes) == 1 and m.fixes[0].addresses == ["Single worker"] and len(m.fixes[0].evidence) == 2
    assert m.summary == "one two"


def test_a_long_tool_result_loses_its_per_second_rows_but_keeps_the_summary(ev):
    full = ev.get_metrics(SPIKE_ID)
    assert "per_second" in full
    small = _shrink(len(json.dumps(full, separators=(",", ":"))) - 1)(full)
    assert "per_second" not in small and small["summary"] == full["summary"] and "omitted" in small
    assert _shrink(10)(full) == {"error": "result too large; ask for a narrower window or journey"}


def test_numbers_in_trimmed_rows_cannot_be_cited_afterwards(ev):
    full = ev.get_metrics(SPIKE_ID)
    rec = Recorder(ev, shrink=_shrink(len(json.dumps(full, separators=(",", ":"))) - 1))
    kept = rec.call("get_metrics", {"experiment_id": SPIKE_ID})
    rec_full = Recorder(ev)
    rec_full.call("get_metrics", {"experiment_id": SPIKE_ID})
    assert "per_second" not in kept and rec.seen.allowed(SPIKE_ID) < rec_full.seen.allowed(SPIKE_ID)


def test_the_shipped_config_batches_the_analyzer_for_small_requests():
    from pathlib import Path
    from chaosai import config
    c = config.load(Path(__file__).parent.parent / "config.yaml")
    assert c.analyzer.batch_size == 1 and c.analyzer.max_tool_output_chars <= 4000
