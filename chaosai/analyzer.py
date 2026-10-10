"""The analyzer agent: a tool-using loop that explains the stored experiments, with every claim checked.

    model  --calls-->  evidence tools (read stored data only)  --results-->  model
    model  --calls-->  submit_analysis
                          |-- valid    -> done
                          '-- invalid  -> the reasons go back as a tool error and the model fixes its answer
                                          (capped by analyzer.max_rejects; e.g. it cited a number no tool returned)

Opus 5.5 rejects a forced tool_choice, so the loop uses tool_choice "auto" and the prompt tells the model
to finish by calling submit_analysis; a turn that ends without it counts as a rejection.

Anything that goes wrong (no credentials, no network, refusal, truncation, the model never producing a valid
answer) falls back to the deterministic rule-based analyzer and records why. A validated answer is cached
on disk, keyed by the model, prompt and a digest of every input file, so a demo survives no Wi-Fi.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from pydantic import ValidationError

from . import rules
from .analysis import SUBMIT_TOOL, Analysis, Recorder, validate
from .config import AnalyzerCfg, llm_label
from .evidence import TOOLS, Evidence
from .llmclient import make_client, model_label
from .planner import FALLBACK_BETA

MAX_TOOL_OUTPUT_CHARS = 24_000
TURN_CAP = 80                      # however large the scope, the loop never runs longer than this


def effective_turns(cfg: AnalyzerCfg, n_experiments: int) -> int:
    """Turn budget: `max_turns` covers up to three experiments, plus `turns_per_experiment` for each one beyond that.
    A model that makes one tool call per turn (gpt-oss does; Gemini batches several) needs about three turns per
    experiment just to read its data, so a flat budget would cut it off before it could answer."""
    return min(TURN_CAP, cfg.max_turns + cfg.turns_per_experiment * max(0, n_experiments - 3))


class AnalyzerError(Exception):
    pass


SYSTEM = """You are a site-reliability engineer analysing chaos experiments run against a small containerised app.

The app: a voting site. The journeys a user sees are `vote` (POST a vote to the vote service) and `result` (load the
result page). The data path is vote -> redis (a list used as a queue) -> worker -> postgres -> result. The result page is
static HTML that gets its numbers over a websocket, so the `result` journey succeeding does NOT prove the database or the
worker are healthy. Every service runs as a single replica.

Each experiment injects one real fault into one container (kill, pause, stop, cpu-limit, memory-limit, network-delay,
network-loss, traffic-spike) and records what happened. Your job: for each experiment say what happened and why, then
rank the system's weaknesses and propose concrete fixes.

How to work
- You may request several tools in one turn, and you should: read many experiments with parallel calls instead of one
  call per turn, because the number of turns is limited.
- Use the tools to read the stored data. Start with list_experiments and get_graph, then look at each experiment
  (get_experiment, get_metrics, get_gauges, get_container_events as needed). Check the `quality` field of every experiment:
  data recorded by an older build, a failed rollback, or an experiment ended by the kill switch is weaker evidence; say so
  rather than leaning on it.
- Separate what was measured from what you infer. If a cause is an inference (not directly measured), say so in the text.
- EVIDENCE RULE: every number in an evidence `claim` must be a value a data tool returned for that experiment (or a
  graph/baseline value). Quote it as returned; do not recompute, round differently, or estimate. A claim with a number no
  tool returned will be rejected. Numbers you propose (replica counts, memory sizes) belong in `change` / `rationale`, not in
  evidence claims.
- Rank weaknesses most severe first (critical, high, medium, low). Severity follows user impact and data integrity.
- Every weakness and every fix needs at least one evidence item. Fixes must be concrete: a docker-compose snippet, a config
  change or a code change, not advice. Say which weaknesses a fix addresses (by exact title).
- hypothesis_held: true/false only if the experiment has a stored hypothesis; otherwise null.
- Finish by calling submit_analysis once. If it is rejected, read the reasons, correct them, and call it again."""


def _prompt(ids: list[str]) -> str:
    return ("Analyse these experiments (use their exact ids):\n" + "\n".join(f"- {i}" for i in ids)
            + "\n\nCover every one of them in `experiments`, then give the ranked `weaknesses` and the `fixes`.")


def _cache_path(cache_dir: Path, cfg: AnalyzerCfg, ids: list[str], fingerprint: str) -> Path:
    raw = "\x00".join((llm_label(cfg), cfg.llm_effort, SYSTEM, _prompt(ids), fingerprint,
                       json.dumps(SUBMIT_TOOL, sort_keys=True), json.dumps(TOOLS, sort_keys=True)))
    return Path(cache_dir) / f"analysis-{hashlib.sha256(raw.encode()).hexdigest()[:24]}.json"


def _block(b, name, default=None):
    return getattr(b, name, default) if not isinstance(b, dict) else b.get(name, default)


def run_agent(ev: Evidence, ids: list[str], client, cfg: AnalyzerCfg) -> tuple[Analysis, dict]:
    """The tool loop. Raises AnalyzerError if it cannot produce a valid analysis."""
    rec = Recorder(ev)
    has_hyp = {i: bool(ev._exps[i].experiment.hypothesis) for i in ev.ids()}
    tools = [*TOOLS, SUBMIT_TOOL]
    messages: list[dict] = [{"role": "user", "content": _prompt(ids)}]
    stats = {"turns": 0, "tool_calls": 0, "rejected": 0, "input_tokens": 0, "output_tokens": 0}
    last_problems: list[str] = []

    limit = effective_turns(cfg, len(ids))
    for _ in range(limit):
        stats["turns"] += 1
        resp = client.beta.messages.create(
            model=cfg.llm_model, max_tokens=16000, system=SYSTEM, tools=tools, messages=messages,
            betas=[FALLBACK_BETA], fallbacks="default",                 # re-run on the fallback model if declined
            output_config={"effort": cfg.llm_effort})
        usage = getattr(resp, "usage", None)
        stats["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        stats["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        if resp.stop_reason == "refusal":                               # branch on stop_reason before reading content
            raise AnalyzerError("the model refused the request")
        if resp.stop_reason == "max_tokens":
            raise AnalyzerError("the response was cut off (max_tokens)")

        messages.append({"role": "assistant", "content": resp.content})  # unchanged, thinking blocks included
        uses = [b for b in resp.content if _block(b, "type") == "tool_use"]
        results: list[dict] = []
        if not uses:
            stats["rejected"] += 1
            last_problems = ["the turn ended without calling submit_analysis"]
            if stats["rejected"] > cfg.max_rejects:
                raise AnalyzerError("the model kept ending turns without submitting an analysis")
            messages.append({"role": "user", "content": "You have not submitted yet. Use the tools if you still need data, "
                                                        "then finish by calling submit_analysis."})
            continue

        for u in uses:
            name, args, uid = _block(u, "name"), _block(u, "input") or {}, _block(u, "id")
            if name != "submit_analysis":
                stats["tool_calls"] += 1
                out = json.dumps(rec.call(name, args))
                if len(out) > MAX_TOOL_OUTPUT_CHARS:
                    out = json.dumps({"error": "result too large; ask for a narrower window or journey"})
                results.append({"type": "tool_result", "tool_use_id": uid, "content": out})
                continue
            try:
                analysis = Analysis(**args, generated_by="llm", model=cfg.llm_model)
                problems = validate(analysis, rec.seen, ids, has_hyp)
            except (ValidationError, TypeError) as e:
                problems = [f"the submission does not match the schema: {str(e)[:400]}"]
            if not problems:
                stats["tool_calls_total"] = rec.calls
                return analysis, stats
            stats["rejected"] += 1
            last_problems = problems
            if stats["rejected"] > cfg.max_rejects:
                raise AnalyzerError("no valid analysis after %d rejections; last problems: %s" % (stats["rejected"], "; ".join(problems)[:600]))
            results.append({"type": "tool_result", "tool_use_id": uid, "is_error": True,
                            "content": "REJECTED. Fix these and call submit_analysis again:\n- " + "\n- ".join(problems[:20])})
        messages.append({"role": "user", "content": results})

    raise AnalyzerError(f"no valid analysis within {limit} turns" + (f"; last problems: {'; '.join(last_problems)[:300]}" if last_problems else ""))


def _by_rules(ev: Evidence, ids: list[str], reason: str | None) -> Analysis:
    rec = Recorder(ev)
    a = rules.analyse(rec, ids)
    problems = validate(a, rec.seen, ids, {i: bool(ev._exps[i].experiment.hypothesis) for i in ev.ids()})
    a.fallback_reason = reason
    if problems:                                                         # a bug in the rules, never hidden
        a.notes.append("RULE-BASED OUTPUT FAILED ITS OWN VALIDATION: " + "; ".join(problems)[:300])
    return a


def analyze(ev: Evidence, ids: list[str], cfg: AnalyzerCfg, use_llm: bool = True, client=None,
            cache_dir: Path | None = None) -> Analysis:
    if not ids:
        raise AnalyzerError("no experiments to analyse (run some with `chaos run` / `chaos agent` / the dashboard)")
    if not (use_llm and cfg.use_llm):
        return _by_rules(ev, ids, "LLM disabled (--offline or analyzer.use_llm: false)")

    cache = _cache_path(cache_dir, cfg, ids, ev.fingerprint(ids)) if cache_dir else None
    if cache and cache.exists():
        try:
            a = Analysis.model_validate_json(json.loads(cache.read_text(encoding="utf-8"))["analysis"])
            a.notes = a.notes + [f"served from cache ({cache.name}); no API call"]
            return a
        except (ValueError, KeyError):
            pass                                                         # corrupt cache entry: ignore it
    try:
        if client is None:
            client = make_client(cfg.provider, cfg.members)
        a, stats = run_agent(ev, ids, client, cfg)
        a.model = model_label(client, a.model)                           # a pool reports the member(s) that really answered
    except AnalyzerError as e:
        return _by_rules(ev, ids, str(e))
    except Exception as e:                                               # no credentials, network, API error, SDK change
        if "authentication method" in str(e):
            return _by_rules(ev, ids, "no Anthropic credentials found (set ANTHROPIC_API_KEY)")
        return _by_rules(ev, ids, f"{type(e).__name__}: {str(e)[:400]}")
    a.notes.append(f"{stats['turns']} turn(s), {stats['tool_calls']} data-tool call(s), {stats['rejected']} rejected submission(s), "
                   f"{stats['input_tokens']} input / {stats['output_tokens']} output tokens")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"model": cfg.llm_model, "analysis": a.model_dump_json(), "saved_at": time.time()}), encoding="utf-8")
    return a
