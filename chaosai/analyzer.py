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
from .analysis import SEVERITIES, SUBMIT_TOOL, Analysis, Evid, Recorder, validate
from .config import AnalyzerCfg, llm_label
from .evidence import TOOLS, Evidence
from .llmclient import make_client, model_label
from .planner import FALLBACK_BETA

BULKY_KEYS = ("per_second", "rows")     # the long per-second series; every tool also returns a complete summary
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


BATCH_NOTE = """

You are analysing one batch of a larger set of experiments; other batches are analysed separately and the results are merged.
- Do NOT call list_experiments (it is not offered). Use exactly the experiment ids you are given. Start with get_graph and
  get_baseline, then read each of your experiments.
- Keep your answer about these experiments only. Name each weakness after the underlying design issue, not after the fault
  (for example "Single worker caps pipeline throughput"), in short plain words, so the same issue found in another batch gets
  the same title and merges."""


def _shrink(limit: int):
    """Make a tool result fit `limit` characters: drop the long per-second series (the summary stays), else refuse."""
    def shrink(out: dict) -> dict:
        if len(json.dumps(out, separators=(",", ":"))) <= limit:
            return out
        dropped = [k for k in BULKY_KEYS if k in out]
        if dropped:
            kept = {k: v for k, v in out.items() if k not in dropped}
            kept["omitted"] = f"{', '.join(dropped)} left out to keep the answer small; the summary above is complete"
            if len(json.dumps(kept, separators=(",", ":"))) <= limit:
                return kept
        return {"error": "result too large; ask for a narrower window or journey"}
    return shrink


def _prompt(ids: list[str]) -> str:
    return ("Analyse these experiments (use their exact ids):\n" + "\n".join(f"- {i}" for i in ids)
            + "\n\nCover every one of them in `experiments`, then give the ranked `weaknesses` and the `fixes`.")


def _cache_path(cache_dir: Path, cfg: AnalyzerCfg, ids: list[str], fingerprint: str, batch: bool = False) -> Path:
    parts = [llm_label(cfg), cfg.llm_effort, SYSTEM, _prompt(ids), fingerprint,
             json.dumps(SUBMIT_TOOL, sort_keys=True), json.dumps(TOOLS, sort_keys=True)]
    if batch:                                         # a batch is cached on its own; an unbatched key is unchanged so older entries still hit
        parts += ["batch", BATCH_NOTE]
    elif cfg.batch_size:
        parts += [f"batched:{cfg.batch_size}"]
    raw = "\x00".join(parts)
    return Path(cache_dir) / f"analysis-{hashlib.sha256(raw.encode()).hexdigest()[:24]}.json"


def _block(b, name, default=None):
    return getattr(b, name, default) if not isinstance(b, dict) else b.get(name, default)


def run_agent(ev: Evidence, ids: list[str], client, cfg: AnalyzerCfg, batch: bool = False) -> tuple[Analysis, dict]:
    """The tool loop. Raises AnalyzerError if it cannot produce a valid analysis. `batch`: one batch of a larger set."""
    rec = Recorder(ev, shrink=_shrink(cfg.max_tool_output_chars))
    has_hyp = {i: bool(ev._exps[i].experiment.hypothesis) for i in ev.ids()}
    tools = [*(t for t in TOOLS if not (batch and t["name"] == "list_experiments")), SUBMIT_TOOL]
    system = SYSTEM + BATCH_NOTE if batch else SYSTEM
    messages: list[dict] = [{"role": "user", "content": _prompt(ids)}]
    stats = {"turns": 0, "tool_calls": 0, "rejected": 0, "input_tokens": 0, "output_tokens": 0}
    last_problems: list[str] = []

    limit = effective_turns(cfg, len(ids))
    for _ in range(limit):
        stats["turns"] += 1
        resp = client.beta.messages.create(
            model=cfg.llm_model, max_tokens=16000, system=system, tools=tools, messages=messages,
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
                out = json.dumps(rec.call(name, args), separators=(",", ":"))
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


def _norm(title: str) -> str:
    return " ".join(title.lower().split())


def _merge(parts: list[Analysis], ids: list[str]) -> Analysis:
    """Combine per-batch analyses. Each part was validated on its own (every cited number came from a tool the model called),
    so merging only joins and de-duplicates: experiments in scope order; a weakness or fix with the same title (ignoring case and
    spacing) becomes one entry with the union of services and evidence; weaknesses stay ranked most severe first."""
    by_id = {e.experiment_id: e for p in parts for e in p.experiments}
    weak: dict[str, object] = {}
    for p in parts:
        for w in p.weaknesses:
            k = _norm(w.title)
            if k not in weak:
                weak[k] = w.model_copy(deep=True)
                continue
            c = weak[k]
            c.services = list(dict.fromkeys([*c.services, *w.services]))
            c.evidence = _dedupe_evidence([*c.evidence, *w.evidence])
            if SEVERITIES.index(w.severity) < SEVERITIES.index(c.severity):
                c.severity, c.description = w.severity, w.description
    weaknesses = sorted(weak.values(), key=lambda w: SEVERITIES.index(w.severity))
    canon = {k: w.title for k, w in weak.items()}
    fixes: dict[str, object] = {}
    for p in parts:
        for f in p.fixes:
            f = f.model_copy(deep=True)
            f.addresses = list(dict.fromkeys(canon.get(_norm(t), t) for t in f.addresses))
            k = _norm(f.title)
            if k not in fixes:
                fixes[k] = f
                continue
            c = fixes[k]
            c.addresses = list(dict.fromkeys([*c.addresses, *f.addresses]))
            c.evidence = _dedupe_evidence([*c.evidence, *f.evidence])
    return Analysis(summary=" ".join(p.summary.strip() for p in parts if p.summary.strip()),
                    experiments=[by_id[i] for i in ids if i in by_id], weaknesses=list(weaknesses), fixes=list(fixes.values()),
                    generated_by="llm" if any(p.generated_by == "llm" for p in parts) else "rules")


def _dedupe_evidence(items: list[Evid]) -> list[Evid]:
    seen, out = set(), []
    for e in items:
        if (e.experiment_id, e.claim) not in seen:
            seen.add((e.experiment_id, e.claim))
            out.append(e)
    return out


def _read_cached(path: Path | None) -> Analysis | None:
    if path is None or not path.exists():
        return None
    try:
        return Analysis.model_validate_json(json.loads(path.read_text(encoding="utf-8"))["analysis"])
    except (ValueError, KeyError):
        return None                                                      # corrupt cache entry: ignore it


def _write_cached(path: Path | None, cfg: AnalyzerCfg, a: Analysis) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": cfg.llm_model, "analysis": a.model_dump_json(), "saved_at": time.time()}), encoding="utf-8")


def run_batched(ev: Evidence, ids: list[str], client, cfg: AnalyzerCfg, cache_dir: Path | None = None) -> tuple[Analysis, dict]:
    """Analyse `cfg.batch_size` experiments at a time, each batch in a fresh small conversation. A pool client starts every
    batch on its next member, so the work is spread across keys and providers, and each request stays far below a free tier's
    per-minute token limit. A batch the model cannot answer is analysed by the rules and said so; a batch already answered
    (cached) is not asked again, so a failed run resumes where it stopped. Raises AnalyzerError if no batch got an LLM answer."""
    size = cfg.batch_size
    batches = [ids[i:i + size] for i in range(0, len(ids), size)]
    parts: list[Analysis] = []
    stats = {"turns": 0, "tool_calls": 0, "rejected": 0, "input_tokens": 0, "output_tokens": 0, "batches": len(batches), "from_cache": 0, "by_rules": 0}
    reasons: list[str] = []
    for b in batches:
        path = _cache_path(cache_dir, cfg, b, ev.fingerprint(b), batch=True) if cache_dir else None
        part = _read_cached(path)
        if part is not None:
            stats["from_cache"] += 1
            parts.append(part)
            continue
        try:
            part, st = run_agent(ev, b, client, cfg, batch=True)
        except Exception as e:                                           # AnalyzerError, quota, network: this batch only
            reasons.append(f"{', '.join(b)}: {str(e)[:200]}")
            stats["by_rules"] += 1
            parts.append(_by_rules(ev, b, str(e)))
            continue
        for k in ("turns", "tool_calls", "rejected", "input_tokens", "output_tokens"):
            stats[k] += st.get(k, 0)
        _write_cached(path, cfg, part)
        parts.append(part)
    if stats["by_rules"] == len(batches):
        raise AnalyzerError("no batch got an answer from the model: " + "; ".join(reasons)[:500])
    a = _merge(parts, ids)
    if reasons:
        a.summary += f" NOTE: {stats['by_rules']} of {len(batches)} batch(es) were analysed by the rule-based analyzer because the model could not answer them ({'; '.join(reasons)[:300]})."
        a.notes.append("partial fallback: " + "; ".join(reasons)[:500])
    return a, stats


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
        if cfg.batch_size and len(ids) > cfg.batch_size:
            a, stats = run_batched(ev, ids, client, cfg, cache_dir)
        else:
            a, stats = run_agent(ev, ids, client, cfg)
        a.model = model_label(client, a.model)                           # a pool reports the member(s) that really answered
    except AnalyzerError as e:
        return _by_rules(ev, ids, str(e))
    except Exception as e:                                               # no credentials, network, API error, SDK change
        if "authentication method" in str(e):
            return _by_rules(ev, ids, "no Anthropic credentials found (set ANTHROPIC_API_KEY)")
        return _by_rules(ev, ids, f"{type(e).__name__}: {str(e)[:400]}")
    if stats.get("batches"):
        a.notes.append(f"{stats['batches']} batch(es) of up to {cfg.batch_size} experiment(s): {stats['from_cache']} from cache, {stats['by_rules']} by rules")
    a.notes.append(f"{stats['turns']} turn(s), {stats['tool_calls']} data-tool call(s), {stats['rejected']} rejected submission(s), "
                   f"{stats['input_tokens']} input / {stats['output_tokens']} output tokens")
    if cache and not stats.get("by_rules"):                              # a partly rule-based result is not cached: a rerun retries those batches
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"model": cfg.llm_model, "analysis": a.model_dump_json(), "saved_at": time.time()}), encoding="utf-8")
    return a
