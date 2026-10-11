"""Experiment planner: rank which failures to try first.

1. heuristic_plan  deterministic risk score, always available (no network, no key):
       score = criticality x fragility x untested_bonus x fault_weight
2. llm_plan        Claude re-ranks the heuristic's candidates and writes falsifiable hypotheses.
                   It can only pick from the candidate list (by id), so it cannot invent targets,
                   faults or parameters; params stay at the catalog defaults. Output is requested
                   as strict JSON and validated; ANY problem (no credentials, network, refusal,
                   bad JSON, unknown/duplicate id, out-of-range duration) falls back to the heuristic
                   and records why. Validated responses are cached on disk so a demo survives no Wi-Fi.

Everything here only proposes. Each experiment still passes safety.Guard.validate at run time.
Hypotheses are predictions derived from the dependency graph, not measurements; the reasons cite
only graph facts and stored results.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from .models import Experiment, ExperimentResult, Graph, Plan, PlannedExperiment, Service

from .config import DEFAULT_FAULTS, PlannerCfg, llm_label
from .llmclient import make_client, model_label

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

FAULT_WEIGHT = {
    "kill": 1.0, "stop": 0.8, "pause": 0.8, "network-delay": 0.7, "network-loss": 0.7,
    "memory-limit": 0.6, "traffic-spike": 0.6, "cpu-limit": 0.5,
}
MAX_PER_TARGET = 2
MAX_PER_FAULT = 3


MAX_ATTEMPTS = 3                  # model answers tried per plan when the client is a pool


class PlannerError(Exception):
    pass


@dataclass
class Candidate:
    id: int
    target: str
    fault: str
    params: dict
    duration_s: int
    score: float
    factors: dict[str, float]
    reason: str
    hypothesis: str
    tested: int = 0


# ---------------------------------------------------------------- history
def load_history(results_dir: Path) -> list[dict]:
    """Summaries of real, completed experiments in results/ (dry runs excluded)."""
    out = []
    for f in sorted(Path(results_dir).glob("*.json")):
        try:
            r = ExperimentResult.model_validate_json(f.read_text(encoding="utf-8"))
        except (ValidationError, ValueError):
            continue                      # graph.json, baseline.json, plan.json, ...
        if r.dry_run:
            continue
        vote_err = max((s.error_rate or 0.0 for s in r.during_by_journey.values()), default=r.during.error_rate or 0.0)
        out.append({
            "id": r.experiment.id, "target": r.experiment.target, "fault": r.experiment.fault,
            "aborted": r.aborted, "abort_reason": r.abort_reason,
            "worst_journey_error_rate": round(vote_err, 3), "recovery_s": r.recovery_s,
            "recovered": r.recovered, "restarts": r.restarts,
        })
    return out


# ---------------------------------------------------------------- heuristic
def applicable(svc: Service, graph: Graph) -> list[str]:
    faults = ["kill", "stop", "pause", "cpu-limit", "network-delay"]
    if svc.fan_in > 0:
        faults.append("network-loss")
    if svc.stateful:
        faults.append("memory-limit")
    if svc.entrypoint:
        faults.append("traffic-spike")
    return faults


def criticality(svc: Service) -> float:
    return 1.0 + (2.0 if svc.on_critical_path else 0.0) + svc.fan_in


def fragility(svc: Service) -> float:
    return (1.0 + (1.0 if svc.replicas < 2 else 0.0) + (1.0 if not svc.healthcheck else 0.0)
            + (1.0 if svc.stateful else 0.0) + (0.5 if not svc.has_limits else 0.0)
            + (0.5 if svc.restart_policy in ("", "no") else 0.0))


def untested_bonus(times_tested: int) -> float:
    return 1.5 if times_tested == 0 else (1.0 if times_tested == 1 else 0.7)


def _facts(svc: Service, graph: Graph) -> list[str]:
    deps = [s.name for s in graph.services.values() if svc.name in s.depends_on]
    facts = []
    if svc.on_critical_path:
        facts.append("on the user critical path")
    if deps:
        facts.append(f"{len(deps)} direct dependent(s): {', '.join(sorted(deps))}")
    if svc.replicas < 2:
        facts.append("single replica")
    if svc.stateful:
        facts.append("stateful")
    if not svc.healthcheck:
        facts.append("no healthcheck")
    if not svc.has_limits:
        facts.append("no resource limits")
    return facts


def _hypothesis(fault: str, svc: Service, graph: Graph) -> str:
    deps = sorted(s.name for s in graph.services.values() if svc.name in s.depends_on)
    who = f"its dependents ({', '.join(deps)})" if deps else "user-facing requests"
    t = svc.name
    return {
        "kill": f"A crash of {t} disrupts {who} until the restart policy ({svc.restart_policy}) revives it.",
        "stop": f"With {t} stopped (no automatic restart), {who} fail until it is started again.",
        "pause": f"A frozen {t} looks alive but never answers; {who} stall or time out.",
        "network-delay": f"Delaying {t}'s network slows {who}; timeouts appear once delay exceeds the caller's timeout.",
        "network-loss": f"Packet loss on {t} causes retries and sporadic errors for {who}.",
        "cpu-limit": f"Throttling {t}'s CPU raises latency for {who} without outright errors.",
        "memory-limit": f"A tight memory cap may OOM-kill {t}; recovery shows whether state and health handling survive a restart.",
        "traffic-spike": f"A traffic surge on {t} raises p95 latency; errors appear if capacity is exceeded.",
    }[fault]


def candidates(graph: Graph, history: list[dict] | None = None, faults: dict | None = None) -> list[Candidate]:
    faults = faults or DEFAULT_FAULTS            # per-fault duration/params (config.yaml `faults`)
    tested: dict[tuple[str, str], int] = {}
    for h in history or []:
        tested[(h["target"], h["fault"])] = tested.get((h["target"], h["fault"]), 0) + 1
    out: list[Candidate] = []
    for svc in graph.services.values():
        crit, frag = criticality(svc), fragility(svc)
        for fault in applicable(svc, graph):
            n = tested.get((svc.name, fault), 0)
            bonus, w = untested_bonus(n), FAULT_WEIGHT[fault]
            out.append(Candidate(
                id=0, target=svc.name, fault=fault, params=dict(faults[fault].params),
                duration_s=faults[fault].duration_s, score=round(crit * frag * bonus * w, 3),
                factors={"criticality": crit, "fragility": frag, "untested": bonus, "fault_weight": w},
                reason=("; ".join(_facts(svc, graph)) or "no notable risk factors")
                       + ("; never tested" if n == 0 else f"; tested {n}x before"),
                hypothesis=_hypothesis(fault, svc, graph), tested=n))
    out.sort(key=lambda c: (-c.score, c.target, c.fault))        # deterministic
    for i, c in enumerate(out):
        c.id = i
    return out


def select(cands: list[Candidate], n: int, max_per_target: int = MAX_PER_TARGET,
           max_per_fault: int = MAX_PER_FAULT) -> list[Candidate]:
    """Greedy by score with variety caps, so the plan is not six variations of one failure."""
    chosen: list[Candidate] = []
    per_target: dict[str, int] = {}
    per_fault: dict[str, int] = {}
    for c in cands:
        if len(chosen) >= n:
            break
        if per_target.get(c.target, 0) >= max_per_target or per_fault.get(c.fault, 0) >= max_per_fault:
            continue
        chosen.append(c)
        per_target[c.target] = per_target.get(c.target, 0) + 1
        per_fault[c.fault] = per_fault.get(c.fault, 0) + 1
    return chosen


def _planned(rank: int, c: Candidate, duration_s: int | None = None, hypothesis: str | None = None,
             reason: str | None = None) -> PlannedExperiment:
    return PlannedExperiment(rank=rank, target=c.target, fault=c.fault, params=c.params,
                             duration_s=duration_s or c.duration_s, hypothesis=hypothesis or c.hypothesis,
                             reason=reason or c.reason, risk_score=c.score, factors=c.factors)


def heuristic_plan(graph: Graph, history: list[dict] | None = None, n: int = 6,
                   fallback_reason: str | None = None, cfg: PlannerCfg | None = None,
                   faults: dict | None = None) -> Plan:
    cfg = cfg or PlannerCfg()
    chosen = select(candidates(graph, history, faults), n, cfg.max_per_target, cfg.max_per_fault)
    return Plan(planner="heuristic", created_at=time.time(), fallback_reason=fallback_reason,
                experiments=[_planned(i + 1, c) for i, c in enumerate(chosen)])


# ---------------------------------------------------------------- LLM
class LLMExperiment(BaseModel):
    candidate_id: int
    duration_s: int
    hypothesis: str
    reason: str


class LLMPlan(BaseModel):
    experiments: list[LLMExperiment]


LLM_SCHEMA = {
    "type": "object",
    "properties": {"experiments": {"type": "array", "items": {
        "type": "object",
        "properties": {"candidate_id": {"type": "integer"}, "duration_s": {"type": "integer"},
                       "hypothesis": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["candidate_id", "duration_s", "hypothesis", "reason"],
        "additionalProperties": False}}},
    "required": ["experiments"], "additionalProperties": False,
}

SYSTEM = """You plan chaos experiments for a small containerized app, in a sandbox, to find resilience weaknesses.

You are given the real dependency graph, a numbered list of candidate experiments (each already scored by a
deterministic risk heuristic), and the results of experiments already run. Choose and order the experiments
that will teach the most.

Rules:
- Pick only from the candidates, by candidate_id. Do not invent targets, faults or parameters.
- Order by expected learning value: weaknesses most likely to hurt users first, but spread across different
  services and failure types, and do not repeat what a stored result already answers.
- hypothesis: one falsifiable sentence stating what you expect to observe (which service or journey degrades,
  and roughly how). Base it only on the graph and stored results; do not claim measurements you were not given.
- reason: one sentence on why this experiment ranks where it does, citing graph facts or stored results.
- duration_s: 5 to 60 seconds."""


def _prompt(graph: Graph, cands: list[Candidate], history: list[dict], n: int) -> str:
    services = {s.name: {
        "depends_on": s.depends_on, "dependents": sorted(o.name for o in graph.services.values() if s.name in o.depends_on),
        "replicas": s.replicas, "healthcheck": s.healthcheck, "restart_policy": s.restart_policy,
        "stateful": s.stateful, "resource_limits": s.has_limits, "on_critical_path": s.on_critical_path,
        "user_facing": s.entrypoint} for s in graph.services.values()}
    return json.dumps({
        "choose_up_to": n, "critical_path": graph.critical_path, "services": services,
        "candidates": [{"candidate_id": c.id, "target": c.target, "fault": c.fault, "params": c.params,
                        "default_duration_s": c.duration_s, "heuristic_score": c.score, "factors": c.factors}
                       for c in cands],
        "stored_results": history,
    }, indent=1, sort_keys=True)


def _cache_path(cache_dir: Path, model: str, system: str, user: str) -> Path:
    key = hashlib.sha256("\n\x00".join((model, system, user)).encode()).hexdigest()[:24]
    return Path(cache_dir) / f"plan-{key}.json"


def _call_model(client, user: str, model: str = MODEL, effort: str = "medium") -> str:
    resp = client.beta.messages.create(
        model=model, max_tokens=8000, system=SYSTEM,
        betas=[FALLBACK_BETA], fallbacks="default",            # re-run on the fallback model if declined
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": LLM_SCHEMA}},
        messages=[{"role": "user", "content": user}])
    if resp.stop_reason == "refusal":                         # branch on stop_reason before reading content
        raise PlannerError("model refused the request")
    if resp.stop_reason == "max_tokens":
        raise PlannerError("response truncated (max_tokens)")
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
    if not text:
        raise PlannerError("no text block in response")
    return text


def _validate(text: str, cands: list[Candidate], n: int) -> list[PlannedExperiment]:
    try:
        plan = LLMPlan.model_validate_json(text)
    except ValidationError as e:
        raise PlannerError(f"response is not the expected JSON: {e.errors()[0]['msg']}") from e
    if not plan.experiments:
        raise PlannerError("empty plan")
    if len(plan.experiments) > n:
        raise PlannerError(f"{len(plan.experiments)} experiments returned, at most {n} allowed")
    by_id = {c.id: c for c in cands}
    seen: set[int] = set()
    out = []
    for i, e in enumerate(plan.experiments):
        if e.candidate_id not in by_id:
            raise PlannerError(f"unknown candidate_id {e.candidate_id}")
        if e.candidate_id in seen:
            raise PlannerError(f"duplicate candidate_id {e.candidate_id}")
        seen.add(e.candidate_id)
        if not 5 <= e.duration_s <= 60:
            raise PlannerError(f"duration_s {e.duration_s} outside 5..60")
        if not e.hypothesis.strip() or not e.reason.strip():
            raise PlannerError("empty hypothesis or reason")
        out.append(_planned(i + 1, by_id[e.candidate_id], e.duration_s, e.hypothesis.strip(), e.reason.strip()))
    return out


def llm_plan(graph: Graph, history: list[dict], n: int = 6, client=None, cache_dir: Path | None = None,
             cfg: PlannerCfg | None = None, faults: dict | None = None) -> Plan:
    """Raises PlannerError on any problem; make_plan() turns that into a heuristic fallback."""
    cfg = cfg or PlannerCfg()
    cands = candidates(graph, history, faults)[:cfg.llm_pool]
    user = _prompt(graph, cands, history, n)
    notes = []
    cache = _cache_path(cache_dir, f"{llm_label(cfg)}|{cfg.llm_effort}", SYSTEM, user) if cache_dir else None
    text, answered_by = None, None
    if cache and cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        text, answered_by = saved["text"], saved.get("model")      # the model that really produced it, not today's configured default
        notes.append(f"LLM response served from cache ({cache.name}); no API call")
    try:
        if text is None:
            if client is None:
                client = make_client(cfg.provider, cfg.members)
            # a pool starts every new conversation on its next member, so a truncated or invalid answer from one member
            # is retried on another instead of dropping straight to the heuristic
            attempts = min(MAX_ATTEMPTS, len(getattr(client, "members", None) or [])) or 1
            problems = []
            for _ in range(attempts):
                try:
                    text = _call_model(client, user, cfg.llm_model, cfg.llm_effort)
                    experiments = _validate(text, cands, n)
                    break
                except PlannerError as e:
                    problems.append(str(e))
                    if len(problems) == attempts:
                        raise PlannerError("; ".join(problems)) if attempts > 1 else e
            notes += [f"attempt {i} was unusable: {m}" for i, m in enumerate(problems, 1)]
        else:
            experiments = _validate(text, cands, n)
    except PlannerError:
        raise
    except Exception as e:                                    # no credentials, network, API error, SDK change ...
        if "authentication method" in str(e):
            raise PlannerError("no Anthropic credentials found (set ANTHROPIC_API_KEY)") from e
        raise PlannerError(f"{type(e).__name__}: {str(e)[:400]}") from e
    answered_by = answered_by or model_label(client, cfg.llm_model)
    if cache and not cache.exists():                          # only validated responses are cached
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"model": answered_by, "text": text}), encoding="utf-8")
    return Plan(planner="llm", model=answered_by, created_at=time.time(), notes=notes, experiments=experiments)


def make_plan(graph: Graph, history: list[dict], n: int = 6, use_llm: bool = True, client=None,
              cache_dir: Path | None = None, cfg: PlannerCfg | None = None, faults: dict | None = None) -> Plan:
    if not use_llm:
        return heuristic_plan(graph, history, n, fallback_reason="LLM disabled (--offline or planner.use_llm: false)",
                              cfg=cfg, faults=faults)
    try:
        return llm_plan(graph, history, n, client, cache_dir, cfg, faults)
    except PlannerError as e:
        return heuristic_plan(graph, history, n, fallback_reason=str(e), cfg=cfg, faults=faults)


def to_experiments(plan: Plan, stamp: str | None = None) -> list[Experiment]:
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    return [Experiment(id=f"{stamp}-{p.rank}-{p.target}-{p.fault}", target=p.target, fault=p.fault,
                       params=p.params, duration_s=p.duration_s, hypothesis=p.hypothesis)
            for p in plan.experiments]
