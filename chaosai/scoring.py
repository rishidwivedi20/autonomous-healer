"""The resilience score: deterministic, computed only from stored measurements.

Each finished experiment is scored 0-100 from four components:

  availability  40  worst user journey's error-rate increase during the fault:   1 - uplift
  latency       20  worst journey's p95 vs its baseline p95 (50 ms slack):        1 at <=1.5x, 0 at >=10x, linear between
  recovery      25  time until users AND the pipeline were back to normal:        1 at <=5 s, 0 at >=60 s, linear between;
                    0 if it did not recover within the timeout
  blast radius  15  collateral damage: the mean of (services other than the target that had to be
                    restarted) and (user journeys with >5% error uplift), each as a fraction of all possible

A component with no usable measurement is NOT guessed: it is left out, the remaining weights are
rescaled, and the gap is listed in `missing`. The overall score is the mean of the experiment scores
weighted by the criticality of the service that was broken (the same criticality the planner uses), so
breaking something important counts for more. Only the latest run of each distinct experiment counts, so
repeating one easy fault cannot inflate the total. An AI may explain this score; it never produces it.
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field

from .models import ExperimentResult, Graph
from .planner import criticality

WEIGHTS = {"availability": 40, "latency": 20, "recovery": 25, "blast_radius": 15}
LATENCY_SLACK_MS = 50.0
LATENCY_OK_RATIO, LATENCY_BAD_RATIO = 1.5, 10.0
RECOVERY_FULL_S, RECOVERY_ZERO_S = 5.0, 60.0
DEGRADED_UPLIFT = 0.05
BANDS = ((85, "good"), (60, "fair"), (0, "poor"))


class Component(BaseModel):
    name: str
    weight: int
    score: Optional[float] = None         # 0..1, None = not measurable
    basis: str = ""                        # the measured numbers behind it, in words


class ExperimentScore(BaseModel):
    experiment_id: str
    target: str
    fault: str
    params: dict = Field(default_factory=dict)
    started_at: float = 0.0
    score: Optional[float] = None         # 0..100, None = nothing measurable
    band: Optional[str] = None
    weight: float = 1.0                    # target criticality
    components: list[Component] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class OverallScore(BaseModel):
    score: Optional[float] = None
    band: Optional[str] = None
    n_experiments: int = 0                 # distinct experiments counted (latest run of each)
    n_runs: int = 0                        # all scored runs
    by_component: dict[str, Optional[float]] = Field(default_factory=dict)   # 0..100, criticality-weighted
    experiments: list[ExperimentScore] = Field(default_factory=list)
    formula: str = ""


def band(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    return next(name for floor, name in BANDS if score >= floor)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _uplifts(r: ExperimentResult) -> dict[str, float]:
    """Per journey: error rate during the fault minus its own baseline error rate (never negative)."""
    out = {}
    for j, d in r.during_by_journey.items():
        if d.n and d.error_rate is not None:
            b = r.baseline_by_journey.get(j)
            base = b.error_rate if b and b.error_rate is not None else 0.0
            out[j] = max(0.0, d.error_rate - base)
    if not out and r.during.n and r.during.error_rate is not None:
        out["all"] = max(0.0, r.during.error_rate - (r.baseline.error_rate or 0.0))
    return out


def _availability(r: ExperimentResult) -> Component:
    c = Component(name="availability", weight=WEIGHTS["availability"])
    up = _uplifts(r)
    if not up:
        c.basis = "no probe results during the fault"
        return c
    j = max(up, key=up.get)
    c.score = _clamp(1 - up[j])
    c.basis = f"worst journey '{j}': error rate up by {up[j]:.0%} during the fault"
    return c


def _latency(r: ExperimentResult) -> Component:
    c = Component(name="latency", weight=WEIGHTS["latency"])
    worst: tuple[float, str, float, float] | None = None
    for j, d in r.during_by_journey.items():
        b = r.baseline_by_journey.get(j)
        if d.p95_ms is None or b is None or b.p95_ms is None:
            continue
        ratio = (d.p95_ms + LATENCY_SLACK_MS) / (b.p95_ms + LATENCY_SLACK_MS)
        if worst is None or ratio > worst[0]:
            worst = (ratio, j, b.p95_ms, d.p95_ms)
    if worst is None:
        c.basis = "no p95 to compare (no successful requests during the fault, or no baseline)"
        return c
    ratio, j, base, during = worst
    c.score = _clamp(1 - (ratio - LATENCY_OK_RATIO) / (LATENCY_BAD_RATIO - LATENCY_OK_RATIO))
    c.basis = f"worst journey '{j}': p95 {base:.0f} ms -> {during:.0f} ms ({ratio:.1f}x with {LATENCY_SLACK_MS:.0f} ms slack)"
    return c


def _recovery(r: ExperimentResult) -> Component:
    c = Component(name="recovery", weight=WEIGHTS["recovery"])
    if r.recovered and r.recovery_s is not None:
        c.score = _clamp(1 - (r.recovery_s - RECOVERY_FULL_S) / (RECOVERY_ZERO_S - RECOVERY_FULL_S))
        c.basis = f"recovered in {r.recovery_s:.0f} s"
        if not r.pipeline_assessed:
            c.basis += " (users only: no pipeline gauges were usable)"
    elif r.abort_reason == "kill switch":
        c.basis = "the operator's kill switch ended the run before recovery was measured"      # not scored: unknown, not failed
    else:
        c.score = 0.0
        c.basis = "did not recover within the timeout"
    return c


def _blast(r: ExperimentResult, graph: Optional[Graph], restarts: dict[str, int]) -> Component:
    c = Component(name="blast_radius", weight=WEIGHTS["blast_radius"])
    parts, notes = [], []
    if graph and len(graph.services) > 1:
        collateral = sorted(s for s, n in restarts.items() if n > 0 and s != r.experiment.target and s in graph.services)
        parts.append(1 - len(collateral) / (len(graph.services) - 1))
        notes.append(f"{len(collateral)} other service(s) restarted" + (f" ({', '.join(collateral)})" if collateral else ""))
    up = _uplifts(r)
    if up:
        degraded = sorted(j for j, u in up.items() if u > DEGRADED_UPLIFT)
        parts.append(1 - len(degraded) / len(up))
        notes.append(f"{len(degraded)} of {len(up)} user journey(s) degraded" + (f" ({', '.join(degraded)})" if degraded else ""))
    if not parts:
        c.basis = "no graph and no probe results"
        return c
    c.score = _clamp(sum(parts) / len(parts))
    c.basis = "; ".join(notes)
    return c


def score_experiment(r: ExperimentResult, graph: Optional[Graph] = None, restarts: Optional[dict[str, int]] = None) -> ExperimentScore:
    """`restarts` overrides r.restarts (the evidence layer recomputes it from the stored events)."""
    comps = [_availability(r), _latency(r), _recovery(r), _blast(r, graph, r.restarts if restarts is None else restarts)]
    have = [c for c in comps if c.score is not None]
    total = sum(c.weight for c in have)
    score = round(100 * sum(c.weight * c.score for c in have) / total, 1) if total else None
    svc = graph.services.get(r.experiment.target) if graph else None
    return ExperimentScore(
        experiment_id=r.experiment.id, target=r.experiment.target, fault=r.experiment.fault,
        params=r.experiment.params, started_at=r.started_at,
        score=score, band=band(score), weight=criticality(svc) if svc else 1.0, components=comps,
        missing=[f"{c.name}: {c.basis}" for c in comps if c.score is None])


def overall(scores: list[ExperimentScore]) -> OverallScore:
    runs = [s for s in scores if s.score is not None]
    latest: dict[str, ExperimentScore] = {}
    for s in sorted(runs, key=lambda s: s.started_at):          # a repeat of the same experiment replaces the older run
        latest[json.dumps([s.target, s.fault, s.params], sort_keys=True)] = s
    scored = list(latest.values())
    out = OverallScore(n_experiments=len(scored), n_runs=len(runs), experiments=scores, formula=(
        "per experiment: 40 availability + 20 latency + 25 recovery + 15 blast radius (unmeasurable parts are dropped and "
        "the rest rescaled); overall: mean of the LATEST run of each distinct experiment (target, fault, params), weighted by the criticality of the service that was broken"))
    if not scored:
        return out
    wsum = sum(s.weight for s in scored)
    out.score = round(sum(s.score * s.weight for s in scored) / wsum, 1)
    out.band = band(out.score)
    for name in WEIGHTS:
        pairs = [(next(c.score for c in s.components if c.name == name), s.weight)
                 for s in scored if any(c.name == name and c.score is not None for c in s.components)]
        out.by_component[name] = round(100 * sum(v * w for v, w in pairs) / sum(w for _, w in pairs), 1) if pairs else None
    return out
