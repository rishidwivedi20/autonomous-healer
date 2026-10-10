"""The analyzer's output schema and the validator that keeps it honest.

An Analysis is rejected (and, for the LLM agent, sent back with the reasons) unless:
  * every experiment in scope is covered, and every cited experiment id exists;
  * every weakness and fix cites at least one piece of evidence;
  * every NUMBER in an evidence claim appears in the output of a data tool the analyzer actually called
    for that experiment (or in a global tool: graph, baseline). So a figure that was not stored cannot be
    cited. Numbers inside `change` snippets and `rationale` text are proposals, not evidence, and are free;
  * weaknesses are ranked most severe first;
  * an experiment without a stored hypothesis does not get a hypothesis verdict.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .evidence import numbers_in

SEVERITIES = ("critical", "high", "medium", "low")
Severity = Literal["critical", "high", "medium", "low"]
TRIVIAL = {0.0, 1.0}                       # "1 restart", "0 errors": too small to be a fabricated figure
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


class Evid(BaseModel):
    experiment_id: str
    claim: str                              # every number here must come from a tool output


class ExpAnalysis(BaseModel):
    experiment_id: str
    hypothesis_held: Optional[bool] = None
    root_cause: str
    blast_radius: list[str] = Field(default_factory=list)   # affected services
    evidence: list[Evid] = Field(default_factory=list)


class Weakness(BaseModel):
    title: str
    severity: Severity
    services: list[str] = Field(default_factory=list)
    description: str
    evidence: list[Evid] = Field(default_factory=list)


class Fix(BaseModel):
    title: str
    change: str                             # concrete: compose/config snippet or code change
    rationale: str
    addresses: list[str] = Field(default_factory=list)       # weakness titles
    evidence: list[Evid] = Field(default_factory=list)


class Analysis(BaseModel):
    summary: str
    experiments: list[ExpAnalysis]
    weaknesses: list[Weakness]
    fixes: list[Fix]
    generated_by: str = "llm"               # llm | rules
    model: Optional[str] = None
    fallback_reason: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- what was cited vs what was returned
class Seen:
    """The numbers the analyzer has actually been shown, per experiment and globally."""

    def __init__(self, known_ids: list[str]):
        self.known = list(known_ids)
        self.by_exp: dict[str, set[float]] = {}
        self.global_: set[float] = set()

    def record(self, args: dict | None, output: dict) -> None:
        nums = numbers_in(output, ignore_ids=set(self.known))
        exp = (args or {}).get("experiment_id")
        if exp and exp in self.known:
            self.by_exp.setdefault(exp, set()).update(nums)
        elif not exp:
            self.global_.update(nums)

    def allowed(self, exp_id: str) -> set[float]:
        return self.by_exp.get(exp_id, set()) | self.global_


def matches(x: float, pool: set[float]) -> bool:
    """Within rounding of a returned value (1%, or 0.05 absolute), so '93 s' matches 93.4 but not 120."""
    return any(abs(x - s) <= max(0.011 * abs(s), 0.051) for s in pool)


def claim_numbers(claim: str, known_ids: list[str]) -> list[float]:
    s = claim
    for i in known_ids:
        s = s.replace(i, " ")
    s = re.sub(r"pid\s*\d+", " ", s)
    s = re.sub(r"\bp\d{1,3}\b", " ", s, flags=re.I)   # percentile labels (p50, p95) are names, not values
    return [float(m) for m in _NUM.findall(s)]


def validate(a: Analysis, seen: Seen, scope_ids: list[str], has_hypothesis: dict[str, bool]) -> list[str]:
    """Return every problem found (empty = acceptable)."""
    problems: list[str] = []
    known = seen.known

    def check_evidence(where: str, evid: list[Evid]):
        if not evid:
            problems.append(f"{where}: cite at least one piece of evidence")
        for e in evid:
            if e.experiment_id not in known:
                problems.append(f"{where}: evidence cites unknown experiment_id {e.experiment_id!r}")
                continue
            if not e.claim.strip():
                problems.append(f"{where}: an evidence claim is empty")
                continue
            pool = seen.allowed(e.experiment_id)
            if not pool:
                problems.append(f"{where}: you cite experiment {e.experiment_id} but never called a data tool for it")
                continue
            bad = [n for n in claim_numbers(e.claim, known) if n not in TRIVIAL and not matches(n, pool)]
            if bad:
                problems.append(f"{where}: claim {e.claim!r} contains number(s) {sorted(set(bad))} that no tool returned "
                                f"for experiment {e.experiment_id}. Cite only values the tools returned, exactly.")

    if not a.summary.strip():
        problems.append("summary is empty")
    covered = [e.experiment_id for e in a.experiments]
    for i in scope_ids:
        if i not in covered:
            problems.append(f"experiments: missing an analysis for {i}")
    for i in covered:
        if i not in scope_ids:
            problems.append(f"experiments: {i!r} is not an experiment in scope")
    if len(set(covered)) != len(covered):
        problems.append("experiments: an experiment_id appears more than once")
    for e in a.experiments:
        if not e.root_cause.strip():
            problems.append(f"experiment {e.experiment_id}: root_cause is empty")
        if e.hypothesis_held is not None and not has_hypothesis.get(e.experiment_id, False):
            problems.append(f"experiment {e.experiment_id}: no hypothesis was stored, so hypothesis_held must be null")
        check_evidence(f"experiment {e.experiment_id}", e.evidence)

    order = [SEVERITIES.index(w.severity) for w in a.weaknesses]
    if order != sorted(order):
        problems.append("weaknesses must be ranked most severe first (critical, high, medium, low)")
    for w in a.weaknesses:
        if not w.services:
            problems.append(f"weakness {w.title!r}: list the affected services")
        check_evidence(f"weakness {w.title!r}", w.evidence)
    if not a.weaknesses:
        problems.append("no weaknesses reported; if the system truly held up, say so in weaknesses with severity low")
    titles = {w.title for w in a.weaknesses}
    for f in a.fixes:
        if not f.change.strip():
            problems.append(f"fix {f.title!r}: `change` must be a concrete snippet or code change")
        for t in f.addresses:
            if t not in titles:
                problems.append(f"fix {f.title!r}: addresses {t!r}, which is not one of the weakness titles")
        check_evidence(f"fix {f.title!r}", f.evidence)
    if not a.fixes:
        problems.append("no fixes proposed")
    return problems


# ---------------------------------------------------------------- the schema sent to the model
_EVID = {"type": "object", "properties": {"experiment_id": {"type": "string"}, "claim": {"type": "string"}},
         "required": ["experiment_id", "claim"], "additionalProperties": False}
_EVID_LIST = {"type": "array", "items": _EVID}
SUBMIT_TOOL = {
    "name": "submit_analysis",
    "description": ("Submit the final analysis. Call this exactly once when done. If it is rejected you get the reasons "
                    "back: fix them and call it again. Every number in an evidence claim must be one a data tool returned."),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "experiments": {"type": "array", "items": {"type": "object", "properties": {
                "experiment_id": {"type": "string"},
                "hypothesis_held": {"type": ["boolean", "null"], "description": "null if no hypothesis was stored"},
                "root_cause": {"type": "string"},
                "blast_radius": {"type": "array", "items": {"type": "string"}, "description": "affected services"},
                "evidence": _EVID_LIST},
                "required": ["experiment_id", "hypothesis_held", "root_cause", "blast_radius", "evidence"],
                "additionalProperties": False}},
            "weaknesses": {"type": "array", "description": "most severe first", "items": {"type": "object", "properties": {
                "title": {"type": "string"}, "severity": {"type": "string", "enum": list(SEVERITIES)},
                "services": {"type": "array", "items": {"type": "string"}}, "description": {"type": "string"},
                "evidence": _EVID_LIST},
                "required": ["title", "severity", "services", "description", "evidence"], "additionalProperties": False}},
            "fixes": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"},
                "change": {"type": "string", "description": "concrete compose/config snippet or code change"},
                "rationale": {"type": "string"},
                "addresses": {"type": "array", "items": {"type": "string"}, "description": "weakness titles this fixes"},
                "evidence": _EVID_LIST},
                "required": ["title", "change", "rationale", "addresses", "evidence"], "additionalProperties": False}},
        },
        "required": ["summary", "experiments", "weaknesses", "fixes"], "additionalProperties": False,
    },
}


class Recorder:
    """Wraps the evidence tools and remembers which numbers each call returned (see Seen)."""

    def __init__(self, ev):
        self.ev = ev
        self.seen = Seen(ev.ids())
        self.calls = 0

    def call(self, name: str, args: dict | None) -> dict:
        self.calls += 1
        out = self.ev.call(name, args)
        if "error" not in out:
            self.seen.record(args, out)
        return out
