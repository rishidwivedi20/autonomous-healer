"""The fixing agent's default job: recommend changes. It does not touch the system.

From the finding agent's analysis it builds a ranked list of recommended changes, each with the weaknesses it addresses, the
evidence, the exact change (a compose / config / code snippet), how to prove it with an experiment, and, where a fix was
measured earlier (results/fixes/), the measured result. Nothing here changes the sandbox or runs an experiment; applying and
proving a fix is a separate, explicit step (`chaos fix`, or `chaos pipeline --validate-fix`; see fixer.py).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .analysis import SEVERITIES, Analysis, Evid
from .fixer import CATALOG, Recipe

MEASURED = ("improved", "worse", "inconclusive")           # a stored fix run with one of these statuses really measured something


class Recommendation(BaseModel):
    rank: int
    title: str
    severity: str                                          # the most severe weakness it addresses
    addresses: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    change: str                                            # concrete snippet or code change
    rationale: str
    evidence: list[Evid] = Field(default_factory=list)
    prove_with: str                                        # how to check that it works
    can_be_validated_by: Optional[str] = None              # a vetted fix the agent can apply and measure, if one matches
    measured: Optional[str] = None                         # the stored measurement of that fix, if one exists
    measured_status: Optional[str] = None


class Recommendations(BaseModel):
    created_at: float
    analysis_by: str
    changes_made: bool = False                             # always False: recommending never changes the system
    items: list[Recommendation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _stored_runs(results_dir: Path) -> list[dict]:
    out = []
    d = results_dir / "fixes"
    for f in sorted(d.glob("*.json"), reverse=True) if d.exists() else []:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def _measured(recipe: Recipe, runs: list[dict]) -> tuple[Optional[str], Optional[str]]:
    run = next((r for r in runs if r.get("recipe_id") == recipe.id and r.get("status") in MEASURED), None)
    if run is None:
        return None, None
    text = run.get("claim") or run.get("detail") or ""
    return f"{run['status']} in the sandbox (run {run['id']}): {text}", run["status"]


def recommend(analysis: Analysis, results_dir: Path | str) -> Recommendations:
    results_dir = Path(results_dir)
    runs = _stored_runs(results_dir)
    weak = {w.title: w for w in analysis.weaknesses}
    items = []
    for f in analysis.fixes:
        addressed = [weak[t] for t in f.addresses if t in weak]
        sev = min((w.severity for w in addressed), key=SEVERITIES.index, default="low")
        services = list(dict.fromkeys(s for w in addressed for s in w.services))
        text = f"{f.title} {f.change} {f.rationale}".lower()
        recipe = next((r for r in CATALOG.values() if r.service in text and ("replica" in text or "scale" in text)), None)   # the fix itself, not just a related weakness
        exps = list(dict.fromkeys(e.experiment_id for e in [*f.evidence, *(e for w in addressed for e in w.evidence)]))
        prove = ("re-run " + ", ".join(exps[:3]) + (f" (and {len(exps) - 3} more)" if len(exps) > 3 else "")
                 + " after the change and compare the resilience score and the metrics quoted in the evidence") if exps else \
                "re-run the experiments that exposed the weakness after the change and compare the resilience score"
        measured, status = _measured(recipe, runs) if recipe else (None, None)
        items.append(Recommendation(rank=0, title=f.title, severity=sev, addresses=f.addresses, services=services, change=f.change,
                                    rationale=f.rationale, evidence=f.evidence, prove_with=prove,
                                    can_be_validated_by=recipe.id if recipe else None, measured=measured, measured_status=status))
    items.sort(key=lambda r: (SEVERITIES.index(r.severity), -len(r.addresses)))          # stable: the analysis order breaks ties
    for n, r in enumerate(items, 1):
        r.rank = n
    notes = ["nothing was changed: these are recommendations; apply them yourself, or let the agent prove one with `chaos fix`"]
    return Recommendations(created_at=time.time(), analysis_by=analysis.generated_by, items=items, notes=notes)


def to_markdown(rec: Recommendations) -> str:
    L = ["# Recommended changes", "",
         f"_Based on an analysis by {rec.analysis_by}. {rec.notes[0] if rec.notes else ''}_", ""]
    if not rec.items:
        return "\n".join(L + ["No changes recommended."])
    for r in rec.items:
        L += [f"## {r.rank}. [{r.severity.upper()}] {r.title}", "",
              f"Addresses: {', '.join(r.addresses) or 'n/a'}" + (f"  \nServices: {', '.join(r.services)}" if r.services else ""), "",
              r.rationale, "", "```", r.change.rstrip(), "```", ""]
        for e in r.evidence:
            L.append(f"- evidence: {e.claim}  [{e.experiment_id}]")
        L += ["", f"Prove it: {r.prove_with}."]
        if r.can_be_validated_by:
            L.append(f"The agent can measure this one: `chaos fix --recipe {r.can_be_validated_by}`.")
        if r.measured:
            L.append(f"Measured earlier: {r.measured}")
        L.append("")
    return "\n".join(L)


def write(rec: Recommendations, results_dir: Path | str) -> dict[str, Path]:
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = {"json": d / "recommendations.json", "md": d / "recommendations.md"}
    paths["json"].write_text(rec.model_dump_json(indent=1), encoding="utf-8")
    paths["md"].write_text(to_markdown(rec), encoding="utf-8")
    return paths
