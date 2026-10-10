"""The finding agent: turn stored experiments into ranked, evidence-backed findings, a resilience score and a report.

It is the second agent of the system, between the injecting agent (agent.py: breaks things safely and records what happened) and
the fixing agent (fixer.py: applies a vetted fix and proves it). Its job:

    read    the stored measurements of the experiments in scope, through read-only evidence tools
    explain the analysis: what happened, which services were affected, weaknesses ranked by severity, a concrete fix and a
            verification re-run for each; the LLM analyzer if a model is configured and answers, otherwise the deterministic
            rule-based analyzer (so it works with no key and no network), and the report says which one was used
    score   the resilience score, computed by code from the stored numbers
    write   results/analysis.json and results/findings.{json,md,html}

Every number in an LLM analysis is checked against the stored data (analysis.py), so this agent cannot cite a measurement that was
not recorded. The fixing agent reads the weaknesses this agent produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import analyzer as analyzer_mod
from . import report as report_mod
from .analysis import Analysis
from .config import AnalyzerCfg
from .evidence import Evidence
from .models import Graph


class FindingError(Exception):
    pass


@dataclass
class Findings:
    ids: list[str]               # the experiments analysed
    analysis: Analysis
    report: dict                 # what findings.json holds
    paths: dict[str, Path]       # md / html / json


class FindingAgent:
    def __init__(self, cfg: AnalyzerCfg, results_dir: Path | str, graph: Optional[Graph], use_llm: bool = True,
                 cache_dir: Path | str | None = None, client=None):
        self.cfg, self.results_dir, self.graph = cfg, Path(results_dir), graph
        self.use_llm, self.cache_dir, self.client = use_llm, cache_dir, client

    def _evidence(self, scope: Optional[list[str]]) -> tuple[Evidence, list[str]]:
        ev = Evidence(self.results_dir)
        ids = ev.ids() if scope is None else [i for i in scope if i in ev.ids()]
        if not ids:
            raise FindingError("no stored experiments to analyse; run `chaos pipeline` (or `chaos autopilot` / `chaos agent`) first")
        return ev, ids

    def analyze(self, scope: Optional[list[str]] = None, use_llm: Optional[bool] = None) -> Analysis:
        ev, ids = self._evidence(scope)
        try:
            return analyzer_mod.analyze(ev, ids, self.cfg, use_llm=self.use_llm if use_llm is None else use_llm,
                                        client=self.client, cache_dir=self.cache_dir)
        except analyzer_mod.AnalyzerError as e:
            raise FindingError(str(e)) from e

    def run(self, scope: Optional[list[str]] = None, analysis: Optional[Analysis] = None) -> Findings:
        """Analyse (unless an analysis is supplied), score, and write the files. The report picks up any validated fixes
        from results/fixes/ (see fixer.py), so call this again after the fixing agent has run to include them."""
        ev, ids = self._evidence(scope)
        analysis = analysis or self.analyze(ids)
        (self.results_dir / "analysis.json").write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
        rep = report_mod.build(ev, analysis, self.graph, ids)
        return Findings(ids=ids, analysis=analysis, report=rep, paths=report_mod.write(self.results_dir, rep))

    def rules_over_history(self) -> Analysis:
        """The deterministic analysis of EVERY stored experiment: what the fixing agent falls back on when this run's
        experiments did not happen to exercise a weakness it knows how to fix."""
        return self.analyze(None, use_llm=False)
