"""Runs tests/report_cases.js with Node and reports each case as its own pytest result."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
CASES = Path(__file__).parent / "report_cases.js"

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


@pytest.fixture(scope="module")
def results():
    p = subprocess.run([NODE, str(CASES)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


NAMES = [
    "score_shows_the_number_the_band_and_every_component",
    "a_component_with_no_data_shows_na_and_an_empty_bar",
    "band_colours",
    "llm_analysis_is_labelled",
    "empty_and_unscored_reports_say_so_instead_of_inventing_a_score",
    "history_lists_newest_first_with_a_hidden_breakdown_row_per_experiment",
    "missing_signals_and_rollback_errors_are_surfaced_in_the_breakdown",
    "findings_show_severity_evidence_and_the_fix_with_its_verification",
    "hostile_text_from_results_or_the_analyzer_is_escaped_everywhere",
    "card_summarises_the_score_for_the_header_row",
]


@pytest.mark.parametrize("name", NAMES)
def test_report_js_case(results, name):
    assert name in results, f"case {name} did not run"
    assert results[name] == "ok", results[name]


def test_every_js_case_is_covered_here(results):
    assert set(results) == set(NAMES)
