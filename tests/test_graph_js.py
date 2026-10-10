"""Runs tests/graph_cases.js with Node and reports each case as its own pytest result."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
CASES = Path(__file__).parent / "graph_cases.js"

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


@pytest.fixture(scope="module")
def results():
    p = subprocess.run([NODE, str(CASES)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


NAMES = [
    "order_follows_critical_path_then_the_rest",
    "voting_graph_is_laid_out_left_to_right_with_straight_neighbour_arrows",
    "headroom_is_added_only_when_an_arrow_has_to_arc",
    "no_text_is_wider_than_its_node",
    "non_adjacent_dependencies_use_arcs_that_alternate_over_and_under",
    "dependencies_not_in_the_graph_are_skipped_instead_of_crashing",
    "blast_radius_of_each_voting_service",
    "blast_radius_is_transitive_and_survives_cycles",
    "health_mapping",
    "idle_picture_shows_every_service_its_warnings_and_the_critical_ring",
    "during_a_redis_fault_the_target_pulses_and_its_dependents_are_at_risk",
    "baseline_phase_flags_nothing_because_no_fault_is_active_yet",
    "recovery_phase_marks_the_target_recovering_without_the_risk_badges",
    "after_an_experiment_services_that_really_restarted_keep_a_badge",
    "user_visible_errors_get_a_badge_on_the_affected_journey",
    "down_and_missing_containers_are_coloured_honestly",
    "hostile_names_are_escaped",
    "empty_or_missing_graphs_render_nothing_and_a_single_service_is_centred",
    "render_sets_the_viewbox_and_the_markup",
]


@pytest.mark.parametrize("name", NAMES)
def test_graph_js_case(results, name):
    assert name in results, f"case {name} did not run"
    assert results[name] == "ok", results[name]


def test_every_js_case_is_covered_here(results):
    assert set(results) == set(NAMES)
