from pathlib import Path

import yaml

from chaosai.discovery import apply_overrides, build_graph, critical_path, discover, to_mermaid

COMPOSE = Path(__file__).resolve().parent.parent / "deploy" / "docker-compose.yml"


def real_graph():
    graph, _ = discover(COMPOSE)
    return graph


def test_services_found():
    assert set(real_graph().services) == {"vote", "redis", "worker", "db", "result"}


def test_edges_match_voting_app_architecture():
    g = real_graph().services
    assert g["vote"].depends_on == ["redis"]
    assert g["worker"].depends_on == ["db", "redis"]
    assert g["result"].depends_on == ["db"]
    assert g["redis"].depends_on == [] and g["db"].depends_on == []


def test_fan_in_out():
    g = real_graph().services
    assert g["redis"].fan_in == 2 and g["db"].fan_in == 2
    assert g["worker"].fan_out == 2 and g["worker"].fan_in == 0


def test_annotations():
    g = real_graph().services
    assert g["redis"].stateful and g["db"].stateful
    assert not g["vote"].stateful and not g["worker"].stateful
    assert g["worker"].healthcheck is False          # known gap, the agent should flag it
    assert g["redis"].healthcheck and g["db"].healthcheck
    assert all(s.replicas == 1 for s in g.values())
    assert all(s.restart_policy == "unless-stopped" for s in g.values())
    assert not any(s.has_limits for s in g.values())


def test_critical_path_includes_worker_bridge():
    assert set(real_graph().critical_path) == {"vote", "redis", "worker", "db", "result"}


def test_env_var_connection_strings_become_edges():
    compose = yaml.safe_load("""
services:
  api: {image: x, ports: ["80:80"], environment: {DB_URL: "postgres://user@pg-main:5432/app", CACHE: "cache:6379"}}
  pg-main: {image: postgres:15}
  cache: {image: redis}
  pg-main-replica: {image: postgres:15}
""")
    g = build_graph(compose).services
    assert g["api"].depends_on == ["cache", "pg-main"]   # not pg-main-replica (boundary match)


def test_depends_on_dict_and_list_forms():
    compose = yaml.safe_load("""
services:
  a: {image: x, depends_on: {b: {condition: service_healthy}}}
  b: {image: x}
  c: {image: x, depends_on: [b]}
""")
    g = build_graph(compose).services
    assert g["a"].depends_on == ["b"] and g["c"].depends_on == ["b"]


def test_extra_live_edges_merged():
    compose = yaml.safe_load("services: {a: {image: x}, b: {image: x}}")
    assert build_graph(compose, {"a": {"b"}}).services["a"].depends_on == ["b"]


def test_overrides_are_applied_and_recorded(tmp_path):
    g = real_graph()
    p = tmp_path / "o.yaml"
    p.write_text("vote: {depends_on_add: [db]}\nworker: {stateful: true}\nghost: {x: 1}\n")
    g, notes = apply_overrides(g, p)
    assert g.services["vote"].depends_on == ["db", "redis"]
    assert g.services["worker"].stateful
    assert g.services["db"].fan_in == 3
    assert any("manual edge" in n for n in notes) and any("unknown service" in n for n in notes)


def test_missing_overrides_file_is_fine(tmp_path):
    g, notes = apply_overrides(real_graph(), tmp_path / "nope.yaml")
    assert notes == []


def test_mermaid_mentions_every_edge():
    m = to_mermaid(real_graph())
    assert "vote --> redis" in m and "worker --> db" in m and "result --> db" in m


def test_critical_path_on_isolated_service_excluded():
    compose = yaml.safe_load("""
services:
  web: {image: x, ports: ["80:80"], depends_on: [db]}
  db: {image: postgres}
  cron: {image: x}
""")
    assert "cron" not in build_graph(compose).critical_path
