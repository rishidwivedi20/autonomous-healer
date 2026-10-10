"""Builds graph.json (the dependency map) from the compose file.

Edge sources, all real configuration:
  1. compose `depends_on`
  2. service names appearing in `environment` values / connection strings
  3. (optional, --live) the environment of the running containers, via the Docker SDK

Limits worth knowing: apps that hardcode a hostname in code (the stock voting app does:
"redis", "db") leave no trace in env vars. Here those edges are covered by depends_on;
anything else can be added by hand in graph.overrides.yaml, which is applied last.
Edges and flags it adds are listed under "manual" in the output so they stay auditable.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from .models import Graph, Service

STATEFUL_IMAGES = ("redis", "postgres", "mysql", "mariadb", "mongo", "memcached", "rabbitmq", "kafka", "etcd")


def _as_list(value) -> list[str]:
    if not value:
        return []
    return list(value.keys()) if isinstance(value, dict) else [str(v) for v in value]


def _env_values(spec: dict) -> list[str]:
    env = spec.get("environment") or {}
    if isinstance(env, dict):
        return [str(v) for v in env.values() if v is not None]
    return [item.split("=", 1)[1] for item in env if "=" in str(item)]


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text) is not None


def _has_limits(spec: dict) -> bool:
    limits = (spec.get("deploy") or {}).get("resources", {}).get("limits")
    return bool(limits or spec.get("mem_limit") or spec.get("cpus") or spec.get("cpu_quota"))


def _healthcheck(spec: dict) -> bool:
    hc = spec.get("healthcheck")
    return bool(hc) and not hc.get("disable", False)


def _stateful(spec: dict, named_volumes: set[str]) -> bool:
    image = str(spec.get("image", ""))
    if any(k in image.split("/")[-1] for k in STATEFUL_IMAGES):
        return True
    for vol in spec.get("volumes") or []:
        src = vol.get("source") if isinstance(vol, dict) else str(vol).split(":", 1)[0]
        if src in named_volumes:
            return True
    return False


def critical_path(services: dict[str, Service], entrypoints: list[str]) -> list[str]:
    """Services a user request/journey needs.

    Start from services that publish a port, add everything they depend on (transitively),
    then add 'bridges': services that depend on two or more already-critical services
    (the voting worker links redis and db, so a vote never reaches the result page without it).
    """
    critical = set(entrypoints)
    stack = list(entrypoints)
    while stack:
        for dep in services[stack.pop()].depends_on:
            if dep not in critical:
                critical.add(dep)
                stack.append(dep)
    changed = True
    while changed:
        changed = False
        for s in services.values():
            if s.name not in critical and len(set(s.depends_on) & critical) >= 2:
                critical.add(s.name)
                changed = True
    order = {name: i for i, name in enumerate(services)}
    return sorted(critical, key=order.get)


def build_graph(compose: dict, extra_edges: dict[str, set[str]] | None = None) -> Graph:
    specs: dict[str, dict] = compose.get("services") or {}
    names = list(specs)
    named_volumes = set((compose.get("volumes") or {}).keys())
    extra_edges = extra_edges or {}

    services: dict[str, Service] = {}
    entrypoints = []
    for name, spec in specs.items():
        deps = set(_as_list(spec.get("depends_on"))) | set(extra_edges.get(name, set()))
        for value in _env_values(spec):
            deps |= {other for other in names if other != name and _mentions(value, other)}
        deps.discard(name)
        if spec.get("ports"):
            entrypoints.append(name)
        services[name] = Service(
            name=name,
            image=str(spec.get("image", "")) or ("build:" + str((spec.get("build") or {}) if isinstance(spec.get("build"), dict) else spec.get("build", ""))),
            replicas=int((spec.get("deploy") or {}).get("replicas", 1)),
            healthcheck=_healthcheck(spec),
            restart_policy=str(spec.get("restart", "no")),
            has_limits=_has_limits(spec),
            stateful=_stateful(spec, named_volumes),
            depends_on=sorted(deps),
            entrypoint=bool(spec.get("ports")),
        )

    for s in services.values():
        s.fan_out = len(s.depends_on)
        s.fan_in = sum(1 for o in services.values() if s.name in o.depends_on)

    path = critical_path(services, entrypoints)
    for name in path:
        services[name].on_critical_path = True
    return Graph(services=services, critical_path=path)


def live_edges(project: str | None = None) -> dict[str, set[str]]:
    """Edges from the environment of running managed containers. Needs Docker; imported lazily."""
    import docker

    client = docker.from_env()
    found: dict[str, set[str]] = {}
    containers = client.containers.list(filters={"label": "chaosai.managed=true"})
    by_service = {c.labels.get("com.docker.compose.service", c.name): c for c in containers}
    names = list(by_service)
    for svc, c in by_service.items():
        for item in c.attrs.get("Config", {}).get("Env") or []:
            value = item.split("=", 1)[1] if "=" in item else ""
            for other in names:
                if other != svc and _mentions(value, other):
                    found.setdefault(svc, set()).add(other)
    return found


def apply_overrides(graph: Graph, path: Path) -> tuple[Graph, list[str]]:
    """graph.overrides.yaml: {service: {depends_on_add: [...], depends_on_remove: [...], <field>: value}}"""
    notes: list[str] = []
    if not path.exists():
        return graph, notes
    overrides = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for name, patch in overrides.items():
        if name not in graph.services:
            notes.append(f"override for unknown service {name!r} ignored")
            continue
        svc = graph.services[name]
        deps = set(svc.depends_on)
        for d in patch.pop("depends_on_add", []):
            deps.add(d)
            notes.append(f"{name} -> {d} (manual edge)")
        for d in patch.pop("depends_on_remove", []):
            deps.discard(d)
            notes.append(f"{name} -x-> {d} (manual removal)")
        for key, val in patch.items():
            setattr(svc, key, val)
            notes.append(f"{name}.{key} = {val!r} (manual)")
        svc.depends_on = sorted(deps)
    for s in graph.services.values():
        s.fan_out = len(s.depends_on)
        s.fan_in = sum(1 for o in graph.services.values() if s.name in o.depends_on)
    return graph, notes


def to_mermaid(graph: Graph) -> str:
    lines = ["graph LR"]
    for s in graph.services.values():
        flags = [f"x{s.replicas}"]
        if not s.healthcheck:
            flags.append("no healthcheck")
        label = f"{s.name}<br/>{', '.join(flags)}"
        lines.append(f'  {s.name}[("{label}")]' if s.stateful else f'  {s.name}["{label}"]')
    for s in graph.services.values():
        for d in s.depends_on:
            lines.append(f"  {s.name} --> {d}")
    crit = [n for n in graph.critical_path]
    if crit:
        lines.append("  classDef crit stroke:#d6336c,stroke-width:3px;")
        lines.append(f"  class {','.join(crit)} crit;")
    return "\n".join(lines)


def discover(compose_path: Path, overrides: Path | None = None, live: bool = False) -> tuple[Graph, list[str]]:
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
    graph = build_graph(compose, live_edges() if live else None)
    notes: list[str] = []
    if overrides:
        graph, notes = apply_overrides(graph, overrides)
    return graph, notes


def write_graph(graph: Graph, out: Path, notes: list[str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = graph.model_dump()
    data["manual"] = notes
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the service dependency map")
    ap.add_argument("--compose", default="deploy/docker-compose.yml", type=Path)
    ap.add_argument("--overrides", default="deploy/graph.overrides.yaml", type=Path)
    ap.add_argument("--out", default="results/graph.json", type=Path)
    ap.add_argument("--live", action="store_true", help="also read env vars from running containers")
    args = ap.parse_args(argv)
    graph, notes = discover(args.compose, args.overrides, args.live)
    write_graph(graph, args.out, notes)
    print(to_mermaid(graph))
    print(f"\ncritical path: {' -> '.join(graph.critical_path)}")
    for n in notes:
        print(f"manual: {n}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
