"""chaos CLI:  discover | baseline | run | abort | reset | status

Run as `python -m chaosai <command>` (or `chaos <command>` via the chaos.cmd wrapper).
Ctrl+C / SIGTERM during `run` engages the kill switch: the experiment aborts, everything is
rolled back, and further runs are refused until `chaos reset`.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import typer
import yaml

from . import discovery
from .models import Experiment, Thresholds
from .observer import DockerWatcher, LoadProbe, ProbeConfig, Store, unhealthy
from .orchestrator import Orchestrator, measure_baseline
from .injector import InjectionError
from .safety import Guard, KillSwitch, SafetyViolation, is_managed

app = typer.Typer(help="Safe chaos experiments against the voting app.", no_args_is_help=True)

RESULTS = Path(os.environ.get("CHAOSAI_RESULTS", "results"))
COMPOSE = Path(os.environ.get("CHAOSAI_COMPOSE", "deploy/docker-compose.yml"))


def kill_switch() -> KillSwitch:
    return KillSwitch(RESULTS / "KILL")


def make_injector(probe=None, dry_run: bool = False):
    from .injector import DockerInjector
    return DockerInjector(state_path=RESULTS / "active_faults.json", probe=probe, dry_run=dry_run)


def managed_from_compose(path: Path) -> set[str]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {n for n, s in (spec.get("services") or {}).items() if is_managed(s.get("labels"))}


def parse_params(items: list[str] | None) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise typer.BadParameter(f"--param expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def fail(msg: str, code: int = 2):
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


# ------------------------------------------------------------------ discover
@app.command()
def discover(live: bool = typer.Option(False, help="also read env vars from running containers"),
             compose: Path = typer.Option(COMPOSE), out: Path = typer.Option(None)):
    """Build the dependency map (results/graph.json) and print it as Mermaid."""
    out = out or RESULTS / "graph.json"
    graph, notes = discovery.discover(compose, compose.parent / "graph.overrides.yaml", live)
    discovery.write_graph(graph, out, notes)
    typer.echo(discovery.to_mermaid(graph))
    typer.echo(f"\ncritical path: {' -> '.join(graph.critical_path)}")
    for n in notes:
        typer.echo(f"manual: {n}")
    typer.echo(f"wrote {out}")


# ------------------------------------------------------------------ baseline
@app.command()
def baseline(seconds: int = typer.Option(120, help="how long to observe steady state"),
             vote_url: str = "http://localhost:8080/", result_url: str = "http://localhost:8081/"):
    """Observe the healthy system and save p50/p95/error rate to results/baseline.json."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
    except Exception as e:
        fail(f"cannot reach Docker: {e}")
    store = Store(RESULTS / "chaosai.db")
    probe = LoadProbe(store, ProbeConfig(vote_url=vote_url, result_url=result_url)).start()
    watcher = DockerWatcher(store, client=client).start()
    try:
        data = measure_baseline(store, seconds, RESULTS, lambda: unhealthy(watcher.snapshot()))
    finally:
        probe.stop()
        watcher.stop()
    o = data["overall"]
    typer.echo(f"baseline over {seconds}s: n={o['n']} error_rate={o['error_rate']} p50={o['p50_ms']}ms p95={o['p95_ms']}ms")
    for k, v in data["by_kind"].items():
        typer.echo(f"  {k:7} n={v['n']} error_rate={v['error_rate']} p95={v['p95_ms']}ms")
    for m in data["missing_signals"]:
        typer.secho(f"  missing: {m}", fg=typer.colors.YELLOW)
    if data["unhealthy_containers"]:
        typer.secho(f"  unhealthy: {', '.join(data['unhealthy_containers'])}", fg=typer.colors.YELLOW)
    typer.echo(f"saved {RESULTS / 'baseline.json'}")


# ------------------------------------------------------------------ run
@app.command()
def run(target: str = typer.Argument(..., help="service name, e.g. redis"),
        fault: str = typer.Argument(..., help="kill|stop|pause|cpu-limit|memory-limit|network-delay|network-loss|traffic-spike"),
        duration: int = typer.Option(30, help="seconds (capped at 60)"),
        param: list[str] = typer.Option(None, "--param", "-p", help="fault parameter key=value (repeatable)"),
        hypothesis: str = typer.Option("", help="what you expect to happen"),
        error_rate: float = typer.Option(None, help="abort threshold (can only be tightened)"),
        p95_ms: float = typer.Option(None, help="abort threshold in ms (can only be tightened)"),
        baseline_s: int = typer.Option(15, help="steady-state window before injecting"),
        recovery_timeout: int = typer.Option(120),
        dry_run: bool = typer.Option(False, help="validate and print the plan; inject nothing"),
        vote_url: str = "http://localhost:8080/", result_url: str = "http://localhost:8081/"):
    """Run one safe experiment and write results/<id>.json."""
    thresholds = Thresholds(**{k: v for k, v in (("error_rate", error_rate), ("p95_ms", p95_ms)) if v is not None})
    exp = Experiment(id=f"{time.strftime('%Y%m%d-%H%M%S')}-{target}-{fault}", target=target, fault=fault,
                     params=parse_params(param), duration_s=duration, hypothesis=hypothesis, thresholds=thresholds)
    kill = kill_switch()

    if dry_run:
        guard = Guard(managed_from_compose(COMPOSE), kill)
        orch = Orchestrator(Store(":memory:"), make_injector(dry_run=True), guard, RESULTS)
        try:
            result = orch.run(exp, dry_run=True)
        except (SafetyViolation, InjectionError) as e:
            fail(f"REJECTED: {e}")
        typer.echo("DRY RUN: nothing was injected.")
        typer.echo(f"  target {result.experiment.target}, fault {result.experiment.fault}, params {result.experiment.params}")
        t = result.experiment.thresholds
        typer.echo(f"  duration {result.experiment.duration_s}s; abort if error_rate>{t.error_rate:.0%} or p95>{t.p95_ms:.0f}ms for {t.breach_seconds}s")
        for n in result.notes:
            typer.echo(f"  {n}")
        return

    try:
        injector = make_injector()
        guard = Guard(set(injector.list_targets()), kill)
    except Exception as e:
        fail(f"cannot reach Docker: {e}")
    store = Store(RESULTS / "chaosai.db")
    probe = LoadProbe(store, ProbeConfig(vote_url=vote_url, result_url=result_url))
    injector.probe = probe                       # traffic-spike needs the live probe
    watcher = DockerWatcher(store, client=injector.client)
    kill.install_signal_handlers()
    probe.start()
    watcher.start()
    orch = Orchestrator(store, injector, guard, RESULTS, health_fn=lambda: unhealthy(watcher.snapshot()),
                        settle_s=probe.cfg.timeout_s + 0.5)
    try:
        result = orch.run(exp, baseline_s=baseline_s, recovery_timeout_s=recovery_timeout)
    except (SafetyViolation, InjectionError) as e:
        fail(f"REFUSED: {e}")
    except BaseException:
        errors = injector.restore_all()      # last line of defense; never leave a fault behind
        typer.secho(f"experiment interrupted; restore_all errors: {errors or 'none'}", fg=typer.colors.RED, err=True)
        raise
    finally:
        probe.stop()
        watcher.stop()

    b, d = result.baseline, result.during
    fmt = lambda s: f"n={s.n} err={s.error_rate} p50={s.p50_ms} p95={s.p95_ms}"
    typer.echo(f"experiment {result.experiment.id}")
    typer.echo(f"  baseline : {fmt(b)}")
    typer.echo(f"  during   : {fmt(d)}")
    typer.echo(f"  aborted  : {result.aborted}" + (f" ({result.abort_reason})" if result.aborted else ""))
    if result.recovered:
        recovery = "%.0fs" % result.recovery_s
    elif result.abort_reason == "kill switch":
        recovery = "not measured (kill switch engaged)"
    else:
        recovery = "NOT recovered within timeout"
    typer.echo(f"  recovery : {recovery}")
    typer.echo(f"  restarts : {result.restarts or 'none observed'}")
    for m in result.missing_signals:
        typer.secho(f"  missing signal: {m}", fg=typer.colors.YELLOW)
    typer.echo(f"  saved    : {RESULTS / (result.experiment.id + '.json')}")
    if result.rollback_errors:
        fail(f"ROLLBACK PROBLEMS: {result.rollback_errors}\nrun `chaos abort` to retry the restore", 3)


# ------------------------------------------------------------------ abort / reset / status
@app.command()
def abort():
    """Kill switch: stop any running experiment, undo every fault, restart anything stopped/paused."""
    kill_switch().engage("chaos abort")          # a running experiment sees this within ~1 s
    typer.echo(f"kill switch engaged ({RESULTS / 'KILL'})")
    try:
        errors = make_injector().restore_all()
    except Exception as e:
        fail(f"kill switch is set, but I could not reach Docker to restore state: {e}", 1)
    if errors:
        fail("restore problems:\n  " + "\n  ".join(errors), 1)
    typer.echo("all faults undone; managed containers running")


@app.command()
def reset():
    """Restore everything, then clear the kill switch so experiments can run again."""
    try:
        errors = make_injector().restore_all()
    except Exception as e:
        fail(f"cannot reach Docker, kill switch left engaged: {e}", 1)
    if errors:
        fail("not clearing the kill switch; restore problems:\n  " + "\n  ".join(errors), 1)
    kill_switch().reset()
    typer.echo("kill switch cleared")


@app.command()
def status():
    """Kill switch, recorded active faults, and container state."""
    typer.echo(f"kill switch : {'ENGAGED' if kill_switch().engaged else 'clear'}")
    state = RESULTS / "active_faults.json"
    faults = json.loads(state.read_text()) if state.exists() else {}
    typer.echo(f"active faults: {[f['fault'] + '@' + f['target'] for f in faults.values()] or 'none'}")
    try:
        watcher = DockerWatcher(Store(":memory:"))
        for svc, (st, health) in sorted(watcher.snapshot().items()):
            typer.echo(f"  {svc:8} {st:10} {health or '-'}")
    except Exception as e:
        typer.secho(f"  docker unavailable: {e}", fg=typer.colors.YELLOW)


if __name__ == "__main__":
    app()
