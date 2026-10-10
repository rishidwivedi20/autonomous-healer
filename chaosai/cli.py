"""chaos CLI:  discover | plan | agent | run | baseline | abort | reset | status | config

Run as `python -m chaosai <command>` (or `chaos <command>` via the chaos.cmd wrapper).
Parameters come from config.yaml (`chaos config` shows the effective values); flags override them.
Ctrl+C / SIGTERM during `run` or `agent` engages the kill switch: the experiment aborts, everything is
rolled back, and further runs are refused until `chaos reset`.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import typer
import yaml

from . import config as config_mod
from . import dashboard as dashboard_mod
from . import discovery, planner
from .agent import InjectorAgent
from .injector import InjectionError
from .models import Experiment, Graph, Thresholds
from .observer import DockerWatcher, LoadProbe, ProbeConfig, Store, unhealthy
from .orchestrator import Orchestrator, measure_baseline
from .safety import ExperimentLock, Guard, KillSwitch, SafetyViolation, is_managed

app = typer.Typer(help="Safe chaos experiments against the voting app.", no_args_is_help=True)

RESULTS = Path(os.environ.get("CHAOSAI_RESULTS", "results"))
COMPOSE = Path(os.environ.get("CHAOSAI_COMPOSE") or config_mod.load().app.compose_file)


def fail(msg: str, code: int = 2):
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def cfg() -> config_mod.Config:
    try:
        return config_mod.load()
    except ValueError as e:
        fail(str(e))


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


def default_thresholds(c: config_mod.Config, error_rate=None, p95_ms=None) -> Thresholds:
    return Thresholds(error_rate=error_rate if error_rate is not None else c.safety.abort_error_rate,
                      p95_ms=p95_ms if p95_ms is not None else c.safety.abort_p95_ms,
                      breach_seconds=c.safety.breach_seconds)


def probe_config(c: config_mod.Config, vote_url=None, result_url=None) -> ProbeConfig:
    return ProbeConfig(vote_url=vote_url or c.app.vote_url, result_url=result_url or c.app.result_url,
                       interval_s=c.probe.interval_s, timeout_s=c.probe.timeout_s)


def orchestrator_kwargs(c: config_mod.Config) -> dict:
    return dict(hold_s=c.experiment.recovery_hold_s, tolerance=c.experiment.recovery_tolerance,
                slack_ms=c.experiment.recovery_latency_slack_ms, queue_slack=c.experiment.recovery_queue_slack,
                votes_fraction=c.experiment.recovery_votes_fraction,
                preflight_max_error_rate=c.safety.preflight_max_error_rate)


def offline_orchestrator(c: config_mod.Config) -> tuple[Orchestrator, Guard]:
    """Dry-run wiring: no Docker, no probe. The allowlist comes from the compose file's labels."""
    guard = Guard(managed_from_compose(COMPOSE), kill_switch(), c.safety.max_duration_s)
    return Orchestrator(Store(":memory:"), make_injector(dry_run=True), guard, RESULTS,
                        **orchestrator_kwargs(c)), guard


def start_live(c: config_mod.Config, kill: KillSwitch, vote_url=None, result_url=None, interval_s=None,
               gauge_every_s=2.0, install_signals=True) -> SimpleNamespace:
    """Connect to Docker, start the probe and watcher, install signal handlers. Exits cleanly without Docker."""
    try:
        injector = make_injector()
        guard = Guard(set(injector.list_targets()), kill, c.safety.max_duration_s,
                      lock=ExperimentLock(RESULTS / "experiment.lock"))
    except Exception as e:
        fail(f"cannot reach Docker: {e}")
    store = Store(RESULTS / "chaosai.db")
    pc = probe_config(c, vote_url, result_url)
    if interval_s:
        pc.interval_s = interval_s
    probe = LoadProbe(store, pc)
    injector.probe = probe                       # traffic-spike needs the live probe
    watcher = DockerWatcher(store, client=injector.client, gauge_every_s=gauge_every_s)
    if install_signals:
        kill.install_signal_handlers()
    probe.start()
    watcher.start()
    health = lambda: unhealthy(watcher.snapshot())
    orch = Orchestrator(store, injector, guard, RESULTS, health_fn=health, settle_s=pc.timeout_s + 0.5,
                        **orchestrator_kwargs(c))
    return SimpleNamespace(store=store, probe=probe, watcher=watcher, injector=injector, guard=guard,
                           orch=orch, health=health, stop=lambda: (probe.stop(), watcher.stop()))


def load_graph(compose: Path) -> Graph:
    graph_file = RESULTS / "graph.json"
    if graph_file.exists():
        return Graph.model_validate(json.loads(graph_file.read_text(encoding="utf-8")))
    graph, notes = discovery.discover(compose, compose.parent / "graph.overrides.yaml")
    discovery.write_graph(graph, graph_file, notes)
    typer.echo(f"(no graph.json yet; discovered it from {compose})")
    return graph


def print_plan(result, n_history: int) -> None:
    who = f"LLM ({result.model})" if result.planner == "llm" else "heuristic"
    typer.echo(f"planner: {who}; {n_history} past experiment(s) considered")
    if result.fallback_reason:
        typer.secho(f"  fell back to heuristic: {result.fallback_reason}", fg=typer.colors.YELLOW)
    for n in result.notes:
        typer.echo(f"  {n}")
    typer.echo("")
    for p in result.experiments:
        f = p.factors
        typer.echo(f"{p.rank}. {p.target} / {p.fault}   score {p.risk_score}  "
                   f"(criticality {f['criticality']:g} x fragility {f['fragility']:g} x untested {f['untested']:g} x fault {f['fault_weight']:g})")
        typer.echo(f"     duration {p.duration_s}s, params {p.params or '-'}")
        typer.echo(f"     why:        {p.reason}")
        typer.echo(f"     hypothesis: {p.hypothesis}")


def make_plan(c: config_mod.Config, top: int | None, offline: bool, compose: Path):
    graph = load_graph(compose)
    history = planner.load_history(RESULTS)
    result = planner.make_plan(graph, history, top or c.planner.top, use_llm=c.planner.use_llm and not offline,
                               cache_dir=RESULTS / "llm_cache", cfg=c.planner, faults=c.faults)
    return result, len(history)


# ------------------------------------------------------------------ config
@app.command("config")
def show_config():
    """Print the effective parameters (config.yaml merged over the defaults)."""
    c = cfg()
    typer.echo(f"# {config_mod.config_path()} {'(found)' if config_mod.config_path().exists() else '(not found: defaults)'}")
    typer.echo(yaml.safe_dump(c.model_dump(mode="json"), sort_keys=False).rstrip())


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


# ------------------------------------------------------------------ plan
@app.command()
def plan(top: int = typer.Option(None, "--top", "-n", help="how many experiments to plan (default: planner.top)"),
         offline: bool = typer.Option(False, help="skip the LLM; use the deterministic heuristic only"),
         dry_run: bool = typer.Option(False, help="print the ranked plan; do not write results/plan.json"),
         compose: Path = typer.Option(COMPOSE)):
    """Rank which experiments to run first (heuristic risk score, re-ranked by Claude when available)."""
    result, n_history = make_plan(cfg(), top, offline, compose)
    print_plan(result, n_history)
    if dry_run:
        typer.echo("\n(dry run: results/plan.json not written)")
        return
    out = RESULTS / "plan.json"
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"\nsaved {out}")


# ------------------------------------------------------------------ agent
@app.command()
def agent(top: int = typer.Option(None, "--top", "-n", help="experiments to plan (default: planner.top)"),
          max_experiments: int = typer.Option(None, "--max", help="cap on experiments actually run (default: agent.max_experiments)"),
          offline: bool = typer.Option(False, help="skip the LLM; plan with the heuristic only"),
          dry_run: bool = typer.Option(False, help="plan and validate everything; inject nothing"),
          yes: bool = typer.Option(False, "--yes", "-y", help="do not ask for confirmation"),
          compose: Path = typer.Option(COMPOSE)):
    """The injector agent: plan, then introduce the faults one at a time through the safety guard."""
    c = cfg()
    result, n_history = make_plan(c, top, offline, compose)
    print_plan(result, n_history)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    exps = [e.model_copy(update={"thresholds": default_thresholds(c)}) for e in planner.to_experiments(result, stamp)]
    acfg = c.agent.model_copy(update={"max_experiments": max_experiments or c.agent.max_experiments})
    n_run = min(len(exps), acfg.max_experiments)

    if dry_run:
        orch, guard = offline_orchestrator(c)
        typer.echo(f"\nDRY RUN: the agent would inject {n_run} fault(s); nothing is injected.")
        run = InjectorAgent(orch, guard, lambda: [], acfg, c.experiment.baseline_s,
                            c.experiment.recovery_timeout_s, log=typer.echo).run(exps, dry_run=True, planner=result.planner)
        if run.status == "stopped":
            raise typer.Exit(1)
        return

    typer.echo(f"\nThe agent will inject {n_run} fault(s) into containers labeled chaosai.managed=true on this machine.")
    if not yes and not typer.confirm("Continue?"):
        fail("cancelled; nothing injected", 1)
    live = start_live(c, kill_switch())
    try:
        run = InjectorAgent(live.orch, live.guard, live.health, acfg, c.experiment.baseline_s,
                            c.experiment.recovery_timeout_s, log=typer.echo).run(exps, planner=result.planner)
    except BaseException:
        errors = live.injector.restore_all()     # last line of defense; never leave a fault behind
        typer.secho(f"agent interrupted; restore_all errors: {errors or 'none'}", fg=typer.colors.RED, err=True)
        raise
    finally:
        live.stop()
    out = RESULTS / "agent-runs" / f"{stamp}.json"     # own folder: results/*.json is experiment history
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    done = [e for e in run.entries if e.status == "done"]
    typer.echo(f"\nagent {run.status}: {len(done)} experiment(s) run, "
               f"{sum(e.aborted for e in done)} auto-aborted, {sum(bool(e.recovered) for e in done)} recovered")
    if run.stop_reason:
        typer.secho(f"stop reason: {run.stop_reason}", fg=typer.colors.YELLOW)
    if run.not_run:
        typer.echo(f"not run: {', '.join(run.not_run)}")
    typer.echo(f"saved {out}")
    if run.status == "stopped":
        raise typer.Exit(1)


# ------------------------------------------------------------------ dashboard
@app.command()
def dashboard(port: int = typer.Option(None, help="default: dashboard.port"),
              rate: float = typer.Option(None, help="probe requests per second per journey (default: dashboard.probe_rate_per_s)"),
              open_browser: bool = typer.Option(False, "--open", help="open the page in your browser")):
    """Live web dashboard: watch every metric move while you inject real faults with the buttons."""
    c = cfg()
    port = port or c.dashboard.port
    c.dashboard.probe_rate_per_s = rate or c.dashboard.probe_rate_per_s
    kill = kill_switch()
    live = start_live(c, kill, interval_s=1.0 / c.dashboard.probe_rate_per_s, gauge_every_s=1.0, install_signals=False)
    table = dashboard_mod.ContainerTable(live.watcher.table)
    table.start()
    web = dashboard_mod.App(c, live.store, live.injector, kill, live.orch, lambda: table.data)
    try:
        server = dashboard_mod.make_server(web, port)
    except OSError as e:
        live.stop()
        table.stop()
        fail(f"cannot listen on 127.0.0.1:{port}: {e}")
    url = f"http://127.0.0.1:{port}/"
    typer.echo(f"dashboard: {url}   (probe {c.dashboard.probe_rate_per_s:g} req/s per journey; Ctrl+C to stop)")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engaged_here = False
        if web.runner.snapshot()["running"]:                       # never leave a fault behind on exit
            typer.echo("stopping the running experiment ...")
            kill.engage("dashboard shutdown")
            engaged_here = True
            if web.runner.thread:
                web.runner.thread.join(timeout=30)
        errors = live.injector.restore_all()
        if errors:
            typer.secho(f"restore problems on exit: {errors}", fg=typer.colors.RED, err=True)
        elif engaged_here:
            kill.reset()
        server.server_close()
        table.stop()
        live.stop()


# ------------------------------------------------------------------ baseline
@app.command()
def baseline(seconds: int = typer.Option(120, help="how long to observe steady state"),
             vote_url: str = typer.Option(None), result_url: str = typer.Option(None)):
    """Observe the healthy system and save p50/p95/error rate to results/baseline.json."""
    c = cfg()
    try:
        import docker
        client = docker.from_env()
        client.ping()
    except Exception as e:
        fail(f"cannot reach Docker: {e}")
    store = Store(RESULTS / "chaosai.db")
    probe = LoadProbe(store, probe_config(c, vote_url, result_url)).start()
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
        duration: int = typer.Option(None, help="seconds (default: faults.<fault>.duration_s; hard cap 60)"),
        param: list[str] = typer.Option(None, "--param", "-p", help="fault parameter key=value (repeatable)"),
        hypothesis: str = typer.Option("", help="what you expect to happen"),
        error_rate: float = typer.Option(None, help="abort threshold (can only be tightened)"),
        p95_ms: float = typer.Option(None, help="abort threshold in ms (can only be tightened)"),
        baseline_s: int = typer.Option(None, help="steady-state window before injecting (default: experiment.baseline_s)"),
        recovery_timeout: int = typer.Option(None, help="default: experiment.recovery_timeout_s"),
        dry_run: bool = typer.Option(False, help="validate and print the plan; inject nothing"),
        vote_url: str = typer.Option(None), result_url: str = typer.Option(None)):
    """Run one safe experiment and write results/<id>.json."""
    c = cfg()
    fc = c.faults.get(fault)
    exp = Experiment(id=f"{time.strftime('%Y%m%d-%H%M%S')}-{target}-{fault}", target=target, fault=fault,
                     params={**(fc.params if fc else {}), **parse_params(param)},
                     duration_s=duration or (fc.duration_s if fc else 30), hypothesis=hypothesis,
                     thresholds=default_thresholds(c, error_rate, p95_ms))
    baseline_s = baseline_s or c.experiment.baseline_s
    recovery_timeout = recovery_timeout or c.experiment.recovery_timeout_s

    if dry_run:
        orch, _ = offline_orchestrator(c)
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

    live = start_live(c, kill_switch(), vote_url, result_url)
    try:
        result = live.orch.run(exp, baseline_s=baseline_s, recovery_timeout_s=recovery_timeout)
    except (SafetyViolation, InjectionError) as e:
        fail(f"REFUSED: {e}")
    except BaseException:
        errors = live.injector.restore_all()      # last line of defense; never leave a fault behind
        typer.secho(f"experiment interrupted; restore_all errors: {errors or 'none'}", fg=typer.colors.RED, err=True)
        raise
    finally:
        live.stop()

    b, d = result.baseline, result.during
    fmt = lambda s: f"n={s.n} err={s.error_rate} p50={s.p50_ms} p95={s.p95_ms}"
    typer.echo(f"experiment {result.experiment.id}")
    typer.echo(f"  baseline : {fmt(b)}")
    typer.echo(f"  during   : {fmt(d)}")
    typer.echo(f"  aborted  : {result.aborted}" + (f" ({result.abort_reason})" if result.aborted else ""))
    if result.recovered:
        recovery = "%.0fs" % result.recovery_s
        if result.pipeline_assessed:
            recovery += f" (probe {result.recovery_probe_s:.0f}s, pipeline {result.recovery_pipeline_s:.0f}s)"
        else:
            recovery += " (probe only: pipeline gauges unavailable)"
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
