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
from .envfile import load_env, mask
from . import dashboard as dashboard_mod
from . import analyzer as analyzer_mod
from . import discovery, planner
from . import report as report_mod
from .evidence import Evidence
from .agent import InjectorAgent
from .injector import InjectionError
from .models import Experiment, Graph, Thresholds
from .observer import DockerWatcher, LoadProbe, ProbeConfig, Store, unhealthy
from .orchestrator import Orchestrator, measure_baseline
from .safety import ExperimentLock, Guard, KillSwitch, SafetyViolation, is_managed

load_env()          # ANTHROPIC_API_KEY from a git-ignored .env, if present

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


# ------------------------------------------------------------------ llm-check
@app.command("llm-check")
def llm_check(send: bool = typer.Option(True, help="--no-send only checks that a key is configured"),
              provider: str = typer.Option(None, help="anthropic | gemini | mistral | openrouter | pool (default: analyzer.provider)"),
              models: bool = typer.Option(False, "--models", help="gemini / mistral / openrouter: list the models this key can use"),
              model: str = typer.Option(None, "--model", help="gemini / mistral / openrouter: model to test (default: analyzer.llm_model when it matches the provider)"),
              match: str = typer.Option(None, "--match", help="with --models: only names containing this text (OpenRouter lists hundreds)")):
    """Check the LLM key: is one configured, and does one tiny request work? (Costs a fraction of a cent or nothing.)"""
    c = cfg()
    provider = provider or c.analyzer.provider
    if provider == "gemini":
        return llm_check_gemini(c, send, models, model, match)
    if provider == "mistral":
        return llm_check_mistral(c, send, models, model, match)
    if provider == "openrouter":
        return llm_check_openrouter(c, send, models, model, match)
    if provider == "pool":
        return llm_check_pool(c, send)
    if provider != "anthropic":
        fail(f"unknown provider {provider!r}; use anthropic, gemini, mistral, openrouter or pool")
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not key:
        fail("no key found. Set ANTHROPIC_API_KEY (see README: 'Getting an Anthropic API key') or put it in .env", 1)
    typer.echo(f"key found: {mask(key)}  (model: {c.analyzer.llm_model})")
    if not send:
        return
    import anthropic
    try:
        resp = anthropic.Anthropic().messages.create(
            model=c.analyzer.llm_model, max_tokens=1024, output_config={"effort": "low"},
            messages=[{"role": "user", "content": "Reply with the single word: ok"}])
    except anthropic.AuthenticationError:
        fail("the key was rejected (401). Check that it is copied completely and not revoked.", 1)
    except anthropic.PermissionDeniedError as e:
        fail(f"the key is valid but not allowed to use this model/workspace (403): {e}", 1)
    except anthropic.RateLimitError:
        fail("rate limited (429); the key works but try again shortly", 1)
    except anthropic.APIStatusError as e:
        fail(f"API error {e.status_code}: {str(e)[:300]}\n(a 400 about credit balance means the account has no credits yet)", 1)
    except anthropic.APIConnectionError as e:
        fail(f"cannot reach the API ({type(e).__name__}); check your network/VPN/proxy", 1)
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
    typer.echo(f"OK: {resp.model} answered {text.strip()!r}; stop_reason={resp.stop_reason}; "
               f"tokens in={resp.usage.input_tokens} out={resp.usage.output_tokens}")


def llm_check_gemini(c: config_mod.Config, send: bool, list_models: bool, model: str | None = None, match: str | None = None) -> None:
    from .gemini import GeminiClient, GeminiError, api_key
    key = api_key()
    if not key:
        fail("no Gemini key found. Create one at https://aistudio.google.com/apikey and put GEMINI_API_KEY=... in .env", 1)
    typer.echo(f"key found: {mask(key)}  (planner model: {c.planner.llm_model}, analyzer model: {c.analyzer.llm_model}; "
               f"provider in config: planner={c.planner.provider}, analyzer={c.analyzer.provider})")
    if not send and not list_models:
        return
    try:
        client = GeminiClient()
        if list_models:
            names = [n for n in client.list_models() if not match or match.lower() in n.lower()]
            typer.echo(f"{len(names)} model(s) this key can call:\n  " + "\n  ".join(names))
        if not send:
            return
        model = model or (c.analyzer.llm_model if c.analyzer.provider == "gemini" else "gemini-3.8-flash")
        resp = client.messages.create(model=model, max_tokens=64, messages=[{"role": "user", "content": "Reply with the single word: ok"}])
    except GeminiError as e:
        hint = {400: "  (a 400 usually means the key is invalid or the model name is wrong: try --models)",
                403: "  (the key is valid but not allowed to use this model/region)",
                404: "  (that model name does not exist for this key: try --models)",
                429: "  (the free tier is rate limited per model; the message says when it resets)"}.get(e.status or 0, "")
        fail(f"Gemini error: {e}{hint}", 1)
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
    typer.echo(f"OK: {model} answered {text.strip()!r}; stop_reason={resp.stop_reason}; "
               f"tokens in={resp.usage.input_tokens} out={resp.usage.output_tokens}")


def llm_check_mistral(c: config_mod.Config, send: bool, list_models: bool, model: str | None = None, match: str | None = None) -> None:
    from .mistral import MistralClient, MistralError, api_key
    key = api_key()
    if not key:
        fail("no Mistral key found. Create one at https://console.mistral.ai/api-keys and put MISTRAL_API_KEY=... in .env", 1)
    typer.echo(f"key found: {mask(key)}  (provider in config: planner={c.planner.provider}, analyzer={c.analyzer.provider})")
    if not send and not list_models:
        return
    try:
        client = MistralClient()
        if list_models:
            names = [n for n in client.list_models() if not match or match.lower() in n.lower()]
            typer.echo(f"{len(names)} chat model(s) with function calling this key can use:\n  " + "\n  ".join(names))
        if not send:
            return
        model = model or (c.analyzer.llm_model if c.analyzer.provider == "mistral" else "mistral-small-latest")
        resp = client.messages.create(model=model, max_tokens=64, messages=[{"role": "user", "content": "Reply with the single word: ok"}])
    except MistralError as e:
        hint = {400: "  (check the model name: try --models)", 401: "  (the key was rejected: check it is copied completely and not revoked)",
                403: "  (the key is valid but not allowed to use this model)", 404: "  (that model name does not exist: try --models)",
                429: "  (rate limited; the free Experiment tier has low limits, wait and retry)"}.get(e.status or 0, "")
        if "0 requests per minute" in str(e):               # waiting cannot help; the message already says what to do
            hint = ""
        fail(f"Mistral error: {e}{hint}", 1)
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
    typer.echo(f"OK: {model} answered {text.strip()!r}; stop_reason={resp.stop_reason}; "
               f"tokens in={resp.usage.input_tokens} out={resp.usage.output_tokens}")


def llm_check_openrouter(c: config_mod.Config, send: bool, list_models: bool, model: str | None = None, match: str | None = None) -> None:
    from .openrouter import OpenRouterClient, OpenRouterError, api_key
    key = api_key()
    if not key:
        fail("no OpenRouter key found. Create one at https://openrouter.ai/keys and put OPENROUTER_API_KEY=... in .env", 1)
    typer.echo(f"key found: {mask(key)}  (provider in config: planner={c.planner.provider}, analyzer={c.analyzer.provider})")
    if not send and not list_models:
        return
    try:
        client = OpenRouterClient()
        if list_models:
            names = [n for n in client.list_models() if not match or match.lower() in n.lower()]
            shown = names[:200]
            typer.echo(f"{len(names)} model(s) that support tools" + (f" matching {match!r}" if match else "") + ":\n  " + "\n  ".join(shown)
                       + (f"\n  ... and {len(names) - len(shown)} more (narrow with --match)" if len(names) > len(shown) else ""))
        if not send:
            return
        model = model or (c.analyzer.llm_model if c.analyzer.provider == "openrouter" else "openai/gpt-oss-120b:free")
        resp = client.messages.create(model=model, max_tokens=256, messages=[{"role": "user", "content": "Reply with the single word: ok"}])
    except OpenRouterError as e:
        hint = {401: "  (the key was rejected: check it is copied completely and not revoked)",
                402: "  (this model needs credits: add some on openrouter.ai, or use a :free model)",
                404: "  (no such model, or no provider supports tools/JSON for it: try --models --match <text>)",
                429: "  (rate limited; free models have per-minute and daily limits)"}.get(e.status or 0, "")
        fail(f"OpenRouter error: {e}{hint}", 1)
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
    typer.echo(f"OK: {model} answered {text.strip()!r}; stop_reason={resp.stop_reason}; "
               f"tokens in={resp.usage.input_tokens} out={resp.usage.output_tokens}")


def llm_check_pool(c: config_mod.Config, send: bool) -> None:
    from .pool import PoolClient, PoolError
    members = c.analyzer.members if c.analyzer.provider == "pool" else c.planner.members
    if not members:
        fail("no pool configured: set `provider: pool` and a `members:` list under analyzer: (and planner:) in config.yaml", 1)
    try:
        pool = PoolClient(members, state_path=RESULTS / "llm_pool.json")
    except PoolError as e:
        fail(str(e), 1)
    typer.echo(f"{len(members)} pool member(s):")
    for label, left, reason in pool.status():
        typer.echo(f"  {label}: " + (f"on cooldown for about {left / 60:.0f} more min ({reason})" if left > 0 else "available"))
    if not send:
        return
    typer.echo("\none tiny request per member (this spends one request of each member's quota):")
    results = pool.check()
    for label, worked, detail in results:
        typer.echo(f"  {'OK  ' if worked else 'FAIL'} {label}: {detail}")
    if not any(w for _, w, _ in results):
        fail("no pool member answered", 1)


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
def execute_agent(c: config_mod.Config, exps: list[Experiment], acfg, planner_name: str, stamp: str):
    """Run the injector agent for real (shared by `agent` and `autopilot`), save the run, print the summary."""
    live = start_live(c, kill_switch())
    try:
        run = InjectorAgent(live.orch, live.guard, live.health, acfg, c.experiment.baseline_s,
                            c.experiment.recovery_timeout_s, log=typer.echo).run(exps, planner=planner_name)
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
    return run


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
    run = execute_agent(c, exps, acfg, result.planner, stamp)
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
    graph_obj, graph_notes = discovery.discover(COMPOSE, COMPOSE.parent / "graph.overrides.yaml")
    discovery.write_graph(graph_obj, RESULTS / "graph.json", graph_notes)     # keep results/graph.json in step with what is drawn
    graph_dump = graph_obj.model_dump()
    findings = report_mod.Live(RESULTS, live.store, lambda: graph_dump)
    web = dashboard_mod.App(c, live.store, live.injector, kill, live.orch, lambda: table.data, graph_fn=lambda: graph_dump,
                            report_fn=findings.get)
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


# ------------------------------------------------------------------ analyze
def print_analysis(a) -> None:
    how = f"LLM agent ({a.model})" if a.generated_by == "llm" else "deterministic rules"
    typer.echo(f"analysis by: {how}")
    if a.fallback_reason:
        typer.secho(f"  rule-based fallback because: {a.fallback_reason}", fg=typer.colors.YELLOW)
    typer.echo(f"\n{a.summary}\n")
    typer.echo("WEAKNESSES (most severe first)")
    for i, w in enumerate(a.weaknesses, 1):
        typer.echo(f"{i}. [{w.severity.upper()}] {w.title}   (services: {', '.join(w.services)})")
        typer.echo(f"     {w.description}")
        for e in w.evidence:
            typer.echo(f"     evidence: {e.claim}   [{e.experiment_id}]")
    typer.echo("\nFIXES")
    for i, f in enumerate(a.fixes, 1):
        typer.echo(f"{i}. {f.title}   (addresses: {'; '.join(f.addresses) or '-'})")
        typer.echo(f"     why: {f.rationale}")
        typer.echo("     change:")
        for line in f.change.splitlines():
            typer.echo(f"       {line}")
    for n in a.notes[:8]:
        typer.echo(f"note: {n}")


@app.command()
def analyze(ids: list[str] = typer.Option(None, "--id", "-i", help="experiment id(s) to analyse (default: all stored experiments)"),
            last: int = typer.Option(None, "--last", help="only the most recent N experiments"),
            offline: bool = typer.Option(False, help="skip the LLM; use the deterministic rule-based analyzer"),
            dry_run: bool = typer.Option(False, help="print the analysis; do not write results/analysis.json")):
    """Explain the stored experiments: per-experiment root cause, ranked weaknesses, and cited fixes."""
    c = cfg()
    ev = Evidence(RESULTS)
    scope = list(ids) if ids else ev.ids()
    unknown = [i for i in scope if i not in ev.ids()]
    if unknown:
        fail(f"unknown experiment id(s): {', '.join(unknown)}\nknown: {', '.join(ev.ids()) or '(none)'}")
    if last:
        scope = scope[-last:]
    try:
        result = analyzer_mod.analyze(ev, scope, c.analyzer, use_llm=not offline, cache_dir=RESULTS / "llm_cache")
    except analyzer_mod.AnalyzerError as e:
        fail(str(e), 1)
    typer.echo(f"{len(scope)} experiment(s) in scope")
    print_analysis(result)
    if dry_run:
        typer.echo("\n(dry run: results/analysis.json not written)")
        return
    out = RESULTS / "analysis.json"
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"\nsaved {out}")


# ------------------------------------------------------------------ report / autopilot
def make_report(c: config_mod.Config, scope: list[str] | None, offline: bool, compose: Path):
    """Analyze the stored experiments, score them, and write results/findings.{json,md,html}."""
    ev = Evidence(RESULTS)
    ids = ev.ids() if scope is None else [i for i in scope if i in ev.ids()]
    if not ids:
        fail("no stored experiments to report on; run `chaos autopilot` (or `chaos agent`) first", 1)
    try:
        analysis = analyzer_mod.analyze(ev, ids, c.analyzer, use_llm=not offline, cache_dir=RESULTS / "llm_cache")
    except analyzer_mod.AnalyzerError as e:
        fail(str(e), 1)
    (RESULTS / "analysis.json").write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    rep = report_mod.build(ev, analysis, load_graph(compose), ids)
    paths = report_mod.write(RESULTS, rep)
    sc = rep["score"]
    how = f"LLM agent ({analysis.model})" if analysis.generated_by == "llm" else "deterministic rules"
    typer.echo(f"\n{len(ids)} experiment(s), analysis by {how}")
    if analysis.fallback_reason:
        typer.secho(f"  rule-based fallback because: {analysis.fallback_reason}", fg=typer.colors.YELLOW)
    if sc["score"] is None:
        typer.echo("resilience score: not computable (nothing measurable)")
    else:
        typer.echo(f"RESILIENCE SCORE {sc['score']:.1f}/100 ({sc['band']})   "
                   + "  ".join(f"{k.replace('_', ' ')} {v if v is None else format(v, '.0f')}" for k, v in sc["by_component"].items()))
    for n, w in enumerate(rep["weaknesses"][:5], 1):
        typer.echo(f"  {n}. [{w['severity'].upper()}] {w['title']}")
    for kind in ("md", "html", "json"):
        typer.echo(f"saved {paths[kind]}")
    return rep


@app.command()
def report(last: int = typer.Option(None, "--last", help="only the most recent N experiments (default: all stored)"),
           offline: bool = typer.Option(False, help="skip the LLM; use the deterministic rule-based analyzer"),
           compose: Path = typer.Option(COMPOSE)):
    """Write the findings report (score, ranked weaknesses, cited fixes) from the stored experiments."""
    c = cfg()
    ids = Evidence(RESULTS).ids()
    make_report(c, ids[-last:] if last else None, offline, compose)


@app.command()
def autopilot(top: int = typer.Option(None, "--top", "-n", help="experiments to plan (default: planner.top)"),
              max_experiments: int = typer.Option(None, "--max", help="cap on experiments actually run (default: agent.max_experiments)"),
              offline: bool = typer.Option(False, help="skip the LLM everywhere; heuristic planner + rule-based analyzer"),
              dry_run: bool = typer.Option(False, help="plan and validate; inject nothing"),
              yes: bool = typer.Option(False, "--yes", "-y", help="do not ask for confirmation"),
              compose: Path = typer.Option(COMPOSE)):
    """Hands-off end to end: plan, inject with the agent, analyze, score, and write the findings report."""
    if dry_run:
        return agent(top=top, max_experiments=max_experiments, offline=offline, dry_run=True, yes=True, compose=compose)
    c = cfg()
    result, n_history = make_plan(c, top, offline, compose)
    print_plan(result, n_history)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    exps = [e.model_copy(update={"thresholds": default_thresholds(c)}) for e in planner.to_experiments(result, stamp)]
    acfg = c.agent.model_copy(update={"max_experiments": max_experiments or c.agent.max_experiments})
    n_run = min(len(exps), acfg.max_experiments)
    typer.echo(f"\nAUTOPILOT: the agent will inject {n_run} fault(s) one at a time into containers labeled "
               f"chaosai.managed=true on this machine, then analyze, score and write the report. Kill switch: Ctrl+C or `chaos abort`.")
    if not yes and not typer.confirm("Continue?"):
        fail("cancelled; nothing injected", 1)
    run = execute_agent(c, exps, acfg, result.planner, stamp)
    done = [e.id for e in run.entries if e.status == "done"]
    if not done:
        fail("the agent ran no experiments, so there is nothing to report", 1)
    typer.echo("\nanalyzing, scoring and writing the report ...")
    make_report(c, done, offline, compose)
    if run.status == "stopped":
        raise typer.Exit(1)


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
