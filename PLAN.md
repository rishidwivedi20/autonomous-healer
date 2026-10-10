# chaos-ai: Implementation Plan

Inject failures into a small microservices app, observe the real impact, then have an AI agent diagnose what broke from stored evidence and recommend concrete, cited fixes.

**Loop:** inject fault -> observe real impact -> agent investigates (using tools) -> diagnosis + ranked fixes citing evidence -> (stretch) apply fix, re-run, score improves.

## Hard rules
1. **Real data only.** No mocks, no synthetic anomalies, no hardcoded scenario metrics in the pipeline. If a signal is unavailable, record it as missing and say so in the report. The agent may only cite numbers that exist in stored results.
2. **Safety first.** `safety.py` is written and tested before any injection code exists.
3. **Minimum footprint.** Laptop-friendly: under 2 GB extra RAM, no Kubernetes, no Prometheus/Grafana, no Chaos Mesh.
4. **Show real output** after every step. Never claim something works without running it.
5. **If blocked,** tell the team lead and propose the simplest alternative instead of expanding scope.

## Stack
Python 3.11+, docker SDK, httpx (async), typer, pydantic v2, sqlite3 (stdlib), pytest, anthropic SDK (Day 2), streamlit (Day 2 stretch). Single process, one SQLite file. `ANTHROPIC_API_KEY` lives in an env var and is never committed.

## Target app
Docker's example-voting-app via Docker Compose (vote -> redis -> worker -> postgres -> result).
- Label `chaosai.managed=true` on every service.
- Healthchecks and `restart: unless-stopped` so recovery is observable.
- ONE replica of each service on purpose (the weakness the agent should find).
- Known gap: the worker has no healthcheck (its image exposes nothing to probe). The agent should flag this.

## Architecture
```
chaosai/
  models.py        pydantic models (shared contract)
  injector.py      Injector interface + DockerInjector
  discovery.py     graph.json from compose file + container env vars
  observer.py      async load probe + docker state/restart watcher -> SQLite
  safety.py        guardrails + kill switch
  orchestrator.py  experiment lifecycle
  planner.py       (Day 2) heuristic + LLM ranking, schema-validated
  analyzer.py      (Day 2) tool-using agent
  scoring.py       (Day 2) resilience score
  cli.py           discover | baseline | run | abort | reset | plan | report
deploy/            docker-compose.yml, Makefile, tasks.ps1 (Windows, no make)
results/           one JSON per experiment
tests/
```

**Fault catalog (DockerInjector):** container kill, stop, pause, CPU limit (`docker update --cpus`), memory limit, network delay/loss via `tc netem` in the container (Pumba if `tc` is unavailable), traffic spike from the load probe.

**Experiment lifecycle:** preflight health check -> baseline window -> inject -> monitor each second -> rollback -> measure recovery until metrics are within 10% of baseline -> write `results/<id>.json`.

**Safety guard:**
- Allowlist: only containers labeled `chaosai.managed=true`.
- One experiment at a time; max duration 60 s.
- Auto-rollback when error rate or p95 breaches the threshold for 3 consecutive seconds. Planners may only tighten thresholds, never loosen them.
- Kill switch: `chaos abort` (file-based, works from another terminal), SIGINT/SIGTERM handler, `try/finally` around every injection (undoes every fault, restarts anything stopped or paused).
- `--dry-run`.

**Result JSON (pydantic):** experiment id, target, fault, params, hypothesis, baseline stats, during-fault stats (error rate, p50/p95 per second), recovery_s, aborted + reason, container events (restarts, health transitions), timestamps, missing signals.

## Status

| Item | Status |
|---|---|
| Repo, venv, pinned `requirements.txt` | Done |
| `deploy/` compose, Makefile, `tasks.ps1` | Written, **never run** (no Docker on the dev laptop) |
| `models.py`, `safety.py`, `tests/test_safety.py` | Written, **tests not run yet** |
| injector, discovery, observer, orchestrator, cli | Not started |

**Blocker:** Docker Desktop (WSL2) must be installed on at least one laptop before Day 1 can proceed.

## Team split (3 people)

Person A owns the shared contract (`models.py` and the interfaces below). Freeze it in the first 30 minutes so B and C can code against it.

### Person A: Infrastructure, safety, orchestration (critical path)
Owns: `deploy/`, `safety.py`, `injector.py`, `orchestrator.py`, `cli.py`

| Day | Tasks | Done when |
|---|---|---|
| 1 | 1. Install Docker, run `tasks.ps1 up`, fix healthchecks and image pulls | Vote and result UIs load, all containers healthy |
| 1 | 2. Run `pytest` on `safety.py`, fix failures | Allowlist, duration cap and threshold tests green |
| 1 | 3. `DockerInjector`: kill, stop, pause, cpus/memory, `tc netem` (Pumba fallback). Every injection in `try/finally`. | Each fault injects and rolls back cleanly |
| 1 | 4. `orchestrator.py` full lifecycle | One experiment runs end to end |
| 1 | 5. `cli.py`: discover, baseline, run, abort, reset, `--dry-run` | Commands work |
| 1 | 6. Checkpoint runs: (a) kill redis, (b) delay that MUST trip the auto-abort, plus kill switch proven mid-experiment | Real output saved, abort proven |
| 2 | Run the 4-6 real experiments, then the fix and re-run (stretch E) | Before/after results saved |

### Person B: Observability and discovery (the data)
Owns: `observer.py`, `discovery.py`, SQLite schema, `results/` format

| Day | Tasks | Done when |
|---|---|---|
| 1 | 1. `discovery.py`: compose file + container env vars -> `graph.json` (replicas, healthcheck, restart policy, limits, stateful, fan-in/out, critical path). Hand-editable, prints Mermaid. | Graph matches the real architecture |
| 1 | 2. Load probe: once a second POST a vote and GET the result page, record latency and status in SQLite | Rows appear in SQLite |
| 1 | 3. Docker watcher: container state, health transitions, restart counts (`ContainerEvent`) | Events appear when a container is killed |
| 1 | 4. Query helpers `window()`, `per_second()`, `events()`. Stats are `None`, not 0, when there is no data. | 2-minute baseline with p50/p95 and error rate saved |
| 2 | Agent get-tools: `get_experiment`, `get_metrics`, `get_container_events`, `get_graph`, `get_baseline`, reading stored data only | Tools return real stored data |
| 2 | Traffic-spike fault (load multiplier in the probe) | Spike visible in metrics |

### Person C: Intelligence and reporting
Owns: `planner.py`, `analyzer.py`, `scoring.py`, report generation, dashboard

| Day | Tasks | Done when |
|---|---|---|
| 1 | 1. Scoring function + unit tests on hand-made inputs: availability 40, latency 20, recovery 25, blast radius 15 (tests only; no fake data in the pipeline) | Scoring tests green |
| 1 | 2. Agent JSON schema (pydantic): `hypothesis_held`, `root_cause`, `blast_radius`, `weaknesses` (ranked), `fixes[{change, evidence}]` | Schema agreed with A and B |
| 1 | 3. README with reproduce-from-scratch steps (under 10 commands), architecture slide | README draft done |
| 2 | 4. `planner.py`: heuristic risk = criticality (on user path, fan-in) x fragility (1 replica, no healthcheck, stateful, no limits) x untested bonus; then LLM ranking, pydantic-validated, heuristic fallback. `plan --dry-run`. | Ranking printed |
| 2 | 5. `analyzer.py`: tool-using agent loop on the Anthropic SDK. Reject and retry any fix citing a number not in stored data. Disk cache + rule-based fallback for no Wi-Fi. | Validated JSON, works offline |
| 2 | 6. `findings.md` and `findings.html` | Report generated from real results |
| 2 | 7. Streamlit dashboard with a visible KILL SWITCH button (stretch) | Button stops a running experiment |

## Contracts to freeze first
```python
# injector.py (A)
class Injector(Protocol):
    def list_targets(self) -> list[str]: ...
    def inject(self, target: str, fault: str, params: dict) -> Handle: ...
    def rollback(self, handle: Handle) -> list[str]: ...   # returns errors, never raises

# observer.py (B)
def window(db, start: float, end: float) -> Stats
def per_second(db, start: float, end: float) -> list[SecondStat]
def events(db, start: float, end: float, service: str | None) -> list[ContainerEvent]

# models.py (A, already written): Experiment, ExperimentResult, Graph, Stats
```

## Scope

**Day 1 (stop and review after):** running app, dependency map, one safe injection, working kill switch. No LLM calls.

**Day 2 (after Day 1 approval):**
- A: planner (heuristic + LLM ranking, validated, with fallback).
- B: run 4-6 real experiments: kill redis, kill worker, kill postgres, network delay on vote, CPU limit on worker, traffic spike.
- C: analyzer agent, resilience score (availability 40 / latency 20 / recovery 25 / blast radius 15, overall weighted by target criticality), `findings.md` and `findings.html`.
- Stretch: apply the top fix automatically, re-run the same experiment, show the score before vs after; then the Streamlit dashboard.

## Timeline

| Block | Team goal |
|---|---|
| Day 1, 0-1 h | Install Docker everywhere. A and B on separate branches. A freezes the contracts. |
| Day 1, 1-6 h | Parallel work; A starts the orchestrator once B's `window()` exists |
| Day 1, 6-8 h | Run checkpoint experiments together (kill redis, auto-abort, kill switch). **Stop and review.** |
| Day 2, 0-4 h | A runs the real experiments while C builds planner and analyzer on stored results |
| Day 2, 4-6 h | Report, scoring, fix and re-run |
| Day 2, 6-6.5 h | **Feature freeze.** Clean run from scratch. |
| Day 2, 6.5-8 h | Demo video (3 min or less, A drives), architecture slide, findings report, team form. **Submit 30 minutes early.** |

## Deliverables (hackathon)
1. Source code: repo link with a README covering setup and how to run.
2. Demo video, 3 minutes or less: inject a failure, observe the impact, report findings.
3. Architecture overview: one page or slide covering components, how the AI makes decisions, safety controls.
4. Findings report: sample output from the sandbox app, weaknesses ranked, with recommended fixes.
5. Team form: team name, members, tools and AI models used, third-party libraries and data used.

## Working rules
- Small commits at each checkpoint.
- Nobody merges injection code before `safety.py` tests pass.
- Pin dependencies in `requirements.txt`.
