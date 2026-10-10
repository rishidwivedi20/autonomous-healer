# chaos-ai

Inject failures into a small microservices app (Docker's example-voting-app), observe the real impact, then (Day 2) have an AI agent diagnose what broke from stored evidence and recommend cited fixes. See [PLAN.md](PLAN.md).

**Status: Day 1 complete and verified on real Docker Desktop (305 unit tests; baseline, redis crash, auto-abort, mid-run kill switch and the injector agent all exercised live).**

## Reproduce from scratch (Windows, PowerShell)

Prerequisite: Docker Desktop running (WSL2 backend), Python 3.11+.

```powershell
git clone https://github.com/rishidwivedi20/autonomous-healer.git chaos-ai; cd chaos-ai
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q                 # 190 tests, no Docker needed
.\deploy\tasks.ps1 up                               # vote :8080, result :8081, waits until healthy
.\chaos.cmd discover                                    # dependency map -> results\graph.json (+ Mermaid)
.\chaos.cmd baseline --seconds 120                      # steady-state p50/p95/error rate -> results\baseline.json
.\chaos.cmd run redis kill --dry-run                    # validate and print the plan, inject nothing
.\chaos.cmd run redis kill --hypothesis "votes fail until redis restarts"
.\chaos.cmd run vote network-delay -p latency_ms=3000   # should trip the auto-abort
.\deploy\tasks.ps1 down
```

(`chaos.cmd` is a wrapper for `python -m chaosai`. In PowerShell type the full `.\chaos.cmd`; plain `.\chaos` is not found. In git-bash use `./chaos`.)

## Commands

| Command | What it does |
|---|---|
| `chaos dashboard [--port N] [--rate R] [--open]` | **Live web page** at http://127.0.0.1:8765: charts for both user journeys, the redis queue and Postgres vote count, container table with restarts, event log, and buttons that inject real faults (plus KILL SWITCH / Reset) |
| `chaos analyze [--offline] [--id ID] [--last N] [--dry-run]` | **Analyzer**: explains the stored experiments (root cause, blast radius, ranked weaknesses, concrete fixes). Every claim cites numbers that the data tools returned. Uses the Claude agent when a key is set, otherwise the deterministic rule-based analyzer |
| `chaos llm-check` | Checks the Anthropic key with one tiny request (prints the key masked) |
| `chaos config` | Prints the effective parameters (`config.yaml` merged over the defaults) |
| `chaos plan [--top N] [--offline] [--dry-run]` | Ranks which experiments to run first (heuristic risk score, re-ranked by Claude when `ANTHROPIC_API_KEY` is set) |
| `chaos agent [--top N] [--max N] [--offline] [--dry-run] [-y]` | **The injector agent**: plans, then introduces the faults one at a time through the safety guard |
| `chaos autopilot [--top N] [--max N] [--offline] [--dry-run] [-y]` | **Hands-off end to end**: plan, inject with the agent, analyze, score, and write the findings report |
| `chaos report [--last N] [--offline]` | Write `results/findings.{md,html,json}` (resilience score, ranked weaknesses, cited fixes with verification re-runs) from the stored experiments |
| `chaos discover [--live]` | Builds the dependency map from the compose file (and running containers with `--live`) |
| `chaos baseline --seconds N` | Observes the healthy system and saves the baseline |
| `chaos run TARGET FAULT [options]` | One experiment: preflight, baseline window, inject, monitor each second, rollback, measure recovery, write `results/<id>.json` |
| `chaos abort` | Kill switch: stops a running experiment, undoes every fault, restarts anything stopped/paused |
| `chaos reset` | Restores everything, then clears the kill switch (runs are refused while it is engaged) |
| `chaos status` | Kill switch, recorded active faults, container state |

Faults: `kill`, `stop`, `pause`, `cpu-limit`, `memory-limit`, `network-delay`, `network-loss`, `traffic-spike`. Parameters via `-p key=value` (e.g. `-p latency_ms=800`, `-p cpus=0.25`, `-p mem=32m`, `-p loss_pct=40`, `-p multiplier=8`).

## Getting an Anthropic API key

The planner's LLM ranking and the analyzer agent need an API key. Without one everything still works: both fall back to deterministic rules and say so.

1. Open the Anthropic Console (https://platform.claude.com) and sign in or create an account. API usage is billed separately from a Claude.ai chat subscription.
2. Add credits under **Settings -> Billing**. A key with no credit balance is rejected.
3. Go to **Settings -> API keys -> Create key**, name it `chaos-ai`, and copy the key at once (it is shown only once). Menu names may differ slightly.
4. Give it to chaos-ai without pasting it anywhere public: run `notepad .env` in the repo folder and save one line `ANTHROPIC_API_KEY=sk-ant-...`. `.env` is git-ignored and only `ANTHROPIC_*` / `CHAOSAI_*` names are read from it. (Alternatively set it for one PowerShell session: `$env:ANTHROPIC_API_KEY = "sk-ant-..."`.)
5. Check it: `.\chaos.cmd llm-check`. Expect `OK: claude-opus-5-5 answered 'ok'`. This costs a fraction of a cent and prints the token counts so you can see what a request costs.
6. Use it: `.\chaos.cmd plan --dry-run` (planner: LLM) and `.\chaos.cmd analyze` (analyzer agent). Validated analyses are cached in `results/llm_cache/`, so re-running on unchanged results makes no API call.

Never commit a key or paste it into a chat. If one leaks, revoke it in the Console and create a new one.

## The analyzer

`chaos analyze` turns the stored results into findings. The agent reads only stored data through read-only tools (`list_experiments`, `get_experiment`, `get_metrics`, `get_gauges`, `get_container_events`, `get_graph`, `get_baseline`) and finishes by submitting a structured analysis: per experiment `hypothesis_held` (only if a hypothesis was stored), `root_cause` and `blast_radius`; weaknesses ranked by severity; and fixes with a concrete change (compose/config snippet) each. A submission is **rejected, with reasons, and the model must retry** unless every experiment is covered, every weakness and fix cites evidence, weaknesses are ranked, and every number in an evidence claim is one the tools actually returned for that experiment. That check is necessary rather than sufficient (a number can coincide with an unrelated stored value), so treat citations as pointers to verify, and note that causes the data cannot measure are labelled as inferences. If anything fails (no key, no network, refusal, no valid answer) the deterministic rule-based analyzer produces the same schema and the result records why.

## Live dashboard

`.\chaos.cmd dashboard` serves http://127.0.0.1:8765 (loopback only). It starts the load probe at `dashboard.probe_rate_per_s` (default 5 requests/s per journey; the CLI's 1/s is too coarse to see a sub-second redis crash) and reads the pipeline gauges every second. A **dependency graph** panel at the top draws the five services along the user's critical path with arrows meaning "depends on": each node's border is the live container health, single replicas and missing healthchecks are flagged, and while a fault runs the faulted service pulses red and everything that depends on it (its blast radius) turns amber; afterwards the services that really restarted keep a badge. (`GET /api/graph` serves the same map as `results/graph.json`, which `chaos dashboard` refreshes from the compose file at start-up.) Below the charts, three **results panels** show the resilience score (with its four component bars and formula), the experiment history (click a row for its score breakdown) and the ranked findings with their evidence and fixes, plus download buttons for the report as HTML, Markdown or JSON (`GET /api/report`, `/report.html`, `/report.md`, `/report.json`). They use the rule-based analysis, rebuild only when an experiment finishes, and never call an LLM; run `chaos report` for the Claude-written version. Press **Crash Redis** (or DB / Worker / Pause DB / Delay Vote) and watch: the red band is the time the fault was active, red dashed lines are real container restarts, and the cards, charts, container table and event log all update once a second. **KILL SWITCH** aborts the running experiment and undoes every fault; **Reset** clears it. Everything shown is read from the same SQLite store the CLI writes; nothing is synthesised.

Safety of the page itself: bound to 127.0.0.1; every request must carry a loopback `Host`; POSTs need `X-Chaos: 1` plus JSON (which forces a CORS preflight no other site can pass) and a same-origin `Origin`; faults go through the same guard, abort rules and rollback as `chaos run`. An OS file lock (`results/experiment.lock`) enforces one experiment at a time across *processes* (two dashboards, or a dashboard plus `chaos run`); it is released automatically if the holder dies. A second dashboard on the same port is refused.

## Parameters

Everything tunable lives in [`config.yaml`](config.yaml) (every value shown there is the default; `chaos config` prints the effective values). Flags override it. Two rules keep it safe: the `safety:` values can only be made *stricter* than the hard limits in `safety.py` (60 s, 50%, 3000 ms, 3 s), and anything looser or any unknown key is an error at load time, not a silent default.

| Section | What it controls |
|---|---|
| `app` | vote/result URLs, compose file |
| `probe` | probe interval and timeout |
| `safety` | max fault duration, abort thresholds, preflight steady-state limit |
| `experiment` | baseline window, recovery timeout / hold / tolerance |
| `faults` | default duration and parameters per fault type (used by the planner and `chaos run`) |
| `planner` | plan size, variety caps, LLM on/off, model, effort |
| `agent` | max experiments, cooldown, health wait, stop-on-abort, refusal limit |

## The injector agent

`chaos autopilot` chains everything with no manual step: it plans, runs the agent below, then analyzes what actually ran, computes the resilience score and writes the report. If the agent stops early it still reports the experiments that finished and exits non-zero.

**Resilience score** (`chaosai/scoring.py`, deterministic, from stored measurements only): per experiment 40 availability (worst journey's error-rate increase) + 20 latency (worst journey's p95 vs baseline, 50 ms slack; full marks up to 1.5x, zero at 10x) + 25 recovery (full marks at 5 s or less, zero at 60 s; users and pipeline) + 15 blast radius (collateral restarts and degraded journeys). A part that cannot be measured is dropped and listed as missing, never guessed; a run cut short by your kill switch has no recovery score. The overall score is the criticality-weighted mean of the latest run of each distinct experiment.

`chaos agent` is the single component that introduces faults autonomously. It does not choose *what* to break (the planner does) or diagnose results (the Day 2 analyzer). For each planned experiment it: waits for every container to be healthy (and refuses to inject into a sick system), runs it through the safety guard and per-second abort rules, rolls back in a `finally`, pauses for a cooldown, and moves on. It stops the whole run on: kill switch, rollback problems, a Docker/injector error, unhealthy containers, or repeated refusals. It asks for confirmation unless `-y`; `--dry-run` plans and validates everything and injects nothing. Run logs go to `results/agent-runs/`.

## Safety

- Only containers labeled `chaosai.managed=true` can be targeted (checked in `safety.py` and again in the injector).
- One experiment at a time; duration capped at 60 s; abort thresholds can only be tightened, never loosened.
- Auto-rollback when error rate > 50% or p95 > 3000 ms for 3 consecutive seconds.
- Kill switch: `chaos abort` (works from another terminal), Ctrl+C / SIGTERM, and `try/finally` around every injection. Undo information is written to `results/active_faults.json` before each fault, so it survives a crash.
- After Ctrl+C or `chaos abort`, run `chaos reset` before the next experiment.

## Layout

```
chaosai/  models  safety  discovery  observer  injector  orchestrator  cli
deploy/   docker-compose.yml  Makefile  tasks.ps1
results/  graph.json  baseline.json  chaosai.db  <experiment-id>.json   (gitignored)
tests/
```

## Known limits and findings from real runs (reported, not hidden)

- **`kill` is a real crash, not `docker kill`.** Docker treats `docker kill` as a manual stop and will NOT revive the container under `restart: unless-stopped` (observed: still down after 25 s). `kill` therefore sends SIGKILL from the Docker VM's pid namespace (`--pid=host` sidecar), which the restart policy does revive. If that is unavailable it falls back to `docker kill` and says so in the result notes; use `stop` for a deliberate manual stop.
- **Redis crash is invisible to the probe.** Redis restarts in about a second, so a 1 req/s probe sees no errors. The impact shows only in the gauges: the worker stalls for roughly 6 s (`redis_queue_len` grows while `db_votes` is flat). The result page is static HTML, so `worker`/`db` faults also show up only in `redis_queue_len` / `db_votes`.
- **Abort, preflight and recovery judge the worst journey** (vote vs result), not the average. A total vote outage with a healthy result page averages only ~46-50% errors.
- **Failures are recorded when a request completes**, so a timeout lands `timeout_s` after it started. All judgements use a window lagged by `timeout_s + 0.5 s`; the auto-abort therefore fires about 6-7 s after a fault begins (about 3.5 s settle + 3 s breach).
- Latency percentiles are over successful requests only; when every request fails they are `null`, not 0.
- **Recovery = probe AND pipeline.** A 5-second hold window counts only if the probe is within 10% of baseline (error rate; p95 + 50 ms slack) **and** the redis queue is back within its baseline maximum (+10% + 2 votes) **and** Postgres is ingesting at >= 50% of the baseline vote rate (an unreadable gauge counts as unhealthy). The result reports each part: `recovery_probe_s`, `recovery_pipeline_s`, and `pipeline_assessed` (false = no usable gauge readings, so the probe alone decided). Measured live: a 20 s `db pause` showed probe recovery 0 s but pipeline recovery 22 s, because 99 votes had queued.
- **Docker cannot remove a memory limit from a running container** (`Memory=0` in an update means "unchanged"; `-1` is rejected with "Minimum memory limit allowed is 6MB"). `memory-limit` therefore restores an originally-unlimited container by lifting the cap to the engine's total RAM (unlimited in practice) and verifies that; `docker compose up -d --force-recreate <service>` restores the literal original config (the data volume is kept). A cap of 16 MB crash-loops Postgres (OOM-killed repeatedly; `result` and `worker` restart once each); recovery took ~49 s because of the vote backlog.
- **The single worker processes ~9.5 votes/s.** A x5 `traffic-spike` (25 votes/s) grew the redis queue to ~309 votes (about 30 s of lag at the peak) with no user-visible errors (vote p95 19 -> 54 ms), and the backlog took ~93 s to drain. Use `experiment.recovery_timeout_s` >= 120 for this fault.
- The tc sidecar image `ghcr.io/alexei-led/pumba-alpine-nettools` has `ENTRYPOINT tail -f /dev/null`, so commands run with `--entrypoint sh`. Netem is tried inside the target first, then in the sidecar; the apply script verifies netem is actually present.

`ANTHROPIC_API_KEY` (Day 2) is read from the environment and never committed.
