# chaos-ai

Inject failures into a small microservices app (Docker's example-voting-app), observe the real impact, then (Day 2) have an AI agent diagnose what broke from stored evidence and recommend cited fixes. See [PLAN.md](PLAN.md).

**Status: Day 1 complete and verified on real Docker Desktop (190 unit tests; baseline, redis crash, auto-abort, mid-run kill switch and the injector agent all exercised live).**

## Reproduce from scratch (Windows, PowerShell)

Prerequisite: Docker Desktop running (WSL2 backend), Python 3.11+.

```powershell
git clone https://github.com/rishidwivedi20/autonomous-healer.git chaos-ai; cd chaos-ai
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q                 # 103 tests, no Docker needed
.\deploy\tasks.ps1 up                               # vote :8080, result :8081, waits until healthy
.\chaos discover                                    # dependency map -> results\graph.json (+ Mermaid)
.\chaos baseline --seconds 120                      # steady-state p50/p95/error rate -> results\baseline.json
.\chaos run redis kill --dry-run                    # validate and print the plan, inject nothing
.\chaos run redis kill --hypothesis "votes fail until redis restarts"
.\chaos run vote network-delay -p latency_ms=3000   # should trip the auto-abort
.\deploy\tasks.ps1 down
```

(`chaos` is `chaos.cmd`, a wrapper for `python -m chaosai`. On git-bash use `./chaos`.)

## Commands

| Command | What it does |
|---|---|
| `chaos config` | Prints the effective parameters (`config.yaml` merged over the defaults) |
| `chaos plan [--top N] [--offline] [--dry-run]` | Ranks which experiments to run first (heuristic risk score, re-ranked by Claude when `ANTHROPIC_API_KEY` is set) |
| `chaos agent [--top N] [--max N] [--offline] [--dry-run] [-y]` | **The injector agent**: plans, then introduces the faults one at a time through the safety guard |
| `chaos discover [--live]` | Builds the dependency map from the compose file (and running containers with `--live`) |
| `chaos baseline --seconds N` | Observes the healthy system and saves the baseline |
| `chaos run TARGET FAULT [options]` | One experiment: preflight, baseline window, inject, monitor each second, rollback, measure recovery, write `results/<id>.json` |
| `chaos abort` | Kill switch: stops a running experiment, undoes every fault, restarts anything stopped/paused |
| `chaos reset` | Restores everything, then clears the kill switch (runs are refused while it is engaged) |
| `chaos status` | Kill switch, recorded active faults, container state |

Faults: `kill`, `stop`, `pause`, `cpu-limit`, `memory-limit`, `network-delay`, `network-loss`, `traffic-spike`. Parameters via `-p key=value` (e.g. `-p latency_ms=800`, `-p cpus=0.25`, `-p mem=32m`, `-p loss_pct=40`, `-p multiplier=8`).

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
- Recovery = 5 consecutive seconds within 10% of baseline (error rate) and 10% + 50 ms slack (p95).
- The tc sidecar image `ghcr.io/alexei-led/pumba-alpine-nettools` has `ENTRYPOINT tail -f /dev/null`, so commands run with `--entrypoint sh`. Netem is tried inside the target first, then in the sidecar; the apply script verifies netem is actually present.

`ANTHROPIC_API_KEY` (Day 2) is read from the environment and never committed.
