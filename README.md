# chaos-ai

Inject failures into a small microservices app (Docker's example-voting-app), observe the real impact, then (Day 2) have an AI agent diagnose what broke from stored evidence and recommend cited fixes. See [PLAN.md](PLAN.md).

**Status: Day 1 code complete and unit-tested (103 tests). Not yet exercised against real Docker.**

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
| `chaos discover [--live]` | Builds the dependency map from the compose file (and running containers with `--live`) |
| `chaos baseline --seconds N` | Observes the healthy system and saves the baseline |
| `chaos run TARGET FAULT [options]` | One experiment: preflight, baseline window, inject, monitor each second, rollback, measure recovery, write `results/<id>.json` |
| `chaos abort` | Kill switch: stops a running experiment, undoes every fault, restarts anything stopped/paused |
| `chaos reset` | Restores everything, then clears the kill switch (runs are refused while it is engaged) |
| `chaos status` | Kill switch, recorded active faults, container state |

Faults: `kill`, `stop`, `pause`, `cpu-limit`, `memory-limit`, `network-delay`, `network-loss`, `traffic-spike`. Parameters via `-p key=value` (e.g. `-p latency_ms=800`, `-p cpus=0.25`, `-p mem=32m`, `-p loss_pct=40`, `-p multiplier=8`).

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

## Known limits (reported, not hidden)

- The result page is static HTML (data arrives over a websocket), so a 200 from it does not prove postgres is up. Two gauges read inside the containers cover the worker/db path: `redis_queue_len` and `db_votes`. `worker` and `db` experiments show up there, not in the probe error rate.
- Latency percentiles are over successful requests only; when every request fails they are `null`, not 0.
- Recovery = 5 consecutive seconds within 10% of baseline (error rate) and 10% + 50 ms slack (p95).
- Untested against real Docker: cpu/memory limit removal (verified after restore, reported if it fails) and the tc sidecar image used when the target has no `tc`.

`ANTHROPIC_API_KEY` (Day 2) is read from the environment and never committed.
