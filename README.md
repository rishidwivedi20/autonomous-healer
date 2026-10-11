# chaos-ai

Inject failures into a small microservices app (Docker's example-voting-app), observe the real impact, then (Day 2) have an AI agent diagnose what broke from stored evidence and recommend cited fixes. See [PLAN.md](PLAN.md).

**Status: three agents (injecting, finding, fixing) built and verified on real Docker Desktop with no API key: eight fault types, safety layer (auto-abort, kill switch, lock), resilience score, findings report, live dashboard, and a fixing agent that proves a fix with the same experiment before and after. Run `python scripts/acceptance.py` for the full acceptance checklist (about 30 minutes; the unit tests need no Docker).**

## Reproduce from scratch (Windows, PowerShell)

Prerequisite: Docker Desktop running (WSL2 backend), Python 3.11+.

```powershell
git clone https://github.com/rishidwivedi20/autonomous-healer.git chaos-ai; cd chaos-ai
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q                 # the unit tests, no Docker needed
.\deploy\tasks.ps1 up                               # vote :8080, result :8081, waits until healthy
.\chaos.cmd discover                                    # dependency map -> results\graph.json (+ Mermaid)
.\chaos.cmd baseline --seconds 120                      # steady-state p50/p95/error rate -> results\baseline.json
.\chaos.cmd run redis kill --dry-run                    # validate and print the plan, inject nothing
.\chaos.cmd run redis kill --hypothesis "votes fail until redis restarts"
.\chaos.cmd run vote network-delay -p latency_ms=3000   # should trip the auto-abort
.\deploy\tasks.ps1 down
```

(`chaos.cmd` is a wrapper for `python -m chaosai`. In PowerShell type the full `.\chaos.cmd`; plain `.\chaos` is not found. In git-bash use `./chaos`.)

## Acceptance run (everything, from scratch, no API key)

```powershell
.\.venv\Scripts\python scripts\acceptance.py --results results-acceptance
```

It strips every API key from the environment and ignores `.env` (`CHAOSAI_NO_DOTENV=1`), so nothing in it can spend credits. It runs the unit tests, then checks the brief's acceptance list against the real sandbox: the documented from-scratch start (`tasks.ps1 down` then `up`), both user journeys, a baseline, all eight fault types (each must be measured, rolled back and leave the sandbox healthy), the kill switch aborting a running experiment, two experiments refusing to overlap, an automatic abort on a threshold breach, refusing to inject into an unhealthy sandbox, the injecting agent, the report (every number must match a stored result), all three agents end to end, and the dashboard (its panels, downloads, security checks and emergency stop). It then verifies the sandbox is back to a clean, healthy state. `--offline-only` skips everything that needs Docker. Results: `acceptance.md` and `acceptance.json`; the exit code is 0 only if every check passed. The run injects real faults into the sandbox containers and rebuilds the sandbox from scratch (removing its demo-vote volume), so run it on a machine where that is fine; `--keep-app` skips the rebuild.

## Commands

| Command | What it does |
|---|---|
| `chaos dashboard [--port N] [--rate R] [--open]` | **Live web page** at http://127.0.0.1:8765: charts for both user journeys, the redis queue and Postgres vote count, container table with restarts, event log, and buttons that inject real faults (plus KILL SWITCH / Reset) |
| `chaos analyze [--offline] [--id ID] [--last N] [--dry-run]` | **Analyzer**: explains the stored experiments (root cause, blast radius, ranked weaknesses, concrete fixes). Every claim cites numbers that the data tools returned. Uses the Claude agent when a key is set, otherwise the deterministic rule-based analyzer |
| `chaos llm-check [--provider anthropic\|gemini\|mistral\|openrouter\|pool] [--models] [--model M] [--match TEXT] [--no-send]` | Checks an LLM provider: is a key configured, which models can it use, does one tiny request work (a key is only ever printed masked); `--provider pool` shows every pool member and its cooldown |
| `chaos config` | Prints the effective parameters (`config.yaml` merged over the defaults) |
| `chaos plan [--top N] [--offline] [--dry-run]` | Ranks which experiments to run first (heuristic risk score, re-ranked by Claude when `ANTHROPIC_API_KEY` is set) |
| `chaos agent [--top N] [--max N] [--offline] [--dry-run] [-y]` | **The injector agent**: plans, then introduces the faults one at a time through the safety guard |
| `chaos autopilot [--top N] [--max N] [--offline] [--dry-run] [-y]` | **Hands-off end to end**: plan, inject with the agent, analyze, score, and write the findings report |
| `chaos pipeline [--top N] [--max N] [--offline] [--no-fix] [--validate-fix] [--keep-fix] [--dry-run] [-y]` | **All three agents, hands-off**: INJECT faults, FIND the weaknesses (analysis, score, report), RECOMMEND ranked changes (changes nothing); `--validate-fix` also applies one vetted fix and measures it before and after; `--offline` needs no API key |
| `chaos recommend [--offline]` | **The fixing agent's main job**: rank recommended changes from the latest findings (`results/recommendations.{md,json}`); changes nothing and runs no experiment |
| `chaos fix [--recipe ID] [--list] [--keep] [--revert] [--offline] [--dry-run] [-y]` | **Opt-in fix validation**: apply one vetted fix, run the same experiment before and after, compare, put the sandbox back; `--revert` undoes anything still applied |
| `chaos report [--last N] [--offline]` | Write `results/findings.{md,html,json}` (resilience score, ranked weaknesses, cited fixes with verification re-runs) from the stored experiments |
| `chaos discover [--live]` | Builds the dependency map from the compose file (and running containers with `--live`) |
| `chaos baseline --seconds N` | Observes the healthy system and saves the baseline |
| `chaos run TARGET FAULT [options]` | One experiment: preflight, baseline window, inject, monitor each second, rollback, measure recovery, write `results/<id>.json` |
| `chaos abort` | Kill switch: stops a running experiment, undoes every fault, restarts anything stopped/paused |
| `chaos reset` | Restores everything, then clears the kill switch (runs are refused while it is engaged) |
| `chaos status` | Kill switch, recorded active faults, container state |

Faults: `kill`, `stop`, `pause`, `cpu-limit`, `memory-limit`, `network-delay`, `network-loss`, `traffic-spike`. Parameters via `-p key=value` (e.g. `-p latency_ms=800`, `-p cpus=0.25`, `-p mem=32m`, `-p loss_pct=40`, `-p multiplier=8`).

## The three agents: inject, find, fix

`chaos pipeline` runs all three hands-off, and `--offline` runs them with no API key and no network (heuristic planner, rule-based analyzer, vetted fixes). Each agent can also be run alone.

| Agent | What it does | Where | With no LLM |
|---|---|---|---|
| **Injecting** | plans the experiments, waits for a healthy system, injects one fault at a time through the safety guard, rolls back, cools down | `agent.py`, `planner.py`, `injector.py`, `safety.py` | heuristic risk ranking |
| **Finding** | reads the stored measurements through read-only evidence tools; ranks weaknesses by severity with cited evidence, a fix and a verification re-run for each; computes the resilience score; writes `findings.{md,html,json}` | `finder.py`, `analyzer.py`, `rules.py`, `scoring.py`, `report.py` | deterministic rule-based analyzer |
| **Fixing** | **recommends** changes: ranks the analysis's proposed fixes by the severity they address, with evidence, the exact change, how to prove it, and any stored measurement. It changes nothing. Optionally (`chaos fix`, `chaos pipeline --validate-fix`) it proves one vetted fix by applying it, re-running the same experiment, comparing, and reverting | `recommend.py`, `fixer.py` | recommending is plain code; the catalogue and the judging are plain code |

### The fixing agent, step by step (opt-in validation)

By default the fixing agent only **recommends** (`chaos recommend`, or step 3 of `chaos pipeline`): no experiment, no change to the sandbox. The steps below run only when you ask for them with `chaos fix` or `chaos pipeline --validate-fix`.

1. **Choose.** It can only pick from `CATALOG` in `fixer.py`, a hard-coded list of reversible fixes (today: `scale-worker`, three worker replicas via `docker compose --scale`). A weakness selects a fix by naming the service and a matching word; an LLM (or a person) only ever supplies an id, never a command. `chaos fix --list` shows the catalogue.
2. **Before.** It runs the fix's verification experiment (for `scale-worker`: a traffic spike on vote at 15 votes/s for 20 s, converted to a multiplier from the probe's own rate) through the normal orchestrator, safety guard and rollback, and measures it.
3. **Check.** If the problem the fix addresses does not show up in that measurement (for `scale-worker`: the redis queue peaking at 20 votes or more), it changes nothing and says so. This is the common case on a healthy system, and it was seen live.
4. **Apply.** It writes the undo to `results/active_fixes.json` first, takes the one-experiment-at-a-time lock, and changes the sandbox with an argument list (no shell) for services labelled `chaosai.managed=true` only, with a replica count from 1 to 5; then waits until the sandbox is healthy.
5. **After.** The same experiment again.
6. **Judge.** `improved` needs BOTH a resilience-score gain of at least 3 points AND a gain of at least 25% in the fix's own metric, with no score component dropping noticeably; a regression is `worse`; anything else is `inconclusive` and is **not claimed**. It is one run each, and the report says so.
7. **Revert**, in a `finally`: the original replica count is restored and verified, unless the fix was measured to help and `--keep` was given. An error or the kill switch cannot leave the change in place; if a revert fails the undo stays on disk and `chaos reset` / `chaos abort` / `chaos fix --revert` retry it.

The two verification runs are stored in `results/fix-runs/<id>/`, apart from the experiment history, so a "fixed" run never replaces the real history that the score is computed from. The validated fix is saved in `results/fixes/`, and appears in `findings.{md,html,json}` and on the dashboard ("Fix validation" panel).

**A real run** (live Docker, no API key): `chaos fix --recipe scale-worker` measured the same spike before and after. Redis queue peak 109 -> 9 votes (-92%), queue back to baseline after 34.3 s -> 12.8 s, recovery time 12 s -> 0 s, other services restarted 0 -> 0, resilience score 93.8 -> 97.1: verdict *improved*, and the sandbox was put back to one worker. An earlier attempt with a smaller spike showed no backlog, and the agent correctly changed nothing.

## Using a free Gemini key instead (Google AI Studio)

The planner and analyzer can run on Gemini instead of Claude. Nothing else changes: the same evidence validator, caches and rule-based fallback apply (`chaosai/gemini.py` is a drop-in for the Anthropic client).

1. Create a key at https://aistudio.google.com/apikey and put `GEMINI_API_KEY=...` in the git-ignored `.env`.
2. In `config.yaml` set `provider: gemini` and a model (for example `llm_model: gemini-3.5-flash`) under both `planner:` and `analyzer:`. A Claude model name with the gemini provider (or the reverse) is rejected at load time.
3. `chaos llm-check --provider gemini --models` lists the models the key can call; `chaos llm-check --provider gemini --model <name>` sends one tiny request. Model availability changes (a model can exist yet be closed to new keys): test the name before relying on it.

**Free-tier limits are tight.** Observed on a real key: **20 generate requests per day per model** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). One analysis is roughly 10 requests (the tool loop makes one request per turn), so expect about one full analysis per model per day, and the planner needs one. When the quota is spent the adapter fails immediately with what ran out and when it resets, and the planner and analyzer fall back to the rules and record why. To make the most of it:

- validated answers are cached on disk (`results/llm_cache/`), so repeating the same analysis costs nothing: run it live once, then replay it;
- limit the scope with `chaos analyze --id <id> ...` or `--last N` (a 6-experiment analysis used about 220k input tokens);
- quotas are per model, so the planner and analyzer can use different models.

Gemini's suggested fixes are validated for evidence (every number it cites must be one a tool returned) but not for realism: it may propose config keys the app does not read. Treat `change` snippets as suggestions to review.

## Using Mistral instead (La Plateforme key)

A third provider, as a backup for the other two (`chaosai/mistral.py`, same drop-in design: same evidence validator, caches and rule-based fallback).

1. Create a key at https://console.mistral.ai/api-keys and put `MISTRAL_API_KEY=...` in the git-ignored `.env`.
2. In `config.yaml` set `provider: mistral` and a model (for example `llm_model: mistral-small-latest`) under both `planner:` and `analyzer:`. A model name from another vendor is rejected at load time.
3. `chaos llm-check --provider mistral --models` lists the chat models with function calling that the key can use; `--model <name>` sends one tiny request.

**A key is not enough: the plan behind it must allow requests.** With a valid key whose plan allowed 0 requests per minute, `--models` worked but every chat call returned HTTP 429 (header `x-ratelimit-limit-req-minute: 0`). The adapter detects that and stops immediately with an instruction instead of retrying: activate the free Experiment plan (or add billing) in the Mistral console. Other 429s are retried with backoff (`Retry-After` is honoured), and any failure falls back to the heuristic planner / rule-based analyzer and records why.

## Using OpenRouter (one key, many models)

A fourth provider (`chaosai/openrouter.py`; it shares the OpenAI-style adapter with Mistral). Model ids look like `vendor/model`, for example `openai/gpt-oss-120b` or its free variant `openai/gpt-oss-120b:free`.

1. Create a key at https://openrouter.ai/keys and put `OPENROUTER_API_KEY=...` in the git-ignored `.env`.
2. In `config.yaml` set `provider: openrouter` and `llm_model: openai/gpt-oss-120b` under `planner:` and `analyzer:`. That paid id is the one verified live (tool calls, tool round trips with reasoning replayed, and strict JSON output all worked; a tiny test costs a fraction of a cent). When tested, the `:free` variant answered HTTP 404 ("unavailable for free"); free variants come and go, so try `--model` first.
3. `chaos llm-check --provider openrouter --models --match gpt-oss` lists the tool-capable models (OpenRouter has hundreds); `--model <id>` sends one tiny request.

What the adapter does for you: it asks OpenRouter to route only to upstream providers that really support tools and JSON schemas (`require_parameters`), sends a reasoning model's `reasoning` fields back on the next turn, retries per-minute limits, and stops at once on the free tier's **daily** limit with when it resets. Use the plain or `:free` id; the `:batch` listing is meant for asynchronous bulk work, not for an interactive tool loop. It is tested offline and was verified live against `openai/gpt-oss-120b`.

## Using Groq (free plan, fast open models)

A fifth provider (`chaosai/groq.py`, same OpenAI-style adapter). Put `GROQ_API_KEY=...` in `.env`; `chaos llm-check --provider groq --models` lists the models, and `openai/gpt-oss-120b` is the one verified live. **Limit measured on a free key: 8000 tokens per minute per model** (response headers `x-ratelimit-limit-tokens`). The planner worked through Groq with real output. The analyzer works too because it is **batched** (`analyzer.batch_size: 1`): each experiment is analysed in a fresh small conversation (the biggest request measured was about 3,200 tokens, against the 8000 cap), tool results are trimmed to their summaries (`max_tool_output_chars`), and the per-experiment parts are merged (same-titled weaknesses and fixes become one entry; the evidence validator ran on every part). With a pool, every batch starts on the next member, so the work spreads across keys and providers; a batch the model cannot answer is analysed by the rules and the summary says so; answered batches are cached, so a rerun resumes where it stopped. Measured live on Groq alone: 3 experiments in about 4 min 45 s (most of it waiting out the per-minute window). A 429 asking for up to 70 s is waited out; a longer wait fails at once.

## Spreading load across models and keys: `provider: pool`

Free tiers are small (Gemini: about 20 requests per model per day; OpenRouter's free models have a daily cap; Mistral depends on the plan) and one analysis takes about 10 requests. A pool makes several members behave like one client (`chaosai/pool.py`):

```yaml
planner:
  provider: pool
  members: &llm_members
    - {provider: gemini, model: gemini-3-flash-preview}
    - {provider: gemini, model: gemini-3.5-flash-lite}
    - {provider: openrouter, model: "openai/gpt-oss-120b"}
    - {provider: mistral, model: mistral-small-latest, key_env: MISTRAL_API_KEY_2}   # key_env: use a different key variable for this member
analyzer:
  provider: pool
  members: *llm_members          # YAML alias: the same list
```

- **Rotation.** Each new run (one planner call, or an analysis from its first request) starts on the next member. The position is saved in `results/llm_pool.json`, so successive commands really spread across members. Consecutive runs also prefer a **different provider** from the one that answered before (the planner, the analysis and the report each go to another provider when one is available; with only one provider left it rotates through that provider's models).
- **Sticky runs.** One analysis stays on its member for every turn, because a tool loop replays provider-specific parts (Gemini's thought signatures).
- **Failover.** If a member fails (rate limit, spent quota, overload, bad key, unknown model, network) it goes on a cooldown (the API's own "retry in ..." when given, otherwise 2 min for a 429, 1 min for a server error, 1 h for a bad key, 6 h for an unknown model; capped at 24 h) and the same request goes to the next member. The cooldown is saved too, so the next run does not try a member known to be out. A run already under way is never moved into Gemini or Anthropic, and when it moves between providers the history is rebuilt (tool-call ids are mapped to the form each API accepts, and one provider's raw reasoning fields are never replayed to another).
- **If every member is out,** the planner or analyzer falls back to the heuristic / rules and records each member's reason.
- **What it does not do.** An analysis that a member returns but our validator rejects repeatedly is not treated as that member failing, so it falls back to the rules instead of trying another member. Several keys only help if they have separate quotas (Gemini's quota is per Google project, not per key); several models of one provider each have their own. Use keys that belong to you or your team.
- **Reporting.** The plan and the analysis record which member(s) actually answered (for example `gemini:gemini-3-flash-preview`).

`chaos llm-check --provider pool --no-send` shows each member and any saved cooldown without sending anything; without `--no-send` it sends one tiny request per member and reports each.

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
