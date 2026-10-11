# Demo runbook (10 minutes)

Goal: prove that **an agent injects failures by itself**, that the system **measures real impact**, and that AI and rules turn it into **ranked weaknesses, a score and recommended changes**, all with a **safety layer** that can stop everything.

The live demo runs from the **dashboard**: one click on **Start the agent**. Everything else is one command: `.\start.cmd`.
Run commands from `C:\Bosch\chaos-ai` in PowerShell. Quote only numbers you can see on your own screen; none are written here.

---

## 1. Before the talk (about 30 minutes ahead)

| # | Do | Check |
|---|---|---|
| 1 | Start Docker Desktop and wait until it is running | `docker ps` works |
| 2 | Run `.\start.cmd` (use `.\start.cmd -Fresh` for a clean sandbox with zero votes) | five steps print, then three URLs answer 200 and the browser opens the dashboard |
| 3 | On the dashboard: the pill is **idle**, the header says nothing about a kill switch, the agent panel shows **Start the agent** enabled | if "kill switch engaged" shows, press **Reset** |
| 4 | Run the unit suite once: `.\.venv\Scripts\python.exe -m pytest -q` | all pass |
| 5 | Check the AI keys without printing them: `.\chaos.cmd llm-check --provider pool` | at least one member answers |
| 6 | **Do one full rehearsal of section 3 and record the screen.** The video is your backup | the plan table fills in and the status chips change |
| 7 | Save a copy of a good earlier result as a fallback: `copy results\findings.html results\demo-findings.html` and `copy results\recommendations.md results\demo-recommendations.md` | files exist |
| 8 | Open three windows: the dashboard (browser), the vote page http://127.0.0.1:8080/ (second tab), and a terminal for `docker events` | |

Keep your free-tier AI quota for the talk: do not run extra AI commands in the last hour. If the AI planner has no quota, the agent falls back to the heuristic planner and says so on the page, which is fine to show.

---

## 2. Rules that keep the demo clean

- **Do not run other chaos commands while the dashboard is up.** Commands such as `agent`, `autopilot`, `pipeline` or `baseline` start their own load probe and mix its measurements with the dashboard's. Use the dashboard's button for injection.
- `.\chaos.cmd recommend`, `.\chaos.cmd abort` and `.\chaos.cmd reset` do not start a probe, so they are safe at any time.
- While the agent runs, the manual fault buttons and Reset are disabled on purpose. The **KILL SWITCH** is always available.
- Never show `.env`.

---

## 3. The 10-minute run

| Time | Segment | What you do | What you say |
|---|---|---|---|
| 0:00 | The problem | Slide: why chaos engineering | "We break a system on purpose, in a safe sandbox, to find weaknesses before real outages do." |
| 1:00 | The system | Dashboard: point at the dependency graph and the containers table; show the vote page working | "A voting app: vote, redis, worker, postgres, result. The agent built this map itself from the compose file. Every number you see is read live from the app." |
| 2:00 | **Live: the agent injects by itself** | See 3a | "I press one button. From here nobody touches anything." |
| 5:00 | Proof it was real | See 3b | "Docker, not our code, reports these events." |
| 6:30 | The results | Scroll to the score, history, findings and fix-validation panels (they refresh when the agent finishes) | "Resilience score, ranked weaknesses, each with evidence from stored numbers." |
| 8:00 | Recommendations | `.\chaos.cmd recommend`, then show `results\recommendations.md` | "The fixing agent recommends changes ranked by severity. It does not change the system by itself." |
| 9:00 | Safety | See 3c, then close | "Every step runs behind an allowlist, a 60 s cap, auto-abort and a kill switch. 35 acceptance checks cover this." |

### 3a. Live injection (the dashboard panel)

1. In **Autonomous injecting agent**, leave **AI planner** checked, set **experiments** to **1** or **2**, and press **Start the agent**.
2. Narrate what the page shows as it happens:
   - the pill goes **planning**, then **injecting 1/N**;
   - the **plan table** appears: each fault and target, **why the agent chose it**, its **hypothesis**, and which planner made it (AI model, or heuristic with the reason);
   - the live line reads "The agent is injecting *fault* into *service*... Nobody pressed a button", and changes from **measuring the steady state**, to **the fault is ACTIVE**, to **measuring recovery**;
   - the header pill shows **FAULT ACTIVE**, the charts get a red band, the graph node pulses, and the containers table and event log show the real restarts;
   - each row ends with a chip: **recovered in N s**, **auto-aborted (reason)**, or **refused by the safety guard**.
3. Each experiment takes a couple of minutes including recovery. Use that time to explain the plan table.

### 3b. Independent proof (terminal, started before 3a)

```powershell
docker events --filter type=container --filter label=chaosai.managed=true
```

Docker itself prints the `kill`, `pause` and `unpause`, `stop` and `start` events as the agent acts. Next to it, keep the vote page open in a tab: during the fault, votes fail or hang, and after the rollback they work again.

Stored evidence afterwards:

```powershell
dir results\*.json | sort LastWriteTime | select -last 3
dir results\agent-runs | sort LastWriteTime | select -last 1
```

Each experiment is a JSON file with its own timestamps, and the agent's run is saved under `results\agent-runs`. The acceptance run checks that every number in the report matches a stored result.

### 3c. Safety

Pick one:
- Press **Start the agent**, and while it is injecting press **KILL SWITCH**. The agent stops ("stopped: kill switch engaged"), every fault is undone, and the pill shows the kill switch. Then press **Reset**.
- Or run the agent normally and point out an **auto-aborted** chip if the planner picked a fault on a user-facing service.

Name the guarantees: only containers labelled `chaosai.managed=true`, one experiment at a time, hard cap of 60 s, abort thresholds that can only be tightened, rollback never raises, fail-closed if the sandbox is not healthy, and manual faults are refused while the agent runs.

---

## 4. If something goes wrong

| Problem | Do |
|---|---|
| Dashboard shows "kill switch engaged" at the start | Press **Reset** on the page. |
| Start button is greyed out | An experiment or the agent is still running, or the kill switch is engaged. Wait, or press **Reset**. |
| The agent refuses to start ("baseline error rate") | A second probe is running. Run `.\stop.cmd`, then `.\start.cmd`. |
| A container is stuck after an abort | `.\chaos.cmd reset` restores every managed container. |
| Docker is slow or unhealthy | `.\stop.cmd -Sandbox`, then `.\start.cmd -Fresh`. |
| The plan says "heuristic planner" | Say so: "no key or no quota, so the deterministic heuristic planned; the page states why." This is designed behaviour. |
| The live run fails | Play the recorded video, then show `results\demo-findings.html` and `results\demo-recommendations.md`. |

---

## 5. Likely questions

- **Is it really injecting, or replaying?** Real Docker faults: SIGKILL, pause, stop, CPU and memory limits, `tc netem` delay and loss, and a traffic multiplier. Docker's own event stream shows them (3b).
- **What does the AI do?** Two things only: the planner chooses and orders experiments from a validated candidate list, and the analyzer explains the stored results. Every number it cites must come from a tool output, or the answer is rejected. It never touches Docker.
- **What if there is no API key?** The whole system still runs. A heuristic ranks the experiments and a rule-based analyzer writes the findings (35 acceptance checks passed with keys removed).
- **Can the fixing agent break the system?** By default it changes nothing: it recommends. The opt-in `chaos fix` applies one vetted change, measures before and after, then puts the system back.
- **Limits, to say honestly:** one app, one machine, eight fault types, one measured fix in the catalogue, free-tier AI quotas, and the resilience score depends on which experiments ran.

---

## 6. Command cheat sheet

```powershell
.\start.cmd                                 # sandbox + dashboard + browser (-Fresh: clean sandbox; -NoBrowser)
.\stop.cmd                                  # undo any fault, stop the dashboard (-Sandbox: stop the containers too)
.\chaos.cmd recommend                       # ranked recommended changes (safe while the dashboard runs)
.\chaos.cmd abort                           # kill switch from a terminal
.\chaos.cmd reset                           # restore everything and clear the kill switch
.\chaos.cmd llm-check --provider pool       # do the AI keys answer?
.\chaos.cmd agent --max 2 --yes             # the same agent from a terminal (only with the dashboard stopped)
.\chaos.cmd pipeline --max 2 --yes          # inject, find, recommend from a terminal (only with the dashboard stopped)
```
