"""Acceptance run: every feature built so far, checked against the brief's acceptance checklist, from a clean results folder
and with no API key (a key is stripped from the environment and .env is ignored, so nothing here can spend credits).

    python scripts/acceptance.py [--results DIR] [--offline-only] [--keep-app] [--only SUBSTRING ...]

What it does to the machine: it runs real faults against the sandbox containers (labelled chaosai.managed=true) and, unless
--keep-app is given, starts the sandbox from scratch with `deploy/tasks.ps1 down` then `up` (the documented reset, which also
removes the sandbox's own demo-vote volume). It restores everything it changed and checks that at the end. Results go to
DIR (default results-acceptance) as acceptance.json and acceptance.md; the exit code is 0 only if every check passed.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PY = sys.executable
SERVICES = {"vote", "redis", "worker", "db", "result"}
NOT_EXPERIMENTS = {"graph.json", "baseline.json", "plan.json", "analysis.json", "findings.json", "active_faults.json", "active_fixes.json"}
KEY_PREFIXES = ("ANTHROPIC_", "GEMINI_", "GOOGLE_API", "MISTRAL_", "OPENROUTER_")
PORT = 8799


class Acceptance:
    def __init__(self, results: Path, offline_only: bool, keep_app: bool, only: list[str]):
        self.results, self.offline_only, self.keep_app, self.only = results.resolve(), offline_only, keep_app, only
        self.checks: list[dict] = []
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(KEY_PREFIXES) and k != "CHAOSAI_CONFIG"}
        self.env.update(CHAOSAI_RESULTS=str(self.results), CHAOSAI_NO_DOTENV="1", PYTHONIOENCODING="utf-8")
        self.docker_exe = None
        try:
            from chaosai.fixer import find_docker
            self.docker_exe = find_docker()
            self.env["PATH"] = str(Path(self.docker_exe).parent) + os.pathsep + self.env.get("PATH", "")
        except Exception:
            pass
        self.proc = None                       # the dashboard
        self.bg = None                         # a background `chaos run` that a check started

    # ------------------------------------------------------------ helpers
    def chaos(self, *args, timeout=900, input=None) -> tuple[int, str]:
        p = subprocess.run([PY, "-m", "chaosai", *args], cwd=ROOT, env=self.env, capture_output=True, text=True, timeout=timeout,
                           input=input, encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")

    def popen(self, *args) -> subprocess.Popen:
        return subprocess.Popen([PY, "-m", "chaosai", *args], cwd=ROOT, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")

    def docker(self, *args, timeout=120) -> tuple[int, str]:
        p = subprocess.run([self.docker_exe or "docker", *args], capture_output=True, text=True, timeout=timeout, env=self.env,
                           encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")

    def containers(self) -> dict[str, str]:
        rc, out = self.docker("ps", "--format", "{{.Names}}|{{.Status}}")
        found = {}
        for line in out.splitlines():
            if line.startswith("chaosai-voting-") and "|" in line:
                name, status = line.split("|", 1)
                found[name] = status
        return found

    def wait_healthy(self, timeout=150) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            c = self.containers()
            svc = {n.removeprefix("chaosai-voting-").rsplit("-", 1)[0] for n in c}
            if svc == SERVICES and all(s.startswith("Up") and "unhealthy" not in s and "starting" not in s and "paused" not in s.lower() for s in c.values()):
                return True
            time.sleep(3)
        return False

    def http(self, method, url, headers=None, body=None, host=None, timeout=15) -> tuple[int, dict, str]:
        u = urlparse(url)
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
        c.putrequest(method, u.path or "/", skip_host=host is not None)
        if host is not None:
            c.putheader("Host", host)
        raw = json.dumps(body).encode() if body is not None else b""
        for k, v in (headers or {}).items():
            c.putheader(k, v)
        c.putheader("Content-Length", str(len(raw)) if method == "POST" else "0")
        c.endheaders(raw if method == "POST" else None)
        r = c.getresponse()
        text = r.read().decode("utf-8", errors="replace")
        return r.status, {k.lower(): v for k, v in r.getheaders()}, text

    def experiment_files(self) -> list[Path]:
        return sorted((p for p in self.results.glob("*.json") if p.name not in NOT_EXPERIMENTS), key=lambda p: p.stat().st_mtime)

    def load(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def new_result(self, before: set[str]) -> dict:
        new = [p for p in self.experiment_files() if p.name not in before]
        if not new:
            raise AssertionError("no result file was written")
        return self.load(new[-1])

    def status(self) -> tuple[bool, str]:
        rc, out = self.chaos("status", timeout=60)
        return rc == 0 and "kill switch : clear" in out and "active faults: none" in out, out

    def settle(self, timeout=150) -> str:
        """After anything that injects: no active faults, kill switch clear, every container healthy. Returns a problem or ''."""
        end = time.time() + timeout
        last = ""
        while time.time() < end:
            ok, last = self.status()
            if ok and self.wait_healthy(15):
                return ""
            time.sleep(3)
        return "sandbox did not settle: " + last.strip().replace("\n", " | ")[:300]

    def check(self, group: str, name: str, fn) -> None:
        if self.only and not any(o.lower() in name.lower() or o.lower() == group.lower() for o in self.only):
            return
        if group == "live" and self.offline_only:
            self.checks.append({"group": group, "name": name, "status": "skipped", "detail": "--offline-only"})
            print(f"  SKIP  {name}")
            return
        t0 = time.time()
        try:
            detail = fn() or ""
            status = "pass"
        except AssertionError as e:
            status, detail = "fail", str(e)
        except Exception as e:
            status, detail = "fail", f"{type(e).__name__}: {e}"
        if status == "fail" and group == "live":
            detail += "   [recovery after the failure: " + (self.recover() or "sandbox clean and healthy") + "]"
        self.checks.append({"group": group, "name": name, "status": status, "detail": detail, "seconds": round(time.time() - t0, 1)})
        print(f"  {status.upper():5} {name}" + (f"   [{detail[:600]}]" if detail and status == "fail" else ""), flush=True)

    def recover(self) -> str:
        """After a failed live check: stop a stray background experiment, undo everything, clear the kill switch, wait until healthy,
        so one failure cannot make every later check fail for the wrong reason. Returns a problem, or '' if the sandbox is clean."""
        if self.bg and self.bg.poll() is None:
            self.bg.kill()
        self.bg = None
        for _ in range(2):
            self.chaos("reset", timeout=180)
        return self.settle(200)

    # ------------------------------------------------------------ the checks
    def run_all(self) -> None:
        self.results.mkdir(parents=True, exist_ok=True)
        try:
            self.offline_checks()
            self.live_checks()
        finally:
            self.cleanup()
        self.write_report()

    def must(self, cond, msg) -> None:
        if not cond:
            raise AssertionError(msg)

    def offline_checks(self) -> None:
        c, g = self.check, "offline"

        def unit_tests():
            p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT, env=self.env, capture_output=True, text=True, timeout=900,
                               encoding="utf-8", errors="replace")
            tail = (p.stdout or "").strip().splitlines()[-1:] or [""]
            self.must(p.returncode == 0, "pytest failed: " + tail[0])
            return tail[0]
        c(g, "the automated test suite passes", unit_tests)

        def readme_commands():
            readme = (ROOT / "README.md").read_text(encoding="utf-8")
            documented = set(re.findall(r"\| `chaos (\w+)", readme))
            rc, out = self.chaos("--help", timeout=60)
            missing = sorted(d for d in documented if d not in out)
            self.must(rc == 0 and documented and not missing, f"documented but missing from `chaos --help`: {missing}")
            return f"{len(documented)} documented commands all exist"
        c(g, "every command the README documents exists", readme_commands)

        def config_loads():
            rc, out = self.chaos("config", timeout=60)
            self.must(rc == 0 and "planner" in out and "fixer" in out, out[-200:])
        c(g, "the configuration loads and validates", config_loads)

        def discovery():
            rc, out = self.chaos("discover", timeout=120)
            self.must(rc == 0, out[-300:])
            graph = self.load(self.results / "graph.json")
            svc = set(graph["services"])
            self.must(svc == SERVICES, f"services {sorted(svc)}")
            self.must(graph["critical_path"] == ["vote", "redis", "worker", "db", "result"], f"critical path {graph['critical_path']}")
            self.must(graph["services"]["worker"]["depends_on"] and graph["services"]["db"]["stateful"], "dependencies/stateful flags missing")
            return "5 services, critical path vote > redis > worker > db > result"
        c(g, "discovery builds the dependency map from the compose file", discovery)

        faults = [("redis", "kill"), ("db", "stop"), ("worker", "pause"), ("worker", "cpu-limit"), ("db", "memory-limit"),
                  ("vote", "network-delay"), ("redis", "network-loss"), ("vote", "traffic-spike")]

        def dry_runs():
            for target, fault in faults:
                rc, out = self.chaos("run", target, fault, "--dry-run", timeout=60)
                self.must(rc == 0 and "would apply" in out, f"{fault} on {target}: {out[-200:]}")
            return f"{len(faults)} fault types validate and print their plan"
        c(g, "all eight fault types validate in a dry run", dry_runs)

        def allowlist():
            rc, out = self.chaos("run", "nginx", "kill", "--dry-run", timeout=60)
            self.must(rc != 0 and "allowlist" in out, out[-200:])
            rc, out = self.chaos("run", "redis", "rm-rf", "--dry-run", timeout=60)
            self.must(rc != 0, "an unknown fault was accepted")
        c(g, "only allowlisted services and known faults are accepted", allowlist)

        def hard_cap():
            rc, out = self.chaos("run", "redis", "kill", "--duration", "600", "--dry-run", timeout=60)
            self.must(rc == 0 and re.search(r"\b60\s*s", out), "duration was not clamped to 60 s: " + out[-300:])
        c(g, "a requested duration is clamped to the 60 s hard cap", hard_cap)

        def thresholds_tighten_only():
            rc, out = self.chaos("run", "redis", "kill", "--error-rate", "0.99", "--p95-ms", "99999", "--dry-run", timeout=60)
            self.must(rc == 0 and ("0.5" in out or "50%" in out) and "3000" in out, "abort thresholds were loosened: " + out[-300:])
        c(g, "abort thresholds can only be tightened, never loosened", thresholds_tighten_only)

        def plan_offline():
            rc, out = self.chaos("plan", "--offline", "--top", "4", timeout=120)
            self.must(rc == 0 and "planner: heuristic" in out and (self.results / "plan.json").exists(), out[-300:])
            plan = self.load(self.results / "plan.json")
            self.must(len(plan["experiments"]) == 4 and all(e["hypothesis"] and e["reason"] for e in plan["experiments"]), "plan lacks hypotheses or reasons")
        c(g, "the planner ranks experiments offline and explains each choice", plan_offline)

        def dry_flows():
            for args in (("agent", "--offline", "--dry-run", "--top", "2"), ("autopilot", "--offline", "--dry-run", "--top", "2"),
                         ("pipeline", "--offline", "--dry-run", "--top", "2"), ("fix", "--recipe", "scale-worker", "--dry-run"), ("fix", "--list")):
                rc, out = self.chaos(*args, timeout=120)
                self.must(rc == 0, f"{' '.join(args)}: {out[-200:]}")
            self.must(not self.experiment_files(), "a dry run wrote an experiment result")
            return "agent, autopilot, pipeline and fix dry runs inject nothing"
        c(g, "the dry runs of every agent change nothing", dry_flows)

        def no_key_fallbacks():
            rc, out = self.chaos("plan", "--top", "3", timeout=180)                      # default config, no key, no .env
            self.must(rc == 0 and "fell back to heuristic" in out, "the planner did not fall back cleanly: " + out[-300:])
            rc, out = self.chaos("llm-check", "--provider", "gemini", timeout=60)
            self.must(rc == 1 and "GEMINI_API_KEY" in out, out[-200:])
            rc, out = self.chaos("llm-check", "--provider", "pool", "--no-send", timeout=60)
            self.must(rc == 0 and "pool member(s)" in out, out[-200:])
            return "with no key the planner falls back to the heuristic and llm-check says what is missing"
        c(g, "with no API key the AI steps fall back instead of failing", no_key_fallbacks)

        def revert_noop():
            rc, out = self.chaos("fix", "--revert", timeout=60)
            self.must(rc == 0 and "reverted" in out, out[-200:])
        c(g, "reverting fixes with nothing recorded is a safe no-op", revert_noop)

    def live_checks(self) -> None:
        c, g = self.check, "live"
        if self.offline_only:
            for name in ("fresh sandbox", "baseline", "faults", "kill switch", "lock", "auto-abort", "fail closed", "agent", "report", "pipeline", "dashboard"):
                c(g, name, lambda: None)
            return

        def fresh_setup():
            self.must(self.docker_exe, "the docker command was not found")
            script = ROOT / "deploy" / "tasks.ps1"
            if not self.keep_app:
                p = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "down"], capture_output=True, text=True,
                                   timeout=300, env=self.env, encoding="utf-8", errors="replace")
                self.must(p.returncode == 0, "tasks.ps1 down failed: " + (p.stderr or p.stdout)[-300:])
            p = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "up"], capture_output=True, text=True,
                               timeout=600, env=self.env, encoding="utf-8", errors="replace")
            self.must(p.returncode == 0, "tasks.ps1 up failed: " + (p.stderr or p.stdout)[-300:])
            self.must(self.wait_healthy(180), f"containers not healthy: {self.containers()}")
            return "5 services healthy" + (" (kept the running app)" if self.keep_app else " after a from-scratch down and up")
        c(g, "the sandbox starts from scratch with the documented commands and is healthy", fresh_setup)

        def journeys():
            s, _, _ = self.http("GET", "http://127.0.0.1:8080/")
            s2, _, _ = self.http("GET", "http://127.0.0.1:8081/")
            self.must(s == 200 and s2 == 200, f"vote {s}, result {s2}")
            s3, _, _ = self.http("POST", "http://127.0.0.1:8080/", headers={"Content-Type": "application/x-www-form-urlencoded"}, body=None)
            self.must(s3 < 500, f"a vote POST returned {s3}")
        c(g, "both user journeys answer", journeys)

        def baseline():
            rc, out = self.chaos("baseline", "--seconds", "30", timeout=240)
            self.must(rc == 0, out[-300:])
            b = self.load(self.results / "baseline.json")
            o = b["overall"]
            self.must(o["n"] > 20 and o["error_rate"] == 0 and o["p95_ms"] is not None, f"baseline {o}")
            return f"n={o['n']} error_rate={o['error_rate']} p50={o['p50_ms']}ms p95={o['p95_ms']}ms"
        c(g, "a baseline is recorded under steady load", baseline)

        faults = [("redis", "kill"), ("db", "stop"), ("worker", "pause"), ("worker", "cpu-limit"), ("db", "memory-limit"),
                  ("vote", "network-delay"), ("redis", "network-loss"), ("vote", "traffic-spike")]
        seen_ids: set[str] = set()
        for target, fault in faults:
            def one(target=target, fault=fault):
                before = {p.name for p in self.experiment_files()}
                extra = ["-p", "multiplier=15"] if fault == "traffic-spike" else []
                rc, out = self.chaos("run", target, fault, "--duration", "10", "--baseline-s", "8", *extra, timeout=400)
                self.must(rc in (0, 1), f"exit {rc}: {out[-300:]}")
                r = self.new_result(before)
                exp = r["experiment"]
                self.must(exp["id"] not in seen_ids and exp["target"] == target and exp["fault"] == fault, "id/target/fault wrong")
                seen_ids.add(exp["id"])
                self.must(not r["rollback_errors"], f"rollback errors: {r['rollback_errors']}")
                self.must(r["baseline"]["n"] > 0 and r["during"]["n"] > 0, "baseline or during window is empty")
                self.must(r["recovered"] or r["aborted"], "neither recovered nor aborted")
                problem = self.settle()
                self.must(not problem, problem)
                return f"recovered={r['recovered']} aborted={r['aborted']} errors_during={r['during']['error_rate']} recovery_s={r['recovery_s']}"
            c(g, f"faults: {fault} on {target} is injected, measured, rolled back and the sandbox is healthy", one)

        def kill_switch():
            before = {p.name for p in self.experiment_files()}
            p = self.bg = self.popen("run", "redis", "pause", "--duration", "40", "--baseline-s", "4")
            end = time.time() + 60
            while time.time() < end and "pause@redis" not in self.chaos("status", timeout=60)[1]:
                time.sleep(2)
            self.must("pause@redis" in self.chaos("status", timeout=60)[1], "the fault never became active")
            rc, out = self.chaos("abort", timeout=120)
            self.must(rc == 0, "abort: " + out[-200:])
            try:
                p.wait(timeout=90)
            except subprocess.TimeoutExpired:
                p.kill()
                raise AssertionError("the running experiment did not stop after the kill switch")
            r = self.new_result(before)
            self.must(r["aborted"] and "kill switch" in (r["abort_reason"] or ""), f"abort_reason={r['abort_reason']}")
            self.must(not r["rollback_errors"], f"rollback errors {r['rollback_errors']}")
            rc, out = self.chaos("reset", timeout=120)
            self.must(rc == 0, "reset: " + out[-200:])
            problem = self.settle()
            self.must(not problem, problem)
            return "the kill switch stopped a running experiment, the pause was undone, reset cleared the switch"
        c(g, "the kill switch aborts a running experiment and everything is restored", kill_switch)

        def lock():
            # a fault that keeps every container healthy, so the second experiment gets past the health check and meets the lock
            p = self.bg = self.popen("run", "worker", "cpu-limit", "--duration", "40", "--baseline-s", "4")
            end = time.time() + 60
            while time.time() < end and "cpu-limit@worker" not in self.chaos("status", timeout=60)[1]:
                time.sleep(2)
            self.must("cpu-limit@worker" in self.chaos("status", timeout=60)[1], "the first fault never became active")
            rc, out = self.chaos("run", "redis", "kill", "--duration", "5", "--baseline-s", "3", timeout=120)
            self.chaos("abort", timeout=120)
            try:
                p.wait(timeout=90)
            except subprocess.TimeoutExpired:
                p.kill()
            self.chaos("reset", timeout=120)
            self.must(rc != 0 and "already running" in out, f"the second experiment was not refused (exit {rc}): {out[-300:]}")
            problem = self.settle()
            self.must(not problem, problem)
            return "a second experiment is refused while one runs, even from another process"
        c(g, "two experiments never overlap", lock)

        def auto_abort():
            before = {p.name for p in self.experiment_files()}
            t0 = time.time()
            rc, out = self.chaos("run", "redis", "stop", "--duration", "30", "--baseline-s", "5", timeout=300)
            r = self.new_result(before)
            self.must(r["aborted"] and "error rate" in (r["abort_reason"] or ""), f"not auto-aborted: {r['abort_reason']}")
            self.must(not r["rollback_errors"] and r["recovered"], f"rollback {r['rollback_errors']} recovered={r['recovered']}")
            problem = self.settle()
            self.must(not problem, problem)
            return f"auto-aborted on a threshold breach ({r['abort_reason']}), rolled back, recovered in {r['recovery_s']} s"
        c(g, "a threshold breach aborts the experiment automatically and rolls back", auto_abort)

        def fail_closed():
            before = {p.name for p in self.experiment_files()}
            self.docker("stop", "chaosai-voting-db-1")
            try:
                rc, out = self.chaos("run", "vote", "kill", "--duration", "5", "--baseline-s", "4", timeout=180)
                refused = rc != 0 and ("not healthy" in out or "steady state" in out)
                wrote = {p.name for p in self.experiment_files()} != before
            finally:
                self.docker("start", "chaosai-voting-db-1")
            self.must(refused, f"an experiment started on an unhealthy sandbox (exit {rc}): {out[-300:]}")
            self.must(not wrote, "a refused experiment still wrote a result")
            problem = self.settle(200)
            self.must(not problem, problem)
            return "with the database stopped the safety layer refused to inject (fail closed)"
        c(g, "it refuses to inject into an unhealthy sandbox (fails closed)", fail_closed)

        def agent():
            before = {p.name for p in self.experiment_files()}
            rc, out = self.chaos("agent", "--offline", "--yes", "--top", "2", "--max", "2", timeout=900)
            self.must(rc == 0, out[-300:])
            new = [p for p in self.experiment_files() if p.name not in before]
            self.must(len(new) == 2, f"{len(new)} results written")
            self.must(list((self.results / "agent-runs").glob("*.json")), "no agent run log")
            problem = self.settle()
            self.must(not problem, problem)
            return "the injecting agent ran 2 planned experiments by itself"
        c(g, "the injecting agent plans and runs experiments with no manual step", agent)

        def report():
            rc, out = self.chaos("report", "--offline", timeout=300)
            self.must(rc == 0, out[-300:])
            rep = self.load(self.results / "findings.json")
            self.must(rep["score"]["score"] is not None and rep["weaknesses"] and rep["fixes"], "score, weaknesses or fixes missing")
            self.must((self.results / "findings.md").exists() and (self.results / "findings.html").exists(), "report files missing")
            ids = {p.stem for p in self.experiment_files()}
            for e in rep["experiments"]:
                self.must(e["id"] in ids, f"the report lists {e['id']}, which is not a stored result")
                stored = self.load(self.results / f"{e['id']}.json")
                self.must(e["recovery_s"] == stored["recovery_s"] and e["aborted"] == stored["aborted"], f"{e['id']}: report differs from the stored result")
            for w in rep["weaknesses"]:
                self.must(w["evidence"] and w["severity"] in ("critical", "high", "medium", "low"), f"weakness without evidence/severity: {w['title']}")
            for f in rep["fixes"]:
                self.must(f["change"].strip() and f["verification"], f"fix without a concrete change or a verification: {f['title']}")
            return f"score {rep['score']['score']} ({rep['score']['band']}), {len(rep['weaknesses'])} weaknesses, every number matches a stored result"
        c(g, "the report is generated from stored results only, with evidence, severity and a verification for each fix", report)

        def pipeline():
            before_fix = set((self.results / "fixes").glob("*.json")) if (self.results / "fixes").exists() else set()
            rc, out = self.chaos("pipeline", "--offline", "--yes", "--top", "2", "--max", "2", timeout=1500)
            self.must(rc == 0, f"exit {rc}: {out[-400:]}")
            for marker in ("AGENT 1/3: INJECTING", "AGENT 2/3: FINDING", "AGENT 3/3: FIXING", "== SUMMARY =="):
                self.must(marker in out, f"missing {marker!r}")
            new = [p for p in (self.results / "fixes").glob("*.json") if p not in before_fix]
            self.must(len(new) == 1, f"{len(new)} fix records written")
            fix = self.load(new[0])
            self.must(fix["status"] in ("improved", "inconclusive", "worse", "skipped"), f"fix status {fix['status']}: {fix['detail']}")
            self.must(not fix["revert_errors"] and (fix["reverted"] or fix["status"] == "skipped"), f"not reverted: {fix['revert_errors']}")
            self.must("Fix validation" in (self.results / "findings.md").read_text(encoding="utf-8"), "the report has no fix validation section")
            self.must(not (self.results / "active_fixes.json").exists(), "a fix is still recorded as applied")
            n_workers = sum(1 for n in self.containers() if "-worker-" in n)
            self.must(n_workers == 1, f"{n_workers} worker containers after the fix was reverted")
            problem = self.settle()
            self.must(not problem, problem)
            return f"all three agents ran; the fix was {fix['status']}: {fix['claim'] or fix['detail']}"[:300]
        c(g, "all three agents run end to end (inject, find, fix) and the sandbox is restored", pipeline)

        def dashboard():
            base = f"http://127.0.0.1:{PORT}"
            self.proc = self.popen("dashboard", "--port", str(PORT))
            end = time.time() + 90
            while time.time() < end:
                try:
                    if self.http("GET", base + "/api/metrics")[0] == 200:
                        break
                except OSError:
                    pass
                time.sleep(2)
            s, _, body = self.http("GET", base + "/api/metrics")
            self.must(s == 200, f"/api/metrics {s}")
            m = json.loads(body)
            self.must({"series", "gauges", "events", "containers", "experiment", "kill_switch"} <= set(m) and set(m["containers"]) == SERVICES, "metrics shape")
            s, _, body = self.http("GET", base + "/api/graph")
            self.must(s == 200 and set(json.loads(body)["services"]) == SERVICES, "graph")
            s, _, body = self.http("GET", base + "/api/report")
            rep = json.loads(body)
            self.must(s == 200 and not rep.get("empty") and rep["score"]["score"] is not None and "fix_validations" in rep, "report panel data")
            for ext, ctype in (("html", "text/html"), ("md", "text/markdown"), ("json", "application/json")):
                s, h, _ = self.http("GET", f"{base}/report.{ext}")
                self.must(s == 200 and h["content-type"].startswith(ctype) and "attachment" in h.get("content-disposition", ""), f"/report.{ext} {s}")
            s, _, page = self.http("GET", base + "/")
            for needle in ("r_score", "r_hist", "r_find", "r_fix", "graph"):
                self.must(needle in page, f"the page lacks {needle}")
            return "metrics, graph, report and the three downloads are served"
        c(g, "the dashboard serves live metrics, the graph, the results panels and report downloads", dashboard)

        def dashboard_security():
            base = f"http://127.0.0.1:{PORT}"
            self.must(self.http("GET", base + "/api/metrics", host="evil.example")[0] == 403, "a foreign Host header was accepted")
            self.must(self.http("POST", base + "/api/inject", body={"target": "redis", "fault": "kill"}, headers={"Content-Type": "application/json"})[0] == 403,
                      "a POST without the X-Chaos header was accepted")
            self.must(self.http("POST", base + "/api/inject", body={"target": "redis", "fault": "kill"},
                                headers={"Content-Type": "application/json", "X-Chaos": "1", "Origin": "http://evil.example"})[0] == 403, "a cross-origin POST was accepted")
            self.must(self.http("POST", base + "/api/inject", body={"target": "nginx", "fault": "kill"},
                                headers={"Content-Type": "application/json", "X-Chaos": "1"})[0] in (400, 409), "an unlisted target was accepted")
            return "foreign host, missing header, cross-origin and unlisted target are all refused"
        c(g, "the dashboard refuses cross-origin, header-less and non-allowlisted requests", dashboard_security)

        def dashboard_kill_switch():
            base = f"http://127.0.0.1:{PORT}"
            hdr = {"Content-Type": "application/json", "X-Chaos": "1"}
            s, _, body = self.http("POST", base + "/api/inject", body={"target": "redis", "fault": "pause"}, headers=hdr)
            self.must(s == 202, f"inject {s}: {body}")
            end = time.time() + 60
            phase = ""
            while time.time() < end and phase != "fault":
                phase = json.loads(self.http("GET", base + "/api/metrics")[2])["experiment"].get("phase", "")
                time.sleep(1)
            self.must(phase == "fault", f"the fault never became active (phase {phase})")
            s, _, body = self.http("POST", base + "/api/abort", body={}, headers=hdr)
            self.must(s == 200, f"abort {s}: {body}")
            end = time.time() + 90
            while time.time() < end and json.loads(self.http("GET", base + "/api/metrics")[2])["experiment"].get("running"):
                time.sleep(2)
            m = json.loads(self.http("GET", base + "/api/metrics")[2])
            self.must(not m["experiment"].get("running") and m["kill_switch"], "the experiment did not stop, or the switch is not shown")
            self.must("pause@redis" not in self.chaos("status", timeout=60)[1], "the pause is still active")
            s, _, body = self.http("POST", base + "/api/reset", body={}, headers=hdr)
            self.must(s == 200 and not json.loads(self.http("GET", base + "/api/metrics")[2])["kill_switch"], f"reset {s}: {body}")
            return "the KILL SWITCH endpoint stopped an injected experiment and Reset cleared it"
        c(g, "the dashboard's emergency stop works while an experiment runs", dashboard_kill_switch)

    def cleanup(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if not self.offline_only and self.docker_exe:
            self.chaos("reset", timeout=180)

        def final():
            problem = self.settle(200)
            self.must(not problem, problem)
            self.must(not (self.results / "active_fixes.json").exists() and not (self.results / "active_faults.json").exists(), "an undo record is left over")
            c = self.containers()
            self.must(len(c) == 5 and sum("-worker-" in n for n in c) == 1, f"containers: {sorted(c)}")
            return "kill switch clear, no active faults, no pending fixes, one worker, all five services healthy"
        if not self.offline_only:
            self.check("live", "the sandbox is back to a clean, healthy state at the end", final)

    def write_report(self) -> None:
        counts = {s: sum(1 for c in self.checks if c["status"] == s) for s in ("pass", "fail", "skipped")}
        (self.results / "acceptance.json").write_text(json.dumps({"counts": counts, "checks": self.checks}, indent=2), encoding="utf-8")
        lines = ["# Acceptance run", "", f"{counts['pass']} passed, {counts['fail']} failed, {counts['skipped']} skipped.", "",
                 "| Result | Check | Detail |", "|---|---|---|"]
        lines += [f"| {c['status'].upper()} | {c['name']} | {str(c['detail']).replace('|', '/')[:300]} |" for c in self.checks]
        (self.results / "acceptance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n{counts['pass']} passed, {counts['fail']} failed, {counts['skipped']} skipped; report: {self.results / 'acceptance.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Acceptance run against the brief's checklist (offline: no API key is used).")
    ap.add_argument("--results", default=str(ROOT / "results-acceptance"), help="a clean results folder (default: results-acceptance)")
    ap.add_argument("--offline-only", action="store_true", help="skip everything that needs Docker")
    ap.add_argument("--keep-app", action="store_true", help="do not tear the sandbox down and rebuild it first")
    ap.add_argument("--only", nargs="*", default=[], help="run only checks whose name contains one of these")
    a = ap.parse_args()
    acc = Acceptance(Path(a.results), a.offline_only, a.keep_app, a.only)
    print(f"acceptance run, results in {acc.results}, no API key\n")
    acc.run_all()
    return 0 if all(c["status"] != "fail" for c in acc.checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
