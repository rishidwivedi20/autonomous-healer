"""Deterministic, rule-based analyzer: same output schema as the LLM agent, no network, no key.

It reads the same evidence tools and builds every claim from the returned values, so its output passes
the same validator the LLM's answers must pass (that is tested). It recognises a fixed set of patterns
(worker throughput ceiling, cascading restarts, pipeline stalls the probe cannot see, ...); anything else
is reported as observed impact without a diagnosis. It is the fallback when there is no key or no
network, and a baseline the LLM's analysis can be compared against.
"""
from __future__ import annotations

from .analysis import Analysis, Evid, ExpAnalysis, Fix, SEVERITIES, Weakness, Recorder

LATENCY_SLACK_MS = 50.0     # same slack the score uses, so tiny baselines do not look like huge ratios
SLOW_RATIO = 5.0            # median p95 during the fault at least this many times the baseline p95


def g(x) -> str:
    return "n/a" if x is None else f"{x:g}"


def _first(d: dict, *keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _clean(items: list) -> list:
    """Prefer experiments whose own data is trustworthy (no rollback problems) as the headline example."""
    return [f for f in items if not f.e.get("rollback_errors")] or items


class _Facts:
    """Everything the tools returned for one experiment."""

    def __init__(self, rec: Recorder, i: str):
        self.id = i
        self.e = rec.call("get_experiment", {"experiment_id": i})
        g_ = rec.call("get_gauges", {"experiment_id": i})
        self.gs = g_.get("summary") if "error" not in g_ else None
        m = rec.call("get_metrics", {"experiment_id": i, "window": "during"})
        self.ms = m.get("summary") if "error" not in m else None
        self.target, self.fault = self.e["target"], self.e["fault"]
        self.restarts = self.e.get("restarts") or {}
        self.backlog = bool(self.gs and self.gs.get("peak_queue") is not None
                            and self.gs["peak_queue"] > (self.gs.get("baseline_queue_max") or 0) + 2)

    def max_err(self, journey: str):
        return _first(self.ms or {}, journey, "max_error_pct")

    def ev(self, claim: str) -> Evid:
        return Evid(experiment_id=self.id, claim=claim)


def _experiment(f: _Facts) -> ExpAnalysis:
    e, ev = f.e, []
    for j in ("vote", "result"):
        s = (f.ms or {}).get(j)
        if s and s.get("seconds_observed"):
            base = _first(e, "baseline_by_journey", j, "p95_ms")
            ev.append(f.ev(f"{j} journey while the fault was active: up to {g(s['max_error_pct'])}% errors in a second "
                           f"({s['seconds_with_errors']} of {s['seconds_observed']} seconds had errors), p95 up to "
                           f"{g(s['max_p95_ms'])} ms" + (f" (baseline p95 {g(base)} ms)" if base is not None else "")))
    if f.backlog:
        gs = f.gs
        back = (f"back to baseline at {g(gs['queue_back_to_baseline_at_s'])} s" if gs.get("queue_back_to_baseline_at_s") is not None
                else "not back to baseline within the stored window")
        ev.append(f.ev(f"redis queue peaked at {g(gs['peak_queue'])} votes at {g(gs['peak_queue_at_s'])} s "
                       f"(baseline max {g(gs['baseline_queue_max'])}), {back}"))
    if f.restarts:
        ev.append(f.ev("container restarts: " + ", ".join(f"{s} x{n}" for s, n in sorted(f.restarts.items()))))
    if e.get("aborted"):
        ev.append(f.ev(f"experiment was aborted: {e['abort_reason']}"))
    if e.get("recovered"):
        ev.append(f.ev(f"recovered after {g(e['recovery_s'])} s (probe {g(e['recovery_probe_s'])} s, pipeline {g(e['recovery_pipeline_s'])} s)"
                       if e.get("pipeline_assessed") else f"recovered after {g(e['recovery_s'])} s (judged by the probe alone)"))
    else:
        ev.append(f.ev("did not recover within the stored window or recovery was not measured"))

    blast = {f.target} | set(f.restarts)
    blast |= {j for j in ("vote", "result") if (f.max_err(j) or 0) > 0}
    if f.backlog:
        blast |= {"worker", "redis"}
    impact = []
    if any((f.max_err(j) or 0) > 0 for j in ("vote", "result")):
        impact.append("user-facing errors were recorded")
    if f.backlog:
        impact.append("votes backed up in redis")
    if set(f.restarts) - {f.target}:
        impact.append("dependent services restarted")
    cause = (f"{f.fault} on {f.target} (active {g(e.get('fault_active_s'))} s): "
             + ("; ".join(impact) if impact else "no user-visible or pipeline impact was recorded") + ".")
    return ExpAnalysis(experiment_id=f.id, hypothesis_held=None, root_cause=cause,
                       blast_radius=sorted(blast), evidence=ev)


def analyse(rec: Recorder, ids: list[str]) -> Analysis:
    graph = rec.call("get_graph", {})
    services = graph.get("services", {}) if "error" not in graph else {}
    facts = [_Facts(rec, i) for i in ids]
    weaknesses: list[Weakness] = []
    fixes: list[Fix] = []

    def add(w: Weakness, fix: Fix):
        weaknesses.append(w)
        fixes.append(fix.model_copy(update={"addresses": [w.title], "evidence": w.evidence}))

    # ---- W1: the worker is the throughput bottleneck
    backlogged = [f for f in facts if f.backlog and f.gs["peak_queue"] >= 20 and f.gs.get("votes_per_s_while_draining")]
    if backlogged:
        worst = max(_clean(backlogged), key=lambda f: f.gs["peak_queue"])
        gs = worst.gs
        evid = [worst.ev(f"redis queue peaked at {g(gs['peak_queue'])} votes (baseline max {g(gs['baseline_queue_max'])}); the worker "
                         f"drained it at {g(gs['votes_per_s_while_draining'])} votes/s against {g(gs['votes_per_s_before'])} votes/s "
                         f"before the fault, a backlog lag of {g(gs['peak_backlog_lag_s'])} s")]
        for f in backlogged:
            if f is not worst and len(evid) < 4:
                evid.append(f.ev(f"after {f.fault} on {f.target} the worker again drained at {g(f.gs['votes_per_s_while_draining'])} votes/s"))
        wrep = _first(services, "worker", "replicas")
        if wrep is not None:
            evid.append(worst.ev(f"worker runs {g(wrep)} replica(s)"))
        lag = gs.get("peak_backlog_lag_s") or 0
        add(Weakness(title="Single worker caps pipeline throughput; any burst or stall builds a backlog",
                     severity="high" if lag >= 20 else "medium", services=["worker", "redis"],
                     description="Votes are consumed by one worker at a fixed rate. When votes arrive faster than that, or the pipeline "
                                 "stalls and resumes, the redis queue grows and votes are applied late, while the user-facing journeys "
                                 "keep returning success.", evidence=evid),
            Fix(title="Run more worker replicas", rationale="Workers pop from a shared redis list, so replicas share the load; "
                "throughput scales with the replica count until postgres becomes the limit.",
                change="worker:\n  deploy:\n    replicas: 3          # was 1; re-run the traffic spike to confirm the drain rate scales", addresses=[], evidence=[]))

    # ---- W2: dependents crash instead of tolerating a lost dependency
    cascades = [f for f in facts if set(f.restarts) - {f.target}]
    if cascades:
        worst = max(_clean(cascades), key=lambda f: sum(f.restarts.values()))
        affected = sorted((set(worst.restarts) - {worst.target}) | {worst.target})
        add(Weakness(title="Dependents exit instead of retrying when a dependency goes away (cascading restarts)",
                     severity="high", services=affected,
                     description="When one service failed, services that depend on it restarted too instead of waiting and reconnecting, "
                                 "widening the blast radius beyond the faulted service.",
                     evidence=[worst.ev(f"{worst.fault} on {worst.target}: restarts " + ", ".join(f"{s} x{n}" for s, n in sorted(worst.restarts.items())))]),
            Fix(title="Retry dependency connections with backoff", rationale="Reconnect-with-backoff turns a dependency outage into a delay "
                "instead of a restart storm. This is an application change in result and worker.",
                change="# application code (worker and result):\n# wrap the database/redis connection in a retry loop with exponential backoff (e.g. 0.5 s doubling to 10 s)\n"
                       "# and only exit after a long, bounded failure window instead of on the first connection error.", addresses=[], evidence=[]))

    # ---- W3: failures the user-facing probe cannot see
    blind = [f for f in facts if f.e.get("pipeline_assessed") and f.e.get("recovery_pipeline_s") is not None
             and f.e.get("recovery_probe_s") is not None and f.e["recovery_pipeline_s"] - f.e["recovery_probe_s"] >= 10]
    if blind:
        worst = max(_clean(blind), key=lambda f: f.e["recovery_pipeline_s"] - f.e["recovery_probe_s"])
        add(Weakness(title="Pipeline stalls are invisible to the user-facing health signals",
                     severity="medium", services=["worker", "redis", "db"],
                     description="Vote requests and the result page kept succeeding while votes were not reaching the database. Nothing a user "
                                 "or a simple health check sees reports this.",
                     evidence=[worst.ev(f"after {worst.fault} on {worst.target}: probe recovery {g(worst.e['recovery_probe_s'])} s but "
                                        f"pipeline recovery {g(worst.e['recovery_pipeline_s'])} s")]),
            Fix(title="Monitor queue depth and ingest rate", rationale="The queue length and Postgres row growth are the signals that "
                "show a stalled pipeline; alert on them (or fail a worker healthcheck when the queue stays high).",
                change="# alert/healthcheck rule (example):\n# redis-cli llen votes   -> alert when above 50 for 30 s\n"
                       "# SELECT count(*) FROM votes -> alert when it does not grow while llen votes > 0", addresses=[], evidence=[]))

    # ---- W4: the worker has no healthcheck
    worker = services.get("worker")
    if worker and not worker.get("has_healthcheck"):
        anchor = next((f for f in facts if f.target == "worker"), facts[0] if facts else None)
        if anchor:
            add(Weakness(title="The worker has no healthcheck, so a stuck worker looks healthy", severity="medium", services=["worker"],
                         description="Docker can only report 'running' for the worker; a hung process or a lost connection is not detected "
                                     "or restarted.",
                         evidence=[anchor.ev(f"worker runs {g(worker['replicas'])} replica(s) and has no healthcheck configured")]),
                Fix(title="Add a worker healthcheck", rationale="Without one, the restart policy only reacts to the process exiting. "
                    "Verify the command exists in the worker image.",
                    change="worker:\n  healthcheck:\n    test: [\"CMD-SHELL\", \"pidof dotnet || exit 1\"]   # confirm pidof exists in the image\n"
                           "    interval: 5s\n    timeout: 3s\n    retries: 3", addresses=[], evidence=[]))

    # ---- W5: redis crash brings back old data (no append-only persistence)
    replay = [f for f in facts if f.target == "redis" and f.fault == "kill" and f.gs and f.gs.get("peak_queue_at_s") is not None
              and f.gs["peak_queue_at_s"] <= 3 and (f.gs.get("peak_queue") or 0) >= 10 and f.gs.get("votes_per_s_before")]
    if replay:
        f = max(replay, key=lambda x: x.gs["peak_queue"])
        add(Weakness(title="A redis crash is not crash-consistent: the queue reappears with stale contents", severity="high",
                     services=["redis", "worker", "db"],
                     description="Right after redis was killed and restarted, the queue held far more votes than the probe could have sent "
                                 "in that time and the vote count in Postgres stood still while it drained. That is consistent with redis "
                                 "reloading an old snapshot (default snapshot persistence, no append-only file), which replays old votes "
                                 "and drops recent ones. This is an inference from the timing, not a directly measured cause.",
                     evidence=[f.ev(f"after the redis kill the queue read {g(f.gs['peak_queue'])} votes at {g(f.gs['peak_queue_at_s'])} s, "
                                    f"although the probe sends about {g(f.gs['votes_per_s_before'])} votes/s")]),
            Fix(title="Use append-only persistence for the vote queue", rationale="AOF with per-second fsync bounds loss to about a "
                "second of votes and avoids reloading a stale snapshot.",
                change="redis:\n  command: [\"redis-server\", \"--appendonly\", \"yes\", \"--appendfsync\", \"everysec\"]\n  volumes: [\"redis-data:/data\"]",
                addresses=[], evidence=[]))

    # ---- W6: a memory cap crash-loops the database
    mem = [f for f in facts if f.fault == "memory-limit" and f.restarts.get(f.target, 0) >= 3]
    if mem:
        f = max(_clean(mem), key=lambda x: x.restarts[x.target])
        add(Weakness(title="No resource sizing: a tight memory cap crash-loops the database", severity="medium", services=[f.target],
                     description="With no memory reservation or limit chosen for the service, its behaviour under memory pressure is "
                                 "unknown, and a small cap sent it into a restart loop that stalled the pipeline.",
                     evidence=[f.ev(f"with a {f.e['params'].get('mem')} memory cap {f.target} restarted {f.restarts[f.target]} times; "
                                    + (f"recovery took {g(f.e['recovery_s'])} s" if f.e.get("recovered")
                                       else "it did not recover within the stored window"))]),
            Fix(title="Size and reserve memory for the database", rationale="Choose a limit with headroom above the working set so "
                "pressure degrades performance instead of killing the process.",
                change="db:\n  deploy:\n    resources:\n      reservations: {memory: 128M}\n      limits: {memory: 512M}", addresses=[], evidence=[]))

    # ---- W7: a network fault on the entry service takes its journey fully down
    for f in facts:
        if f.fault in ("network-delay", "network-loss") and (f.max_err(f.target) or 0) >= 50:
            add(Weakness(title=f"A degraded network on {f.target} takes the whole {f.target} journey down", severity="high", services=[f.target],
                         description="With the only replica slowed, requests time out instead of being served by a healthy instance.",
                         evidence=[f.ev(f"{f.fault} on {f.target}: {f.target} journey reached {g(f.max_err(f.target))}% errors in a second")]),
                Fix(title=f"Run two {f.target} replicas behind a proxy", rationale="A load balancer can route around one degraded instance.",
                    change=f"{f.target}:\n  deploy:\n    replicas: 2\n# put a reverse proxy (nginx/traefik) in front and drop the fixed host port on the replicas",
                    addresses=[], evidence=[]))
            break

    # ---- W8: network trouble on a dependency makes users wait seconds instead of failing fast
    slow = []
    for f in facts:
        if f.fault not in ("network-delay", "network-loss"):
            continue
        for j in ("vote", "result"):
            s = (f.ms or {}).get(j) or {}
            base, med = _first(f.e, "baseline_by_journey", j, "p95_ms"), s.get("median_p95_ms")
            if base is None or med is None or (f.max_err(j) or 0) >= 50:
                continue                                  # a journey that went fully down is W7's finding
            ratio = (med + LATENCY_SLACK_MS) / (base + LATENCY_SLACK_MS)
            if ratio >= SLOW_RATIO:
                slow.append((ratio, f, j, s, base))
    if slow:
        pool = [x for x in slow if not x[1].e.get("rollback_errors")] or slow
        best: dict[str, tuple] = {}
        for x in sorted(pool, key=lambda x: -x[0]):       # worst journey per experiment, worst experiments first
            best.setdefault(x[1].id, x)
        shown = list(best.values())[:3]
        evid = [f.ev(f"{f.fault} on {f.target}: {j} journey p95 had a median of {g(s['median_p95_ms'])} ms and reached "
                     f"{g(s['max_p95_ms'])} ms, against a baseline p95 of {g(base)} ms (no errors, just waiting)")
                for _, f, j, s, base in shown]
        top = shown[0]
        targets = sorted({x[1].target for x in shown})
        add(Weakness(title="Network trouble on a dependency makes users wait seconds instead of failing fast",
                     severity="high" if top[0] >= 20 else "medium",
                     services=sorted(set(targets) | {x[2] for x in shown}),
                     description="With the network to a dependency delayed or lossy, requests did not fail: they waited, so user latency rose "
                                 "many times over while the error rate stayed low. A health check that only looks at errors misses this, "
                                 "and callers hold their connections the whole time.",
                     evidence=evid),
            Fix(title="Set short timeouts and fail fast on dependency calls",
                rationale="A bounded timeout (with at most one retry) turns an unbounded wait into a quick, explicit failure or fallback, "
                          "so a slow dependency cannot stall every user request.",
                change="# in the caller (e.g. the vote service's redis client):\n"
                       "redis = Redis(host=\"redis\", socket_connect_timeout=0.5, socket_timeout=0.5)\n"
                       "# on timeout: return a quick 503 / 'try again' instead of holding the request open",
                addresses=[], evidence=[]))

    if not weaknesses and facts:
        f = facts[0]
        weaknesses.append(Weakness(title="No weakness was detected by the rules", severity="low", services=[f.target],
                                   description="None of the known patterns matched the stored experiments.",
                                   evidence=[f.ev(f"{f.fault} on {f.target} caused no recorded impact pattern the rules recognise")]))
        fixes.append(Fix(title="Run more varied experiments", change="chaos plan --offline && chaos agent --offline",
                         rationale="More faults give the rules more to match.", addresses=[weaknesses[0].title], evidence=weaknesses[0].evidence))

    weaknesses.sort(key=lambda w: SEVERITIES.index(w.severity))
    rank = {w.title: i for i, w in enumerate(weaknesses)}
    fixes.sort(key=lambda x: rank.get(x.addresses[0], 99))
    counts = {s: sum(1 for w in weaknesses if w.severity == s) for s in SEVERITIES}
    notes = [f"{f.id}: {q}" for f in facts for q in f.e.get("quality", [])][:12]
    return Analysis(
        summary=(f"Rule-based analysis of {len(facts)} experiment(s): {len(weaknesses)} weakness(es) "
                 + "(" + ", ".join(f"{n} {s}" for s, n in counts.items() if n) + "). "
                 + (f"Most severe: {weaknesses[0].title}." if weaknesses else "")),
        experiments=[_experiment(f) for f in facts], weaknesses=weaknesses, fixes=fixes,
        generated_by="rules", notes=notes)
