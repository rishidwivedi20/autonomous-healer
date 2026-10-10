"""Observation layer: everything measured lands in one SQLite file.

  LoadProbe      once a second: POST a vote and GET the result page (async httpx)
  DockerWatcher  container state / health / restart count transitions, plus two pipeline
                 gauges read from inside the containers (redis queue length, rows in postgres)
  query helpers  window(), per_second(), events(), gauge_series(): the only way later code
                 (orchestrator, analyzer tools) reads data back

Known blind spots, reported rather than hidden (see missing_signals()):
  * The result page is static HTML; data arrives over a websocket. A 200 from it does not
    prove the database is up. The pipeline gauges are what reveal worker/db trouble.
  * Latency percentiles are computed over successful requests only. Failures are counted in
    error_rate; a fault that makes every request fail leaves p50/p95 as None, not 0.
"""
from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .models import ContainerEvent, SecondStat, Stats

SCHEMA = """
CREATE TABLE IF NOT EXISTS probes (
  t REAL NOT NULL, kind TEXT NOT NULL, ok INTEGER NOT NULL,
  status INTEGER, latency_ms REAL NOT NULL, error TEXT);
CREATE INDEX IF NOT EXISTS probes_t ON probes(t);
CREATE TABLE IF NOT EXISTS events (
  t REAL NOT NULL, service TEXT NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_t ON events(t);
CREATE TABLE IF NOT EXISTS gauges (
  t REAL NOT NULL, name TEXT NOT NULL, value REAL);   -- NULL value = could not be read
CREATE INDEX IF NOT EXISTS gauges_t ON gauges(t);
"""


class Store:
    """Thread-safe SQLite wrapper (probe thread, watcher thread and main thread all write/read)."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            if self.path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)

    def execute(self, sql: str, args: tuple = ()) -> None:
        with self._lock:
            self._db.execute(sql, args)
            self._db.commit()

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        with self._lock:
            self._db.executemany(sql, rows)
            self._db.commit()

    def query(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def close(self) -> None:
        with self._lock:
            self._db.close()


# ---------------------------------------------------------------- stats
def percentile(sorted_values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; None for no data."""
    if not sorted_values:
        return None
    rank = max(1, -(-int(q * 100) * len(sorted_values) // 100))  # ceil(q*n)
    return sorted_values[min(rank, len(sorted_values)) - 1]


def stats_of(rows: list[tuple[int, float]]) -> Stats:
    """rows = [(ok, latency_ms)]"""
    if not rows:
        return Stats()
    ok_lat = sorted(lat for ok, lat in rows if ok)
    return Stats(
        n=len(rows),
        error_rate=sum(1 for ok, _ in rows if not ok) / len(rows),
        p50_ms=percentile(ok_lat, 0.50),
        p95_ms=percentile(ok_lat, 0.95),
    )


# ---------------------------------------------------------------- query helpers
def window(store: Store, start: float, end: float, kind: str | None = None) -> Stats:
    sql = "SELECT ok, latency_ms FROM probes WHERE t >= ? AND t < ?"
    args: tuple = (start, end)
    if kind:
        sql += " AND kind = ?"
        args += (kind,)
    return stats_of(store.query(sql, args))


def by_journey(store: Store, start: float, end: float) -> dict[str, Stats]:
    kinds = [k for (k,) in store.query("SELECT DISTINCT kind FROM probes WHERE t >= ? AND t < ? ORDER BY kind", (start, end))]
    return {k: window(store, start, end, k) for k in kinds}


def worst_window(store: Store, start: float, end: float) -> Stats:
    """Stats of the WORST user journey, not the average: if every vote fails while the result page
    is fine, the combined error rate is only 50%, but the vote journey is 100% down. Safety checks
    (abort, preflight) and recovery use this so a total outage of one journey is never diluted."""
    per = [s for s in by_journey(store, start, end).values() if s.n]
    if not per:
        return Stats()
    p50 = [s.p50_ms for s in per if s.p50_ms is not None]
    p95 = [s.p95_ms for s in per if s.p95_ms is not None]
    return Stats(n=sum(s.n for s in per), error_rate=max(s.error_rate for s in per),
                 p50_ms=max(p50) if p50 else None, p95_ms=max(p95) if p95 else None)


def per_second(store: Store, start: float, end: float, kind: str | None = None) -> list[SecondStat]:
    """One entry per whole second from `start`; t is seconds since start. Empty seconds have n=0."""
    sql = "SELECT t, ok, latency_ms FROM probes WHERE t >= ? AND t < ?"
    args: tuple = (start, end)
    if kind:
        sql += " AND kind = ?"
        args += (kind,)
    buckets: dict[int, list[tuple[int, float]]] = {}
    for t, ok, lat in store.query(sql, args):
        buckets.setdefault(int(t - start), []).append((ok, lat))
    out = []
    for sec in range(int(end - start)):
        s = stats_of(buckets.get(sec, []))
        out.append(SecondStat(t=float(sec), n=s.n, error_rate=s.error_rate, p50_ms=s.p50_ms, p95_ms=s.p95_ms))
    return out


def events(store: Store, start: float, end: float, service: str | None = None) -> list[ContainerEvent]:
    sql = "SELECT t, service, kind, value FROM events WHERE t >= ? AND t <= ?"
    args: tuple = (start, end)
    if service:
        sql += " AND service = ?"
        args += (service,)
    return [ContainerEvent(t=t, service=s, kind=k, value=v) for t, s, k, v in store.query(sql + " ORDER BY t", args)]


def gauge_series(store: Store, name: str, start: float, end: float) -> list[tuple[float, float | None]]:
    return store.query("SELECT t, value FROM gauges WHERE name = ? AND t >= ? AND t <= ? ORDER BY t", (name, start, end))


def restart_counts(store: Store, start: float, end: float) -> dict[str, int]:
    """Automatic restarts observed in [start, end].

    Docker resets RestartCount to 0 when a container is started again after a manual stop (seen live:
    db 20 -> 0), so this sums only the INCREASES between consecutive readings; a decrease is a reset and
    counting resumes from the new value. (Taking last minus first went negative for `stop` faults.)"""
    out: dict[str, int] = {}
    for (svc,) in store.query("SELECT DISTINCT service FROM events WHERE kind='restart_count'"):
        before = store.query(
            "SELECT value FROM events WHERE service=? AND kind='restart_count' AND t < ? ORDER BY t DESC LIMIT 1",
            (svc, start))
        inside = store.query(
            "SELECT value FROM events WHERE service=? AND kind='restart_count' AND t >= ? AND t <= ? ORDER BY t",
            (svc, start, end))
        prev = int(before[0][0]) if before else None
        total = 0
        for (v,) in inside:
            v = int(v)
            if prev is not None and v > prev:
                total += v - prev
            prev = v                                   # a drop is a reset: continue from the new value
        if total:
            out[svc] = total
    return out


def unhealthy(snapshot: dict[str, tuple[str, str | None]]) -> list[str]:
    """snapshot: service -> (state, health). Healthy = running and (no healthcheck or 'healthy')."""
    bad = []
    for svc, (state, health) in sorted(snapshot.items()):
        if state != "running" or health not in (None, "healthy"):
            bad.append(svc)
    return bad


def missing_signals(store: Store, start: float, end: float, expected_gauges: tuple[str, ...] = ()) -> list[str]:
    """What could not be measured in this window, for the report to state plainly."""
    missing = []
    if window(store, start, end).n == 0:
        missing.append("no probe samples in window")
    for name in expected_gauges:
        series = gauge_series(store, name, start, end)
        if not series:
            missing.append(f"gauge {name}: no readings")
        elif all(v is None for _, v in series):
            missing.append(f"gauge {name}: every read failed")
    # Events are stored on change only, so a quiet window has none. Gauge rows are written every
    # couple of seconds (NULL when unreadable), so they are the watcher's heartbeat.
    if not events(store, start, end) and not store.query(
            "SELECT 1 FROM gauges WHERE t >= ? AND t <= ? LIMIT 1", (start, end)):
        missing.append("watcher produced no events or gauge readings (not running?)")
    return missing


# ---------------------------------------------------------------- load probe
@dataclass
class ProbeConfig:
    vote_url: str = "http://127.0.0.1:8080/"
    result_url: str = "http://127.0.0.1:8081/"
    interval_s: float = 1.0
    timeout_s: float = 3.0
    choices: tuple[str, str] = ("a", "b")
    max_connections: int = 200


POOL_RESET_AFTER = 20      # consecutive PoolTimeouts (the probe could not even send) before it builds a fresh client


class LoadProbe:
    """Background thread. Each tick sends `round(multiplier)` rounds of (vote POST, result GET).
    `multiplier` > 1 is the traffic-spike fault."""

    def __init__(self, store: Store, cfg: ProbeConfig | None = None):
        self.store = store
        self.cfg = cfg or ProbeConfig()
        self.multiplier = 1.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._n = 0
        self._pool_stalls = 0       # consecutive PoolTimeouts
        self.client_resets = 0

    def start(self) -> "LoadProbe":
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True, name="probe")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.cfg.timeout_s + 5)

    async def _one(self, http: httpx.AsyncClient, kind: str, method: str, url: str, data: dict | None) -> None:
        t0 = time.time()
        start = time.perf_counter()
        status, error = None, None
        if kind == "vote":
            http.cookies.clear()    # every vote is a new voter; one persistent voter_id would just update one row
        try:
            r = await http.request(method, url, data=data)
            status = r.status_code
            ok = 200 <= status < 400
            if not ok:
                error = f"HTTP {status}"
        except httpx.HTTPError as e:
            ok, error = False, type(e).__name__
        self._pool_stalls = self._pool_stalls + 1 if error == "PoolTimeout" else 0
        latency = (time.perf_counter() - start) * 1000
        self.store.execute("INSERT INTO probes VALUES (?,?,?,?,?,?)", (t0, kind, int(ok), status, latency, error))

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        pending: set[asyncio.Task] = set()
        http = self._client()
        try:
            next_tick = loop.time()
            while not self._stop.is_set():
                if self._pool_stalls >= POOL_RESET_AFTER:
                    # every connection in the pool is stuck, so nothing can be sent any more (seen after a machine stall):
                    # start a fresh client and close the old one once its in-flight requests have timed out
                    old, http = http, self._client()
                    self._pool_stalls, self.client_resets = 0, self.client_resets + 1
                    retire = asyncio.create_task(self._retire(old))
                    pending.add(retire)
                    retire.add_done_callback(pending.discard)
                for _ in range(max(1, round(self.multiplier))):
                    choice = self.cfg.choices[self._n % 2]
                    self._n += 1
                    for coro in (
                        self._one(http, "vote", "POST", self.cfg.vote_url, {"vote": choice}),
                        self._one(http, "result", "GET", self.cfg.result_url, None),
                    ):
                        task = asyncio.create_task(coro)
                        pending.add(task)
                        task.add_done_callback(pending.discard)
                next_tick += self.cfg.interval_s
                await asyncio.sleep(max(0.0, next_tick - loop.time()))
            if pending:
                await asyncio.wait(pending, timeout=self.cfg.timeout_s + 1)
        finally:
            try:
                await http.aclose()
            except Exception:
                pass

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.cfg.timeout_s, limits=httpx.Limits(max_connections=self.cfg.max_connections))

    async def _retire(self, client: httpx.AsyncClient) -> None:
        await asyncio.sleep(self.cfg.timeout_s + 0.5)
        try:
            await client.aclose()
        except Exception:
            pass


# ---------------------------------------------------------------- docker watcher
GAUGE_COMMANDS = {
    # name: (service, command). Real reads from inside the containers; failure is stored as NULL.
    "redis_queue_len": ("redis", ["redis-cli", "llen", "votes"]),
    "db_votes": ("db", ["psql", "-U", "postgres", "-tAc", "SELECT count(*) FROM votes"]),
}


WORST_STATE_FIRST = ("dead", "exited", "restarting", "removing", "paused", "created")


def combine_state(states: list[str]) -> str:
    """One state for a service with several replicas: 'running' only if every replica is; otherwise the worst one."""
    for s in WORST_STATE_FIRST:
        if s in states:
            return s
    others = [s for s in states if s != "running"]
    return others[0] if others else "running"


def combine_health(healths: list[str | None]) -> str | None:
    """One health for a service with several replicas: None if none has a healthcheck, else unhealthy > starting > healthy."""
    known = [h for h in healths if h]
    if not known:
        return None
    for h in ("unhealthy", "starting"):
        if h in known:
            return h
    return "healthy" if all(h == "healthy" for h in known) else known[0]


class DockerWatcher:
    """Polls managed containers; stores an event only when something changes (plus a first snapshot)."""

    def __init__(self, store: Store, client=None, interval_s: float = 1.0, gauge_every_s: float = 2.0,
                 label: str = "chaosai.managed=true"):
        if client is None:
            import docker
            client = docker.from_env()
        self.store, self.client, self.label = store, client, label
        self.interval_s, self.gauge_every_s = interval_s, gauge_every_s
        self._last: dict[tuple[str, str], str] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _services(self) -> dict[str, dict]:
        """service -> {state, health, restarts}, combined over ALL replicas of that service. Keying by service alone would let replicas
        overwrite each other, and replicas with different restart counts would look like a restart on every poll."""
        agg: dict[str, dict] = {}
        for c in self.client.containers.list(all=True, filters={"label": self.label}):
            c.reload()
            st = c.attrs.get("State", {})
            e = agg.setdefault(c.labels.get("com.docker.compose.service", c.name), {"states": [], "healths": [], "restarts": 0})
            e["states"].append(st.get("Status", "unknown"))
            e["healths"].append((st.get("Health") or {}).get("Status"))
            e["restarts"] += int(c.attrs.get("RestartCount", 0))
        return {svc: {"state": combine_state(e["states"]), "health": combine_health(e["healths"]), "restarts": e["restarts"]}
                for svc, e in agg.items()}

    def snapshot(self) -> dict[str, tuple[str, str | None]]:
        """service -> (state, health). health None = no replica has a healthcheck. With several replicas the worst one decides."""
        return {svc: (v["state"], v["health"]) for svc, v in self._services().items()}

    def table(self) -> dict[str, dict]:
        """service -> {state, health, restarts} for the dashboard's container table (restarts summed over replicas)."""
        return self._services()

    def poll_once(self) -> None:
        now = time.time()
        for svc, v in self._services().items():
            current = {"state": v["state"], "restart_count": str(v["restarts"])}
            if v["health"] is not None:
                current["health"] = v["health"]
            for kind, value in current.items():
                if self._last.get((svc, kind)) != value:
                    self._last[(svc, kind)] = value
                    self.store.execute("INSERT INTO events VALUES (?,?,?,?)", (now, svc, kind, value))

    def read_gauges(self) -> None:
        now = time.time()
        for name, (svc, cmd) in GAUGE_COMMANDS.items():
            value = None
            try:
                c = self.client.containers.get(self._container_of(svc))
                code, out = c.exec_run(cmd)
                if code == 0:
                    value = float(out.decode().strip())
            except Exception:
                value = None
            self.store.execute("INSERT INTO gauges VALUES (?,?,?)", (now, name, value))

    def _container_of(self, svc: str) -> str:
        for c in self.client.containers.list(all=True, filters={"label": self.label}):
            if c.labels.get("com.docker.compose.service") == svc:
                return c.id
        raise LookupError(svc)

    def start(self) -> "DockerWatcher":
        def loop():
            last_gauge = 0.0
            while not self._stop.is_set():
                try:
                    self.poll_once()
                    if time.time() - last_gauge >= self.gauge_every_s:
                        last_gauge = time.time()
                        self.read_gauges()
                except Exception:
                    pass  # Docker daemon hiccup; the gap shows up as missing events, not a crash
                self._stop.wait(self.interval_s)
        self._thread = threading.Thread(target=loop, daemon=True, name="docker-watcher")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
