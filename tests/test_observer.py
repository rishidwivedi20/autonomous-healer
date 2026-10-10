import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from chaosai.observer import (by_journey, worst_window, DockerWatcher, LoadProbe, ProbeConfig, Store, events, gauge_series, missing_signals,
                              per_second, percentile, restart_counts, stats_of, unhealthy, window)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


def add(store, t, ok=1, lat=10.0, kind="vote"):
    store.execute("INSERT INTO probes VALUES (?,?,?,?,?,?)", (t, kind, ok, 200 if ok else 500, lat, None))


# ---- stats
def test_percentile_nearest_rank():
    v = sorted(float(i) for i in range(1, 101))
    assert percentile(v, 0.50) == 50.0
    assert percentile(v, 0.95) == 95.0
    assert percentile([], 0.5) is None
    assert percentile([7.0], 0.95) == 7.0


def test_stats_none_when_no_data():
    s = stats_of([])
    assert s.n == 0 and s.error_rate is None and s.p95_ms is None


def test_latency_excludes_failures_but_error_rate_counts_them():
    s = stats_of([(1, 10.0), (1, 20.0), (0, 5000.0), (0, 1.0)])
    assert s.error_rate == 0.5
    assert s.p95_ms == 20.0


def test_all_failures_gives_none_latency():
    s = stats_of([(0, 3000.0)] * 3)
    assert s.error_rate == 1.0 and s.p50_ms is None and s.p95_ms is None


# ---- queries
def test_window_boundaries_and_kind_filter(store):
    add(store, 100.0)
    add(store, 100.5, ok=0)
    add(store, 101.0, kind="result")
    add(store, 102.0)            # end is exclusive
    assert window(store, 100.0, 102.0).n == 3
    assert window(store, 100.0, 102.0, kind="vote").n == 2
    assert window(store, 100.0, 102.0, kind="vote").error_rate == 0.5


def test_per_second_buckets_include_empty_seconds(store):
    add(store, 100.1)
    add(store, 100.9, ok=0)
    add(store, 102.5)
    rows = per_second(store, 100.0, 103.0)
    assert [r.n for r in rows] == [2, 0, 1]
    assert rows[0].error_rate == 0.5 and rows[1].error_rate is None
    assert [r.t for r in rows] == [0.0, 1.0, 2.0]


def test_events_filter(store):
    store.execute("INSERT INTO events VALUES (1,'redis','state','running')")
    store.execute("INSERT INTO events VALUES (2,'redis','state','exited')")
    store.execute("INSERT INTO events VALUES (2,'db','state','running')")
    assert [e.value for e in events(store, 0, 5, "redis")] == ["running", "exited"]
    assert len(events(store, 0, 5)) == 3


def test_restart_counts(store):
    for t, v in [(1, "0"), (10, "1"), (20, "2")]:
        store.execute("INSERT INTO events VALUES (?,?,?,?)", (t, "redis", "restart_count", v))
    store.execute("INSERT INTO events VALUES (1,'db','restart_count','0')")
    assert restart_counts(store, 5, 30) == {"redis": 2}
    assert restart_counts(store, 12, 30) == {"redis": 1}
    assert restart_counts(store, 0, 0.5) == {}


def test_unhealthy():
    snap = {"vote": ("running", "healthy"), "worker": ("running", None), "db": ("running", "starting"),
            "redis": ("exited", "healthy"), "result": ("paused", None)}
    assert unhealthy(snap) == ["db", "redis", "result"]


def test_missing_signals_reports_gaps(store):
    miss = missing_signals(store, 0, 10, ("db_votes",))
    assert "no probe samples in window" in miss
    assert any("db_votes" in m for m in miss)
    assert any("watcher produced no" in m for m in miss)
    store.execute("INSERT INTO gauges VALUES (1,'db_votes',NULL)")
    assert any("every read failed" in m for m in missing_signals(store, 0, 10, ("db_votes",)))


# ---- load probe against a real local HTTP server
class Handler(BaseHTTPRequestHandler):
    mode = "ok"
    seen: list = []
    cookies_received: list = []

    def log_message(self, *a): pass

    def do_GET(self):
        self._reply()

    def do_POST(self):
        Handler.cookies_received.append(self.headers.get("Cookie"))
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        Handler.seen.append(body)
        self._reply()

    def _reply(self):
        if Handler.mode == "500":
            self.send_response(500)
        else:
            self.send_response(200)
        self.send_header("Set-Cookie", "voter_id=abc123; Path=/")   # like the real vote app
        self.end_headers()
        self.wfile.write(b"x")


@pytest.fixture
def server():
    Handler.mode, Handler.seen, Handler.cookies_received = "ok", [], []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/"
    srv.shutdown()


def run_probe(store, url, seconds, **kw):
    cfg = ProbeConfig(vote_url=url, result_url=url, interval_s=0.2, timeout_s=1.0)
    p = LoadProbe(store, cfg)
    for k, v in kw.items():
        setattr(p, k, v)
    p.start()
    deadline = time.time() + 5.0                      # a loaded machine can take a while to start the first request
    while not store.query("SELECT 1 FROM probes LIMIT 1") and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(seconds)
    p.stop()
    return p


def test_probe_records_successes_and_posts_votes(store, server):
    run_probe(store, server, 1.0)
    s = window(store, 0, time.time() + 1)
    assert s.n >= 6 and s.error_rate == 0.0 and s.p95_ms is not None
    assert {k for (k,) in store.query("SELECT DISTINCT kind FROM probes")} == {"vote", "result"}
    assert any(b in ("vote=a", "vote=b") for b in Handler.seen)


def test_probe_counts_http_500_as_error(store, server):
    Handler.mode = "500"
    run_probe(store, server, 0.8)
    s = window(store, 0, time.time() + 1)
    assert s.n > 0 and s.error_rate == 1.0


def test_probe_counts_connection_refused_as_error(store):
    run_probe(store, "http://127.0.0.1:1/", 0.8)
    s = window(store, 0, time.time() + 1)
    assert s.n > 0 and s.error_rate == 1.0
    assert store.query("SELECT DISTINCT error FROM probes")[0][0] is not None


def test_multiplier_sends_more_traffic(store, server):
    run_probe(store, server, 1.0)
    base = window(store, 0, time.time() + 1).n
    store2 = Store(":memory:")
    run_probe(store2, server, 1.0, multiplier=5)
    assert window(store2, 0, time.time() + 1).n > base * 3


# ---- docker watcher with a fake client (the real one needs Docker)
class FakeContainer:
    def __init__(self, svc, status="running", health="healthy", restarts=0, gauge=("0\n", 0)):
        self.labels = {"com.docker.compose.service": svc}
        self.name, self.id = svc, svc
        self.attrs = {}
        self.status, self.health, self.restarts, self.gauge = status, health, restarts, gauge
        self.reload()

    def reload(self):
        st = {"Status": self.status}
        if self.health:
            st["Health"] = {"Status": self.health}
        self.attrs = {"State": st, "RestartCount": self.restarts}

    def exec_run(self, cmd):
        code, out = self.gauge[1], self.gauge[0]
        return code, out.encode()


class FakeClient:
    def __init__(self, *cs):
        self.cs = {c.id: c for c in cs}
        self.containers = self

    def list(self, all=False, filters=None):
        return list(self.cs.values())

    def get(self, cid):
        return self.cs[cid]


def test_watcher_records_changes_only(store):
    redis = FakeContainer("redis")
    w = DockerWatcher(store, client=FakeClient(redis, FakeContainer("worker", health=None)))
    w.poll_once()
    first = len(events(store, 0, time.time() + 1))
    assert first == 5           # redis: state/restart/health; worker: state/restart
    w.poll_once()
    assert len(events(store, 0, time.time() + 1)) == first      # nothing changed -> nothing stored
    redis.status, redis.health, redis.restarts = "exited", "unhealthy", 1
    w.poll_once()
    vals = [(e.kind, e.value) for e in events(store, 0, time.time() + 1, "redis")]
    assert ("state", "exited") in vals and ("restart_count", "1") in vals and ("health", "unhealthy") in vals


def test_watcher_snapshot_feeds_unhealthy(store):
    w = DockerWatcher(store, client=FakeClient(FakeContainer("db", health="starting"),
                                               FakeContainer("worker", health=None)))
    assert unhealthy(w.snapshot()) == ["db"]


def test_gauges_store_value_and_null_on_failure(store):
    redis = FakeContainer("redis", gauge=("7\n", 0))
    db = FakeContainer("db", gauge=("", 1))       # psql failed
    DockerWatcher(store, client=FakeClient(redis, db)).read_gauges()
    assert gauge_series(store, "redis_queue_len", 0, time.time() + 1)[0][1] == 7.0
    assert gauge_series(store, "db_votes", 0, time.time() + 1)[0][1] is None


def test_every_vote_is_a_new_voter(store, server):
    """The real vote app keys rows by the voter_id cookie; reusing one would update a single row and
    the db_votes gauge could never show progress."""
    run_probe(store, server, 2.0)
    assert len(Handler.cookies_received) >= 3
    assert all(c is None for c in Handler.cookies_received)


def test_quiet_window_with_live_watcher_is_not_reported_missing(store):
    """Events are change-only; a healthy quiet window has none. Gauge rows prove the watcher ran."""
    add(store, 5.0)
    store.execute("INSERT INTO gauges VALUES (5,'redis_queue_len',0)")
    store.execute("INSERT INTO gauges VALUES (5,'db_votes',10)")
    assert missing_signals(store, 0, 10, ("redis_queue_len", "db_votes")) == []


def test_worst_window_is_not_diluted_by_a_healthy_journey(store):
    for t in (1.0, 2.0, 3.0):
        add(store, t, ok=0, kind="vote")
        add(store, t, ok=1, kind="result", lat=5.0)
    assert window(store, 0, 10).error_rate == 0.5              # the average hides it
    w = worst_window(store, 0, 10)
    assert w.error_rate == 1.0 and w.p95_ms == 5.0 and w.n == 6
    assert set(by_journey(store, 0, 10)) == {"vote", "result"}


def test_worst_window_empty(store):
    assert worst_window(store, 0, 10).n == 0


# ---- restart counting must survive Docker resetting RestartCount on a manual start
def test_restart_count_reset_after_a_manual_stop_is_not_negative(store):
    """Real case: db restart_count was 20, the `stop` fault ended with a manual start, Docker reset it to 0."""
    for t, v in [(1, "20"), (40, "0")]:
        store.execute("INSERT INTO events VALUES (?,?,?,?)", (t, "db", "restart_count", v))
    assert restart_counts(store, 10, 60) == {}


def test_restarts_after_a_reset_are_still_counted(store):
    for t, v in [(1, "20"), (40, "0"), (45, "1"), (50, "2")]:      # reset, then two real automatic restarts
        store.execute("INSERT INTO events VALUES (?,?,?,?)", (t, "db", "restart_count", v))
    assert restart_counts(store, 10, 60) == {"db": 2}


def test_increases_before_and_after_a_reset_are_summed(store):
    for t, v in [(1, "5"), (20, "7"), (30, "0"), (35, "3")]:
        store.execute("INSERT INTO events VALUES (?,?,?,?)", (t, "db", "restart_count", v))
    assert restart_counts(store, 10, 60) == {"db": 2 + 3}
