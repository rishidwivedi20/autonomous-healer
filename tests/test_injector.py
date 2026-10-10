import json

import pytest

from chaosai.injector import DockerInjector, InjectionError, netem_script


class FakeContainer:
    def __init__(self, svc, managed=True, status="running", exec_ok=True, host=None):
        self.name = self.id = svc
        self.labels = {"com.docker.compose.service": svc}
        if managed:
            self.labels["chaosai.managed"] = "true"
        self.status = status
        self.exec_ok = exec_ok
        self.host = host if host is not None else {}
        self.calls: list = []
        self.update_error = None
        self.attrs = {}
        self.reload()

    def reload(self):
        self.attrs = {"State": {"Status": self.status}, "HostConfig": dict(self.host)}

    def kill(self): self.calls.append("kill"); self.status = "exited"
    def stop(self, timeout=10): self.calls.append("stop"); self.status = "exited"
    def start(self): self.calls.append("start"); self.status = "running"
    def pause(self): self.calls.append("pause"); self.status = "paused"
    def unpause(self): self.calls.append("unpause"); self.status = "running"

    def update(self, **kw):
        self.calls.append(("update", kw))
        if self.update_error:
            err, self.update_error = self.update_error, None      # fail once, so the restore can succeed
            raise RuntimeError(err)
        if "cpu_quota" in kw:
            self.host["CpuQuota"] = 0 if kw["cpu_quota"] == -1 else kw["cpu_quota"]
            self.host["CpuPeriod"] = kw["cpu_period"]
        if "mem_limit" in kw:
            ml = kw["mem_limit"]
            self.host["Memory"] = 0 if ml == -1 else (33554432 if ml == "32m" else ml)

    def exec_run(self, cmd, user=None):
        self.calls.append(("exec", cmd[-1]))
        if not self.exec_ok:
            return 1, b"RTNETLINK answers: Operation not permitted"
        return 0, b""


class FakeClient:
    def __init__(self, *cs):
        self.cs = list(cs)
        self.containers = self
        self.ran = []

    def list(self, all=False, filters=None):
        return [c for c in self.cs if c.labels.get("chaosai.managed") == "true"]

    def run(self, image, cmd, **kw):
        self.ran.append((image, cmd, kw))
        return b""


@pytest.fixture
def world(tmp_path):
    redis, vote, other = FakeContainer("redis"), FakeContainer("vote"), FakeContainer("not-ours", managed=False)
    client = FakeClient(redis, vote, other)
    inj = DockerInjector(client, state_path=tmp_path / "faults.json")
    return inj, client, redis, vote, other, tmp_path / "faults.json"


# ---- targets / allowlist
def test_list_targets_only_labeled(world):
    inj, *_ = world
    assert inj.list_targets() == ["redis", "vote"]


def test_unlabeled_container_cannot_be_targeted(world):
    inj, _, _, _, other, _ = world
    with pytest.raises(InjectionError, match="not a running/known container labeled"):
        inj.inject("not-ours", "kill")
    assert other.calls == []


def test_unknown_fault_rejected(world):
    with pytest.raises(InjectionError, match="unknown fault"):
        world[0].inject("redis", "explode")


# ---- kill / stop / pause + rollback
def test_kill_then_rollback_restarts(world):
    inj, _, redis, *_, state = world
    h = inj.inject("redis", "kill")
    assert redis.status == "exited" and state.exists()
    assert inj.rollback(h) == []
    assert redis.status == "running" and not state.exists()


def test_rollback_after_restart_policy_already_revived_it(world):
    inj, _, redis, *_ = world
    h = inj.inject("redis", "kill")
    redis.status = "running"                    # docker restart policy beat us to it
    assert inj.rollback(h) == []
    assert "start" not in redis.calls


def test_stop_and_pause_roll_back(world):
    inj, _, redis, vote, *_ = world
    h1, h2 = inj.inject("redis", "stop"), inj.inject("vote", "pause")
    assert redis.status == "exited" and vote.status == "paused"
    assert inj.rollback(h1) == [] and inj.rollback(h2) == []
    assert redis.status == "running" and vote.status == "running"


# ---- cpu / memory
def test_cpu_limit_applied_and_removed(world):
    inj, _, redis, *_ = world
    h = inj.inject("redis", "cpu-limit", {"cpus": 0.25})
    assert redis.host["CpuQuota"] == 25000
    assert inj.rollback(h) == []
    assert redis.host["CpuQuota"] == 0


def test_cpu_limit_restores_preexisting_limit(tmp_path):
    c = FakeContainer("redis", host={"CpuQuota": 50000, "CpuPeriod": 100000})
    inj = DockerInjector(FakeClient(c), state_path=tmp_path / "f.json")
    h = inj.inject("redis", "cpu-limit", {"cpus": 0.1})
    assert inj.rollback(h) == []
    assert c.host["CpuQuota"] == 50000


def test_memory_limit_applied_and_removed(world):
    inj, _, redis, *_ = world
    h = inj.inject("redis", "memory-limit")
    assert redis.host["Memory"] == 33554432
    assert inj.rollback(h) == []
    assert redis.host["Memory"] == 0


def test_limit_not_actually_removed_is_reported_not_assumed(world):
    inj, _, redis, *_, state = world
    h = inj.inject("redis", "cpu-limit")
    redis.update = lambda **kw: None            # docker silently ignores the restore
    errors = inj.rollback(h)
    assert errors and "not removed" in errors[0]
    assert state.exists()                       # kept so `chaos abort` can retry


# ---- network
def test_netem_via_exec_when_possible(world):
    inj, client, redis, *_ = world
    h = inj.inject("redis", "network-delay", {"latency_ms": 800, "jitter_ms": 50})
    assert any(c[0] == "exec" and "delay 800ms 50ms" in c[1] for c in redis.calls if isinstance(c, tuple))
    assert client.ran == []
    assert inj.rollback(h) == []
    assert any(c[0] == "exec" and "qdisc del" in c[1] for c in redis.calls if isinstance(c, tuple))


def test_netem_falls_back_to_sidecar_sharing_netns(tmp_path):
    c = FakeContainer("redis", exec_ok=False)
    client = FakeClient(c)
    inj = DockerInjector(client, state_path=tmp_path / "f.json")
    h = inj.inject("redis", "network-loss", {"loss_pct": 40})
    image, cmd, kw = client.ran[0]
    assert "loss 40.0%" in cmd[-1]
    assert kw["network_mode"] == "container:redis" and kw["cap_add"] == ["NET_ADMIN"]
    assert "chaosai.managed" not in kw["labels"]            # sidecar must never be an injection target
    assert inj.rollback(h) == []
    assert "qdisc del" in client.ran[1][1][-1]               # removal also via sidecar


def test_netem_script_shape():
    assert 'netem delay 100ms' in netem_script("delay 100ms")
    assert "qdisc del" in netem_script(None)
    assert '"$i" = lo' in netem_script("delay 1ms")          # never touch loopback


# ---- failure handling
def test_partial_failure_is_undone_before_raising(world):
    inj, _, redis, *_, state = world
    redis.update_error = "docker said no"
    with pytest.raises(InjectionError, match="docker said no"):
        inj.inject("redis", "cpu-limit")
    assert not state.exists()                    # undone cleanly, nothing left to restore later


def test_rollback_continues_past_failures_and_never_raises(world):
    inj, _, redis, *_ = world
    h = inj.inject("redis", "kill")
    h.undo.insert(0, lambda: (_ for _ in ()).throw(RuntimeError("boom")))   # runs last
    h.undo.append(lambda: (_ for _ in ()).throw(RuntimeError("bang")))      # runs first
    errors = inj.rollback(h)
    assert len(errors) == 2 and redis.status == "running"


# ---- write-ahead state file + kill-switch restore
def test_state_written_before_fault_applied(world):
    inj, _, redis, *_, state = world
    seen = {}
    orig_kill = redis.kill
    redis.kill = lambda: (seen.update(json.loads(state.read_text())), orig_kill())
    inj.inject("redis", "kill")
    assert any(v["target"] == "redis" for v in seen.values())


def test_restore_all_from_state_after_crash(world, tmp_path):
    inj, client, redis, vote, _, state = world
    inj.inject("redis", "cpu-limit")
    inj.inject("vote", "pause")
    inj2 = DockerInjector(client, state_path=state)           # a fresh process: no handles, only the file
    assert inj2.restore_all() == []
    assert redis.host["CpuQuota"] == 0 and vote.status == "running"
    assert not state.exists()


def test_restore_all_restarts_stopped_containers_even_without_state(world):
    inj, _, redis, *_ = world
    redis.status = "exited"
    assert inj.restore_all() == []
    assert redis.status == "running"


# ---- traffic spike
def test_traffic_spike_needs_probe(world):
    with pytest.raises(InjectionError, match="LoadProbe"):
        world[0].inject("vote", "traffic-spike")


def test_traffic_spike_sets_and_restores_multiplier(tmp_path):
    class P: multiplier = 1.0
    p = P()
    inj = DockerInjector(FakeClient(FakeContainer("vote")), state_path=tmp_path / "f.json", probe=p)
    h = inj.inject("vote", "traffic-spike", {"multiplier": 8})
    assert p.multiplier == 8.0
    assert inj.rollback(h) == [] and p.multiplier == 1.0


# ---- dry run
def test_dry_run_touches_nothing(tmp_path):
    c = FakeContainer("redis")
    inj = DockerInjector(FakeClient(c), state_path=tmp_path / "f.json", dry_run=True)
    h = inj.inject("redis", "kill")
    assert c.calls == [] and h.dry_run and "dry-run" in h.actions[0]
    assert not (tmp_path / "f.json").exists()
    assert inj.rollback(h) == []
