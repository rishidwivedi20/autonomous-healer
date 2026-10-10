import subprocess
import sys
import textwrap

import pytest

from chaosai.models import Experiment
from chaosai.safety import ExperimentLock, Guard, KillSwitch, SafetyViolation


def test_second_holder_is_refused_until_the_first_releases(tmp_path):
    a, b = ExperimentLock(tmp_path / "x.lock"), ExperimentLock(tmp_path / "x.lock")
    assert a.acquire() is True
    assert b.acquire() is False
    a.release()
    assert b.acquire() is True
    b.release()


def test_release_is_idempotent_and_safe_without_acquire(tmp_path):
    lock = ExperimentLock(tmp_path / "x.lock")
    lock.release()
    assert lock.acquire() and (lock.release() or True)
    lock.release()


def test_holder_pid_is_reported_for_a_useful_message(tmp_path):
    a = ExperimentLock(tmp_path / "x.lock")
    a.acquire()
    assert a.holder().startswith("pid ")
    a.release()


def test_guard_refuses_a_second_process_and_releases_on_end(tmp_path):
    kill = KillSwitch(tmp_path / "KILL")
    g1 = Guard({"redis"}, kill, lock=ExperimentLock(tmp_path / "e.lock"))
    g2 = Guard({"redis"}, kill, lock=ExperimentLock(tmp_path / "e.lock"))      # stands in for another process
    g1.begin()
    with pytest.raises(SafetyViolation, match="another process"):
        g2.begin()
    assert g2._running is False                       # a refused begin() leaves no half-state behind
    g1.end()
    g2.begin()
    g2.end()


def test_a_crashed_holder_cannot_leave_the_lock_stuck(tmp_path):
    """The OS drops the lock when the process dies, so there is never a stale lock to clean up."""
    path = tmp_path / "e.lock"
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(tmp_path.parent.parent.parent)!r})
        from chaosai.safety import ExperimentLock
        l = ExperimentLock({str(path)!r})
        assert l.acquire()
        print("held", flush=True)
        os._exit(0)                                    # die without releasing
    """)
    repo = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    code = code.replace(repr(str(tmp_path.parent.parent.parent)), repr(repo))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert "held" in out.stdout, out.stderr
    assert ExperimentLock(path).acquire() is True


def test_a_live_holder_in_another_process_blocks_us(tmp_path):
    path = tmp_path / "e.lock"
    repo = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {repo!r})
        from chaosai.safety import ExperimentLock
        l = ExperimentLock({str(path)!r})
        assert l.acquire()
        print("held", flush=True)
        sys.stdin.readline()
    """)
    p = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "held"
        assert ExperimentLock(path).acquire() is False
        g = Guard({"redis"}, KillSwitch(tmp_path / "KILL"), lock=ExperimentLock(path))
        with pytest.raises(SafetyViolation, match="another process"):
            g.begin()
    finally:
        p.communicate("\n", timeout=15)
    assert ExperimentLock(path).acquire() is True


def test_dashboard_port_cannot_be_shared_by_two_servers(tmp_path):
    from pathlib import Path
    from chaosai import config
    from chaosai.dashboard import App, make_server
    from chaosai.observer import Store

    class Inj:
        def restore_all(self):
            return []

    class Orch:
        injector = Inj()

    app = App(config.load(Path("none.yaml")), Store(tmp_path / "t.db"), Inj(), KillSwitch(tmp_path / "K"), Orch(), lambda: {})
    first = make_server(app, 0)
    try:
        with pytest.raises(OSError):
            make_server(app, first.server_port)
    finally:
        first.server_close()
