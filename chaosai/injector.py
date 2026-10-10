"""Fault injection behind a small interface. DockerInjector is the only implementation;
a KubernetesInjector can implement the same Protocol later.

Safety properties, all independent of the Guard in safety.py (defense in depth):
  * Only containers labeled chaosai.managed=true can be resolved as targets.
  * Undo information is written to a state file BEFORE a fault is applied, so rollback
    works after a crash and from another process (`chaos abort` -> restore_all()).
  * rollback()/restore_all() never raise: they continue past failures and return the errors.
  * A fault that fails halfway through is undone before inject() raises.

Untested against real Docker as of writing (not installed on the dev machine): cpu/memory
limit removal and the tc sidecar. Limit restores are therefore VERIFIED by re-inspecting the
container, and any mismatch is returned as a rollback error instead of being assumed fine.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

LABEL = "chaosai.managed=true"
SERVICE_LABEL = "com.docker.compose.service"
NETTOOLS_IMAGE = "ghcr.io/alexei-led/pumba-alpine-nettools:latest"   # small image with tc; used as netns sidecar

FAULTS = {
    "kill": "SIGKILL the container (restart policy should bring it back)",
    "stop": "graceful stop; stays down until rollback (restart policy does not revive manual stops)",
    "pause": "freeze all processes; container looks up but does not respond",
    "cpu-limit": "throttle CPU (params: cpus, default 0.1)",
    "memory-limit": "cap memory (params: mem, default '32m')",
    "network-delay": "tc netem delay (params: latency_ms=500, jitter_ms=0)",
    "network-loss": "tc netem packet loss (params: loss_pct=30)",
    "traffic-spike": "multiply load-probe traffic (params: multiplier=5); needs a LoadProbe",
}


class InjectionError(Exception):
    pass


class Injector(Protocol):
    def list_targets(self) -> list[str]: ...
    def inject(self, target: str, fault: str, params: dict | None = None) -> "Handle": ...
    def rollback(self, handle: "Handle") -> list[str]: ...


@dataclass
class Handle:
    id: str
    target: str
    fault: str
    params: dict
    container_id: str = ""
    started_at: float = 0.0
    actions: list[str] = field(default_factory=list)          # human-readable log (also used by --dry-run)
    undo: list[Callable[[], None]] = field(default_factory=list)
    dry_run: bool = False


def netem_script(args: str | None) -> str:
    """Shell snippet applying (args) or removing (None) netem on every non-loopback interface."""
    if args is None:
        body = 'tc qdisc del dev "$i" root 2>/dev/null || true'
    else:
        body = f'tc qdisc replace dev "$i" root netem {args}'
    return f'for i in $(ls /sys/class/net); do [ "$i" = lo ] || {body}; done'


class DockerInjector:
    def __init__(self, client=None, state_path: Path | str = "results/active_faults.json",
                 probe=None, dry_run: bool = False, nettools_image: str = NETTOOLS_IMAGE):
        if client is None and not dry_run:
            import docker
            client = docker.from_env()
        self.client = client
        self.state_path = Path(state_path)
        self.probe = probe
        self.dry_run = dry_run
        self.nettools_image = nettools_image

    # ------------------------------------------------------------ targets
    def _managed(self) -> dict[str, object]:
        """service name -> container, labeled containers only."""
        found = self.client.containers.list(all=True, filters={"label": LABEL})
        return {c.labels.get(SERVICE_LABEL, c.name): c for c in found}

    def list_targets(self) -> list[str]:
        return sorted(self._managed())

    # ------------------------------------------------------------ inject
    def inject(self, target: str, fault: str, params: dict | None = None) -> Handle:
        params = dict(params or {})
        if fault not in FAULTS:
            raise InjectionError(f"unknown fault {fault!r}; known: {', '.join(FAULTS)}")
        h = Handle(id=uuid.uuid4().hex[:8], target=target, fault=fault, params=params,
                   started_at=time.time(), dry_run=self.dry_run)

        if self.dry_run:
            h.actions.append(self._describe(fault, target, params))
            return h

        if fault == "traffic-spike":
            return self._traffic_spike(h)

        managed = self._managed()
        if target not in managed:
            raise InjectionError(f"{target!r} is not a running/known container labeled {LABEL}")
        c = managed[target]
        h.container_id = c.id
        c.reload()
        try:
            getattr(self, "_" + fault.replace("-", "_"))(c, h)
        except Exception as e:
            errors = self.rollback(h)
            raise InjectionError(f"{fault} on {target} failed: {e}" + (f" (rollback issues: {errors})" if errors else "")) from e
        return h

    def _describe(self, fault: str, target: str, params: dict) -> str:
        return f"[dry-run] would apply {fault} to {target} with {params or 'default params'}: {FAULTS[fault]}"

    # ------------------------------------------------------------ individual faults
    # Each writes its undo to the state file first, registers an undo callable, then acts.
    def _kill(self, c, h: Handle) -> None:
        self._record(h, {"restore": "ensure_running"})
        h.undo.append(lambda: self._ensure_running(c))
        c.kill()
        h.actions.append(f"killed {h.target}")

    def _stop(self, c, h: Handle) -> None:
        self._record(h, {"restore": "ensure_running"})
        h.undo.append(lambda: self._ensure_running(c))
        c.stop(timeout=int(h.params.get("timeout_s", 5)))
        h.actions.append(f"stopped {h.target}")

    def _pause(self, c, h: Handle) -> None:
        self._record(h, {"restore": "unpause"})
        h.undo.append(lambda: self._unpause(c))
        c.pause()
        h.actions.append(f"paused {h.target}")

    def _cpu_limit(self, c, h: Handle) -> None:
        cpus = float(h.params.get("cpus", 0.1))
        host = c.attrs.get("HostConfig", {})
        orig = {"cpu_period": host.get("CpuPeriod") or 0, "cpu_quota": host.get("CpuQuota") or 0}
        self._record(h, {"restore": "cpu", **orig})
        h.undo.append(lambda: self._restore_cpu(c, orig))
        period = 100_000
        c.update(cpu_period=period, cpu_quota=max(1000, int(cpus * period)))
        h.actions.append(f"cpu limit {cpus} on {h.target}")

    def _memory_limit(self, c, h: Handle) -> None:
        mem = h.params.get("mem", "32m")
        host = c.attrs.get("HostConfig", {})
        orig = {"memory": host.get("Memory") or 0, "memswap": host.get("MemorySwap") or 0}
        self._record(h, {"restore": "memory", **orig})
        h.undo.append(lambda: self._restore_memory(c, orig))
        c.update(mem_limit=mem, memswap_limit=mem)
        h.actions.append(f"memory limit {mem} on {h.target}")

    def _network_delay(self, c, h: Handle) -> None:
        latency = int(h.params.get("latency_ms", 500))
        jitter = int(h.params.get("jitter_ms", 0))
        args = f"delay {latency}ms" + (f" {jitter}ms" if jitter else "")
        self._netem(c, h, args)
        h.actions.append(f"network delay {latency}ms (+/-{jitter}ms) on {h.target}")

    def _network_loss(self, c, h: Handle) -> None:
        loss = float(h.params.get("loss_pct", 30))
        self._netem(c, h, f"loss {loss}%")
        h.actions.append(f"network loss {loss}% on {h.target}")

    def _traffic_spike(self, h: Handle) -> Handle:
        if self.probe is None:
            raise InjectionError("traffic-spike needs a LoadProbe (pass probe= to DockerInjector)")
        mult = float(h.params.get("multiplier", 5))
        before = self.probe.multiplier
        self._record(h, {"restore": "none"})        # in-process only; nothing to restore after a crash
        h.undo.append(lambda: setattr(self.probe, "multiplier", before))
        self.probe.multiplier = mult
        h.actions.append(f"traffic x{mult}")
        return h

    # ------------------------------------------------------------ netem: exec first, sidecar fallback
    def _netem(self, c, h: Handle, args: str) -> None:
        method = self._run_tc(c, netem_script(args), prefer="exec")
        self._record(h, {"restore": "netem", "method": method})
        h.undo.append(lambda: self._clear_netem(c, method))
        h.actions.append(f"tc netem via {method}")

    def _run_tc(self, c, script: str, prefer: str) -> str:
        """Returns the method that worked: 'exec' (inside the target) or 'sidecar' (shared netns)."""
        if prefer == "exec":
            try:
                code, out = c.exec_run(["sh", "-c", script], user="root")
                if code == 0:
                    return "exec"
            except Exception:
                pass    # no shell / no tc / no NET_ADMIN: fall through to the sidecar
        self.client.containers.run(
            self.nettools_image, ["sh", "-c", script], network_mode=f"container:{c.id}",
            cap_add=["NET_ADMIN"], remove=True, detach=False, labels={"chaosai.sidecar": "true"})
        return "sidecar"

    def _clear_netem(self, c, method: str) -> None:
        self._run_tc(c, netem_script(None), prefer="exec" if method == "exec" else "sidecar")

    # ------------------------------------------------------------ restore primitives (idempotent)
    @staticmethod
    def _ensure_running(c) -> None:
        c.reload()
        status = c.attrs.get("State", {}).get("Status")
        if status == "paused":
            c.unpause()
        elif status != "running":
            c.start()

    @staticmethod
    def _unpause(c) -> None:
        c.reload()
        if c.attrs.get("State", {}).get("Status") == "paused":
            c.unpause()

    @staticmethod
    def _restore_cpu(c, orig: dict) -> None:
        c.update(cpu_period=orig["cpu_period"] or 100_000, cpu_quota=orig["cpu_quota"] or -1)
        c.reload()
        quota = c.attrs.get("HostConfig", {}).get("CpuQuota") or 0
        if (orig["cpu_quota"] or 0) <= 0 and quota > 0:
            raise RuntimeError(f"cpu limit not removed (CpuQuota={quota}); recreate the container")

    def _restore_memory(self, c, orig: dict) -> None:
        mem = orig["memory"] if orig["memory"] > 0 else -1
        swap = orig["memswap"] if orig["memswap"] != 0 else -1
        c.update(mem_limit=mem, memswap_limit=swap)
        c.reload()
        now = c.attrs.get("HostConfig", {}).get("Memory") or 0
        if orig["memory"] <= 0 and now > 0:
            raise RuntimeError(f"memory limit not removed (Memory={now}); recreate the container")

    # ------------------------------------------------------------ rollback
    def rollback(self, handle: Handle) -> list[str]:
        """Undo everything this handle did. Never raises; returns the errors it hit."""
        errors: list[str] = []
        for undo in reversed(handle.undo):
            try:
                undo()
            except Exception as e:
                errors.append(f"{handle.fault}/{handle.target}: {e}")
        handle.undo.clear()
        if not errors:
            self._forget(handle)
        return errors

    def restore_all(self) -> list[str]:
        """Kill-switch path (works from another process): undo every recorded fault, then make sure
        every managed container is running and unpaused."""
        errors: list[str] = []
        if self.dry_run:
            return errors
        managed = self._managed()
        for entry in list(self._load().values()):
            c = managed.get(entry["target"])
            if c is None:
                errors.append(f"{entry['target']}: container gone, cannot restore {entry['fault']}")
                continue
            try:
                c.reload()
                r = entry["restore"]
                if r == "cpu":
                    self._restore_cpu(c, {"cpu_period": entry["cpu_period"], "cpu_quota": entry["cpu_quota"]})
                elif r == "memory":
                    self._restore_memory(c, {"memory": entry["memory"], "memswap": entry["memswap"]})
                elif r == "netem":
                    self._clear_netem(c, entry.get("method", "sidecar"))
                elif r in ("ensure_running", "unpause"):
                    self._ensure_running(c)
            except Exception as e:
                errors.append(f"{entry['target']}/{entry['fault']}: {e}")
        for svc, c in managed.items():       # belt and braces
            try:
                self._ensure_running(c)
            except Exception as e:
                errors.append(f"{svc}: could not ensure running: {e}")
        if self.probe is not None:
            self.probe.multiplier = 1.0
        if not errors:
            self.state_path.unlink(missing_ok=True)
        return errors

    # ------------------------------------------------------------ write-ahead state file
    def _load(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _record(self, h: Handle, restore: dict) -> None:
        state = self._load()
        state[h.id] = {"target": h.target, "fault": h.fault, "started_at": h.started_at, **restore}
        self._save(state)

    def _forget(self, h: Handle) -> None:
        state = self._load()
        if state.pop(h.id, None) is not None:
            if state:
                self._save(state)
            else:
                self.state_path.unlink(missing_ok=True)
