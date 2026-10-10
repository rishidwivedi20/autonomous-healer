"""Tunable parameters, loaded from config.yaml (path override: CHAOSAI_CONFIG).

Every default here equals the value the code used before parameters existed, so a missing or
empty config.yaml changes nothing. Two rules keep config from becoming a safety hole:
  * Safety limits can only be TIGHTENED below the hard limits in safety.py; a looser value is an
    error at load time (not silently clamped), and safety.Guard enforces the hard limits anyway.
  * Unknown keys are errors, so a typo (`abort_eror_rate`) cannot silently fall back to a default.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .safety import MAX_DURATION_S, MAX_THRESHOLDS

ROOT = Path(__file__).resolve().parent.parent


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppCfg(_Base):
    vote_url: str = "http://localhost:8080/"
    result_url: str = "http://localhost:8081/"
    compose_file: str = "deploy/docker-compose.yml"


class ProbeCfg(_Base):
    interval_s: float = Field(1.0, gt=0)
    timeout_s: float = Field(3.0, gt=0)


class SafetyCfg(_Base):
    """May only be stricter than the hard limits in safety.py."""
    max_duration_s: int = Field(MAX_DURATION_S, ge=1)
    abort_error_rate: float = Field(MAX_THRESHOLDS.error_rate, gt=0, le=1)
    abort_p95_ms: float = Field(MAX_THRESHOLDS.p95_ms, gt=0)
    breach_seconds: int = Field(MAX_THRESHOLDS.breach_seconds, ge=1)
    preflight_max_error_rate: float = Field(0.05, ge=0, le=1)

    @model_validator(mode="after")
    def _only_tighter(self):
        if self.max_duration_s > MAX_DURATION_S:
            raise ValueError(f"safety.max_duration_s {self.max_duration_s} exceeds the hard limit {MAX_DURATION_S}")
        if self.abort_error_rate > MAX_THRESHOLDS.error_rate:
            raise ValueError(f"safety.abort_error_rate {self.abort_error_rate} is looser than the hard limit {MAX_THRESHOLDS.error_rate}")
        if self.abort_p95_ms > MAX_THRESHOLDS.p95_ms:
            raise ValueError(f"safety.abort_p95_ms {self.abort_p95_ms} is looser than the hard limit {MAX_THRESHOLDS.p95_ms}")
        if self.breach_seconds > MAX_THRESHOLDS.breach_seconds:
            raise ValueError(f"safety.breach_seconds {self.breach_seconds} is looser than the hard limit {MAX_THRESHOLDS.breach_seconds}")
        return self


class ExperimentCfg(_Base):
    baseline_s: int = Field(15, ge=1)
    recovery_timeout_s: int = Field(120, ge=1)
    recovery_hold_s: int = Field(5, ge=1)
    recovery_tolerance: float = Field(0.10, ge=0)
    recovery_latency_slack_ms: float = Field(50.0, ge=0)


class FaultCfg(_Base):
    duration_s: int = Field(30, ge=1)
    params: dict = Field(default_factory=dict)


DEFAULT_FAULTS: dict[str, FaultCfg] = {
    "kill": FaultCfg(duration_s=30),
    "stop": FaultCfg(duration_s=30),
    "pause": FaultCfg(duration_s=20),
    "network-delay": FaultCfg(duration_s=30, params={"latency_ms": 500}),
    "network-loss": FaultCfg(duration_s=30, params={"loss_pct": 30}),
    "cpu-limit": FaultCfg(duration_s=30, params={"cpus": 0.1}),
    "memory-limit": FaultCfg(duration_s=30, params={"mem": "32m"}),
    "traffic-spike": FaultCfg(duration_s=30, params={"multiplier": 5}),
}


class PlannerCfg(_Base):
    top: int = Field(6, ge=1)
    max_per_target: int = Field(2, ge=1)
    max_per_fault: int = Field(3, ge=1)
    use_llm: bool = True
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_pool: int = Field(14, ge=1)


class AgentCfg(_Base):
    max_experiments: int = Field(6, ge=1)
    cooldown_s: float = Field(10.0, ge=0)          # pause between experiments
    health_wait_s: float = Field(60.0, ge=0)       # how long to wait for containers to be healthy before each one
    stop_on_abort: bool = False                    # stop the run after an auto-aborted experiment
    max_consecutive_skips: int = Field(2, ge=1)    # stop if this many experiments in a row are refused


class Config(_Base):
    app: AppCfg = Field(default_factory=AppCfg)
    probe: ProbeCfg = Field(default_factory=ProbeCfg)
    safety: SafetyCfg = Field(default_factory=SafetyCfg)
    experiment: ExperimentCfg = Field(default_factory=ExperimentCfg)
    faults: dict[str, FaultCfg] = Field(default_factory=dict)
    planner: PlannerCfg = Field(default_factory=PlannerCfg)
    agent: AgentCfg = Field(default_factory=AgentCfg)

    @field_validator("faults")
    @classmethod
    def _known_faults_within_limits(cls, v):
        for name, f in v.items():
            if name not in DEFAULT_FAULTS:
                raise ValueError(f"faults.{name}: unknown fault (known: {', '.join(DEFAULT_FAULTS)})")
            if f.duration_s > MAX_DURATION_S:
                raise ValueError(f"faults.{name}.duration_s {f.duration_s} exceeds the hard limit {MAX_DURATION_S}")
        return v

    @model_validator(mode="after")
    def _fill_fault_defaults(self):
        for name, default in DEFAULT_FAULTS.items():
            self.faults.setdefault(name, default.model_copy(deep=True))
        return self


def config_path() -> Path:
    return Path(os.environ.get("CHAOSAI_CONFIG") or ROOT / "config.yaml")


def load(path: Path | str | None = None) -> Config:
    """Read the config file (defaults if it does not exist). Raises ValueError with a clear message."""
    p = Path(path) if path else config_path()
    data = {}
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    try:
        return Config.model_validate(data)
    except Exception as e:
        raise ValueError(f"invalid config {p}: {e}") from e
