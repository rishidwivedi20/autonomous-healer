"""Pydantic models persisted to results/ and SQLite. Real measurements only."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Thresholds(BaseModel):
    error_rate: float = 0.5      # fraction of failed probes (0-1)
    p95_ms: float = 3000.0
    breach_seconds: int = 3      # consecutive seconds over a threshold before auto-abort


class Stats(BaseModel):
    n: int = 0
    error_rate: Optional[float] = None
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None


class SecondStat(BaseModel):
    t: float          # seconds since experiment start
    n: int
    error_rate: Optional[float]
    p50_ms: Optional[float]
    p95_ms: Optional[float]


class ContainerEvent(BaseModel):
    t: float          # epoch seconds
    service: str
    kind: str         # state | health | restart_count
    value: str


class Service(BaseModel):
    name: str
    image: str = ""
    replicas: int = 1
    healthcheck: bool = False
    restart_policy: str = "no"
    has_limits: bool = False
    stateful: bool = False
    depends_on: list[str] = Field(default_factory=list)
    fan_in: int = 0
    fan_out: int = 0
    on_critical_path: bool = False


class Graph(BaseModel):
    services: dict[str, Service]
    critical_path: list[str] = Field(default_factory=list)


class Experiment(BaseModel):
    id: str
    target: str
    fault: str
    params: dict = Field(default_factory=dict)
    duration_s: int = 30
    hypothesis: str = ""
    thresholds: Thresholds = Field(default_factory=Thresholds)


class ExperimentResult(BaseModel):
    experiment: Experiment
    started_at: float
    ended_at: float
    dry_run: bool = False
    baseline: Stats
    during: Stats
    per_second: list[SecondStat] = Field(default_factory=list)
    recovery_s: Optional[float] = None        # None = did not recover / not measured
    recovered: bool = False
    aborted: bool = False
    abort_reason: Optional[str] = None
    container_events: list[ContainerEvent] = Field(default_factory=list)
    restarts: dict[str, int] = Field(default_factory=dict)
    rollback_errors: list[str] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)
