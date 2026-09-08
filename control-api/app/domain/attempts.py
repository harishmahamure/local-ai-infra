from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AttemptStatus(str, Enum):
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    LOST = "LOST"


@dataclass
class JobAttempt:
    attempt_id: str
    job_id: str
    attempt_number: int
    status: AttemptStatus
    compute_node: str = "gpu-box"
    runtime: str | None = None
    model: str | None = None
    variant: str | None = None
    workflow: str | None = None
    worker_id: str | None = None
    queued_at: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    peak_vram: float | None = None
    execution_time: float | None = None
    result: dict[str, Any] | None = None
    error_code: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)
    lease_expires_at: str | None = None
    profile: str | None = None
    queue: str = "gpu"

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "job_id": self.job_id,
            "attempt_number": self.attempt_number,
            "status": self.status.value,
            "compute_node": self.compute_node,
            "runtime": self.runtime,
            "model": self.model,
            "variant": self.variant,
            "workflow": self.workflow,
            "worker": self.worker_id,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "peak_vram": self.peak_vram,
            "execution_time": self.execution_time,
            "result": self.result,
            "error_code": self.error_code,
            "error_details": dict(self.error_details),
        }
