from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import DomainError, ErrorCode


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}


LEGAL_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.CANCELLED}),
    JobStatus.RUNNING: frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}


def can_transition(current: JobStatus, target: JobStatus) -> bool:
    if current == target:
        return True
    return target in LEGAL_TRANSITIONS.get(current, frozenset())


@dataclass
class Job:
    job_id: str
    operation: str
    status: JobStatus
    phase: str
    progress: float
    inputs: dict[str, Any]
    parameters: dict[str, Any]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: dict[str, Any] | None = None
    asset_ids: list[str] = field(default_factory=list)
    idempotency_key: str | None = None
    cancel_requested: bool = False
    seed: int | None = None

    def transition(self, status: JobStatus, *, phase: str | None = None, progress: float | None = None) -> None:
        if not can_transition(self.status, status):
            raise DomainError(
                ErrorCode.INVALID_REQUEST,
                f"Illegal transition {self.status.value} -> {status.value}",
                {"job_id": self.job_id, "from": self.status.value, "to": status.value},
            )
        self.status = status
        if phase is not None:
            self.phase = phase
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))

    def to_public_dict(self, *, queue_position: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "operation": self.operation,
            "status": self.status.value,
            "phase": self.phase,
            "progress": self.progress,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "asset_ids": list(self.asset_ids),
            "seed": self.seed,
            "cancel_requested": self.cancel_requested,
        }
        if queue_position is not None:
            payload["queue_position"] = queue_position
        return payload
