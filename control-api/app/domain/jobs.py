from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import DomainError, ErrorCode


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    RESOLVING = "RESOLVING"
    WAITING_FOR_RESOURCE = "WAITING_FOR_RESOURCE"
    RUNNING = "RUNNING"
    POST_PROCESSING = "POST_PROCESSING"
    QC = "QC"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}

    @property
    def public(self) -> str:
        return self.value.lower() if self == JobStatus.QUEUED else self.value


class RetryStrategy(str, Enum):
    SAME = "same"
    NEW_SEED = "new_seed"
    OVERRIDE = "override"
    RERUN_STAGE = "rerun_stage"


@dataclass
class JobResult:
    asset_ids: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    job_id: str
    operation: str
    preset: str
    status: JobStatus
    phase: str
    progress: float
    inputs: dict[str, Any]
    parameters: dict[str, Any]
    client_context: dict[str, Any]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: dict[str, Any] | None = None
    result: JobResult | None = None
    workflow_id: str | None = None
    workflow_version: str | None = None
    model_ids: list[str] = field(default_factory=list)
    retry_count: int = 0
    parent_job_id: str | None = None
    idempotency_key: str | None = None
    trace_id: str = ""
    cancel_requested: bool = False
    output: dict[str, Any] | None = None

    def request_cancel(self) -> None:
        if self.status.terminal:
            raise DomainError(ErrorCode.INVALID_REQUEST, f"Job {self.job_id} is already {self.status.value}")
        self.cancel_requested = True
        if self.status in {JobStatus.QUEUED, JobStatus.VALIDATING, JobStatus.RESOLVING}:
            self.status = JobStatus.CANCELLED
            self.phase = "cancelled"
            self.error = {"code": ErrorCode.CANCELLED.value, "message": "Job cancelled", "retryable": False}

    def mark(self, status: JobStatus, *, phase: str | None = None, progress: float | None = None) -> None:
        self.status = status
        if phase is not None:
            self.phase = phase
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))

    def to_public_dict(self) -> dict[str, Any]:
        status = "queued" if self.status == JobStatus.QUEUED else self.status.value
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "status": status,
            "operation": self.operation,
            "preset": self.preset,
            "phase": self.phase,
            "progress": self.progress,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "client_context": self.client_context,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": None,
            "workflow_id": self.workflow_id,
            "workflow_version": self.workflow_version,
            "model_ids": list(self.model_ids),
            "retry_count": self.retry_count,
            "parent_job_id": self.parent_job_id,
            "trace_id": self.trace_id,
        }
        if self.result is not None:
            payload["result"] = {
                "asset_ids": list(self.result.asset_ids),
                "manifest": dict(self.result.manifest),
            }
        return payload
