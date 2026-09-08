from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import DomainError, ErrorCode
from .priority import Priority
from .queues import QueueClass


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    BLOCKED = "BLOCKED"
    VALIDATING = "VALIDATING"
    RESOLVING = "RESOLVING"
    WAITING_FOR_RESOURCE = "WAITING_FOR_RESOURCE"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    POST_PROCESSING = "POST_PROCESSING"
    QC = "QC"
    RETRY_WAIT = "RETRY_WAIT"
    CANCELLING = "CANCELLING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}

    @property
    def public(self) -> str:
        if self == JobStatus.QUEUED:
            return "queued"
        if self == JobStatus.DISPATCHED:
            return JobStatus.WAITING_FOR_RESOURCE.value
        return self.value


LEGAL_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.BLOCKED: frozenset({JobStatus.QUEUED, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.QUEUED: frozenset(
        {JobStatus.DISPATCHED, JobStatus.WAITING_FOR_RESOURCE, JobStatus.CANCELLED, JobStatus.BLOCKED, JobStatus.RUNNING}
    ),
    JobStatus.VALIDATING: frozenset({JobStatus.RESOLVING, JobStatus.QUEUED, JobStatus.CANCELLED}),
    JobStatus.RESOLVING: frozenset({JobStatus.QUEUED, JobStatus.CANCELLED}),
    JobStatus.WAITING_FOR_RESOURCE: frozenset(
        {JobStatus.RUNNING, JobStatus.DISPATCHED, JobStatus.QUEUED, JobStatus.CANCELLED, JobStatus.FAILED}
    ),
    JobStatus.DISPATCHED: frozenset(
        {JobStatus.RUNNING, JobStatus.QUEUED, JobStatus.CANCELLED, JobStatus.WAITING_FOR_RESOURCE, JobStatus.FAILED}
    ),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.POST_PROCESSING,
            JobStatus.RETRY_WAIT,
            JobStatus.QUEUED,
            JobStatus.FAILED,
            JobStatus.CANCELLING,
            JobStatus.COMPLETED,
            JobStatus.QC,
        }
    ),
    JobStatus.POST_PROCESSING: frozenset(
        {JobStatus.COMPLETED, JobStatus.RETRY_WAIT, JobStatus.FAILED, JobStatus.CANCELLING, JobStatus.QC}
    ),
    JobStatus.QC: frozenset({JobStatus.COMPLETED, JobStatus.FAILED}),
    JobStatus.RETRY_WAIT: frozenset({JobStatus.QUEUED, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.CANCELLING: frozenset({JobStatus.CANCELLED, JobStatus.FAILED}),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED: frozenset({JobStatus.QUEUED}),
    JobStatus.CANCELLED: frozenset({JobStatus.QUEUED}),
}


def can_transition(current: JobStatus, target: JobStatus) -> bool:
    if current == target:
        return True
    return target in LEGAL_TRANSITIONS.get(current, frozenset())


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
    priority: Priority = Priority.NORMAL
    queue: QueueClass = QueueClass.GPU
    queued_at: str | None = None
    completed_at: str | None = None
    max_retries: int = 3
    current_attempt: str | None = None
    batch_id: str | None = None
    next_attempt_at: str | None = None
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    compute_node: str = "gpu-box"
    resolved_plan: dict[str, Any] = field(default_factory=dict)
    error_summary: str | None = None

    def request_cancel(self) -> None:
        if self.status.terminal:
            raise DomainError(ErrorCode.INVALID_REQUEST, f"Job {self.job_id} is already {self.status.value}")
        self.cancel_requested = True
        if self.status in {
            JobStatus.QUEUED,
            JobStatus.VALIDATING,
            JobStatus.RESOLVING,
            JobStatus.BLOCKED,
            JobStatus.RETRY_WAIT,
        }:
            self.transition(JobStatus.CANCELLED, phase="cancelled")
            self.error = {"code": ErrorCode.CANCELLED.value, "message": "Job cancelled", "retryable": False}

    def mark(self, status: JobStatus, *, phase: str | None = None, progress: float | None = None) -> None:
        self.transition(status, phase=phase, progress=progress)

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

    def to_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status.public,
            "operation": self.operation,
            "preset": self.preset,
            "phase": self.phase,
            "progress": self.progress,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "client_context": self.client_context,
            "created_at": self.created_at,
            "queued_at": self.queued_at or self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "completed_at": self.completed_at or self.finished_at,
            "error": self.error,
            "error_summary": self.error_summary,
            "result": None,
            "workflow_id": self.workflow_id,
            "workflow_version": self.workflow_version,
            "model_ids": list(self.model_ids),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "current_attempt": self.current_attempt,
            "parent_job_id": self.parent_job_id,
            "trace_id": self.trace_id,
            "priority": self.priority.value,
            "queue": self.queue.value,
            "batch_id": self.batch_id,
            "cancel_requested": self.cancel_requested,
            "compute_node": self.compute_node,
        }
        if self.result is not None:
            payload["result"] = {
                "asset_ids": list(self.result.asset_ids),
                "manifest": dict(self.result.manifest),
            }
        return payload
