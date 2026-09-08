from __future__ import annotations

from enum import Enum


class JobEventType(str, Enum):
    JOB_CREATED = "JOB_CREATED"
    JOB_QUEUED = "JOB_QUEUED"
    RESOURCE_WAIT_STARTED = "RESOURCE_WAIT_STARTED"
    ATTEMPT_STARTED = "ATTEMPT_STARTED"
    MODEL_LOADING = "MODEL_LOADING"
    INFERENCE_STARTED = "INFERENCE_STARTED"
    POST_PROCESSING_STARTED = "POST_PROCESSING_STARTED"
    ATTEMPT_FAILED = "ATTEMPT_FAILED"
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    JOB_CANCEL_REQUESTED = "JOB_CANCEL_REQUESTED"
    JOB_CANCELLED = "JOB_CANCELLED"
    JOB_COMPLETED = "JOB_COMPLETED"
    JOB_FAILED = "JOB_FAILED"
    PRIORITY_CHANGED = "PRIORITY_CHANGED"
    JOB_REQUEUED = "JOB_REQUEUED"
    DEPENDENCY_UNBLOCKED = "DEPENDENCY_UNBLOCKED"
    LEASE_ACQUIRED = "LEASE_ACQUIRED"
    LEASE_LOST = "LEASE_LOST"
    OOM_FALLBACK = "OOM_FALLBACK"
    ARTIFACT_RECOVERED = "ARTIFACT_RECOVERED"

    @property
    def legacy(self) -> str:
        return {
            JobEventType.JOB_CREATED: "job.created",
            JobEventType.JOB_QUEUED: "job.queued",
            JobEventType.RESOURCE_WAIT_STARTED: "job.waiting_for_resource",
            JobEventType.ATTEMPT_STARTED: "attempt.started",
            JobEventType.MODEL_LOADING: "model.loading",
            JobEventType.INFERENCE_STARTED: "generation.started",
            JobEventType.POST_PROCESSING_STARTED: "postprocess.started",
            JobEventType.ATTEMPT_FAILED: "attempt.failed",
            JobEventType.RETRY_SCHEDULED: "retry.scheduled",
            JobEventType.JOB_CANCEL_REQUESTED: "job.cancel_requested",
            JobEventType.JOB_CANCELLED: "job.cancelled",
            JobEventType.JOB_COMPLETED: "job.completed",
            JobEventType.JOB_FAILED: "job.failed",
            JobEventType.PRIORITY_CHANGED: "job.priority_changed",
            JobEventType.JOB_REQUEUED: "job.requeued",
            JobEventType.DEPENDENCY_UNBLOCKED: "job.unblocked",
            JobEventType.LEASE_ACQUIRED: "lease.acquired",
            JobEventType.LEASE_LOST: "lease.lost",
            JobEventType.OOM_FALLBACK: "retry.oom_fallback",
            JobEventType.ARTIFACT_RECOVERED: "artifact.recovered",
        }[self]


def event_name(value: str | JobEventType) -> str:
    if isinstance(value, JobEventType):
        return value.legacy
    return value
