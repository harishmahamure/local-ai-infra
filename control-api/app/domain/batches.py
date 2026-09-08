from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BatchStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"


@dataclass
class Batch:
    batch_id: str
    capability: str
    preset: str
    created_at: str
    status: BatchStatus = BatchStatus.QUEUED
    total: int = 0
    queued: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    job_ids: list[str] = field(default_factory=list)
    priority: str = "NORMAL"
    cancel_requested: bool = False

    def refresh_status(self) -> None:
        if self.cancelled and self.completed + self.failed + self.cancelled >= self.total:
            self.status = BatchStatus.CANCELLED
            return
        if self.failed and self.completed + self.failed + self.cancelled >= self.total:
            self.status = BatchStatus.PARTIAL if self.completed else BatchStatus.FAILED
            return
        if self.completed >= self.total and self.total > 0:
            self.status = BatchStatus.COMPLETED
            return
        if self.running or self.completed or self.failed:
            self.status = BatchStatus.RUNNING
            return
        self.status = BatchStatus.QUEUED

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "capability": self.capability,
            "preset": self.preset,
            "status": self.status.value,
            "created_at": self.created_at,
            "priority": self.priority,
            "total": self.total,
            "queued": self.queued,
            "running": self.running,
            "completed": self.completed,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "job_ids": list(self.job_ids),
        }
