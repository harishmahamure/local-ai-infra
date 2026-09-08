from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WorkerStatus(str, Enum):
    STARTING = "STARTING"
    ALIVE = "ALIVE"
    DRAINING = "DRAINING"
    UNAVAILABLE = "UNAVAILABLE"
    DEAD = "DEAD"


@dataclass
class WorkerRecord:
    worker_id: str
    role: str
    status: WorkerStatus = WorkerStatus.STARTING
    current_job_id: str | None = None
    current_attempt_id: str | None = None
    runtime_status: str = "idle"
    gpu_status: str | None = None
    last_heartbeat: str | None = None
    started_at: str | None = None
    hostname: str = "gpu-box"
    queues: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "role": self.role,
            "status": self.status.value,
            "current_job": self.current_job_id,
            "current_attempt": self.current_attempt_id,
            "runtime_status": self.runtime_status,
            "gpu_status": self.gpu_status,
            "last_heartbeat": self.last_heartbeat,
            "started_at": self.started_at,
            "hostname": self.hostname,
            "queues": list(self.queues),
        }
