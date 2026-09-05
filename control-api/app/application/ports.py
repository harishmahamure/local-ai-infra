from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..domain.artifacts import Artifact
from ..domain.jobs import Job
from ..domain.models import ModelState
from ..domain.runtime import ResourceLease, ResourceRequirement
from ..domain.workflows import WorkflowDefinition


class Clock(Protocol):
    def now_iso(self) -> str: ...

    def now_ts(self) -> float: ...


class IDGenerator(Protocol):
    def job_id(self) -> str: ...

    def asset_id(self) -> str: ...

    def trace_id(self) -> str: ...


class JobRepository(Protocol):
    def save(self, job: Job) -> None: ...

    def get(self, job_id: str) -> Job | None: ...

    def list_active(self) -> list[Job]: ...


class ArtifactRepository(Protocol):
    def save(self, artifact: Artifact) -> None: ...

    def get(self, asset_id: str) -> Artifact | None: ...

    def delete(self, asset_id: str) -> None: ...


class IdempotencyRepository(Protocol):
    def get(self, key: str) -> tuple[str, str] | None:
        """Return (request_hash, job_id) if present."""

    def put(self, key: str, request_hash: str, job_id: str) -> None: ...


class RuntimeStateRepository(Protocol):
    def get(self, key: str) -> str | None: ...

    def put(self, key: str, value: str) -> None: ...


class ArtifactStorage(Protocol):
    def write(self, asset_id: str, data: bytes) -> str: ...

    def read(self, asset_id: str) -> bytes: ...

    def delete(self, asset_id: str) -> None: ...

    def writable(self) -> bool: ...


class EventPublisher(Protocol):
    def publish(self, job_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def history(self, job_id: str) -> list[dict[str, Any]]: ...

    def subscribe(self, job_id: str) -> Any: ...


class MediaInspector(Protocol):
    def inspect_image(self, data: bytes, mime_type: str) -> dict[str, Any]: ...


class ModelDownloader(Protocol):
    def start(self, model_ids: list[str]) -> dict[str, Any]: ...

    def status(self) -> dict[str, Any]: ...


class ModelRuntime(Protocol):
    def disk_status(self) -> dict[str, str]: ...

    def load(self, model_id: str) -> dict[str, Any]: ...

    def unload(self, model_id: str) -> dict[str, Any]: ...

    def state(self, model_id: str) -> ModelState: ...


class ResourceScheduler(Protocol):
    def acquire(self, job_id: str, requirement: ResourceRequirement) -> ResourceLease: ...

    def release(self, lease: ResourceLease) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...

    def interrupt(self, job_id: str) -> None: ...


@dataclass
class ExecutionOutput:
    data: bytes
    mime_type: str
    artifact_type: str
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None


@dataclass
class ExecutionRequest:
    job: Job
    workflow: WorkflowDefinition
    plan: dict[str, Any]
    input_files: dict[str, bytes] = field(default_factory=dict)
    cancel_check: Callable[[], bool] | None = None
    on_phase: Callable[[str, float], None] | None = None


@dataclass
class ExecutionResult:
    outputs: list[ExecutionOutput]
    model_ids: list[str]
    seed: int | None = None
    duration_ms: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class WorkflowExecutor(Protocol):
    def execute(self, request: ExecutionRequest) -> ExecutionResult: ...

    def cancel(self, job_id: str) -> None: ...

    def health(self) -> dict[str, Any]: ...
