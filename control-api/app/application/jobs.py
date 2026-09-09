from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterator

from ..domain.errors import DomainError, ErrorCode
from ..domain.jobs import Job, JobStatus
from ..domain.operations import OPERATIONS, operation_capabilities, operation_catalog
from ..infrastructure.assets import AssetStore
from ..infrastructure.gpu_scheduler import GpuScheduler
from ..infrastructure.job_store import JobStore
from ..infrastructure.job_worker import JobWorker
from .image_ops import plan_operation


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobService:
    def __init__(
        self,
        *,
        store: JobStore,
        assets: AssetStore,
        scheduler: GpuScheduler,
        worker: JobWorker,
        disk_status: Any = None,
    ) -> None:
        self.store = store
        self.assets = assets
        self.scheduler = scheduler
        self.worker = worker
        self._disk_status = disk_status

    def recover_and_start(self) -> None:
        self.store.recover_stale()
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()

    def list_operations(self) -> dict[str, Any]:
        status = {}
        if self._disk_status:
            try:
                status = self._disk_status()
            except Exception:
                status = {}
        ops = []
        for item in operation_catalog():
            missing = [bid for bid in item["requiredBundles"] if status and status.get(bid) not in {None, "complete", "ok"}]
            # If we have a status map, missing bundles that are absent entirely also count.
            if status:
                missing = [bid for bid in item["requiredBundles"] if status.get(bid, "missing") != "complete"]
            ops.append({**item, "available": not missing, "missingBundles": missing})
        return {"operations": ops, "capabilities": operation_capabilities()}

    def submit(self, body: dict[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        operation = str(body.get("operation") or "").strip()
        if operation not in OPERATIONS:
            raise DomainError(ErrorCode.UNSUPPORTED_OPERATION, f"Unknown operation {operation}")
        if idempotency_key:
            existing = self.store.get_by_idempotency(idempotency_key)
            if existing:
                return existing.to_public_dict(queue_position=self.store.queue_position(existing.job_id))
        steps, _refs, _bundles = plan_operation(operation, body)
        seed = steps[0].seed if steps else None
        inputs = {k: v for k, v in body.items() if k != "operation"}
        job = Job(
            job_id=f"job_{uuid.uuid4().hex}",
            operation=operation,
            status=JobStatus.QUEUED,
            phase="queued",
            progress=0.0,
            inputs=inputs,
            parameters={},
            created_at=_now(),
            idempotency_key=idempotency_key,
            seed=seed,
        )
        self.store.save(job)
        return job.to_public_dict(queue_position=self.store.queue_position(job.job_id))

    def get(self, job_id: str) -> dict[str, Any]:
        job = self.store.get(job_id)
        if job is None:
            raise DomainError(ErrorCode.INVALID_REQUEST, f"Unknown job {job_id}")
        return job.to_public_dict(queue_position=self.store.queue_position(job.job_id))

    def list_jobs(self, *, status: str | None = None, cursor: str | None = None, limit: int = 20) -> dict[str, Any]:
        if status:
            try:
                JobStatus(status)
            except ValueError as exc:
                raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unknown status {status}") from exc
        jobs, next_cursor = self.store.list_jobs(status=status, cursor=cursor, limit=limit)
        return {
            "data": [job.to_public_dict(queue_position=self.store.queue_position(job.job_id)) for job in jobs],
            "pagination": {"nextCursor": next_cursor, "hasMore": next_cursor is not None},
        }

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.store.get(job_id)
        if job is None:
            raise DomainError(ErrorCode.INVALID_REQUEST, f"Unknown job {job_id}")
        if job.status.terminal:
            raise DomainError(ErrorCode.INVALID_REQUEST, f"Job {job_id} is already {job.status.value}")
        job.cancel_requested = True
        if job.status == JobStatus.QUEUED:
            job.transition(JobStatus.CANCELLED, phase="cancelled", progress=1.0)
            job.finished_at = _now()
            job.error = {"code": ErrorCode.CANCELLED.value, "message": "Job cancelled", "retryable": False}
        else:
            self.worker.interrupt_active()
        self.store.save(job)
        return job.to_public_dict()

    def events(self, job_id: str) -> Iterator[dict[str, Any]]:
        last: str | None = None
        while True:
            job = self.store.get(job_id)
            if job is None:
                raise DomainError(ErrorCode.INVALID_REQUEST, f"Unknown job {job_id}")
            payload = job.to_public_dict(queue_position=self.store.queue_position(job.job_id))
            marker = f"{payload['status']}:{payload['phase']}:{payload['progress']}"
            if marker != last:
                yield payload
                last = marker
            if job.status.terminal:
                return
            yield from ()
            import time

            time.sleep(0.6)

    def snapshot(self) -> dict[str, Any]:
        return {
            **self.scheduler.snapshot(),
            "queueDepth": self.store.queued_count(),
        }
