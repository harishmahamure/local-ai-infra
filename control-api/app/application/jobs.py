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

    def delete_job(self, job_id: str, *, delete_assets: bool = True) -> dict[str, Any]:
        job = self.store.get(job_id)
        if job is None:
            raise DomainError(ErrorCode.INVALID_REQUEST, f"Unknown job {job_id}")
        if job.status == JobStatus.RUNNING:
            raise DomainError(ErrorCode.INVALID_REQUEST, "Job is running; cancel it first")
        outputs = list(job.asset_ids)
        if not self.store.delete_job(job_id):
            raise DomainError(ErrorCode.INVALID_REQUEST, f"Unknown job {job_id}")
        removed: list[str] = []
        if delete_assets and self.assets is not None:
            for asset_id in outputs:
                try:
                    self.assets.delete(asset_id)
                    removed.append(asset_id)
                except DomainError as exc:
                    if exc.code != ErrorCode.ASSET_NOT_FOUND:
                        raise
        return {"job_id": job_id, "deleted": True, "deleted_assets": removed}

    def delete_jobs(self, *, status: str | None = None, delete_assets: bool = True) -> dict[str, Any]:
        if status:
            try:
                parsed = JobStatus(status)
            except ValueError as exc:
                raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unknown status {status}") from exc
            if parsed == JobStatus.RUNNING:
                raise DomainError(ErrorCode.INVALID_REQUEST, "Cannot bulk-delete running jobs; cancel them first")
        deleted_jobs: list[str] = []
        deleted_assets: list[str] = []
        skipped_running = 0
        for job in self.store.list_all_jobs(status=status):
            if job.status == JobStatus.RUNNING:
                skipped_running += 1
                continue
            result = self.delete_job(job.job_id, delete_assets=delete_assets)
            deleted_jobs.append(result["job_id"])
            deleted_assets.extend(result["deleted_assets"])
        return {
            "deleted": True,
            "deleted_jobs": deleted_jobs,
            "deleted_assets": deleted_assets,
            "skipped_running": skipped_running,
        }

    def delete_job_assets(self, job_id: str) -> dict[str, Any]:
        job = self.store.get(job_id)
        if job is None:
            raise DomainError(ErrorCode.INVALID_REQUEST, f"Unknown job {job_id}")
        if self.assets is None:
            raise DomainError(ErrorCode.INTERNAL_ERROR, "Asset store is not enabled")
        removed: list[str] = []
        for asset_id in list(job.asset_ids):
            try:
                self.assets.delete(asset_id)
                removed.append(asset_id)
            except DomainError as exc:
                if exc.code != ErrorCode.ASSET_NOT_FOUND:
                    raise
        return {"job_id": job_id, "deleted": True, "deleted_assets": removed}

    def delete_all_assets(self) -> dict[str, Any]:
        if self.assets is None:
            raise DomainError(ErrorCode.INTERNAL_ERROR, "Asset store is not enabled")
        protected: set[str] = set()
        for job in self.store.list_all_jobs(status=JobStatus.RUNNING.value):
            protected.update(job.asset_ids)
        removed: list[str] = []
        skipped = 0
        for record in self.store.list_all_assets():
            asset_id = str(record["asset_id"])
            if asset_id in protected:
                skipped += 1
                continue
            try:
                self.assets.delete(asset_id)
                removed.append(asset_id)
            except DomainError as exc:
                if exc.code != ErrorCode.ASSET_NOT_FOUND:
                    raise
        return {"deleted": True, "deleted_assets": removed, "skipped_running": skipped}

    def delete_asset(self, asset_id: str) -> dict[str, Any]:
        if self.assets is None:
            raise DomainError(ErrorCode.INTERNAL_ERROR, "Asset store is not enabled")
        return self.assets.delete(asset_id)

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
