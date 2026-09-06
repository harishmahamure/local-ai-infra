from __future__ import annotations

import queue
import threading
from typing import Any

from ..application.ports import ExecutionRequest, WorkflowExecutor
from ..application.resolvers import resolve_preset, resolve_workflow
from ..domain.artifacts import Artifact, ArtifactType, Lineage, ReproducibilityManifest
from ..domain.errors import DomainError, ErrorCode
from ..domain.jobs import Job, JobResult, JobStatus
from ..application.catalog import CatalogRegistry
from ..infrastructure.clock import expires_at
from ..infrastructure.comfyui.plans import build_plan
from .storage import sha256_bytes


class JobWorker:
    def __init__(
        self,
        *,
        jobs,
        artifacts,
        storage,
        events,
        scheduler,
        executor: WorkflowExecutor,
        catalog: CatalogRegistry,
        clock,
        ids,
        ttl_hours: int,
        inspector,
    ) -> None:
        self.jobs = jobs
        self.artifacts = artifacts
        self.storage = storage
        self.events = events
        self.scheduler = scheduler
        self.executor = executor
        self.catalog = catalog
        self.clock = clock
        self.ids = ids
        self.ttl_hours = ttl_hours
        self.inspector = inspector
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.alive = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="engine-gpu-worker", daemon=True)
        self._thread.start()
        self.alive = True

    def stop(self) -> None:
        self._stop.set()
        self.alive = False

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)
        if hasattr(self.scheduler, "set_queue_depth"):
            self.scheduler.set_queue_depth(self._queue.qsize())

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run(job_id)
            except Exception:
                pass
            finally:
                if hasattr(self.scheduler, "set_queue_depth"):
                    self.scheduler.set_queue_depth(self._queue.qsize())

    def _run(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        if job.cancel_requested or job.status == JobStatus.CANCELLED:
            job.status = JobStatus.CANCELLED
            job.phase = "cancelled"
            job.finished_at = self.clock.now_iso()
            self.jobs.save(job)
            self.events.publish(job.job_id, "job.cancelled")
            return
        try:
            self._execute_job(job)
        except DomainError as exc:
            if exc.code == ErrorCode.CANCELLED:
                job.status = JobStatus.CANCELLED
                job.phase = "cancelled"
                job.error = {"code": exc.code.value, "message": exc.message, "retryable": False}
                self.events.publish(job.job_id, "job.cancelled")
            else:
                job.status = JobStatus.FAILED
                job.phase = "failed"
                job.error = {"code": exc.code.value, "message": exc.message, "retryable": exc.code.retryable, "details": exc.details}
                self.events.publish(job.job_id, "job.failed", {"error": job.error})
            job.finished_at = self.clock.now_iso()
            self.jobs.save(job)
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.phase = "failed"
            job.error = {"code": ErrorCode.INTERNAL_ERROR.value, "message": str(exc), "retryable": True}
            job.finished_at = self.clock.now_iso()
            self.jobs.save(job)
            self.events.publish(job.job_id, "job.failed", {"error": job.error})

    def _execute_job(self, job: Job) -> None:
        job.mark(JobStatus.WAITING_FOR_RESOURCE, phase="waiting_for_resource", progress=0.05)
        job.started_at = self.clock.now_iso()
        self.jobs.save(job)
        self.events.publish(job.job_id, "job.waiting_for_resource")

        operation = self.catalog.operation(job.operation)
        preset = resolve_preset(self.catalog, operation, job.preset)
        workflow = resolve_workflow(self.catalog, operation, preset)
        loras = job.parameters.get("_resolved_loras") or []
        plan = build_plan(job, workflow, preset, loras)

        lease = self.scheduler.acquire(job.job_id, workflow.resource)
        try:
            job.mark(JobStatus.RUNNING, phase="model_loading", progress=0.1)
            self.jobs.save(job)
            self.events.publish(job.job_id, "model.loading")

            input_files = self._load_inputs(job)

            def cancel_check() -> bool:
                latest = self.jobs.get(job.job_id)
                return bool(latest and latest.cancel_requested)

            def on_phase(phase: str, progress: float) -> None:
                latest = self.jobs.get(job.job_id) or job
                latest.mark(JobStatus.RUNNING, phase=phase, progress=progress)
                self.jobs.save(latest)
                self.events.publish(job.job_id, phase if phase.startswith("generation.") else "generation.progress", {"progress": progress})

            result = self.executor.execute(
                ExecutionRequest(
                    job=job,
                    workflow=workflow,
                    plan=plan,
                    input_files=input_files,
                    cancel_check=cancel_check,
                    on_phase=on_phase,
                )
            )
            job = self.jobs.get(job.job_id) or job
            if job.cancel_requested:
                raise DomainError(ErrorCode.CANCELLED, "Job cancelled")

            job.mark(JobStatus.POST_PROCESSING, phase="persist", progress=0.9)
            self.jobs.save(job)
            self.events.publish(job.job_id, "postprocess.started")

            asset_ids = self._persist_outputs(job, workflow, plan, result, loras)
            job.model_ids = list(result.model_ids)
            manifest = ReproducibilityManifest(
                job_id=job.job_id,
                operation=job.operation,
                preset=job.preset,
                workflow_id=workflow.id,
                workflow_version=workflow.version,
                model_ids=list(result.model_ids),
                lora_ids=[x.get("id") for x in loras],
                lora_strengths=[float(x.get("strength") or 0) for x in loras],
                seed=result.seed,
                prompt=job.inputs.get("prompt"),
                negative_prompt=job.inputs.get("negative_prompt"),
                steps=plan.get("steps"),
                cfg=plan.get("cfg"),
                width=plan.get("width"),
                height=plan.get("height"),
                fps=plan.get("fps"),
                execution_duration_ms=result.duration_ms,
            ).to_dict()
            warnings = list(result.extra.get("warnings") or [])
            if plan.get("video_ignored"):
                warning = "audio.foley is prompt-only in Phase A; video input was ignored"
                if warning not in warnings:
                    warnings.append(warning)
            if warnings:
                manifest["warnings"] = warnings
            job.result = JobResult(asset_ids=asset_ids, manifest=manifest)
            job.mark(JobStatus.COMPLETED, phase="done", progress=1.0)
            job.finished_at = self.clock.now_iso()
            self.jobs.save(job)
            self.events.publish(job.job_id, "job.completed", {"asset_ids": asset_ids})
        finally:
            self.scheduler.release(lease)

    def _load_inputs(self, job: Job) -> dict[str, bytes]:
        files: dict[str, bytes] = {}
        for key, value in job.inputs.items():
            if isinstance(value, list):
                for index, item in enumerate(value):
                    asset_id = _asset_ref(item)
                    if not asset_id:
                        continue
                    artifact = self.artifacts.get(asset_id)
                    if artifact is None or artifact.deleted:
                        raise DomainError(ErrorCode.ASSET_NOT_FOUND, f"Asset {asset_id} not found")
                    files[f"{key}_{index}"] = self.storage.read(asset_id)
                continue
            asset_id = _asset_ref(value)
            if not asset_id:
                continue
            artifact = self.artifacts.get(asset_id)
            if artifact is None or artifact.deleted:
                raise DomainError(ErrorCode.ASSET_NOT_FOUND, f"Asset {asset_id} not found")
            files[key] = self.storage.read(asset_id)
        return files

    def _persist_outputs(self, job: Job, workflow, plan: dict[str, Any], result, loras) -> list[str]:
        ids: list[str] = []
        now = self.clock.now_iso()
        sources = [_asset_ref(v) for v in job.inputs.values() if _asset_ref(v)]
        for output in result.outputs:
            asset_id = self.ids.asset_id()
            checksum = self.storage.write(asset_id, output.data)
            meta = self.inspector.inspect_image(output.data, output.mime_type) if output.artifact_type == "IMAGE" else {}
            artifact = Artifact(
                asset_id=asset_id,
                type=ArtifactType[output.artifact_type] if output.artifact_type in ArtifactType.__members__ else ArtifactType.TEMPORARY,
                mime_type=output.mime_type,
                checksum=checksum or sha256_bytes(output.data),
                size_bytes=len(output.data),
                created_at=now,
                expires_at=expires_at(now, self.ttl_hours),
                width=meta.get("width") or output.width,
                height=meta.get("height") or output.height,
                duration_seconds=output.duration_seconds,
                lineage=Lineage(
                    source_asset_ids=sources,
                    parent_job_id=job.job_id,
                    model_id=(result.model_ids or [None])[0],
                    workflow_id=workflow.id,
                    workflow_version=workflow.version,
                    seed=result.seed,
                    prompt=job.inputs.get("prompt"),
                    loras=list(loras),
                    generation_settings={"steps": plan.get("steps"), "cfg": plan.get("cfg")},
                ),
                manifest={"job_id": job.job_id, "operation": job.operation, "preset": job.preset},
            )
            self.artifacts.save(artifact)
            ids.append(asset_id)
        return ids


def _asset_ref(value: Any) -> str | None:
    if isinstance(value, dict) and value.get("asset_id"):
        return str(value["asset_id"])
    return None
