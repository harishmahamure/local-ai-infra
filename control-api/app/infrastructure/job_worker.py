from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable

from ..application.image_ops import plan_operation, step_to_plan
from ..domain.errors import DomainError, ErrorCode
from ..domain.jobs import Job, JobStatus
from .comfy import graphs as qwen_graph
from .assets import AssetStore
from .comfy_client import ComfyClient
from .gpu_scheduler import GpuScheduler
from .job_store import JobStore

IMAGE_TIMEOUT_SEC = 600


def _collect_files(outputs: dict[str, Any]) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    for out in outputs.values():
        for item in out.get("images") or []:
            filename = item.get("filename")
            if filename:
                files.append(
                    {
                        "filename": filename,
                        "subfolder": item.get("subfolder", ""),
                        "folder_type": item.get("type", "output"),
                    }
                )
    return files


def _history_failed(entry: dict[str, Any]) -> bool:
    status = entry.get("status") or {}
    return status.get("status_str") == "error"


class JobWorker:
    def __init__(
        self,
        *,
        jobs: JobStore,
        assets: AssetStore,
        scheduler: GpuScheduler,
        catalog: Any,
        client: ComfyClient | None = None,
        now_iso: Callable[[], str],
        disk_status: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        self.jobs = jobs
        self.assets = assets
        self.scheduler = scheduler
        self.catalog = catalog
        self.client = client or ComfyClient()
        self._now = now_iso
        self._disk_status = disk_status
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._active_client_job: str | None = None
        self.alive = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="image-gpu-worker", daemon=True)
        self._thread.start()
        self.alive = True

    def stop(self) -> None:
        self._stop.set()
        self.alive = False

    def interrupt_active(self) -> None:
        try:
            self.client.interrupt()
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.jobs.claim_next()
            except Exception:
                threading.Event().wait(0.5)
                continue
            if job is None:
                threading.Event().wait(0.4)
                continue
            if job.status == JobStatus.CANCELLED:
                continue
            try:
                self._run(job)
            except Exception:
                pass

    def _run(self, job: Job) -> None:
        if job.cancel_requested:
            job.transition(JobStatus.CANCELLED, phase="cancelled", progress=1.0)
            job.finished_at = self._now()
            job.error = {"code": ErrorCode.CANCELLED.value, "message": "Job cancelled", "retryable": False}
            self.jobs.save(job)
            return
        try:
            self._execute(job)
        except DomainError as exc:
            latest = self.jobs.get(job.job_id) or job
            if exc.code == ErrorCode.CANCELLED:
                latest.transition(JobStatus.CANCELLED, phase="cancelled", progress=1.0)
            else:
                latest.transition(JobStatus.FAILED, phase="failed", progress=1.0)
            latest.error = {
                "code": exc.code.value,
                "message": exc.message,
                "retryable": exc.code.retryable,
                "details": exc.details,
            }
            latest.finished_at = self._now()
            self.jobs.save(latest)
        except Exception as exc:
            latest = self.jobs.get(job.job_id) or job
            latest.transition(JobStatus.FAILED, phase="failed", progress=1.0)
            latest.error = {"code": ErrorCode.INTERNAL_ERROR.value, "message": str(exc), "retryable": True}
            latest.finished_at = self._now()
            self.jobs.save(latest)

    def _execute(self, job: Job) -> None:
        body = {**job.inputs, **job.parameters, "seed": job.seed}
        steps, refs, bundles = plan_operation(job.operation, body)
        self._assert_bundles(bundles)
        job.phase = "loading_profile"
        job.progress = 0.08
        self.jobs.save(job)

        lease = self.scheduler.acquire(job.job_id)
        try:
            if not self.client.is_ready():
                raise DomainError(ErrorCode.COMFYUI_UNAVAILABLE, "ComfyUI is not reachable")
            uploaded = self._upload_refs(job, refs)
            outputs: list[str] = []
            previous_name: str | None = None
            base_name: str | None = None
            total = max(1, len(steps))
            for index, step in enumerate(steps):
                self._check_cancel(job.job_id)
                job.phase = f"rendering {index + 1}/{total}"
                job.progress = 0.15 + (index / total) * 0.7
                self.jobs.save(job)
                missing = self.client.missing_nodes(list(step.required_nodes))
                if missing:
                    raise DomainError(
                        ErrorCode.COMFYUI_UNAVAILABLE,
                        f"ComfyUI is missing nodes: {', '.join(missing)}",
                    )
                plan = step_to_plan(
                    step,
                    uploaded=uploaded,
                    previous_name=previous_name,
                    base_name=base_name,
                )
                graph = qwen_graph.build(plan)
                files = self._run_graph(job.job_id, graph)
                for item in files:
                    raw, mime = self.client.fetch_media(item["filename"], item["subfolder"], item["folder_type"])
                    record = self.assets.write(raw, mime_type=mime, filename=item["filename"])
                    outputs.append(record["asset_id"])
                    reuploaded = self.client.upload_image(raw, f"{job.job_id}_step{index}.png")
                    previous_name = str(reuploaded.get("name") or reuploaded.get("filename") or f"{job.job_id}_step{index}.png")
                    if base_name is None:
                        base_name = previous_name
            job = self.jobs.get(job.job_id) or job
            self._check_cancel(job.job_id)
            job.phase = "saving"
            job.progress = 0.95
            job.asset_ids = outputs
            job.transition(JobStatus.SUCCEEDED, phase="done", progress=1.0)
            job.finished_at = self._now()
            self.jobs.save(job)
        finally:
            self.scheduler.release(lease)

    def _assert_bundles(self, bundles: list[str]) -> None:
        if not bundles:
            return
        status = self._disk_status() if self._disk_status else {}
        missing = [mid for mid in bundles if status.get(mid, "complete") not in {"complete", "ok"}]
        # If disk_status is empty (tests / no scanner), skip.
        if not status:
            return
        if missing:
            raise DomainError(
                ErrorCode.MODEL_NOT_AVAILABLE,
                f"Required models not on disk: {', '.join(missing)}",
                {"bundles": missing},
            )

    def _upload_refs(self, job: Job, refs: dict[str, str]) -> dict[str, str]:
        uploaded: dict[str, str] = {}
        for key, asset_id in refs.items():
            raw, _record = self.assets.read(asset_id)
            name = f"{job.job_id}_{key}.png"
            result = self.client.upload_image(raw, name)
            uploaded[key] = str(result.get("name") or result.get("filename") or name)
        return uploaded

    def _check_cancel(self, job_id: str) -> None:
        latest = self.jobs.get(job_id)
        if latest and latest.cancel_requested:
            try:
                self.client.interrupt()
            except Exception:
                pass
            raise DomainError(ErrorCode.CANCELLED, "Job cancelled")

    def _run_graph(self, job_id: str, graph: dict[str, Any]) -> list[dict[str, str]]:
        client_id = str(uuid.uuid4())
        try:
            queued = self.client.queue_prompt(graph, client_id=client_id)
        except Exception as exc:
            raise DomainError(ErrorCode.GENERATION_FAILED, str(exc)) from exc
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            raise DomainError(ErrorCode.GENERATION_FAILED, "ComfyUI did not return prompt_id")
        self._active_client_job = job_id
        deadline = time.monotonic() + IMAGE_TIMEOUT_SEC
        try:
            while True:
                self._check_cancel(job_id)
                if time.monotonic() > deadline:
                    try:
                        self.client.interrupt()
                    except Exception:
                        pass
                    raise DomainError(ErrorCode.GENERATION_TIMEOUT, "Generation timed out")
                try:
                    hist = self.client.get_history(prompt_id)
                except Exception as exc:
                    raise DomainError(ErrorCode.GENERATION_FAILED, str(exc)) from exc
                entry = hist.get(prompt_id)
                if not entry:
                    threading.Event().wait(2)
                    continue
                if _history_failed(entry):
                    status = entry.get("status") or {}
                    detail = str(status.get("messages") or "ComfyUI execution error")
                    raise DomainError(ErrorCode.GENERATION_FAILED, detail)
                files = _collect_files(entry.get("outputs") or {})
                if files:
                    return files
                threading.Event().wait(2)
        finally:
            self._active_client_job = None
