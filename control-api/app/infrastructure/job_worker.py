from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable

from ..application.image_ops import plan_job, step_to_plan
from ..domain.errors import DomainError, ErrorCode
from ..domain.jobs import Job, JobStatus
from .comfy import graphs as qwen_graph
from .comfy import ltx_graph
from .comfy import ltx_shot_graph
from .comfy import post_graph
from .assets import AssetStore
from .comfy_client import ComfyClient
from .gpu_scheduler import GpuScheduler
from .job_store import JobStore
from .. import config

IMAGE_TIMEOUT_SEC = 600
VIDEO_TIMEOUT_SEC = 1200


def _collect_files(outputs: dict[str, Any]) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    for out in outputs.values():
        for key in ("images", "videos", "gifs"):
            for item in out.get(key) or []:
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
        self._ltx_client: ComfyClient | None = None
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
        if self._ltx_client is not None:
            try:
                self._ltx_client.interrupt()
            except Exception:
                pass

    def _client_for(self, profile: str) -> ComfyClient:
        if profile == "comfy-ltx":
            if self._ltx_client is None:
                self._ltx_client = ComfyClient(port=config.COMFY_LTX_PORT)
            return self._ltx_client
        return self.client

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
        steps, refs, bundles = plan_job(job.operation, body)
        self._assert_bundles(bundles)
        job.phase = "loading_profile"
        job.progress = 0.08
        self.jobs.save(job)

        profile = steps[0].profile if steps else "comfyui"
        client = self._client_for(profile)
        lease = self.scheduler.acquire(job.job_id, profile=profile)
        try:
            if not client.is_ready():
                raise DomainError(ErrorCode.COMFYUI_UNAVAILABLE, "ComfyUI is not reachable")
            upload_cache: dict[tuple[str, str], str] = {}
            uploaded = self._upload_refs(job, refs, client, cache=upload_cache)
            outputs: list[str] = []
            previous_name: str | None = None
            previous_raw: bytes | None = None
            previous_mime: str | None = None
            base_name: str | None = None
            total = max(1, len(steps))
            for index, step in enumerate(steps):
                if step.profile != lease.profile:
                    lease = self.scheduler.switch(lease, step.profile)
                    client = self._client_for(step.profile)
                    if not client.is_ready():
                        raise DomainError(ErrorCode.COMFYUI_UNAVAILABLE, "ComfyUI is not reachable")
                    uploaded = self._upload_refs(job, refs, client, cache=upload_cache)
                    if previous_name and previous_raw and previous_mime:
                        previous_name = self._upload_media(client, previous_raw, previous_mime, f"{job.job_id}_prev")
                        uploaded["__previous__"] = previous_name
                self._check_cancel(job.job_id, client)
                job.phase = f"rendering {index + 1}/{total}"
                job.progress = 0.15 + (index / total) * 0.7
                self.jobs.save(job)
                missing = client.missing_nodes(list(step.required_nodes))
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
                if step.mode in {"i2v", "t2v"}:
                    graph = ltx_graph.build(plan)
                elif step.mode == "shot":
                    graph = ltx_shot_graph.build(plan)
                elif step.mode == "post":
                    graph = post_graph.build(plan)
                else:
                    graph = qwen_graph.build(plan)
                timeout = VIDEO_TIMEOUT_SEC if step.mode in {"i2v", "t2v", "shot", "post"} else IMAGE_TIMEOUT_SEC
                files = self._run_graph(job.job_id, graph, client, timeout_sec=timeout)
                for item in files:
                    raw, mime = client.fetch_media(item["filename"], item["subfolder"], item["folder_type"])
                    record = self.assets.write(raw, mime_type=mime, filename=item["filename"])
                    outputs.append(record["asset_id"])
                    previous_raw, previous_mime = raw, mime
                    previous_name = self._upload_media(client, raw, mime, f"{job.job_id}_step{index}")
                    uploaded["__previous__"] = previous_name
                    if mime.startswith("image/") and base_name is None:
                        base_name = previous_name
            job = self.jobs.get(job.job_id) or job
            self._check_cancel(job.job_id, client)
            job.phase = "saving"
            job.progress = 0.95
            job.asset_ids = outputs
            warnings = [note for step in steps for note in step.warnings]
            if warnings:
                job.parameters = {**job.parameters, "warnings": warnings}
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

    def _upload_media(self, client: ComfyClient, raw: bytes, mime: str, stem: str) -> str:
        video = mime.startswith("video/")
        name = f"{stem}.mp4" if video else f"{stem}.png"
        result = client.upload_video(raw, name) if video else client.upload_image(raw, name)
        return str(result.get("name") or result.get("filename") or name)

    def _upload_refs(
        self,
        job: Job,
        refs: dict[str, str],
        client: ComfyClient,
        cache: dict[tuple[str, str], str] | None = None,
    ) -> dict[str, str]:
        uploaded: dict[str, str] = {}
        client_key = f"{client.lan_ip}:{client.port}"
        for key, asset_id in refs.items():
            cache_key = (client_key, asset_id)
            if cache is not None and cache_key in cache:
                uploaded[key] = cache[cache_key]
                continue
            raw, record = self.assets.read(asset_id)
            mime = str(record.get("mime_type") or "")
            name = self._upload_media(client, raw, mime, f"{job.job_id}_{key}")
            uploaded[key] = name
            if cache is not None:
                cache[cache_key] = name
        return uploaded

    def _check_cancel(self, job_id: str, client: ComfyClient | None = None) -> None:
        latest = self.jobs.get(job_id)
        if latest and latest.cancel_requested:
            active = client or self.client
            try:
                active.interrupt()
            except Exception:
                pass
            raise DomainError(ErrorCode.CANCELLED, "Job cancelled")

    def _run_graph(
        self,
        job_id: str,
        graph: dict[str, Any],
        client: ComfyClient | None = None,
        *,
        timeout_sec: int = IMAGE_TIMEOUT_SEC,
    ) -> list[dict[str, str]]:
        active = client or self.client
        client_id = str(uuid.uuid4())
        try:
            queued = active.queue_prompt(graph, client_id=client_id)
        except Exception as exc:
            raise DomainError(ErrorCode.GENERATION_FAILED, str(exc)) from exc
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            raise DomainError(ErrorCode.GENERATION_FAILED, "ComfyUI did not return prompt_id")
        self._active_client_job = job_id
        deadline = time.monotonic() + timeout_sec
        try:
            while True:
                self._check_cancel(job_id, active)
                if time.monotonic() > deadline:
                    try:
                        active.interrupt()
                    except Exception:
                        pass
                    raise DomainError(ErrorCode.GENERATION_TIMEOUT, "Generation timed out")
                try:
                    hist = active.get_history(prompt_id)
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
