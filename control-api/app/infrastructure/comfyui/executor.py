from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from ... import comfy_client, ltx_graph, qwen_graph
from ...application.ports import ExecutionOutput, ExecutionRequest, ExecutionResult
from ...domain.errors import DomainError, ErrorCode
from .plans import build_plan


def _collect_files(outputs: dict[str, Any]) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    for out in outputs.values():
        for key in ("images", "gifs", "videos"):
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


class ComfyUIExecutor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, tuple[Any, str]] = {}

    def health(self) -> dict[str, Any]:
        return {
            "comfy": comfy_client.is_ready(),
            "comfy_ltx": comfy_client.ltx_is_ready(),
        }

    def cancel(self, job_id: str) -> None:
        with self._lock:
            pair = self._active.get(job_id)
        if not pair:
            return
        client, _prompt_id = pair
        try:
            client.interrupt()
        except Exception:
            pass

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        workflow = request.workflow
        job = request.job
        start = time.monotonic()
        client = comfy_client.ltx if workflow.builder.startswith("ltx_") else comfy_client.default
        if not client.is_ready():
            raise DomainError(ErrorCode.COMFYUI_UNAVAILABLE, "ComfyUI is not reachable")

        plan = dict(request.plan)
        if workflow.builder == "qwen_upscale":
            image_bytes = request.input_files.get("image") or request.input_files.get("start_image")
            if not image_bytes:
                raise DomainError(ErrorCode.INVALID_REQUEST, "An input image asset is required")
            uploaded = client.upload_image(image_bytes, f"{job.job_id}.png")
            plan["image_name"] = uploaded.get("name") or uploaded.get("filename") or f"{job.job_id}.png"
        if workflow.builder in {"qwen_edit", "qwen_control", "qwen_layered"}:
            self._upload_qwen_assets(client, job.job_id, request.input_files, plan, workflow.builder)
        if workflow.builder.startswith("ltx_"):
            self._upload_ltx_assets(client, job.job_id, request.input_files, plan)

        candidate_count = 1
        if workflow.supports_batch:
            candidate_count = int(job.inputs.get("candidate_count") or job.parameters.get("candidate_count") or 1)
            candidate_count = max(1, min(8, candidate_count))

        outputs: list[ExecutionOutput] = []
        seed0 = int(plan.get("seed") or 0)
        for index in range(candidate_count):
            if request.cancel_check and request.cancel_check():
                raise DomainError(ErrorCode.CANCELLED, "Job cancelled")
            item_plan = dict(plan)
            item_plan["seed"] = seed0 + index
            if request.on_phase:
                request.on_phase("generation.started", 0.1 + (index / max(candidate_count, 1)) * 0.7)
            graph = self._build_graph(workflow.builder, item_plan)
            outputs.extend(self._run_graph(client, job.job_id, graph, request, workflow.resource.timeout_seconds))

        elapsed = int((time.monotonic() - start) * 1000)
        model_ids = list(workflow.required_models)
        if workflow.builder.startswith("ltx_") and plan.get("refine"):
            model_ids.append("ltx-2.5-studio")
        return ExecutionResult(outputs=outputs, model_ids=model_ids, seed=seed0, duration_ms=elapsed, extra={"plan": plan})

    def _build_graph(self, builder: str, plan: dict[str, Any]) -> dict[str, Any]:
        if builder == "qwen_txt2img":
            return qwen_graph.build_txt2img(plan)
        if builder == "qwen_edit":
            return qwen_graph.build_edit(plan)
        if builder == "qwen_control":
            return qwen_graph.build_control(plan)
        if builder == "qwen_layered":
            missing = [name for name in qwen_graph.LAYERED_REQUIRED_NODES if not comfy_client.has_node(name)]
            if missing:
                raise DomainError(
                    ErrorCode.WORKFLOW_INVALID,
                    "Qwen-Image-Layered nodes are missing on ComfyUI "
                    f"({', '.join(missing)}). Update primary ComfyUI to a release that includes layered nodes.",
                )
            return qwen_graph.build_layered(plan)
        if builder == "qwen_upscale":
            return qwen_graph.build_upscale(plan)
        if builder.startswith("ltx_"):
            mode = {
                "ltx_t2v": "t2v",
                "ltx_i2v": "i2v",
                "ltx_a2v": "a2v",
                "ltx_flf2v": "flf2v",
                "ltx_lipsync": "lipsync",
                "ltx_motion_transfer": "motion_transfer",
            }.get(builder)
            if not mode:
                raise DomainError(ErrorCode.WORKFLOW_INVALID, f"Unknown builder {builder}")
            return ltx_graph.build({**plan, "mode": mode})
        raise DomainError(ErrorCode.WORKFLOW_INVALID, f"Unknown builder {builder}")

    def _upload_qwen_assets(
        self,
        client: Any,
        job_id: str,
        files: dict[str, bytes],
        plan: dict[str, Any],
        builder: str,
    ) -> None:
        names: list[str] = []
        for index in range(3):
            raw = files.get(f"reference_images_{index}")
            if not raw:
                continue
            uploaded = client.upload_image(raw, f"{job_id}_ref{index}.png")
            names.append(str(uploaded.get("name") or uploaded.get("filename") or f"{job_id}_ref{index}.png"))
        primary = files.get("image") or files.get("control_image")
        if primary:
            uploaded = client.upload_image(primary, f"{job_id}.png")
            name = str(uploaded.get("name") or uploaded.get("filename") or f"{job_id}.png")
            if name not in names:
                names.insert(0, name)
        if not names:
            raise DomainError(ErrorCode.INVALID_REQUEST, "An input image asset is required")
        plan["image_name"] = names[0]
        plan["image_names"] = names
        if builder == "qwen_edit" and len(names) < 1:
            raise DomainError(ErrorCode.INVALID_REQUEST, "image.edit requires at least one reference image")
        if builder == "qwen_control" and not (files.get("control_image") or files.get("image")):
            raise DomainError(ErrorCode.INVALID_REQUEST, "image.controlled requires control_image")

    def _upload_ltx_assets(self, client: Any, job_id: str, files: dict[str, bytes], plan: dict[str, Any]) -> None:
        mapping = {
            "start_image": ("image_name", client.upload_image, f"{job_id}_start.png"),
            "image": ("image_name", client.upload_image, f"{job_id}.png"),
            "end_image": ("end_image_name", client.upload_image, f"{job_id}_end.png"),
            "middle_image": ("middle_image_name", client.upload_image, f"{job_id}_mid.png"),
            "audio": ("audio_name", client.upload_image, f"{job_id}.wav"),
            "reference_video": ("video_name", client.upload_video, f"{job_id}.mp4"),
        }
        for key, (plan_key, upload, filename) in mapping.items():
            raw = files.get(key)
            if not raw:
                continue
            uploaded = upload(raw, filename)
            plan[plan_key] = uploaded.get("name") or uploaded.get("filename") or filename

    def _run_graph(
        self,
        client: Any,
        job_id: str,
        graph: dict[str, Any],
        request: ExecutionRequest,
        timeout_sec: int,
    ) -> list[ExecutionOutput]:
        client_id = str(uuid.uuid4())
        try:
            queued = client.queue_prompt(graph, client_id=client_id)
        except Exception as exc:
            raise DomainError(ErrorCode.COMFYUI_ERROR, str(exc)) from exc
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            raise DomainError(ErrorCode.COMFYUI_ERROR, "ComfyUI did not return prompt_id")
        with self._lock:
            self._active[job_id] = (client, prompt_id)
        deadline = time.monotonic() + timeout_sec
        try:
            while True:
                if request.cancel_check and request.cancel_check():
                    try:
                        client.interrupt()
                    except Exception:
                        pass
                    raise DomainError(ErrorCode.CANCELLED, "Job cancelled")
                if time.monotonic() > deadline:
                    raise DomainError(ErrorCode.GENERATION_TIMEOUT, "Generation timed out")
                try:
                    hist = client.get_history(prompt_id)
                except Exception as exc:
                    raise DomainError(ErrorCode.COMFYUI_ERROR, str(exc)) from exc
                entry = hist.get(prompt_id)
                if not entry:
                    threading.Event().wait(2)
                    continue
                if _history_failed(entry):
                    status = entry.get("status") or {}
                    raise DomainError(ErrorCode.COMFYUI_ERROR, str(status.get("messages") or "ComfyUI execution error"))
                files = _collect_files(entry.get("outputs") or {})
                if files:
                    return self._fetch_outputs(client, files)
                threading.Event().wait(2)
        finally:
            with self._lock:
                self._active.pop(job_id, None)

    def _fetch_outputs(self, client: Any, files: list[dict[str, str]]) -> list[ExecutionOutput]:
        outputs: list[ExecutionOutput] = []
        for item in files:
            raw, mime = client.fetch_media(item["filename"], item["subfolder"], item["folder_type"])
            kind = "VIDEO" if mime.startswith("video/") else "IMAGE"
            outputs.append(ExecutionOutput(data=raw, mime_type=mime, artifact_type=kind))
        return outputs
