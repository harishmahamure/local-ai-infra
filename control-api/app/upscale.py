"""RealESRGAN upscale jobs — ComfyUI only, no LLM planner."""

from __future__ import annotations

import base64
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import comfy_client, config, job_progress, presets, qwen_graph, runtime

JOBS_PATH = Path(config.LOGS) / "upscale-jobs.json"
JOBS_ROOT = Path(config.LOGS) / "jobs"
GENERATE_JOBS_ROOT = Path(config.LOGS) / "jobs"
UPSCALE_BUNDLE = "upscalers-esrgan"
VALID_SCALES = frozenset({2, 4})


class UpscaleError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jobs() -> dict[str, Any]:
    if JOBS_PATH.exists():
        try:
            return json.loads(JOBS_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {"jobs": {}}


def _save_jobs(data: dict[str, Any]) -> None:
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    disk_jobs: dict[str, Any] = {}
    for jid, job in data.get("jobs", {}).items():
        j = {k: v for k, v in job.items() if k != "images"}
        j["images"] = [
            {k: v for k, v in img.items() if k != "data"}
            for img in (job.get("images") or [])
        ]
        disk_jobs[jid] = j
    JOBS_PATH.write_text(json.dumps({"jobs": disk_jobs}, indent=2) + "\n")


def _job_images_dir(job_id: str) -> Path:
    path = JOBS_ROOT / job_id / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _image_url(job_id: str, filename: str) -> str:
    return f"/api/v1/upscale/{job_id}/images/{filename}"


def _update_job(job_id: str, **fields: Any) -> dict[str, Any]:
    store = _load_jobs()
    job = store["jobs"][job_id]
    job.update(fields)
    store["jobs"][job_id] = job
    _save_jobs(store)
    return job


def _bundle_map() -> dict[str, dict]:
    return {b["id"]: b for b in runtime.get_models().get("bundles", [])}


def _bundles_ready(bundle_ids: list[str]) -> tuple[bool, list[str]]:
    bmap = _bundle_map()
    missing = []
    for bid in bundle_ids:
        b = bmap.get(bid)
        if not b or b.get("status") != "complete":
            missing.append(bid)
    return len(missing) == 0, missing


def _ensure_comfy_loaded() -> None:
    status = runtime.get_status()
    profile = status.get("profile", "none")
    if profile != "comfyui" or status.get("loadState") != "LOADED":
        raise UpscaleError(
            "COMFY_NOT_LOADED",
            "ComfyUI profile is not loaded.",
            409,
        )
    if not comfy_client.is_ready():
        raise UpscaleError("COMFY_NOT_LOADED", "ComfyUI is not responding on port 8188", 409)


def _start_comfy() -> None:
    status = runtime.get_status()
    if status.get("profile") == "comfyui" and status.get("loadState") == "LOADED":
        return
    runtime.start_profile("comfy")


def _resolve_image_field(
    *,
    image: str | None,
    source_job_id: str | None,
    source_filename: str | None,
) -> str:
    if image:
        return image
    if source_job_id and source_filename:
        if ".." in source_filename or "/" in source_filename or "\\" in source_filename:
            raise UpscaleError("VALIDATION_ERROR", "Invalid source filename", 422)
        for subdir in ("images",):
            path = GENERATE_JOBS_ROOT / source_job_id / subdir / source_filename
            if path.is_file():
                raw = path.read_bytes()
                ext = path.suffix.lstrip(".") or "png"
                b64 = base64.b64encode(raw).decode("ascii")
                mime = "jpeg" if ext in ("jpg", "jpeg") else ext
                return f"data:image/{mime};base64,{b64}"
        raise UpscaleError("SOURCE_NOT_FOUND", f"Source image not found: {source_filename}", 404)
    raise UpscaleError("IMAGE_REQUIRED", "image or sourceJobId+sourceFilename is required", 422)


def _resolve_image_name(image_field: str) -> str:
    raw, suggested = comfy_client.decode_image_field(image_field)
    if raw:
        upload = comfy_client.upload_image(raw, suggested)
        return upload.get("name") or suggested
    if suggested:
        return suggested
    raise UpscaleError("IMAGE_UPLOAD_FAILED", "Could not decode or upload image", 422)


def _normalize_scale(scale: int) -> int:
    try:
        value = int(scale)
    except (TypeError, ValueError):
        raise UpscaleError("VALIDATION_ERROR", "scale must be 2 or 4", 422) from None
    if value not in VALID_SCALES:
        raise UpscaleError("VALIDATION_ERROR", "scale must be 2 or 4", 422)
    return value


def _build_plan(image_name: str, scale: int) -> dict[str, Any]:
    plan = presets.resolve(
        "upscale_only",
        refined_prompt="",
        overrides={"mode": "upscale"},
        image_name=image_name,
    )
    plan["upscale_scale"] = scale
    return plan


def _persist_outputs(job_id: str, history_entry: dict[str, Any]) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    outputs = history_entry.get("outputs") or {}
    img_idx = 0
    out_dir = _job_images_dir(job_id)

    for out in outputs.values():
        for img in out.get("images") or []:
            filename = img.get("filename")
            if not filename:
                continue
            subfolder = img.get("subfolder", "")
            folder_type = img.get("type", "output")
            try:
                raw, mime = comfy_client.fetch_image(filename, subfolder, folder_type)
            except Exception:
                continue

            stored_name = f"up_{img_idx:02d}_{Path(filename).name}"
            out_path = out_dir / stored_name
            out_path.write_bytes(raw)
            images.append(
                {
                    "index": img_idx,
                    "filename": stored_name,
                    "mime": mime,
                    "url": _image_url(job_id, stored_name),
                    "bytes": len(raw),
                }
            )
            img_idx += 1
    return images


def _history_entry_status(entry: dict[str, Any]) -> str:
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        return "failed"
    if entry.get("outputs"):
        for out in entry["outputs"].values():
            if out.get("images"):
                return "completed"
    return "running"


def _execute_upscale(job_id: str, plan: dict[str, Any]) -> list[dict[str, Any]]:
    graph = qwen_graph.build(plan)
    client_id = str(uuid.uuid4())
    queued = comfy_client.queue_prompt(graph, client_id=client_id)
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise UpscaleError("QUEUE_FAILED", "ComfyUI did not return prompt_id", 500)

    _update_job(job_id, promptId=prompt_id, clientId=client_id, phase="running")

    while True:
        try:
            hist = comfy_client.get_history(prompt_id)
        except Exception as exc:
            raise UpscaleError("COMFY_HISTORY_FAILED", str(exc), 500) from exc

        entry = hist.get(prompt_id)
        if not entry:
            threading.Event().wait(2)
            continue

        state = _history_entry_status(entry)
        if state == "failed":
            status = entry.get("status") or {}
            raise UpscaleError(
                "COMFY_EXEC_FAILED",
                str(status.get("messages") or "ComfyUI execution error"),
                500,
            )

        images = _persist_outputs(job_id, entry)
        if images:
            return images

        threading.Event().wait(2)


def _run_job(job_id: str) -> None:
    store = _load_jobs()
    job = store["jobs"].get(job_id)
    if not job:
        return

    try:
        scale = _normalize_scale(job.get("scale", 4))
        image_field = job.get("imageField") or ""

        ready, missing = _bundles_ready([UPSCALE_BUNDLE])
        if not ready:
            raise UpscaleError(
                "MODELS_MISSING",
                f"Required bundles not on disk: {', '.join(missing)}",
                409,
            )

        _update_job(job_id, status="switching", phase="switching")
        _start_comfy()
        _ensure_comfy_loaded()

        image_name = _resolve_image_name(image_field)
        plan = _build_plan(image_name, scale)
        _update_job(job_id, status="running", phase="running", plan=plan, scale=scale)

        images = _execute_upscale(job_id, plan)
        _update_job(
            job_id,
            status="completed",
            phase="done",
            images=images,
            finishedAt=_now(),
            error=None,
        )
    except UpscaleError as exc:
        _update_job(job_id, status="failed", error=exc.message, finishedAt=_now(), phase="failed")
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), finishedAt=_now(), phase="failed")


def submit(
    *,
    image: str | None = None,
    source_job_id: str | None = None,
    source_filename: str | None = None,
    scale: int = 4,
) -> dict[str, Any]:
    scale = _normalize_scale(scale)
    image_field = _resolve_image_field(
        image=image,
        source_job_id=source_job_id,
        source_filename=source_filename,
    )

    job_id = str(uuid.uuid4())
    _job_images_dir(job_id)

    job = {
        "jobId": job_id,
        "status": "switching",
        "phase": "switching",
        "scale": scale,
        "imageField": image_field,
        "plan": None,
        "promptId": None,
        "clientId": None,
        "createdAt": _now(),
        "finishedAt": None,
        "images": [],
        "error": None,
        "storageDir": str(JOBS_ROOT / job_id / "images"),
    }

    store = _load_jobs()
    store.setdefault("jobs", {})[job_id] = job
    _save_jobs(store)

    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()
    return job_progress.attach(job, "upscale", "GET /api/v1/upscale/{jobId}")


def get_job_image_path(job_id: str, filename: str) -> Path | None:
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        return None
    store = _load_jobs()
    if job_id not in store.get("jobs", {}):
        return None
    path = JOBS_ROOT / job_id / "images" / filename
    if path.is_file():
        return path
    return None


def get_job(job_id: str) -> dict[str, Any]:
    store = _load_jobs()
    job = store.get("jobs", {}).get(job_id)
    if not job:
        raise UpscaleError("JOB_NOT_FOUND", f"Unknown job: {job_id}", 404)
    return job_progress.attach(job, "upscale", "GET /api/v1/upscale/{jobId}")


def get_capabilities() -> dict[str, Any]:
    bmap = _bundle_map()
    bundle = bmap.get(UPSCALE_BUNDLE, {})
    return {
        "ready": bundle.get("status") == "complete",
        "bundle": UPSCALE_BUNDLE,
        "models": {
            "x2": "RealESRGAN_x2plus.pth",
            "x4": "RealESRGAN_x4plus.pth",
        },
        "scales": sorted(VALID_SCALES),
        "defaultScale": 4,
        "note": "Upload an image or reference a prior generate job output; ComfyUI runs RealESRGAN.",
    }
