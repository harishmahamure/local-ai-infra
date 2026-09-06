"""LTX-2.5 video + audio job orchestration: direct params -> ComfyUI-ltx execution."""

from __future__ import annotations

import base64
import copy
import json
import os
import random
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import comfy_client, config, job_progress, ltx_graph, ltx_loras, ltx_presets, runtime

JOBS_PATH = Path(config.LOGS) / "ltx-video-jobs.json"
JOBS_ROOT = Path(config.LOGS) / "jobs"
GENERATE_JOBS_ROOT = Path(config.LOGS) / "jobs"
LTX_BUNDLE = "ltx-2.5-distilled"
LTX_STUDIO_BUNDLE = "ltx-2.5-studio"
LTX_PROMPT_ENHANCER_BUNDLE = "ltx-2.5-prompt-enhancer"
LTX_IC_UNION_BUNDLE = ltx_loras.IC_UNION_BUNDLE
LTX_IC_LIPDUB_BUNDLE = ltx_loras.IC_LIPDUB_BUNDLE
LTX_IC_MOTION_BUNDLE = ltx_loras.IC_MOTION_BUNDLE
LTX_BUNDLES = [
    LTX_BUNDLE,
    LTX_STUDIO_BUNDLE,
    LTX_PROMPT_ENHANCER_BUNDLE,
    LTX_IC_UNION_BUNDLE,
    LTX_IC_LIPDUB_BUNDLE,
    LTX_IC_MOTION_BUNDLE,
]
LTX_MODES = ("t2v", "i2v", "a2v", "flf2v", "lipsync", "motion_transfer")
MAX_DURATION_SECONDS = ltx_graph.MAX_DURATION_SECONDS
_PROGRESS_ENDPOINT = "GET /api/v1/ltx-video/{jobId}"

_LTX_OVERRIDE_KEYS = (
    "width",
    "height",
    "length",
    "fps",
    "steps",
    "video_cfg",
    "audio_cfg",
    "negative_prompt",
    "sampler_name",
    "max_shift",
    "base_shift",
    "terminal",
    "stretch",
    "strength",
    "refine",
    "refine_steps",
    "refine_denoise",
    "tiled_decode",
    "tile_size",
    "tile_overlap",
    "temporal_size",
    "temporal_overlap",
    "weight_dtype",
    "prompt_enhance",
    "prompt_enhance_max_length",
    "camera_motion",
    "ic_lora",
    "control_type",
    "lora_strength",
    "ic_lora_strength",
)


class LtxVideoError(Exception):
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
    JOBS_PATH.write_text(json.dumps(data, indent=2) + "\n")


def _job_videos_dir(job_id: str) -> Path:
    path = JOBS_ROOT / job_id / "videos"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _video_url(job_id: str, filename: str) -> str:
    return f"/api/v1/ltx-video/{job_id}/videos/{filename}"


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


def _ensure_ltx_checkpoints() -> None:
    script = config.LINK_LTX_CHECKPOINTS_SH
    if not script.is_file():
        return
    env = {
        **os.environ,
        "COMFYUI_MODELS": os.environ.get("COMFYUI_MODELS", str(Path.home() / "ComfyUI" / "models")),
    }
    subprocess.run(["bash", str(script)], check=False, env=env)


def _ensure_ltx_loaded() -> None:
    try:
        runtime.wait_for_profile("comfy-ltx", timeout_sec=90.0)
    except RuntimeError as exc:
        raise LtxVideoError("COMFY_LTX_NOT_LOADED", str(exc), 409) from exc
    if not comfy_client.ltx_is_ready():
        raise LtxVideoError(
            "COMFY_LTX_NOT_LOADED",
            "ComfyUI LTX is not responding on port 8189",
            409,
        )


def _start_comfy_ltx() -> None:
    status = runtime.get_status()
    if status.get("profile") == "comfyui-ltx" and status.get("loadState") == "LOADED":
        return
    runtime.start_profile("comfy-ltx")
    runtime.wait_for_profile("comfy-ltx", timeout_sec=90.0)


def _resolve_image_name(image_field: str | None) -> str | None:
    if not image_field:
        return None
    raw, suggested = comfy_client.decode_image_field(image_field)
    if raw:
        upload = comfy_client.ltx.upload_image(raw, suggested)
        return upload.get("name") or suggested
    return suggested


def _resolve_video_name(video_field: str | None) -> str | None:
    if not video_field:
        return None
    raw, suggested = comfy_client.decode_video_field(video_field)
    if raw:
        upload = comfy_client.ltx.upload_video(raw, suggested)
        return upload.get("name") or suggested
    return suggested


def _resolve_audio_name(audio_field: str | None) -> str | None:
    if not audio_field:
        return None
    raw, suggested = comfy_client.decode_audio_field(audio_field)
    if raw:
        upload = comfy_client.ltx.upload_image(raw, suggested)
        return upload.get("name") or suggested
    return suggested


def _resolve_image_field(
    *,
    image: str | None,
    source_job_id: str | None,
    source_filename: str | None,
) -> str | None:
    if image:
        return image
    if source_job_id and source_filename:
        if ".." in source_filename or "/" in source_filename or "\\" in source_filename:
            raise LtxVideoError("VALIDATION_ERROR", "Invalid source filename", 422)
        path = GENERATE_JOBS_ROOT / source_job_id / "images" / source_filename
        if not path.is_file():
            raise LtxVideoError("SOURCE_NOT_FOUND", f"Source image not found: {path.name}", 404)
        raw = path.read_bytes()
        ext = path.suffix.lstrip(".") or "png"
        b64 = base64.b64encode(raw).decode("ascii")
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        return f"data:image/{mime};base64,{b64}"
    return None


def _apply_duration(plan: dict[str, Any], duration: float | None) -> dict[str, Any]:
    out = copy.deepcopy(plan)
    if duration is None:
        return out
    fps = float(out.get("fps", ltx_graph.DEFAULT_FPS))
    capped = min(float(duration), MAX_DURATION_SECONDS)
    out["length"] = ltx_graph.duration_to_length(capped, fps)
    return out


def _build_plan(job: dict[str, Any]) -> dict[str, Any]:
    explicit = job.get("explicitPlan")
    if explicit:
        plan = copy.deepcopy(explicit)
        plan["prompt"] = plan.get("prompt") or job.get("prompt") or ""
        plan.setdefault(
            "audio_prompt",
            job.get("audioPrompt") or "silent, no sound, no music, no background audio",
        )
        plan["mode"] = job.get("mode") or plan.get("mode") or "t2v"
        plan = _apply_lora_decision(plan, job)
        return _apply_duration(plan, job.get("duration"))

    mode = job.get("mode") or "t2v"
    orientation = job.get("orientation")
    speed = job.get("speed")
    preset_id = job.get("preset") or "ltx_reel"
    if orientation and not job.get("preset"):
        orient = ltx_presets.LTX_ORIENTATIONS.get(orientation)
        if orient:
            preset_id = orient["preset"]
    if speed == "quality" and not job.get("preset"):
        preset_id = "ltx_quality"
    elif speed == "fast" and not job.get("preset"):
        preset_id = "ltx_fast"
    overrides = {k: job[k] for k in _LTX_OVERRIDE_KEYS if job.get(k) is not None}
    if job.get("negativePrompt") is not None:
        overrides["negative_prompt"] = job["negativePrompt"]
    if job.get("videoCfg") is not None:
        overrides["video_cfg"] = job["videoCfg"]
    if job.get("audioCfg") is not None:
        overrides["audio_cfg"] = job["audioCfg"]
    if job.get("samplerName") is not None:
        overrides["sampler_name"] = job["samplerName"]
    if job.get("maxShift") is not None:
        overrides["max_shift"] = job["maxShift"]
    if job.get("baseShift") is not None:
        overrides["base_shift"] = job["baseShift"]
    if job.get("refineSteps") is not None:
        overrides["refine_steps"] = job["refineSteps"]
    if job.get("refineDenoise") is not None:
        overrides["refine_denoise"] = job["refineDenoise"]
    if job.get("tiledDecode") is not None:
        overrides["tiled_decode"] = job["tiledDecode"]
    if job.get("promptEnhance") is not None:
        overrides["prompt_enhance"] = job["promptEnhance"]
    if job.get("cameraMotion") is not None:
        overrides["camera_motion"] = job["cameraMotion"]
    if job.get("icLora") is not None:
        overrides["ic_lora"] = job["icLora"]
    if job.get("controlType") is not None:
        overrides["control_type"] = job["controlType"]
    if job.get("loraStrength") is not None:
        overrides["lora_strength"] = job["loraStrength"]
    if job.get("icLoraStrength") is not None:
        overrides["ic_lora_strength"] = job["icLoraStrength"]

    plan = ltx_presets.resolve(
        preset_id,
        prompt=job.get("prompt") or "",
        audio_prompt=job.get("audioPrompt") or "",
        mode=mode,
        orientation=orientation,
        speed=speed,
        overrides=overrides,
        seed=job.get("seed"),
    )
    plan = _apply_lora_decision(plan, job)
    return _apply_duration(plan, job.get("duration"))


def _apply_lora_decision(plan: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(plan)
    decision = ltx_loras.decide(
        job.get("prompt") or out.get("prompt") or "",
        camera_motion=job.get("cameraMotion") or out.get("camera_motion") or "auto",
        ic_lora=job.get("icLora") or out.get("ic_lora") or "auto",
        control_type=job.get("controlType") or out.get("control_type") or "auto",
        has_reference_video=bool(job.get("videoField") or out.get("video_name")),
        lora_strength=float(job.get("loraStrength") or out.get("lora_strength") or 1.0),
        ic_lora_strength=float(job.get("icLoraStrength") or out.get("ic_lora_strength") or 1.0),
    )
    out.update(decision)
    hint = decision.get("prompt_hint")
    if hint:
        existing = (out.get("prompt") or "").rstrip()
        if hint.lower() not in existing.lower():
            out["prompt"] = f"{existing}, {hint}" if existing else hint
    mode = str(job.get("mode") or out.get("mode") or "t2v")
    if mode == "lipsync":
        out["ic_loras"] = [ltx_loras.explicit_ic_lora("lipdub", float(job.get("icLoraStrength") or 1.0))]
        out["ic_enabled"] = True
        out["ic_lora"] = "lipdub"
        out["ic_source"] = "override"
    elif mode == "motion_transfer":
        out["ic_loras"] = [ltx_loras.explicit_ic_lora("motion_track", float(job.get("icLoraStrength") or 1.0))]
        out["ic_enabled"] = True
        out["ic_lora"] = "motion_track"
        out["ic_guide_raw"] = True
        out["ic_source"] = "override"
    out = _drop_uninstalled_auto_loras(out)
    return out


def _drop_uninstalled_auto_loras(plan: dict[str, Any]) -> dict[str, Any]:
    """Keep explicit LoRA choices; skip prompt-auto picks whose weights are not on disk."""
    models_root = _models_root()
    kept_loras = []
    for lora in plan.get("loras") or []:
        path = models_root / "loras" / str(lora.get("name") or "")
        if path.is_file():
            kept_loras.append(lora)
            continue
        plan["lora_skip"] = f"weights missing: {lora.get('name')}"
    plan["loras"] = kept_loras

    kept_ic = []
    for lora in plan.get("ic_loras") or []:
        path = models_root / "loras" / str(lora.get("name") or "")
        if path.is_file() or plan.get("ic_source") == "override":
            kept_ic.append(lora)
            continue
        plan["ic_skip"] = f"weights missing: {lora.get('name')}"
    plan["ic_loras"] = kept_ic
    plan["ic_enabled"] = bool(kept_ic)
    if not kept_ic:
        plan["ic_lora"] = None
        plan["control_type"] = None

    bundles: list[str] = []
    for lora in kept_ic:
        name = str(lora.get("name") or "")
        if "lipdub" in name:
            bundles.append(ltx_loras.IC_LIPDUB_BUNDLE)
        elif "motion-track" in name:
            bundles.append(ltx_loras.IC_MOTION_BUNDLE)
        else:
            bundles.append(ltx_loras.IC_UNION_BUNDLE)
    plan["lora_bundles"] = list(dict.fromkeys(bundles))
    return plan


def _validate_refine(plan: dict[str, Any]) -> None:
    if not plan.get("refine"):
        return
    models_root = Path(os.environ.get("COMFYUI_MODELS", Path.home() / "ComfyUI" / "models"))
    upscaler_path = models_root / "latent_upscale_models" / ltx_graph.M["latent_upscaler"]
    if not upscaler_path.is_file():
        raise LtxVideoError(
            "MODELS_MISSING",
            f"refine=true requires latent upscaler on disk: {upscaler_path.name}. "
            "Run `./bin/ai download ltx-2.5-studio`.",
            409,
        )


def _models_root() -> Path:
    return Path(os.environ.get("COMFYUI_MODELS", Path.home() / "ComfyUI" / "models"))


def _validate_loras(plan: dict[str, Any]) -> None:
    models_root = _models_root()
    for lora in plan.get("loras") or []:
        name = lora.get("name")
        if not name:
            continue
        path = models_root / "loras" / name
        if not path.is_file():
            raise LtxVideoError(
                "MODELS_MISSING",
                f"LoRA missing on disk: {name}.",
                409,
            )
    if not plan.get("ic_enabled"):
        return
    for lora in plan.get("ic_loras") or []:
        name = lora.get("name")
        if not name:
            continue
        path = models_root / "loras" / name
        if not path.is_file():
            name_s = str(name)
            if "lipdub" in name_s:
                bundle = ltx_loras.IC_LIPDUB_BUNDLE
            elif "motion-track" in name_s:
                bundle = ltx_loras.IC_MOTION_BUNDLE
            else:
                bundle = ltx_loras.IC_UNION_BUNDLE
            raise LtxVideoError(
                "MODELS_MISSING",
                f"IC-LoRA missing on disk: {name}. Run `./bin/ai download {bundle}`.",
                409,
            )
    if not comfy_client.ltx_has_node("LTXICLoRALoaderModelOnly"):
        raise LtxVideoError(
            "COMFY_NODE_MISSING",
            "ComfyUI LTX is missing LTXICLoRALoaderModelOnly — install ComfyUI-LTXVideo in ComfyUI-ltx.",
            409,
        )
    if not comfy_client.ltx_has_node("LTXAddVideoICLoRAGuide"):
        raise LtxVideoError(
            "COMFY_NODE_MISSING",
            "ComfyUI LTX is missing LTXAddVideoICLoRAGuide — install ComfyUI-LTXVideo in ComfyUI-ltx.",
            409,
        )
    control = str(plan.get("control_type") or "depth")
    if control == "depth" and not comfy_client.ltx_has_node("VideoDepthAnythingProcess"):
        raise LtxVideoError(
            "COMFY_NODE_MISSING",
            "Depth IC-LoRA needs VideoDepthAnythingProcess — install ComfyUI-Video-Depth-Anything.",
            409,
        )
    if control == "canny" and not comfy_client.ltx_has_node("CannyEdgePreprocessor"):
        raise LtxVideoError(
            "COMFY_NODE_MISSING",
            "Canny IC-LoRA needs CannyEdgePreprocessor (comfyui_controlnet_aux).",
            409,
        )
    if control == "pose" and not comfy_client.ltx_has_node("DWPreprocessor"):
        raise LtxVideoError(
            "COMFY_NODE_MISSING",
            "Pose IC-LoRA needs DWPreprocessor (comfyui_controlnet_aux).",
            409,
        )


def _validate_prompt_enhance(plan: dict[str, Any]) -> None:
    if not plan.get("prompt_enhance"):
        return
    models_root = Path(os.environ.get("COMFYUI_MODELS", Path.home() / "ComfyUI" / "models"))
    enhancer_path = models_root / "text_encoders" / ltx_graph.M["prompt_enhancer"]
    if not enhancer_path.is_file():
        raise LtxVideoError(
            "MODELS_MISSING",
            f"promptEnhance=true requires prompt enhancer on disk: {enhancer_path.name}. "
            "Run `./bin/ai download ltx-2.5-prompt-enhancer`.",
            409,
        )
    if not comfy_client.ltx_has_node("TextGenerateLTX2Prompt"):
        raise LtxVideoError(
            "COMFY_NODE_MISSING",
            "ComfyUI LTX is missing TextGenerateLTX2Prompt — update ComfyUI-ltx to v0.32+",
            409,
        )


def _collect_output_files(outputs: dict[str, Any]) -> list[dict[str, str]]:
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


def _persist_videos(job_id: str, history_entry: dict[str, Any]) -> list[dict[str, Any]]:
    videos: list[dict[str, Any]] = []
    outputs = history_entry.get("outputs") or {}
    out_dir = _job_videos_dir(job_id)

    for idx, item in enumerate(_collect_output_files(outputs)):
        filename = item["filename"]
        try:
            raw, mime = comfy_client.ltx.fetch_media(
                filename,
                item["subfolder"],
                item["folder_type"],
            )
        except Exception:
            continue
        stored_name = f"{idx:02d}_{Path(filename).name}"
        if not stored_name.endswith((".mp4", ".webm", ".mov", ".gif")):
            stored_name = f"{stored_name}.mp4" if mime == "video/mp4" else stored_name
        out_path = out_dir / stored_name
        out_path.write_bytes(raw)
        videos.append(
            {
                "filename": stored_name,
                "mime": mime,
                "url": _video_url(job_id, stored_name),
                "bytes": len(raw),
            }
        )
    return videos


def _history_entry_status(entry: dict[str, Any]) -> str:
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        return "failed"
    if entry.get("outputs") and _collect_output_files(entry["outputs"]):
        return "completed"
    return "running"


def _execute_ltx(
    job_id: str,
    plan: dict[str, Any],
    tracker: job_progress.ProgressTracker,
) -> list[dict[str, Any]]:
    graph, node_meta = ltx_graph.build_with_meta(plan)
    tracker.bind_nodes(node_meta)
    for optional_stage in ("prompt_enhance", "ic_guide", "refine"):
        tracker.skip_optional_stage(optional_stage)
    tracker.set_pipeline_stage("load_models", state="running", detail="Queueing LTX graph")
    client_id = str(uuid.uuid4())
    queued = comfy_client.ltx.queue_prompt(graph, client_id=client_id)
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise LtxVideoError("QUEUE_FAILED", "ComfyUI LTX did not return prompt_id", 500)

    _update_job(job_id, promptId=prompt_id, clientId=client_id, phase="running")

    while True:
        try:
            hist = comfy_client.ltx.get_history(prompt_id)
        except Exception as exc:
            raise LtxVideoError("COMFY_HISTORY_FAILED", str(exc), 500) from exc

        entry = hist.get(prompt_id)
        if not entry:
            threading.Event().wait(3)
            continue

        state = _history_entry_status(entry)
        if state == "failed":
            status = entry.get("status") or {}
            raise LtxVideoError(
                "COMFY_EXEC_FAILED",
                str(status.get("messages") or "ComfyUI LTX execution error"),
                500,
            )

        videos = _persist_videos(job_id, entry)
        if videos:
            tracker.complete_all()
            return videos

        threading.Event().wait(3)


def _run_job(job_id: str) -> None:
    store = _load_jobs()
    job = store["jobs"].get(job_id)
    if not job:
        return

    tracker = job_progress.ProgressTracker(job_progress.LTX_STAGES)
    try:
        mode = job.get("mode") or "t2v"
        image_field = job.get("imageField")
        plan = _build_plan(job)
        _validate_refine(plan)

        if plan.get("seed") is None:
            plan["seed"] = random.randint(0, 2**31 - 1)

        ready, missing = _bundles_ready(plan.get("bundles") or [LTX_BUNDLE])
        if not ready:
            raise LtxVideoError(
                "MODELS_MISSING",
                f"Required bundles not on disk: {', '.join(missing)}",
                409,
            )

        tracker.set_pipeline_stage("switching", state="running", detail="Starting ComfyUI LTX")
        _update_job(job_id, status="switching", phase="switching", plan=plan)
        _start_comfy_ltx()
        _ensure_ltx_loaded()
        _ensure_ltx_checkpoints()
        _validate_prompt_enhance(plan)
        _validate_loras(plan)

        if mode in {"i2v", "flf2v", "lipsync"} or (mode in {"a2v", "motion_transfer"} and image_field):
            image_name = _resolve_image_name(image_field)
            if mode in {"i2v", "flf2v", "lipsync"} and not image_name:
                raise LtxVideoError("IMAGE_UPLOAD_FAILED", "Start image upload to ComfyUI LTX failed", 500)
            if image_name:
                plan["image_name"] = image_name

        if job.get("endImageField"):
            end_name = _resolve_image_name(job.get("endImageField"))
            if mode == "flf2v" and not end_name:
                raise LtxVideoError("IMAGE_UPLOAD_FAILED", "Last-frame upload to ComfyUI LTX failed", 500)
            if end_name:
                plan["end_image_name"] = end_name

        if job.get("middleImageField"):
            mid_name = _resolve_image_name(job.get("middleImageField"))
            if mid_name:
                plan["middle_image_name"] = mid_name

        if job.get("audioField"):
            audio_name = _resolve_audio_name(job.get("audioField"))
            if mode in {"a2v", "lipsync"} and not audio_name:
                raise LtxVideoError("AUDIO_UPLOAD_FAILED", "Audio upload to ComfyUI LTX failed", 500)
            if audio_name:
                plan["audio_name"] = audio_name

        if plan.get("ic_enabled") and (mode == "motion_transfer" or job.get("videoField")):
            video_name = _resolve_video_name(job.get("videoField"))
            if mode == "motion_transfer" and not video_name:
                raise LtxVideoError(
                    "VIDEO_UPLOAD_FAILED",
                    "Reference video upload to ComfyUI LTX failed",
                    500,
                )
            if video_name:
                plan["video_name"] = video_name
            elif mode != "lipsync":
                raise LtxVideoError(
                    "VIDEO_UPLOAD_FAILED",
                    "Reference video upload to ComfyUI LTX failed",
                    500,
                )

        _update_job(job_id, status="running", phase="running", plan=plan)
        videos = _execute_ltx(job_id, plan, tracker)
        _update_job(
            job_id,
            status="completed",
            phase="done",
            videos=videos,
            finishedAt=_now(),
            error=None,
        )
    except LtxVideoError as exc:
        tracker.fail_current(exc.message)
        _update_job(job_id, status="failed", error=exc.message, finishedAt=_now(), phase="failed")
    except Exception as exc:
        tracker.fail_current(str(exc))
        _update_job(job_id, status="failed", error=str(exc), finishedAt=_now(), phase="failed")


def submit(
    *,
    mode: str = "t2v",
    prompt: str = "",
    audio_prompt: str | None = None,
    image: str | None = None,
    end_image: str | None = None,
    middle_image: str | None = None,
    audio: str | None = None,
    source_job_id: str | None = None,
    source_filename: str | None = None,
    duration: float | None = None,
    preset: str | None = None,
    orientation: str | None = None,
    speed: str | None = None,
    plan: dict[str, Any] | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    length: int | None = None,
    fps: float | None = None,
    steps: int | None = None,
    video_cfg: float | None = None,
    audio_cfg: float | None = None,
    negative_prompt: str | None = None,
    sampler_name: str | None = None,
    max_shift: float | None = None,
    base_shift: float | None = None,
    terminal: float | None = None,
    stretch: bool | None = None,
    strength: float | None = None,
    refine: bool | None = None,
    refine_steps: int | None = None,
    refine_denoise: float | None = None,
    tiled_decode: bool | None = None,
    prompt_enhance: bool | None = None,
    camera_motion: str | None = None,
    reference_video: str | None = None,
    ic_lora: str | None = None,
    control_type: str | None = None,
    lora_strength: float | None = None,
    ic_lora_strength: float | None = None,
) -> dict[str, Any]:
    mode = mode if mode in LTX_MODES else "t2v"
    if mode in {"i2v", "flf2v", "lipsync"} and not (image or (source_job_id and source_filename)):
        raise LtxVideoError(
            "IMAGE_REQUIRED",
            "image or sourceJobId+sourceFilename is required",
            422,
        )
    if mode == "flf2v" and not end_image:
        raise LtxVideoError("IMAGE_REQUIRED", "endImage is required for first/last-frame video", 422)
    if mode in {"a2v", "lipsync"} and not audio:
        raise LtxVideoError("AUDIO_REQUIRED", "audio is required", 422)
    if mode == "motion_transfer" and not reference_video:
        raise LtxVideoError("VIDEO_REQUIRED", "referenceVideo is required for motion transfer", 422)

    image_field = _resolve_image_field(
        image=image,
        source_job_id=source_job_id,
        source_filename=source_filename,
    )
    video_field = reference_video

    job_id = str(uuid.uuid4())
    _job_videos_dir(job_id)

    job = {
        "jobId": job_id,
        "status": "switching",
        "phase": "switching",
        "mode": mode,
        "prompt": prompt,
        "audioPrompt": audio_prompt,
        "duration": duration,
        "preset": preset,
        "orientation": orientation,
        "speed": speed,
        "imageField": image_field,
        "endImageField": end_image,
        "middleImageField": middle_image,
        "audioField": audio,
        "explicitPlan": copy.deepcopy(plan) if plan else None,
        "seed": seed,
        "width": width,
        "height": height,
        "length": length,
        "fps": fps,
        "steps": steps,
        "videoCfg": video_cfg,
        "audioCfg": audio_cfg,
        "negativePrompt": negative_prompt,
        "samplerName": sampler_name,
        "maxShift": max_shift,
        "baseShift": base_shift,
        "terminal": terminal,
        "stretch": stretch,
        "strength": strength,
        "refine": refine,
        "refineSteps": refine_steps,
        "refineDenoise": refine_denoise,
        "tiledDecode": tiled_decode,
        "promptEnhance": prompt_enhance,
        "cameraMotion": camera_motion,
        "videoField": video_field,
        "icLora": ic_lora,
        "controlType": control_type,
        "loraStrength": lora_strength,
        "icLoraStrength": ic_lora_strength,
        "plan": None,
        "promptId": None,
        "clientId": None,
        "createdAt": _now(),
        "finishedAt": None,
        "videos": [],
        "error": None,
        "storageDir": str(JOBS_ROOT / job_id / "videos"),
    }

    store = _load_jobs()
    store.setdefault("jobs", {})[job_id] = job
    _save_jobs(store)

    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()
    return _public_job(job)


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in job.items() if k not in ("imageField", "videoField", "endImageField", "middleImageField", "audioField")}
    return job_progress.attach(out, "ltx", _PROGRESS_ENDPOINT)


def get_job_video_path(job_id: str, filename: str) -> Path | None:
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        return None
    store = _load_jobs()
    if job_id not in store.get("jobs", {}):
        return None
    path = JOBS_ROOT / job_id / "videos" / filename
    if path.is_file():
        return path
    return None


def get_job(job_id: str) -> dict[str, Any]:
    store = _load_jobs()
    job = store.get("jobs", {}).get(job_id)
    if not job:
        raise LtxVideoError("JOB_NOT_FOUND", f"Unknown job: {job_id}", 404)
    return _public_job(job)


def get_capabilities() -> dict[str, Any]:
    bmap = _bundle_map()
    bundle_ready = bmap.get(LTX_BUNDLE, {}).get("status") == "complete"
    studio_ready = bmap.get(LTX_STUDIO_BUNDLE, {}).get("status") == "complete"
    enhancer_ready = bmap.get(LTX_PROMPT_ENHANCER_BUNDLE, {}).get("status") == "complete"
    enhancer_nodes = (
        comfy_client.ltx_has_node("TextGenerateLTX2Prompt") if comfy_client.ltx_is_ready() else False
    )
    prompt_enhance_ready = enhancer_ready and enhancer_nodes
    presets = []
    for p in ltx_presets.list_presets():
        ready = all(bmap.get(b, {}).get("status") == "complete" for b in p.get("bundles") or [])
        presets.append({**p, "ready": ready})

    nodes_ok = comfy_client.ltx_has_node("LTXVDualCFGGuider") if comfy_client.ltx_is_ready() else False
    refine_nodes = (
        comfy_client.ltx_has_node("LTXVLatentUpsampler")
        and comfy_client.ltx_has_node("SplitSigmasDenoise")
        if comfy_client.ltx_is_ready()
        else False
    )
    refine_ready = refine_nodes and studio_ready

    models_root = _models_root()
    ic_loras = []
    for spec in ltx_loras.IC_LORAS.values():
        path = models_root / "loras" / spec["file"]
        ic_loras.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "file": spec["file"],
                "bundle": spec["bundle"],
                "needsVideo": True,
                "ready": path.is_file(),
            }
        )
    ic_nodes = (
        comfy_client.ltx_has_node("LTXICLoRALoaderModelOnly")
        and comfy_client.ltx_has_node("LTXAddVideoICLoRAGuide")
        if comfy_client.ltx_is_ready()
        else False
    )
    ic_union_ready = any(item["id"] == "union" and item["ready"] for item in ic_loras)

    note = "Text or image to video with synchronized audio. Pass prompt and audioPrompt verbatim."
    if not bundle_ready:
        note += (
            " Weights missing: accept https://huggingface.co/Lightricks/LTX-2.5 license, set HF_TOKEN, "
            "then run `./bin/ai download ltx-2.5-distilled`."
        )
    elif not studio_ready:
        note += " For quality/refine (LTX Studio workflow), run `./bin/ai download ltx-2.5-studio`."
    if not prompt_enhance_ready:
        note += " Optional prompt enhancer (ComfyUI prompt_enhance): `./bin/ai download ltx-2.5-prompt-enhancer`."
    if not ic_union_ready:
        note += " Union IC-LoRA: `./bin/ai download ltx-iclora-union`."
    if not any(item["id"] == "lipdub" and item["ready"] for item in ic_loras):
        note += " LipDub: `./bin/ai download ltx-iclora-lipdub`."
    if not any(item["id"] == "motion_track" and item["ready"] for item in ic_loras):
        note += " Motion Track: `./bin/ai download ltx-iclora-motion-track`."

    return {
        "ready": bundle_ready and nodes_ok,
        "bundles": LTX_BUNDLES,
        "bundleReady": bundle_ready,
        "studioReady": studio_ready,
        "promptEnhancerReady": enhancer_ready,
        "promptEnhanceReady": prompt_enhance_ready,
        "nodesReady": nodes_ok,
        "refineReady": refine_ready,
        "icLoras": ic_loras,
        "icLoraReady": ic_union_ready and ic_nodes,
        "icNodesReady": ic_nodes,
        "cameraMotions": ltx_loras.list_camera_motions(),
        "controlTypes": ltx_loras.list_control_types(),
        "workflow": {
            "id": "ltx-2.5-quality-t2v",
            "label": "LTX Studio quality (2-stage refine)",
            "comfyBlueprint": "LTX-2.5-Quality-T2V.json",
            "preset": "ltx_studio",
            "speed": "quality",
            "defaults": {
                "refine": True,
                "tiledDecode": True,
                "samplerName": "euler_ancestral",
                "orientation": "landscape",
            },
        },
        "modes": list(LTX_MODES),
        "lipdubReady": any(item["id"] == "lipdub" and item["ready"] for item in ic_loras) and ic_nodes,
        "motionTrackReady": any(item["id"] == "motion_track" and item["ready"] for item in ic_loras) and ic_nodes,
        "maxDurationSeconds": MAX_DURATION_SECONDS,
        "presets": presets,
        "orientations": ltx_presets.list_orientations(),
        "speedModes": ltx_presets.list_speed_modes(),
        "parameters": ltx_presets.LTX_PARAMETER_RANGES,
        "executorProfile": "comfy-ltx",
        "model": "LTX-2.5 22B distilled int8 + Gemma4 TE (local)",
        "note": (
            note
            + " Standard and Quality use 2-stage LTX Studio refine (sharp). Fast is single-pass preview only. "
            "Use detailed prompts, or set promptEnhance=true (requires ltx-2.5-prompt-enhancer). "
            "cameraMotion appends camera language to the prompt (LTX-2 19B camera LoRAs are not used on 2.5). "
            "LTX-2.3 IC-LoRAs need a referenceVideo. Distilled LoRA is not applied on the distilled transformer."
        ),
    }
