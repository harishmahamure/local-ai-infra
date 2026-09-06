"""Image generation job orchestration: LLM plan -> ComfyUI execute."""

from __future__ import annotations

import base64
import copy
import json
import random
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import comfy_client, config, job_progress, planner, presets, qwen_graph, runtime
from .application import image_flows

JOBS_PATH = Path(config.LOGS) / "generate-jobs.json"
JOBS_ROOT = Path(config.LOGS) / "jobs"
MAX_BATCH_COUNT = 10
_FAST_GEN_MODES = frozenset({"txt2img", "bg_replace"})
_FAST_GEN_STEPS = 4
_FAST_GEN_CFG = 1.0
_STYLE_LORA_MARKERS = (
    "advertisement",
    "poster",
    "flat-cartoon",
    "flat_cartoon",
    "eligen",
    "realism",
)
_SLOW_PRESET_PREFIXES = ("infographic", "scene_", "character_master")


class GenerateError(Exception):
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


def _strip_image_payload(img: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in img.items() if k != "data"}


def _strip_item_for_disk(item: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in item.items() if k != "images"}
    out["images"] = [_strip_image_payload(img) for img in (item.get("images") or [])]
    return out


def _save_jobs(data: dict[str, Any]) -> None:
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    disk_jobs: dict[str, Any] = {}
    for jid, job in data.get("jobs", {}).items():
        j = {k: v for k, v in job.items() if k not in ("images", "items")}
        j["images"] = [_strip_image_payload(img) for img in (job.get("images") or [])]
        j["items"] = [_strip_item_for_disk(it) for it in (job.get("items") or [])]
        disk_jobs[jid] = j
    JOBS_PATH.write_text(json.dumps({"jobs": disk_jobs}, indent=2) + "\n")


def _job_images_dir(job_id: str) -> Path:
    path = JOBS_ROOT / job_id / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _image_url(job_id: str, filename: str) -> str:
    return f"/api/v1/generate/{job_id}/images/{filename}"


def _flatten_item_images(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for item in items:
        flat.extend(item.get("images") or [])
    return flat


def _count_item_outcomes(items: list[dict[str, Any]]) -> tuple[int, int]:
    completed = sum(1 for it in items if it.get("status") == "completed")
    failed = sum(1 for it in items if it.get("status") == "failed")
    return completed, failed


def _job_status_from_items(items: list[dict[str, Any]]) -> str:
    completed, failed = _count_item_outcomes(items)
    total = len(items)
    if completed > 0:
        return "completed"
    if failed == total:
        return "failed"
    return "running"


def _update_job(job_id: str, **fields: Any) -> dict[str, Any]:
    store = _load_jobs()
    job = store["jobs"][job_id]
    if "items" in fields:
        fields["images"] = _flatten_item_images(fields["items"])
        fields["completedCount"], fields["failedCount"] = _count_item_outcomes(fields["items"])
    job.update(fields)
    store["jobs"][job_id] = job
    _save_jobs(store)
    return job


def _has_style_lora(plan: dict[str, Any]) -> bool:
    for lora in plan.get("loras") or []:
        name = str(lora.get("name", "")).lower()
        if any(marker in name for marker in _STYLE_LORA_MARKERS):
            return True
    return False


def _uses_lightning_lora(plan: dict[str, Any]) -> bool:
    for lora in plan.get("loras") or []:
        name = str(lora.get("name", "")).lower()
        if "lightning" in name or "turbo" in name:
            return True
    return False


def _can_use_fast_path(plan: dict[str, Any]) -> bool:
    mode = plan.get("mode", "txt2img")
    if mode == "bg_replace":
        return True
    if mode not in _FAST_GEN_MODES:
        return False
    if _has_style_lora(plan):
        return False
    preset_id = str(plan.get("preset", "general"))
    if preset_id.startswith(_SLOW_PRESET_PREFIXES):
        return False
    return True


def _ensure_lightning_lora(plan: dict[str, Any]) -> None:
    if _uses_lightning_lora(plan):
        return
    loras = list(plan.get("loras") or [])
    loras.append({"name": presets.LORA["lightning"], "strength": 1.0})
    plan["loras"] = loras


def _apply_generation_options(
    plan: dict[str, Any],
    *,
    fast_gen: bool | None,
    steps: int | None,
) -> dict[str, Any]:
    """Apply fast-gen defaults or an explicit step override (None steps = auto)."""
    resolved = copy.deepcopy(plan)

    if steps is not None:
        resolved["steps"] = steps
        if steps <= 8 and resolved.get("mode") == "txt2img" and not _has_style_lora(resolved):
            _ensure_lightning_lora(resolved)
            resolved["cfg"] = min(float(resolved.get("cfg", _FAST_GEN_CFG)), _FAST_GEN_CFG)
        return resolved

    if fast_gen is not True or not _can_use_fast_path(resolved):
        return resolved

    if resolved.get("mode") == "bg_replace":
        resolved["steps"] = _FAST_GEN_STEPS
        resolved["cfg"] = _FAST_GEN_CFG
        return resolved

    _ensure_lightning_lora(resolved)
    resolved["steps"] = _FAST_GEN_STEPS
    resolved["cfg"] = _FAST_GEN_CFG
    return resolved


def _apply_upscale_option(plan: dict[str, Any], upscale: bool | None) -> dict[str, Any]:
    if upscale is None:
        return plan
    resolved = copy.deepcopy(plan)
    if resolved.get("mode") == "upscale":
        return resolved
    resolved["upscale"] = upscale
    return resolved


def _normalize_items(
    *,
    prompts: list[dict[str, Any]] | None,
    prompt: str,
    image: str | None,
    plan: dict[str, Any] | None,
    seed: int | None,
    width: int | None,
    height: int | None,
    count: int,
    fast_gen_mode: bool | None = None,
    steps: int | None = None,
    upscale: bool | None = None,
    flow: str | None = None,
    extra_images: list[str] | None = None,
    control_type: str | None = None,
    control_strength: float | None = None,
    layers: int | None = None,
) -> list[dict[str, Any]]:
    """Build normalized job items from prompts[] or prompt+count shorthand."""
    if prompts:
        if not prompts:
            raise GenerateError("VALIDATION_ERROR", "prompts array must not be empty", 422)
        if len(prompts) > MAX_BATCH_COUNT:
            raise GenerateError(
                "VALIDATION_ERROR",
                f"prompts array exceeds max batch size ({MAX_BATCH_COUNT})",
                422,
            )
        items: list[dict[str, Any]] = []
        for i, raw in enumerate(prompts):
            item_prompt = raw.get("prompt") or ""
            item_plan = raw.get("plan")
            item_flow = raw.get("flow") or flow
            if not item_prompt and not item_plan and item_flow != "layered":
                raise GenerateError(
                    "VALIDATION_ERROR",
                    f"prompts[{i}] requires prompt or plan",
                    422,
                )
            item_steps = raw.get("steps", steps)
            if item_steps == "auto":
                item_steps = None
            item_upscale = raw.get("upscale", upscale)
            if item_upscale == "auto":
                item_upscale = None
            item_fast = raw.get("fastGenMode", fast_gen_mode)
            if item_fast == "auto":
                item_fast = None
            item_images = [img for img in (raw.get("images") or extra_images or []) if img]
            items.append(
                {
                    "index": i,
                    "prompt": item_prompt,
                    "imageField": raw.get("image"),
                    "imageFields": item_images,
                    "explicitPlan": item_plan,
                    "flow": item_flow,
                    "controlType": raw.get("controlType", control_type),
                    "controlStrength": raw.get("controlStrength", control_strength),
                    "layers": raw.get("layers", layers),
                    "seed": raw.get("seed"),
                    "width": raw.get("width"),
                    "height": raw.get("height"),
                    "fastGenMode": item_fast,
                    "stepsOverride": item_steps,
                    "upscaleOverride": item_upscale,
                    "status": "pending",
                    "plan": None,
                    "images": [],
                    "error": None,
                }
            )
        return items

    if not prompt and not plan and flow != "layered":
        raise GenerateError("VALIDATION_ERROR", "prompt, plan, or prompts is required", 422)

    batch_count = max(1, min(int(count), MAX_BATCH_COUNT))
    extra = [img for img in (extra_images or []) if img]
    items = []
    for i in range(batch_count):
        item_seed = (int(seed) + i) if seed is not None else None
        items.append(
            {
                "index": i,
                "prompt": prompt,
                "imageField": image,
                "imageFields": extra,
                "explicitPlan": plan,
                "flow": flow,
                "controlType": control_type,
                "controlStrength": control_strength,
                "layers": layers,
                "seed": item_seed,
                "width": width,
                "height": height,
                "fastGenMode": fast_gen_mode,
                "stepsOverride": steps,
                "upscaleOverride": upscale,
                "status": "pending",
                "plan": None,
                "images": [],
                "error": None,
            }
        )
    return items


def _needs_llm_planning(items: list[dict[str, Any]]) -> bool:
    return any(not it.get("explicitPlan") and not it.get("flow") for it in items)


def _items_need_vision_planning(items: list[dict[str, Any]]) -> bool:
    for item in items:
        if item.get("explicitPlan") or item.get("flow"):
            continue
        if item.get("imageField"):
            return True
    return False


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


def _collect_required_bundles(items: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    for item in items:
        plan = item.get("plan") or item.get("explicitPlan") or {}
        for bid in plan.get("bundles") or []:
            seen.add(bid)
    return sorted(seen)


def _ensure_comfy_loaded() -> None:
    status = runtime.get_status()
    profile = status.get("profile", "none")
    if profile != "comfyui" or status.get("loadState") != "LOADED":
        raise GenerateError(
            "COMFY_NOT_LOADED",
            "ComfyUI profile is not loaded.",
            409,
        )
    if not comfy_client.is_ready():
        raise GenerateError("COMFY_NOT_LOADED", "ComfyUI is not responding on port 8188", 409)


def _start_planner(*, has_image: bool) -> str:
    return planner.start_planner_profile(has_image=has_image)


def _start_comfy() -> None:
    status = runtime.get_status()
    if status.get("profile") == "comfyui" and status.get("loadState") == "LOADED":
        return
    runtime.start_profile("comfy")


def _image_to_data_url(image_field: str) -> tuple[str | None, str | None]:
    """Return (data_url for planner, comfy_upload_name)."""
    if not image_field:
        return None, None
    raw, suggested = comfy_client.decode_image_field(image_field)
    if raw:
        b64 = base64.b64encode(raw).decode("ascii")
        ext = "png"
        if suggested.endswith(".jpg") or suggested.endswith(".jpeg"):
            ext = "jpeg"
        data_url = f"data:image/{ext};base64,{b64}"
        return data_url, None
    return None, suggested


def _resolve_image_name(image_field: str | None) -> str | None:
    if not image_field:
        return None
    _data_url, pre_uploaded = _image_to_data_url(image_field)
    if pre_uploaded:
        return pre_uploaded
    raw, suggested = comfy_client.decode_image_field(image_field)
    if raw:
        upload = comfy_client.upload_image(raw, suggested)
        return upload.get("name") or suggested
    return suggested


_CONTROL_PREPROCESSORS: dict[str, tuple[str, ...]] = {
    "pose": ("DWPreprocessor", "OpenposePreprocessor"),
    "depth": ("DepthAnythingV2Preprocessor", "MiDaS Depth Approximation"),
    "canny": ("Canny",),
}


def _control_preprocessor_available(control_type: str | None) -> bool:
    for node in _CONTROL_PREPROCESSORS.get(control_type or "pose", ()):
        if comfy_client.has_node(node):
            return True
    return False


def _resolve_depth_control_type() -> str | None:
    if comfy_client.has_node("DepthAnythingV2Preprocessor"):
        return "depth"
    if comfy_client.has_node("MiDaS Depth Approximation"):
        return "depth_midas"
    return None


def _adjust_plan_for_comfy(plan: dict[str, Any], *, has_image: bool) -> dict[str, Any]:
    """Downgrade unsupported control graphs; prefer img2img when preprocessors are missing."""
    adjusted = copy.deepcopy(plan)
    if not has_image or adjusted.get("mode") != "control":
        return adjusted

    control_type = adjusted.get("control_type") or "pose"
    if control_type == "depth":
        resolved = _resolve_depth_control_type()
        if resolved:
            adjusted["control_type"] = resolved
            return adjusted

    if _control_preprocessor_available(control_type):
        return adjusted

    # Pose/depth preprocessors are not installed on this ComfyUI box.
    adjusted["mode"] = "txt2img"
    adjusted.setdefault("denoise", 0.75)
    adjusted.pop("control_type", None)
    adjusted.pop("control_strength", None)
    notes = adjusted.setdefault("runtime_notes", [])
    if isinstance(notes, list):
        notes.append(
            f"Control mode ({control_type}) unavailable on ComfyUI; fell back to img2img."
        )
    return adjusted


def _apply_item_overrides(plan: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(plan)
    if item.get("seed") is not None:
        resolved["seed"] = int(item["seed"])
    elif resolved.get("seed") is None:
        resolved["seed"] = random.randint(0, 2**31 - 1)
    if item.get("width") is not None:
        resolved["width"] = item["width"]
    if item.get("height") is not None:
        resolved["height"] = item["height"]
    return resolved


def _plan_item(item: dict[str, Any]) -> dict[str, Any]:
    explicit = item.get("explicitPlan")
    prompt = item.get("prompt") or ""
    image_field = item.get("imageField")
    data_url, _ = _image_to_data_url(image_field) if image_field else (None, None)
    fast_gen = item.get("fastGenMode")
    steps_override = item.get("stepsOverride")
    upscale_override = item.get("upscaleOverride")
    if steps_override == "auto":
        steps_override = None
    if upscale_override == "auto":
        upscale_override = None

    if explicit:
        plan = copy.deepcopy(explicit)
        plan["prompt"] = plan.get("prompt") or prompt
        if steps_override is not None:
            plan = _apply_generation_options(
                plan,
                fast_gen=None,
                steps=steps_override,
            )
        plan = _apply_upscale_option(plan, upscale_override)
    elif item.get("flow"):
        extras = [f for f in (item.get("imageFields") or []) if f]
        image_count = (1 if image_field else 0) + len(extras)
        if image_field and extras and extras[0] == image_field:
            image_count = len(extras)
        try:
            plan = image_flows.resolve_flow_plan(
                str(item["flow"]),
                prompt=prompt,
                image_count=image_count,
                control_type=item.get("controlType"),
                control_strength=item.get("controlStrength"),
                layers=item.get("layers"),
                width=item.get("width"),
                height=item.get("height"),
                seed=item.get("seed"),
                upscale=upscale_override,
            )
        except image_flows.FlowError as exc:
            raise GenerateError("VALIDATION_ERROR", str(exc), 422) from exc
    else:
        plan = planner.plan(prompt=prompt, image_data_url=data_url, fast_gen=fast_gen is True)
        plan = _apply_generation_options(
            plan,
            fast_gen=fast_gen,
            steps=steps_override,
        )
        plan = _apply_upscale_option(plan, upscale_override)
    return _apply_item_overrides(plan, item)


def _item_image_fields(item: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    primary = item.get("imageField")
    extras = [f for f in (item.get("imageFields") or []) if f]
    if primary:
        fields.append(primary)
    for extra in extras:
        if extra not in fields:
            fields.append(extra)
    return fields


def _ensure_layered_nodes() -> None:
    missing = [name for name in qwen_graph.LAYERED_REQUIRED_NODES if not comfy_client.has_node(name)]
    if missing:
        raise GenerateError(
            "WORKFLOW_INVALID",
            "Qwen-Image-Layered nodes are missing on ComfyUI "
            f"({', '.join(missing)}). Update primary ComfyUI to a release that includes layered nodes.",
            409,
        )


def _persist_outputs(job_id: str, batch_index: int, history_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch ComfyUI outputs, write to jobs/{job_id}/images/, return metadata."""
    images: list[dict[str, Any]] = []
    outputs = history_entry.get("outputs") or {}
    img_idx = 0
    out_dir = _job_images_dir(job_id)

    for _nid, out in outputs.items():
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

            stored_name = f"{batch_index:04d}_{img_idx:02d}_{Path(filename).name}"
            out_path = out_dir / stored_name
            out_path.write_bytes(raw)
            images.append(
                {
                    "index": batch_index,
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


def _execute_comfy_batch(
    job_id: str,
    plan: dict[str, Any],
    batch_index: int,
    total: int,
) -> list[dict[str, Any]]:
    """Queue one graph, poll until done, persist outputs to disk."""
    graph = qwen_graph.build(plan)
    client_id = str(uuid.uuid4())
    queued = comfy_client.queue_prompt(graph, client_id=client_id)
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise GenerateError("QUEUE_FAILED", "ComfyUI did not return prompt_id", 500)

    _update_job(
        job_id,
        promptId=prompt_id,
        clientId=client_id,
        phase=f"running {batch_index + 1}/{total}",
    )

    while True:
        try:
            hist = comfy_client.get_history(prompt_id)
        except Exception as exc:
            raise GenerateError("COMFY_HISTORY_FAILED", str(exc), 500) from exc

        entry = hist.get(prompt_id)
        if not entry:
            threading.Event().wait(2)
            continue

        state = _history_entry_status(entry)
        if state == "failed":
            status = entry.get("status") or {}
            raise GenerateError(
                "COMFY_EXEC_FAILED",
                str(status.get("messages") or "ComfyUI execution error"),
                500,
            )

        images = _persist_outputs(job_id, batch_index, entry)
        if images:
            return images

        threading.Event().wait(2)


def _refresh_job(job: dict[str, Any]) -> dict[str, Any]:
    """Refresh a single in-flight ComfyUI prompt when worker is unavailable."""
    prompt_id = job.get("promptId")
    items = job.get("items") or []
    total = int(job.get("total") or len(items) or 1)
    if total > 1 or not prompt_id or job.get("status") in ("completed", "failed", "planning", "switching"):
        return job

    try:
        hist = comfy_client.get_history(prompt_id)
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["finishedAt"] = _now()
        return job

    entry = hist.get(prompt_id)
    if not entry:
        job["status"] = "running"
        return job

    state = _history_entry_status(entry)
    if state == "failed":
        job["status"] = "failed"
        job["error"] = str((entry.get("status") or {}).get("messages") or "ComfyUI execution error")
        job["finishedAt"] = _now()
        return job

    images = _persist_outputs(job.get("jobId", ""), 0, entry)
    if images and items:
        items[0]["status"] = "completed"
        items[0]["images"] = images
        job["items"] = items
        job["images"] = images
        job["completedCount"] = 1
        job["status"] = "completed"
        job["finishedAt"] = _now()
    elif images:
        job["status"] = "completed"
        job["images"] = images
        job["completedCount"] = 1
        job["finishedAt"] = _now()
    else:
        job["status"] = "running"
    return job


def _run_job(job_id: str) -> None:
    store = _load_jobs()
    job = store["jobs"].get(job_id)
    if not job:
        return

    items: list[dict[str, Any]] = copy.deepcopy(job.get("items") or [])
    total = len(items)

    try:
        # Phase A — plan all items while the planner profile stays loaded (one switch).
        needs_planning = _needs_llm_planning(items)
        _update_job(job_id, status="planning", phase=f"planning 0/{total}", items=items)
        if needs_planning:
            _start_planner(has_image=_items_need_vision_planning(items))

        for i, item in enumerate(items):
            item["status"] = "planning"
            _update_job(job_id, phase=f"planning {i + 1}/{total}", items=items)

            try:
                item["plan"] = _plan_item(item)
                item["status"] = "planned"
            except Exception as exc:
                item["status"] = "failed"
                item["error"] = str(exc)
            items[i] = item
            _update_job(job_id, items=items)

        planned_items = [it for it in items if it.get("plan")]
        if not planned_items:
            raise GenerateError("PLANNING_FAILED", "All items failed during planning", 500)

        all_bundles = _collect_required_bundles(planned_items)
        ready, missing = _bundles_ready(all_bundles)
        if not ready:
            raise GenerateError("MODELS_MISSING", f"Required bundles not on disk: {', '.join(missing)}", 409)

        # Phase B — execute all planned items while Comfy stays loaded (one profile switch).
        _update_job(job_id, status="switching", phase="switching", items=items)
        _start_comfy()
        _ensure_comfy_loaded()
        _update_job(job_id, status="running", phase=f"running 0/{total}", items=items)

        for i, item in enumerate(items):
            if item.get("status") == "failed" and not item.get("plan"):
                continue
            if not item.get("plan"):
                continue

            item["status"] = "running"
            _update_job(job_id, phase=f"running {i + 1}/{total}", items=items)

            try:
                plan_i = copy.deepcopy(item["plan"])
                image_fields = _item_image_fields(item)
                image_names = [_resolve_image_name(field) for field in image_fields]
                if any(field and not name for field, name in zip(image_fields, image_names)):
                    raise GenerateError(
                        "IMAGE_UPLOAD_FAILED",
                        "Reference image could not be uploaded to ComfyUI",
                        500,
                    )
                uploaded = [name for name in image_names if name]
                image_name = uploaded[0] if uploaded else None
                plan_i["image_name"] = image_name
                plan_i["image_names"] = uploaded
                if plan_i.get("mode") == "layered":
                    _ensure_layered_nodes()
                if item.get("flow") == "control":
                    if not image_name:
                        raise GenerateError("VALIDATION_ERROR", "control requires a guide image", 422)
                    plan_i = _adjust_plan_for_comfy(plan_i, has_image=True)
                    if plan_i.get("mode") != "control":
                        raise GenerateError(
                            "WORKFLOW_INVALID",
                            "Control preprocessor nodes are missing on ComfyUI",
                            409,
                        )
                else:
                    plan_i = _adjust_plan_for_comfy(plan_i, has_image=bool(image_name))
                batch_images = _execute_comfy_batch(job_id, plan_i, item["index"], total)
                item["images"] = batch_images
                item["status"] = "completed"
                item["error"] = None
            except GenerateError as exc:
                item["status"] = "failed"
                item["error"] = exc.message
            except Exception as exc:
                item["status"] = "failed"
                item["error"] = str(exc)

            items[i] = item
            completed, failed = _count_item_outcomes(items)
            final_status = _job_status_from_items(items)
            _update_job(
                job_id,
                items=items,
                status=final_status if (completed + failed) >= total else "running",
                phase="done" if (completed + failed) >= total else f"running {completed + failed}/{total}",
                finishedAt=_now() if (completed + failed) >= total else None,
                error=None if completed > 0 else job.get("error"),
            )

    except GenerateError as exc:
        _update_job(job_id, status="failed", error=exc.message, finishedAt=_now(), phase="failed", items=items)
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), finishedAt=_now(), phase="failed", items=items)


def submit(
    *,
    prompt: str = "",
    image: str | None = None,
    plan: dict[str, Any] | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    count: int = 1,
    prompts: list[dict[str, Any]] | None = None,
    fast_gen_mode: bool | None = None,
    steps: int | None = None,
    upscale: bool | None = None,
    flow: str | None = None,
    extra_images: list[str] | None = None,
    control_type: str | None = None,
    control_strength: float | None = None,
    layers: int | None = None,
) -> dict[str, Any]:
    items = _normalize_items(
        prompts=prompts,
        prompt=prompt,
        image=image,
        plan=plan,
        seed=seed,
        width=width,
        height=height,
        count=count,
        fast_gen_mode=fast_gen_mode,
        steps=steps,
        upscale=upscale,
        flow=flow,
        extra_images=extra_images,
        control_type=control_type,
        control_strength=control_strength,
        layers=layers,
    )
    for item in items:
        if not item.get("flow") or item.get("explicitPlan"):
            continue
        try:
            _plan_item(item)
        except GenerateError:
            raise
        except image_flows.FlowError as exc:
            raise GenerateError("VALIDATION_ERROR", str(exc), 422) from exc

    job_id = str(uuid.uuid4())
    _job_images_dir(job_id)
    total = len(items)

    job = {
        "jobId": job_id,
        "total": total,
        "completedCount": 0,
        "failedCount": 0,
        "status": "planning",
        "phase": "planning",
        "items": items,
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
    return job_progress.attach(job, "generate", "GET /api/v1/generate/{jobId}")


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
        raise GenerateError("JOB_NOT_FOUND", f"Unknown job: {job_id}", 404)

    if job.get("status") not in ("planning", "switching", "failed", "completed"):
        job = _refresh_job(job)
        store["jobs"][job_id] = job
        _save_jobs(store)
    return job_progress.attach(job, "generate", "GET /api/v1/generate/{jobId}")


def get_capabilities() -> dict[str, Any]:
    bmap = _bundle_map()
    preset_list = []
    for p in presets.list_presets():
        bundles = p.get("bundles") or []
        ready = all(bmap.get(b, {}).get("status") == "complete" for b in bundles)
        preset_list.append({**p, "ready": ready})

    modes = ["txt2img", "edit", "control", "bg_replace", "painterly", "upscale", "layered"]
    flows = []
    for spec in image_flows.flow_catalog():
        ready = all(bmap.get(b, {}).get("status") == "complete" for b in spec["bundles"])
        flows.append({**spec, "ready": ready})
    loras = [
        {"id": "lightning", "file": presets.LORA["lightning"], "bundle": "qwen-image-2512-lightning-lora"},
        {"id": "advertisement", "file": presets.LORA["advertisement"], "bundle": "qwen-lora-advertisement"},
        {"id": "poster", "file": presets.LORA["poster"], "bundle": "qwen-lora-poster"},
        {"id": "flat_cartoon", "file": presets.LORA["flat_cartoon"], "bundle": "qwen-lora-flat-cartoon"},
        {"id": "eligen", "file": presets.LORA["eligen"], "bundle": "qwen-lora-eligen-poster"},
        {"id": "realism", "file": presets.LORA["realism"], "bundle": "qwen-lora-realism"},
    ]
    for l in loras:
        l["ready"] = bmap.get(l["bundle"], {}).get("status") == "complete"

    return {
        "modes": modes,
        "flows": flows,
        "presets": preset_list,
        "loras": loras,
        "plannerProfileText": config.PLANNER_PROFILE_TEXT,
        "plannerProfileVision": config.PLANNER_PROFILE_VISION,
        "plannerProfile": config.PLANNER_PROFILE_TEXT,
        "executorProfile": "comfy",
        "maxBatchCount": MAX_BATCH_COUNT,
        "supportsPromptArray": True,
        "fastGenModeDefault": "auto",
        "fastGenModeOptions": ["auto", True, False],
        "stepsDefault": "auto",
        "stepsOptions": [2, 4, 8, 20, 30],
        "upscaleDefault": "auto",
        "defaultWidth": presets.DEFAULT_WIDTH,
        "defaultHeight": presets.DEFAULT_HEIGHT,
        "batchNote": (
            "Multi-prompt jobs load the planner once (Gemma for text and vision), "
            "then Comfy once to execute (2 profile switches total)."
        ),
    }
