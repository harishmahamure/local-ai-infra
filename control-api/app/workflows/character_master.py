"""WF_01 Character Master — locked Qwen 2512 identity sheet."""

from __future__ import annotations

import copy
import json
import random
import threading
import uuid
from pathlib import Path
from typing import Any

from .. import comfy_client, config, job_progress, planner, presets, qwen_graph, runtime
from . import assets, types

JOBS_PATH = Path(config.LOGS) / "character-master-jobs.json"
JOBS_ROOT = Path(config.LOGS) / "jobs"
MASTER_FILENAME_PREFIX = "wf01_character_master"


class WorkflowError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


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
    return f"/api/v1/workflows/character-master/{job_id}/images/{filename}"


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
        fields["candidates"] = _candidates_from_items(fields["items"])
    job.update(fields)
    store["jobs"][job_id] = job
    _save_jobs(store)
    return job


def _candidates_from_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        images = item.get("images") or []
        image = images[0] if images else None
        out.append(
            {
                "index": item.get("index"),
                "seed": item.get("seed") if item.get("seed") is not None else (item.get("plan") or {}).get("seed"),
                "status": item.get("status"),
                "filename": image.get("filename") if image else None,
                "url": image.get("url") if image else None,
                "error": item.get("error"),
            }
        )
    return out


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


def bundle_complete(bundle_id: str) -> bool:
    b = _bundle_map().get(bundle_id)
    return bool(b and b.get("status") == "complete")


def _installed_hashes() -> dict[str, str]:
    """Map basename -> sha256 from catalog/installed.json when present."""
    path = Path(config.ROOT / "catalog" / "installed.json")
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    installed = raw.get("installed") or raw
    if not isinstance(installed, dict):
        return {}
    hashes: dict[str, str] = {}
    for entry in installed.values():
        if not isinstance(entry, dict):
            continue
        for file_meta in entry.get("files") or []:
            if not isinstance(file_meta, dict):
                continue
            digest = file_meta.get("sha256")
            local = file_meta.get("local")
            if digest and local:
                hashes[Path(str(local)).name] = str(digest)
    return hashes


def model_hashes_for_plan(plan: dict[str, Any]) -> dict[str, str | None]:
    known = _installed_hashes()
    names = [
        qwen_graph.M["qwen_unet"],
        qwen_graph.M["qwen_clip"],
        qwen_graph.M["qwen_vae"],
    ]
    for lora in plan.get("loras") or []:
        name = lora.get("name")
        if name:
            names.append(str(name))
    return {name: known.get(name) for name in names}


def compose_user_prompt(description: str, style: str | None) -> str:
    parts = [
        "Cinematic character portrait, identity lock sheet.",
        description.strip(),
    ]
    if style and style.strip():
        parts.append(f"Style: {style.strip()}.")
    parts.append(
        "Single consistent identity: face, age, hair, build, primary attire, "
        "recognizable silhouette. Neutral pose, even lighting, no text, "
        "no watermark, no extra people."
    )
    return " ".join(parts)


def locked_preset_id(quality: str) -> str:
    return types.PRESET_BY_QUALITY[quality]


def assert_master_loras_allowed(plan: dict[str, Any], *, quality: str) -> None:
    if quality != "master":
        return
    for lora in plan.get("loras") or []:
        name = str(lora.get("name") or "")
        if types.lora_is_forbidden_on_master(name):
            raise WorkflowError(
                "MASTER_LORA_FORBIDDEN",
                f"MASTER quality forbids Lightning/Turbo LoRA: {name}",
                422,
            )


def build_locked_plan(
    *,
    quality: str,
    prompt: str,
    negative_prompt: str | None = None,
    seed: int | None = None,
    width: int,
    height: int,
    realism_lora: float | None = None,
    denoise: float | None = None,
    upscale: bool = False,
) -> dict[str, Any]:
    preset_id = locked_preset_id(quality)
    overrides: dict[str, Any] = {
        "mode": "txt2img",
        "upscale": upscale,
        "loras": [{"name": l.name, "strength": l.strength} for l in presets.PRESETS[preset_id].loras],
    }
    if negative_prompt is not None:
        overrides["negative_prompt"] = negative_prompt
    if denoise is not None:
        overrides["denoise"] = denoise
    if realism_lora is not None:
        loras = list(overrides["loras"])
        loras.append({"name": presets.LORA["realism"], "strength": float(realism_lora)})
        overrides["loras"] = loras
        if "qwen-lora-realism" not in presets.PRESETS[preset_id].bundles:
            overrides["bundles"] = list(presets.PRESETS[preset_id].bundles) + ["qwen-lora-realism"]

    plan = presets.resolve(
        preset_id,
        refined_prompt=prompt,
        overrides=overrides,
        seed=seed,
        width=width,
        height=height,
    )
    if overrides.get("bundles"):
        plan["bundles"] = overrides["bundles"]
    plan["mode"] = "txt2img"
    plan["filename_prefix"] = MASTER_FILENAME_PREFIX
    plan["sampler_name"] = types.SAMPLER_NAME
    plan["scheduler"] = types.SCHEDULER_NAME
    assert_master_loras_allowed(plan, quality=quality)
    return plan


def validate_request(body: dict[str, Any]) -> dict[str, Any]:
    description = str(body.get("characterDescription") or "").strip()
    if not description:
        raise WorkflowError("VALIDATION_ERROR", "characterDescription is required", 422)

    quality = str(body.get("quality") or types.DEFAULT_QUALITY).lower()
    if quality not in types.QUALITY_VALUES:
        raise WorkflowError("VALIDATION_ERROR", "quality must be draft or master", 422)

    aspect = str(body.get("aspectRatio") or types.DEFAULT_ASPECT)
    if aspect not in types.ASPECT_VALUES:
        raise WorkflowError(
            "VALIDATION_ERROR",
            f"aspectRatio must be one of {', '.join(types.ASPECT_VALUES)}",
            422,
        )

    default_w, default_h = types.dimensions_for_aspect(aspect)
    width = body.get("width")
    height = body.get("height")
    try:
        width = int(width) if width is not None else default_w
        height = int(height) if height is not None else default_h
    except (TypeError, ValueError) as exc:
        raise WorkflowError("VALIDATION_ERROR", "width and height must be integers", 422) from exc
    if not (256 <= width <= 4096 and 256 <= height <= 4096):
        raise WorkflowError("VALIDATION_ERROR", "width/height must be between 256 and 4096", 422)

    raw_count = body.get("candidateCount")
    if raw_count is None:
        candidate_count = types.DEFAULT_CANDIDATES[quality]
    else:
        try:
            candidate_count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise WorkflowError("VALIDATION_ERROR", "candidateCount must be an integer", 422) from exc
    if candidate_count < 1 or candidate_count > types.MAX_CANDIDATES:
        raise WorkflowError(
            "VALIDATION_ERROR",
            f"candidateCount must be between 1 and {types.MAX_CANDIDATES}",
            422,
        )

    seed = body.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise WorkflowError("VALIDATION_ERROR", "seed must be an integer", 422) from exc

    character_id = body.get("characterId")
    if character_id:
        character_id = str(character_id).strip()
        if not assets.is_safe_character_id(character_id):
            raise WorkflowError(
                "VALIDATION_ERROR",
                "characterId must be 1-64 chars: letters, digits, . _ -",
                422,
            )
    else:
        character_id = str(uuid.uuid4())

    realism = body.get("realismLora", False)
    realism_strength: float | None = None
    if realism not in (None, False, 0, 0.0, "false", "0"):
        try:
            realism_strength = float(realism)
        except (TypeError, ValueError) as exc:
            raise WorkflowError("VALIDATION_ERROR", "realismLora must be a number 0-1 or false", 422) from exc
        if not (0.0 < realism_strength <= 1.0):
            if realism_strength == 0.0:
                realism_strength = None
            else:
                raise WorkflowError("VALIDATION_ERROR", "realismLora must be between 0 and 1", 422)

    image = body.get("image")
    denoise = body.get("denoise")
    if image and denoise is None:
        denoise = types.DEFAULT_DENOISE
    if denoise is not None:
        try:
            denoise = float(denoise)
        except (TypeError, ValueError) as exc:
            raise WorkflowError("VALIDATION_ERROR", "denoise must be a number", 422) from exc
        if not (0.0 <= denoise <= 1.0):
            raise WorkflowError("VALIDATION_ERROR", "denoise must be between 0 and 1", 422)
        if not image:
            raise WorkflowError("VALIDATION_ERROR", "denoise requires an image reference", 422)

    style = body.get("style")
    style = str(style).strip() if style else None

    return {
        "characterDescription": description,
        "style": style,
        "quality": quality,
        "aspectRatio": aspect,
        "width": width,
        "height": height,
        "seed": seed,
        "candidateCount": candidate_count,
        "image": image,
        "denoise": denoise,
        "realismLora": realism_strength,
        "upscale": bool(body.get("upscale", False)),
        "characterId": character_id,
        "userPrompt": compose_user_prompt(description, style),
    }


def planner_profile_for(*, quality: str, has_image: bool) -> str:
    return planner.planner_profile(has_image=has_image)


def _resolve_image_name(image_field: str | None) -> str | None:
    if not image_field:
        return None
    raw, suggested = comfy_client.decode_image_field(image_field)
    if raw:
        upload = comfy_client.upload_image(raw, suggested)
        return upload.get("name") or suggested
    return suggested


def _image_to_data_url(image_field: str | None) -> str | None:
    if not image_field:
        return None
    raw, suggested = comfy_client.decode_image_field(image_field)
    if not raw:
        return None
    import base64

    ext = "jpeg" if suggested.endswith((".jpg", ".jpeg")) else "png"
    return f"data:image/{ext};base64,{base64.b64encode(raw).decode('ascii')}"


def _ensure_comfy_loaded() -> None:
    status = runtime.get_status()
    profile = status.get("profile", "none")
    if profile != "comfyui" or status.get("loadState") != "LOADED":
        raise WorkflowError("COMFY_NOT_LOADED", "ComfyUI profile is not loaded.", 409)
    if not comfy_client.is_ready():
        raise WorkflowError("COMFY_NOT_LOADED", "ComfyUI is not responding on port 8188", 409)


def _start_comfy() -> None:
    status = runtime.get_status()
    if status.get("profile") == "comfyui" and status.get("loadState") == "LOADED":
        return
    runtime.start_profile("comfy")


def _persist_outputs(
    job_id: str,
    character_id: str,
    batch_index: int,
    seed: int | None,
    history_entry: dict[str, Any],
) -> list[dict[str, Any]]:
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
            assets.persist_candidate(character_id, batch_index, seed, raw)
            images.append(
                {
                    "index": batch_index,
                    "filename": stored_name,
                    "mime": mime,
                    "url": _image_url(job_id, stored_name),
                    "bytes": len(raw),
                    "assetFilename": assets.candidate_filename(batch_index, seed),
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
    character_id: str,
    plan: dict[str, Any],
    batch_index: int,
    total: int,
) -> list[dict[str, Any]]:
    graph = qwen_graph.build(plan)
    client_id = str(uuid.uuid4())
    queued = comfy_client.queue_prompt(graph, client_id=client_id)
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise WorkflowError("QUEUE_FAILED", "ComfyUI did not return prompt_id", 500)

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
            raise WorkflowError("COMFY_HISTORY_FAILED", str(exc), 500) from exc

        entry = hist.get(prompt_id)
        if not entry:
            threading.Event().wait(2)
            continue

        state = _history_entry_status(entry)
        if state == "failed":
            status = entry.get("status") or {}
            raise WorkflowError(
                "COMFY_EXEC_FAILED",
                str(status.get("messages") or "ComfyUI execution error"),
                500,
            )

        images = _persist_outputs(job_id, character_id, batch_index, plan.get("seed"), entry)
        if images:
            return images

        threading.Event().wait(2)


def _enrich_prompt(req: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """Use the planner only to enrich prose. Preset/LoRA picks are discarded."""
    has_image = bool(req.get("image"))
    profile = planner_profile_for(quality=req["quality"], has_image=has_image)
    planner.start_planner_profile(has_image=has_image, profile=profile)
    data_url = _image_to_data_url(req.get("image"))
    planned = planner.plan(
        prompt=req["userPrompt"],
        image_data_url=data_url,
        fast_gen=req["quality"] == "draft",
        profile=profile,
    )
    refined = planned.get("prompt") or req["userPrompt"]
    planner_meta = {
        "profile": profile,
        "presetIgnored": planned.get("preset"),
        "confidence": (planned.get("planner") or {}).get("confidence"),
    }
    return refined, planner_meta


def _run_job(job_id: str) -> None:
    store = _load_jobs()
    job = store["jobs"].get(job_id)
    if not job:
        return

    items: list[dict[str, Any]] = copy.deepcopy(job.get("items") or [])
    total = len(items)
    req = job.get("request") or {}
    character_id = job["characterId"]
    quality = job["quality"]

    try:
        _update_job(job_id, status="planning", phase="planning 0/1", items=items)
        try:
            refined, planner_meta = _enrich_prompt(req)
        except Exception as exc:
            refined = req.get("userPrompt") or ""
            planner_meta = {"error": str(exc), "fallback": True}

        base_plan = build_locked_plan(
            quality=quality,
            prompt=refined,
            seed=req.get("seed"),
            width=req["width"],
            height=req["height"],
            realism_lora=req.get("realismLora"),
            denoise=req.get("denoise") if req.get("image") else None,
            upscale=bool(req.get("upscale")),
        )
        ready, missing = _bundles_ready(base_plan.get("bundles") or [])
        if not ready:
            raise WorkflowError("MODELS_MISSING", f"Required bundles not on disk: {', '.join(missing)}", 409)

        metadata = types.generation_metadata(
            job_id=job_id,
            quality=quality,
            model_ids=list(base_plan.get("bundles") or []),
            model_hashes=model_hashes_for_plan(base_plan),
            lora_ids=[str(l.get("name")) for l in (base_plan.get("loras") or [])],
            lora_weights=[float(l.get("strength", 1.0)) for l in (base_plan.get("loras") or [])],
            seed=req.get("seed"),
            prompt=refined,
            negative_prompt=str(base_plan.get("negative_prompt") or ""),
            width=int(base_plan["width"]),
            height=int(base_plan["height"]),
            steps=int(base_plan["steps"]),
            cfg=float(base_plan["cfg"]),
            denoise=float(base_plan.get("denoise", 1.0 if not req.get("image") else types.DEFAULT_DENOISE)),
            character_id=character_id,
            aspect_ratio=req["aspectRatio"],
            reference_asset_id=None,
        )
        assets.write_metadata(character_id, metadata)
        _update_job(job_id, plan=base_plan, metadata=metadata, planner=planner_meta, items=items)

        _update_job(job_id, status="switching", phase="switching", items=items)
        _start_comfy()
        _ensure_comfy_loaded()
        _update_job(job_id, status="running", phase=f"running 0/{total}", items=items)

        image_name = None
        if req.get("image"):
            image_name = _resolve_image_name(req.get("image"))
            if not image_name:
                raise WorkflowError("IMAGE_UPLOAD_FAILED", "Reference image could not be uploaded to ComfyUI", 500)

        for i, item in enumerate(items):
            item["status"] = "running"
            _update_job(job_id, phase=f"running {i + 1}/{total}", items=items)
            plan_i = copy.deepcopy(base_plan)
            if item.get("seed") is not None:
                plan_i["seed"] = int(item["seed"])
            elif plan_i.get("seed") is None:
                plan_i["seed"] = random.randint(0, 2**31 - 1)
                item["seed"] = plan_i["seed"]
            plan_i["image_name"] = image_name
            item["plan"] = plan_i
            try:
                item["images"] = _execute_comfy_batch(job_id, character_id, plan_i, item["index"], total)
                item["status"] = "completed"
                item["error"] = None
            except WorkflowError as exc:
                item["status"] = "failed"
                item["error"] = exc.message
            except Exception as exc:
                item["status"] = "failed"
                item["error"] = str(exc)
            items[i] = item
            completed, failed = _count_item_outcomes(items)
            done = completed + failed >= total
            _update_job(
                job_id,
                items=items,
                status=_job_status_from_items(items) if done else "running",
                phase="done" if done else f"running {completed + failed}/{total}",
                finishedAt=types.utc_now() if done else None,
                metadata=metadata,
            )

    except WorkflowError as exc:
        _update_job(job_id, status="failed", error=exc.message, finishedAt=types.utc_now(), phase="failed", items=items)
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), finishedAt=types.utc_now(), phase="failed", items=items)


def submit(body: dict[str, Any]) -> dict[str, Any]:
    req = validate_request(body)
    job_id = str(uuid.uuid4())
    _job_images_dir(job_id)
    assets.character_dir(req["characterId"])

    items = []
    for i in range(req["candidateCount"]):
        item_seed = (int(req["seed"]) + i) if req["seed"] is not None else None
        items.append(
            {
                "index": i,
                "seed": item_seed,
                "status": "pending",
                "plan": None,
                "images": [],
                "error": None,
            }
        )

    job = {
        "jobId": job_id,
        "workflowId": types.WORKFLOW_ID,
        "workflowVersion": types.WORKFLOW_VERSION,
        "characterId": req["characterId"],
        "quality": req["quality"],
        "total": len(items),
        "completedCount": 0,
        "failedCount": 0,
        "status": "planning",
        "phase": "planning",
        "items": items,
        "candidates": _candidates_from_items(items),
        "selected": None,
        "promptId": None,
        "clientId": None,
        "createdAt": types.utc_now(),
        "finishedAt": None,
        "images": [],
        "error": None,
        "request": req,
        "plan": None,
        "metadata": None,
        "storageDir": str(JOBS_ROOT / job_id / "images"),
        "assetDir": str(assets.character_dir(req["characterId"])),
    }

    store = _load_jobs()
    store.setdefault("jobs", {})[job_id] = job
    _save_jobs(store)

    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()
    return job_progress.attach(job, "character_master", "GET /api/v1/workflows/character-master/{jobId}")


def get_job(job_id: str) -> dict[str, Any]:
    store = _load_jobs()
    job = store.get("jobs", {}).get(job_id)
    if not job:
        raise WorkflowError("JOB_NOT_FOUND", f"Unknown job: {job_id}", 404)
    return job_progress.attach(job, "character_master", "GET /api/v1/workflows/character-master/{jobId}")


def get_job_image_path(job_id: str, filename: str) -> Path | None:
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        return None
    store = _load_jobs()
    if job_id not in store.get("jobs", {}):
        return None
    path = JOBS_ROOT / job_id / "images" / filename
    return path if path.is_file() else None


def select_candidate(job_id: str, candidate_index: int) -> dict[str, Any]:
    store = _load_jobs()
    job = store.get("jobs", {}).get(job_id)
    if not job:
        raise WorkflowError("JOB_NOT_FOUND", f"Unknown job: {job_id}", 404)

    items = job.get("items") or []
    match = next((it for it in items if it.get("index") == candidate_index), None)
    if not match or match.get("status") != "completed" or not (match.get("images") or []):
        raise WorkflowError("CANDIDATE_NOT_FOUND", f"No completed candidate at index {candidate_index}", 404)

    image = match["images"][0]
    job_file = JOBS_ROOT / job_id / "images" / image["filename"]
    if not job_file.is_file():
        raise WorkflowError("CANDIDATE_NOT_FOUND", "Candidate image file is missing", 404)

    character_id = job["characterId"]
    metadata = job.get("metadata") or assets.load_metadata(character_id) or {}
    master = assets.promote_candidate(
        character_id,
        job_file,
        metadata,
        selected_index=candidate_index,
    )
    selected = {
        "index": candidate_index,
        "filename": image["filename"],
        "url": image.get("url"),
        "assetId": character_id,
        "masterFilename": assets.MASTER_FILENAME,
        "masterPath": str(master),
        "masterUrl": f"/api/v1/assets/characters/{character_id}/files/{assets.MASTER_FILENAME}",
    }
    metadata = dict(metadata)
    metadata["selected_candidate_index"] = candidate_index
    job = _update_job(job_id, selected=selected, metadata=metadata, assetId=character_id)
    return job_progress.attach(job, "character_master", "GET /api/v1/workflows/character-master/{jobId}")


def get_character_file(character_id: str, filename: str) -> Path | None:
    if not assets.is_safe_character_id(character_id):
        return None
    if not filename or ".." in filename or filename.startswith("/"):
        return None
    directory = assets.assets_root() / character_id
    if filename == assets.MASTER_FILENAME:
        path = directory / filename
    else:
        path = directory / "candidates" / Path(filename).name
    return path if path.is_file() else None


def get_capabilities() -> dict[str, Any]:
    bmap = _bundle_map()
    draft = presets.get_preset("character_master_draft")
    master = presets.get_preset("character_master_master")
    draft_ready = bool(draft) and all(bmap.get(b, {}).get("status") == "complete" for b in draft.bundles)
    master_ready = bool(master) and all(bmap.get(b, {}).get("status") == "complete" for b in master.bundles)
    return {
        "id": types.WORKFLOW_ID,
        "version": types.WORKFLOW_VERSION,
        "ready": draft_ready or master_ready,
        "draftReady": draft_ready,
        "masterReady": master_ready,
        "presets": {
            "draft": "character_master_draft",
            "master": "character_master_master",
        },
        "defaultAspectRatio": types.DEFAULT_ASPECT,
        "aspectRatios": list(types.ASPECT_VALUES),
        "defaultCandidateCount": types.DEFAULT_CANDIDATES,
        "maxCandidateCount": types.MAX_CANDIDATES,
        "plannerDraft": "gemma (text + vision)",
        "plannerMaster": "gemma (text + vision)",
        "bundles": {
            "draft": list(draft.bundles) if draft else [],
            "master": list(master.bundles) if master else [],
        },
    }
