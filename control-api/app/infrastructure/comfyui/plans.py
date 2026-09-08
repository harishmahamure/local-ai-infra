from __future__ import annotations

import random
from typing import Any

from ...application.image_flows import (
    DEFAULT_CHARACTER_DENOISE,
    DEFAULT_IMG2IMG_DENOISE,
    FlowError,
    compose_face_lock_prompts,
    domain_flow_error,
    resolve_flow_plan,
)
from ...application.resolvers import resolve_dimensions
from ...domain.errors import DomainError, ErrorCode
from ...domain.jobs import Job
from ...domain.presets import Preset
from ...domain.workflows import WorkflowDefinition

LTX_ASPECT = {
    "9:16": (704, 1216),
    "16:9": (1216, 704),
    "1:1": (896, 896),
    "2:3": (704, 1056),
    "4:5": (768, 960),
}


def _seed(job: Job) -> int:
    value = job.parameters.get("seed")
    if value is None:
        return random.randint(0, 2**31 - 1)
    return int(value)


def build_plan(job: Job, workflow: WorkflowDefinition, preset: Preset, loras: list[dict[str, Any]]) -> dict[str, Any]:
    if workflow.builder == "qwen_txt2img":
        return _qwen_txt2img(job, preset, loras)
    if workflow.builder == "qwen_edit":
        return _qwen_edit(job, preset)
    if workflow.builder == "qwen_control":
        return _qwen_control(job, preset)
    if workflow.builder == "qwen_layered":
        return _qwen_layered(job, preset)
    if workflow.builder == "qwen_upscale":
        return _qwen_upscale(job, preset)
    if workflow.builder.startswith("ltx_"):
        return _ltx(job, preset, workflow.builder)
    if workflow.builder.startswith("ace_step"):
        return _ace(job, preset, workflow.builder)
    if workflow.builder.startswith("tts_"):
        return _tts(job, preset)
    if workflow.builder.startswith("ffmpeg_"):
        return _ffmpeg(job, preset, workflow.builder)
    raise ValueError(f"Unknown builder: {workflow.builder}")


def _qwen_txt2img(job: Job, preset: Preset, loras: list[dict[str, Any]]) -> dict[str, Any]:
    width, height = resolve_dimensions(job.inputs, preset)
    graph_loras = [{"name": item["name"], "strength": item["strength"]} for item in loras]
    if preset.id == "draft" and not graph_loras:
        graph_loras = [{"name": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors", "strength": 1.0}]
    plan: dict[str, Any] = {
        "mode": "txt2img",
        "prompt": str(job.inputs.get("prompt") or ""),
        "negative_prompt": str(job.inputs.get("negative_prompt") or job.parameters.get("negative_prompt") or ""),
        "width": width,
        "height": height,
        "steps": int(job.parameters.get("steps") or preset.parameters.get("steps") or 30),
        "cfg": float(job.parameters.get("cfg") or preset.parameters.get("cfg") or 4.0),
        "seed": _seed(job),
        "loras": graph_loras,
        "upscale": False,
        "filename_prefix": "engine",
    }
    face_lock = preset.id == "face_lock"
    if face_lock:
        plan["prompt"], plan["negative_prompt"] = compose_face_lock_prompts(
            str(job.inputs.get("prompt") or ""),
            str(job.inputs.get("negative_prompt") or job.parameters.get("negative_prompt") or ""),
        )
    has_image = bool(job.inputs.get("image") or job.inputs.get("reference_images"))
    if face_lock and not has_image:
        raise DomainError(ErrorCode.INVALID_REQUEST, "face_lock requires a reference image")
    if has_image:
        denoise = job.inputs.get("denoise")
        if denoise is None:
            denoise = job.parameters.get("denoise")
        if denoise is None:
            denoise = preset.parameters.get("denoise")
        default = DEFAULT_CHARACTER_DENOISE if face_lock else DEFAULT_IMG2IMG_DENOISE
        plan["denoise"] = float(denoise) if denoise is not None else default
    return plan


def _reference_count(job: Job) -> int:
    refs = job.inputs.get("reference_images")
    if isinstance(refs, list) and refs:
        return len(refs)
    if job.inputs.get("image"):
        return 1
    return 0


def _overlay_numeric(plan: dict[str, Any], job: Job, _preset: Preset) -> dict[str, Any]:
    """Keep flow-locked steps/CFG. Seed and negative prompt come from the job."""
    if job.inputs.get("negative_prompt") or job.parameters.get("negative_prompt"):
        plan["negative_prompt"] = str(job.inputs.get("negative_prompt") or job.parameters.get("negative_prompt") or "")
    plan["seed"] = _seed(job)
    plan["filename_prefix"] = "engine"
    return plan


def _qwen_edit(job: Job, preset: Preset) -> dict[str, Any]:
    count = _reference_count(job)
    flow = "merge" if count >= 2 else "text_edit"
    try:
        plan = resolve_flow_plan(
            flow,
            prompt=str(job.inputs.get("prompt") or ""),
            image_count=count,
            width=job.inputs.get("width") or job.parameters.get("width"),
            height=job.inputs.get("height") or job.parameters.get("height"),
            seed=_seed(job),
        )
    except FlowError as exc:
        raise domain_flow_error(exc) from exc
    return _overlay_numeric(plan, job, preset)


def _qwen_control(job: Job, preset: Preset) -> dict[str, Any]:
    try:
        plan = resolve_flow_plan(
            "control",
            prompt=str(job.inputs.get("prompt") or ""),
            image_count=1 if job.inputs.get("control_image") or job.inputs.get("image") else 0,
            control_type=job.inputs.get("control_type"),
            control_strength=job.inputs.get("control_strength") or job.parameters.get("control_strength"),
            width=job.inputs.get("width") or job.parameters.get("width"),
            height=job.inputs.get("height") or job.parameters.get("height"),
            seed=_seed(job),
        )
    except FlowError as exc:
        raise domain_flow_error(exc) from exc
    return _overlay_numeric(plan, job, preset)


def _qwen_layered(job: Job, preset: Preset) -> dict[str, Any]:
    layers = job.inputs.get("layers")
    try:
        plan = resolve_flow_plan(
            "layered",
            prompt=str(job.inputs.get("prompt") or ""),
            image_count=1 if job.inputs.get("image") else 0,
            layers=int(layers) if layers is not None else None,
            width=job.inputs.get("width") or job.parameters.get("width"),
            height=job.inputs.get("height") or job.parameters.get("height"),
            seed=_seed(job),
        )
    except FlowError as exc:
        raise domain_flow_error(exc) from exc
    return _overlay_numeric(plan, job, preset)


def _qwen_upscale(job: Job, preset: Preset) -> dict[str, Any]:
    scale = int(job.inputs.get("scale") or job.parameters.get("scale") or preset.parameters.get("scale") or 4)
    return {
        "mode": "upscale",
        "upscale_scale": 2 if scale <= 2 else 4,
        "filename_prefix": "engine_upscale",
    }


def _ltx(job: Job, preset: Preset, builder: str) -> dict[str, Any]:
    aspect = str(job.inputs.get("aspect_ratio") or "9:16")
    width, height = LTX_ASPECT.get(aspect, LTX_ASPECT["9:16"])
    if job.inputs.get("width") and job.inputs.get("height"):
        width, height = int(job.inputs["width"]), int(job.inputs["height"])
    duration = float(job.inputs.get("duration_seconds") or 4.0)
    fps = 24.0
    from ... import ltx_graph

    refine = bool(preset.parameters.get("refine")) if preset.id == "master" else False
    steps = int(job.parameters.get("steps") or (8 if refine else preset.parameters.get("steps") or 20))
    if preset.id == "draft":
        steps = int(job.parameters.get("steps") or 8)
    mode = {
        "ltx_t2v": "t2v",
        "ltx_i2v": "i2v",
        "ltx_a2v": "a2v",
        "ltx_flf2v": "flf2v",
        "ltx_lipsync": "lipsync",
        "ltx_motion_transfer": "motion_transfer",
    }.get(builder, "t2v")
    audio_prompt = str(job.inputs.get("audio_prompt") or "").strip()
    if not audio_prompt:
        audio_prompt = "silent, no sound, no music, no background audio"
    from ... import ltx_loras

    plan = {
        "mode": mode,
        "prompt": str(job.inputs.get("prompt") or ""),
        "audio_prompt": audio_prompt,
        "negative_prompt": str(job.inputs.get("negative_prompt") or job.parameters.get("negative_prompt") or ""),
        "width": width,
        "height": height,
        "length": ltx_graph.duration_to_length(duration, fps),
        "fps": fps,
        "steps": steps,
        "video_cfg": 1.0,
        "audio_cfg": 1.0,
        "refine": refine,
        "refine_steps": 3,
        "refine_denoise": 0.4,
        "sampler_name": "euler_ancestral",
        "tiled_decode": False,
        "prompt_enhance": False,
        "seed": _seed(job),
        "filename_prefix": "engine_ltx",
        "loras": [],
        "ic_enabled": False,
        "ic_loras": [],
    }
    if mode == "lipsync":
        plan["ic_loras"] = [ltx_loras.explicit_ic_lora("lipdub")]
        plan["ic_enabled"] = True
    if mode == "motion_transfer":
        plan["ic_loras"] = [ltx_loras.explicit_ic_lora("motion_track")]
        plan["ic_enabled"] = True
        plan["ic_guide_raw"] = True
    return plan


def _ace(job: Job, preset: Preset, builder: str) -> dict[str, Any]:
    kind = {
        "ace_step_music": "music",
        "ace_step_sfx": "sfx",
        "ace_step_ambience": "ambience",
        "ace_step_foley": "foley",
    }.get(builder, "music")
    prompt = str(job.inputs.get("prompt") or "")
    if kind == "music" and (job.inputs.get("no_vocals") is True or preset.parameters.get("no_vocals")):
        prompt = f"{prompt}, instrumental, no vocals".strip(", ")
    if kind == "sfx" and "sfx" not in prompt.lower():
        prompt = f"[sfx] {prompt}".strip()
    if kind == "ambience" and "ambience" not in prompt.lower():
        prompt = f"ambience, {prompt}".strip()
    duration = (
        job.inputs.get("duration_seconds")
        or job.parameters.get("duration_seconds")
        or preset.parameters.get("duration_seconds")
        or (24 if kind == "music" else 4)
    )
    return {
        "kind": kind,
        "prompt": prompt,
        "duration_seconds": float(duration),
        "mood": job.inputs.get("mood"),
        "seed": _seed(job),
        "steps": int(job.parameters.get("steps") or preset.parameters.get("steps") or 8),
        "cfg": float(job.parameters.get("cfg") or preset.parameters.get("cfg") or 4.0),
        "filename_prefix": f"engine_ace_{kind}",
        "video_ignored": kind == "foley" and bool(job.inputs.get("video")),
    }


def _tts(job: Job, preset: Preset) -> dict[str, Any]:
    language = str(job.inputs.get("language") or job.parameters.get("language") or preset.parameters.get("language") or "en")
    return {
        "text": str(job.inputs.get("text") or ""),
        "language": language,
        "voice": job.inputs.get("voice") or preset.parameters.get("voice"),
        "style": job.inputs.get("style") or preset.parameters.get("style"),
        "speaking_rate": float(job.parameters.get("speaking_rate") or preset.parameters.get("speaking_rate") or 1.0),
        "seed": _seed(job),
    }


def _ffmpeg(job: Job, preset: Preset, builder: str) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "builder": builder,
        "target_lufs": job.parameters.get("target_lufs")
        or job.inputs.get("target_lufs")
        or preset.parameters.get("target_lufs")
        or -16,
        "true_peak_db": job.inputs.get("true_peak_db") or -1.5,
        "duration_seconds": job.inputs.get("duration_seconds"),
        "stems": job.inputs.get("stems") or [],
        "clips": job.inputs.get("clips") or [],
        "transition": job.inputs.get("transition") or "cut",
        "transition_seconds": job.inputs.get("transition_seconds") or 0,
        "audio_offset_seconds": job.inputs.get("audio_offset_seconds") or 0,
        "container": job.inputs.get("container") or "mp4",
    }
    return plan
