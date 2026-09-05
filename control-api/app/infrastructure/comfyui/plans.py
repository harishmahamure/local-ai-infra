from __future__ import annotations

import random
from typing import Any

from ...application.resolvers import resolve_dimensions
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
    if workflow.builder == "qwen_upscale":
        return _qwen_upscale(job, preset)
    if workflow.builder.startswith("ltx_"):
        return _ltx(job, preset, workflow.builder)
    raise ValueError(f"Unknown builder: {workflow.builder}")


def _qwen_txt2img(job: Job, preset: Preset, loras: list[dict[str, Any]]) -> dict[str, Any]:
    width, height = resolve_dimensions(job.inputs, preset)
    graph_loras = [{"name": item["name"], "strength": item["strength"]} for item in loras]
    if preset.id == "draft" and not graph_loras:
        graph_loras = [{"name": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors", "strength": 1.0}]
    return {
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
