"""Presets for LTX-2.5 video + audio generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import ltx_graph


@dataclass
class LtxPreset:
    id: str
    label: str
    when_to_use: str
    width: int
    height: int
    length: int
    fps: float
    steps: int
    video_cfg: float
    audio_cfg: float
    bundles: list[str]
    refine: bool = False
    refine_steps: int = 3
    refine_denoise: float = 0.4
    sampler_name: str = "euler_ancestral"
    tiled_decode: bool = False


LTX_PRESETS: dict[str, LtxPreset] = {
    "ltx_reel": LtxPreset(
        id="ltx_reel",
        label="Reel (9:16 portrait)",
        when_to_use="Short vertical clips for social / phone viewing",
        width=704,
        height=1216,
        length=97,
        fps=24.0,
        steps=20,
        video_cfg=1.0,
        audio_cfg=1.0,
        bundles=["ltx-2.5-distilled"],
        sampler_name="euler_ancestral",
    ),
    "ltx_landscape": LtxPreset(
        id="ltx_landscape",
        label="Landscape (16:9)",
        when_to_use="Widescreen b-roll and scene clips",
        width=1216,
        height=704,
        length=97,
        fps=24.0,
        steps=20,
        video_cfg=1.0,
        audio_cfg=1.0,
        bundles=["ltx-2.5-distilled"],
        sampler_name="euler_ancestral",
    ),
    "ltx_square": LtxPreset(
        id="ltx_square",
        label="Square (1:1)",
        when_to_use="Square social posts and product loops",
        width=896,
        height=896,
        length=97,
        fps=24.0,
        steps=20,
        video_cfg=1.0,
        audio_cfg=1.0,
        bundles=["ltx-2.5-distilled"],
        sampler_name="euler_ancestral",
    ),
    "ltx_quality": LtxPreset(
        id="ltx_quality",
        label="Quality (portrait + refine)",
        when_to_use="Higher detail with two-stage latent upscale refine",
        width=704,
        height=1216,
        length=97,
        fps=24.0,
        steps=8,
        video_cfg=1.0,
        audio_cfg=1.0,
        bundles=["ltx-2.5-studio"],
        refine=True,
        refine_steps=3,
        refine_denoise=0.4,
        sampler_name="euler_ancestral",
        tiled_decode=True,
    ),
    "ltx_studio": LtxPreset(
        id="ltx_studio",
        label="Studio (landscape + refine)",
        when_to_use="Same pipeline as ComfyUI LTX-2.5-Quality-T2V.json (1216×704, 2-stage refine)",
        width=1216,
        height=704,
        length=97,
        fps=24.0,
        steps=8,
        video_cfg=1.0,
        audio_cfg=1.0,
        bundles=["ltx-2.5-studio"],
        refine=True,
        refine_steps=3,
        refine_denoise=0.4,
        sampler_name="euler_ancestral",
        tiled_decode=True,
    ),
    "ltx_fast": LtxPreset(
        id="ltx_fast",
        label="Fast (portrait)",
        when_to_use="Quick preview — same HD resolution, fewer steps",
        width=704,
        height=1216,
        length=97,
        fps=24.0,
        steps=12,
        video_cfg=1.0,
        audio_cfg=1.0,
        bundles=["ltx-2.5-distilled"],
    ),
}


def list_orientations() -> list[dict[str, Any]]:
    return [
        {"id": o["id"], "label": o["label"], "width": o["width"], "height": o["height"]}
        for o in LTX_ORIENTATIONS.values()
    ]


def list_speed_modes() -> list[dict[str, Any]]:
    return [
        {
            "id": m["id"],
            "label": m["label"],
            "steps": m["steps"],
            "video_cfg": m["video_cfg"],
            "refine": m.get("refine", False),
            "tiled_decode": m.get("tiled_decode", False),
            "sampler_name": m.get("sampler_name"),
        }
        for m in LTX_SPEED_MODES.values()
    ]

NEG_DEFAULT = (
    "blurry, low quality, still frame, watermark, overlay, titles, subtitles, "
    "distorted audio, buzzing, loud music when silence requested"
)

# Base dimensions per orientation (all divisible by 64 for refine compatibility).
LTX_ORIENTATIONS: dict[str, dict[str, Any]] = {
    "portrait": {
        "id": "portrait",
        "label": "Portrait (9:16)",
        "width": 704,
        "height": 1216,
        "preset": "ltx_reel",
    },
    "landscape": {
        "id": "landscape",
        "label": "Landscape (16:9)",
        "width": 1216,
        "height": 704,
        "preset": "ltx_landscape",
    },
    "square": {
        "id": "square",
        "label": "Square (1:1)",
        "width": 896,
        "height": 896,
        "preset": "ltx_square",
    },
}

# Standard = LTX Studio 2-stage refine (sharp). Fast = single-pass preview only.
LTX_SPEED_MODES: dict[str, dict[str, Any]] = {
    "fast": {
        "id": "fast",
        "label": "Fast (preview, softer)",
        "steps": 12,
        "video_cfg": 1.0,
        "refine": False,
        "tiled_decode": False,
        "sampler_name": "euler_ancestral",
    },
    "standard": {
        "id": "standard",
        "label": "Standard (2-stage refine)",
        "steps": 8,
        "video_cfg": 1.0,
        "refine": True,
        "refine_steps": 3,
        "refine_denoise": 0.4,
        "tiled_decode": True,
        "sampler_name": "euler_ancestral",
    },
    "quality": {
        "id": "quality",
        "label": "Quality (LTX Studio)",
        "steps": 8,
        "video_cfg": 1.0,
        "refine": True,
        "refine_steps": 3,
        "refine_denoise": 0.4,
        "tiled_decode": True,
        "sampler_name": "euler_ancestral",
    },
}

LTX_PARAMETER_RANGES = {
    "width": {"min": 256, "max": 2048, "align": 32},
    "height": {"min": 256, "max": 2048, "align": 32},
    "length": {"min": 17, "max": ltx_graph.MAX_LENGTH, "align": "8k+1"},
    "fps": {"min": 16.0, "max": 30.0},
    "steps": {"min": 1, "max": 60},
    "refine_steps": {"min": 1, "max": 40},
    "video_cfg": {"min": 1.0, "max": 15.0},
    "audio_cfg": {"min": 1.0, "max": 15.0},
    "refine_denoise": {"min": 0.05, "max": 1.0},
    "strength": {"min": 0.0, "max": 1.0},
    "prompt_enhance_max_length": {"min": 64, "max": 4096},
}


def _resolve_bundles(
    preset: LtxPreset,
    *,
    refine: bool,
    prompt_enhance: bool,
) -> list[str]:
    bundles = ["ltx-2.5-studio"] if refine else list(preset.bundles)
    if prompt_enhance:
        bundles.append("ltx-2.5-prompt-enhancer")
    return list(dict.fromkeys(bundles))


def list_presets() -> list[dict[str, Any]]:
    return [
        {
            "id": p.id,
            "label": p.label,
            "when_to_use": p.when_to_use,
            "bundles": p.bundles,
            "defaults": {
                "width": p.width,
                "height": p.height,
                "length": p.length,
                "fps": p.fps,
                "steps": p.steps,
                "video_cfg": p.video_cfg,
                "audio_cfg": p.audio_cfg,
                "refine": p.refine,
                "refine_steps": p.refine_steps,
                "refine_denoise": p.refine_denoise,
                "sampler_name": p.sampler_name,
                "tiled_decode": p.tiled_decode,
            },
        }
        for p in LTX_PRESETS.values()
    ]


def resolve(
    preset_id: str,
    *,
    prompt: str,
    audio_prompt: str = "",
    mode: str = "t2v",
    orientation: str | None = None,
    speed: str | None = None,
    overrides: dict[str, Any] | None = None,
    seed: int | None = None,
    image_name: str | None = None,
) -> dict[str, Any]:
    preset = LTX_PRESETS.get(preset_id) or LTX_PRESETS["ltx_reel"]
    ov = overrides or {}
    orient = LTX_ORIENTATIONS.get(orientation or "") if orientation else None
    speed_mode = LTX_SPEED_MODES.get(speed or "standard")

    width = int(ov.get("width", orient["width"] if orient else preset.width))
    height = int(ov.get("height", orient["height"] if orient else preset.height))
    refine_default = bool(speed_mode["refine"]) if speed_mode else preset.refine
    prompt_enhance_default = bool(ov.get("prompt_enhance", False))

    plan: dict[str, Any] = {
        "preset": preset.id,
        "orientation": orient["id"] if orient else None,
        "speed": speed_mode["id"] if speed_mode else None,
        "mode": mode,
        "prompt": prompt,
        "audio_prompt": audio_prompt or "silent, no sound, no music, no background audio",
        "negative_prompt": ov.get("negative_prompt", NEG_DEFAULT),
        "width": width,
        "height": height,
        "length": int(ov.get("length", preset.length)),
        "fps": float(ov.get("fps", preset.fps)),
        "steps": int(ov.get("steps", speed_mode["steps"] if speed_mode else preset.steps)),
        "video_cfg": float(ov.get("video_cfg", speed_mode["video_cfg"] if speed_mode else preset.video_cfg)),
        "audio_cfg": float(ov.get("audio_cfg", preset.audio_cfg)),
        "refine": bool(ov.get("refine", refine_default)),
        "refine_steps": int(
            ov.get(
                "refine_steps",
                speed_mode.get("refine_steps", preset.refine_steps) if speed_mode else preset.refine_steps,
            )
        ),
        "refine_denoise": float(
            ov.get(
                "refine_denoise",
                speed_mode.get("refine_denoise", preset.refine_denoise) if speed_mode else preset.refine_denoise,
            )
        ),
        "sampler_name": str(
            ov.get(
                "sampler_name",
                speed_mode.get("sampler_name", preset.sampler_name) if speed_mode else preset.sampler_name,
            )
        ),
        "tiled_decode": bool(
            ov.get(
                "tiled_decode",
                speed_mode.get("tiled_decode", preset.tiled_decode) if speed_mode else preset.tiled_decode,
            )
        ),
        "prompt_enhance": bool(ov.get("prompt_enhance", prompt_enhance_default)),
        "bundles": _resolve_bundles(
            preset,
            refine=bool(ov.get("refine", refine_default)),
            prompt_enhance=bool(ov.get("prompt_enhance", prompt_enhance_default)),
        ),
        "filename_prefix": ov.get("filename_prefix", "ltx_video"),
    }

    for key in (
        "max_shift",
        "base_shift",
        "terminal",
        "stretch",
        "weight_dtype",
        "tile_size",
        "tile_overlap",
        "temporal_size",
        "temporal_overlap",
        "prompt_enhance_max_length",
        "prompt_enhance_temperature",
        "prompt_enhance_top_k",
        "prompt_enhance_top_p",
        "prompt_enhance_min_p",
        "prompt_enhance_repetition_penalty",
        "prompt_enhance_presence_penalty",
        "prompt_enhance_seed",
        "prompt_enhance_thinking",
        "prompt_enhance_use_template",
    ):
        if key in ov:
            plan[key] = ov[key]

    if mode == "i2v":
        plan["image_name"] = image_name
        plan["strength"] = float(ov.get("strength", 1.0))

    seed_val = seed if seed is not None else ov.get("seed")
    if seed_val is not None:
        plan["seed"] = int(seed_val)

    return plan
