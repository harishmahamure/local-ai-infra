"""Curated image-generation recipes ported from legacy workflow defaults."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

NEG_DEFAULT = "blurry, low quality, watermark, text, logo, deformed, ugly, bad anatomy"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
NEG_INFOGRAPHIC = (
    "blurry text, misspelled words, garbled letters, overlapping text, "
    "low contrast text, watermark, messy layout, low quality"
)

LORA = {
    "lightning": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
    "turbo": "Wuli-Qwen-Image-2512-Turbo-LoRA-2steps-V1.0-bf16.safetensors",
    "advertisement": "qwen-advertisement-lora.safetensors",
    "poster": "qwen-poster-lora.safetensors",
    "flat_cartoon": "qwen-flat-cartoon-lora.safetensors",
    "eligen": "qwen-eligen-poster-lora.safetensors",
    "realism": "qwen-realism-lora.safetensors",
}


@dataclass
class LoraSpec:
    name: str
    strength: float


@dataclass
class Preset:
    id: str
    label: str
    when_to_use: str
    mode: str  # txt2img | edit | control | bg_replace | painterly | upscale
    loras: list[LoraSpec] = field(default_factory=list)
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    steps: int = 30
    cfg: float = 4.0
    negative_prompt: str = NEG_DEFAULT
    upscale: bool = False
    control_type: str | None = None  # pose | depth | canny
    control_strength: float = 0.85
    bundles: list[str] = field(default_factory=list)
    hint: str = ""


PRESETS: dict[str, Preset] = {
    "deity_wallpaper": Preset(
        id="deity_wallpaper",
        label="Deity wallpaper (Qwen + Lightning)",
        when_to_use="Hindu deity phone wallpaper, portrait 9:16, Sanskrit/Devanagari text",
        mode="txt2img",
        loras=[LoraSpec(LORA["lightning"], 1.0)],
        width=1152,
        height=2016,
        steps=4,
        cfg=1.0,
        upscale=True,
        bundles=["qwen-image-2512-fp8", "qwen-image-2512-lightning-lora", "upscalers-esrgan"],
        hint="Traditional Hindu deity art, ornate jewelry, divine aura, Devanagari script if requested",
    ),
    "painterly": Preset(
        id="painterly",
        label="Painterly traditional art (Chroma1-HD)",
        when_to_use="Painterly, oil/watercolor traditional art, classical deity portraits",
        mode="painterly",
        width=1152,
        height=2016,
        steps=30,
        cfg=4.0,
        upscale=True,
        bundles=["chroma1-hd", "upscalers-esrgan"],
        hint="Painterly brushwork, rich pigments, traditional art composition",
    ),
    "desktop_wallpaper": Preset(
        id="desktop_wallpaper",
        label="Desktop wallpaper (Qwen fast)",
        when_to_use="16:9 landscape desktop wallpaper, scenic backgrounds",
        mode="txt2img",
        loras=[LoraSpec(LORA["lightning"], 1.0)],
        width=1920,
        height=1080,
        steps=4,
        cfg=1.0,
        upscale=True,
        bundles=["qwen-image-2512-fp8", "qwen-image-2512-lightning-lora", "upscalers-esrgan"],
    ),
    "infographic_text": Preset(
        id="infographic_text",
        label="Infographic with legible text",
        when_to_use="Charts, typography, data visualization without style LoRA",
        mode="txt2img",
        width=1024,
        height=1448,
        steps=30,
        cfg=4.0,
        negative_prompt=NEG_INFOGRAPHIC,
        bundles=["qwen-image-2512-fp8"],
        hint="Clear readable typography, structured layout, high contrast labels",
    ),
    "infographic_corp_ad": Preset(
        id="infographic_corp_ad",
        label="Corporate ad poster",
        when_to_use="Corporate posters, infrastructure announcements, photo + icons + headline",
        mode="txt2img",
        loras=[LoraSpec(LORA["advertisement"], 0.85)],
        width=1024,
        height=1280,
        steps=30,
        cfg=4.5,
        negative_prompt=NEG_INFOGRAPHIC,
        bundles=["qwen-image-2512-fp8", "qwen-lora-advertisement"],
    ),
    "infographic_mixed_photo": Preset(
        id="infographic_mixed_photo",
        label="Photo + graphic hybrid poster",
        when_to_use="Photorealistic subject embedded in designed poster layout",
        mode="txt2img",
        loras=[LoraSpec(LORA["advertisement"], 0.75), LoraSpec(LORA["realism"], 0.55)],
        width=1024,
        height=1280,
        steps=32,
        cfg=4.5,
        negative_prompt=NEG_INFOGRAPHIC,
        bundles=["qwen-image-2512-fp8", "qwen-lora-advertisement", "qwen-lora-realism"],
    ),
    "infographic_poster_layout": Preset(
        id="infographic_poster_layout",
        label="Typography-first poster",
        when_to_use="Headline + subhead + body text regions, event posters",
        mode="txt2img",
        loras=[LoraSpec(LORA["poster"], 0.9)],
        width=1024,
        height=1448,
        steps=35,
        cfg=5.0,
        negative_prompt=NEG_INFOGRAPHIC,
        bundles=["qwen-image-2512-fp8", "qwen-lora-poster"],
    ),
    "infographic_flat_icons": Preset(
        id="infographic_flat_icons",
        label="Flat icons & data panels",
        when_to_use="Flat vector icons, maps, stat panels, diagram slides",
        mode="txt2img",
        loras=[LoraSpec(LORA["flat_cartoon"], 0.8)],
        width=1024,
        height=1024,
        steps=28,
        cfg=3.5,
        negative_prompt=NEG_INFOGRAPHIC,
        bundles=["qwen-image-2512-fp8", "qwen-lora-flat-cartoon"],
    ),
    "infographic_eligen": Preset(
        id="infographic_eligen",
        label="Multi-block text poster (EliGen)",
        when_to_use="Multi-block project overview slides, bilingual signage",
        mode="txt2img",
        loras=[LoraSpec(LORA["eligen"], 0.85)],
        width=1024,
        height=1448,
        steps=32,
        cfg=4.5,
        negative_prompt=NEG_INFOGRAPHIC,
        bundles=["qwen-image-2512-fp8", "qwen-lora-eligen-poster"],
    ),
    "infographic_edit": Preset(
        id="infographic_edit",
        label="Edit infographic labels",
        when_to_use="Rewrite labels/values on an existing infographic image",
        mode="edit",
        steps=20,
        cfg=4.0,
        bundles=["qwen-image-2512-fp8", "qwen-image-edit-2511-fp8"],
    ),
    "diagram_flat": Preset(
        id="diagram_flat",
        label="Flat diagram / flowchart",
        when_to_use="Flowcharts, flat vector diagrams, icon sets",
        mode="txt2img",
        loras=[LoraSpec(LORA["flat_cartoon"], 0.8)],
        width=1024,
        height=1024,
        steps=28,
        cfg=3.5,
        bundles=["qwen-image-2512-fp8", "qwen-lora-flat-cartoon"],
    ),
    "machine_studio": Preset(
        id="machine_studio",
        label="Product / machine studio shot",
        when_to_use="Studio product photography, industrial machinery on clean background",
        mode="txt2img",
        loras=[LoraSpec(LORA["lightning"], 1.0)],
        width=1024,
        height=1024,
        steps=4,
        cfg=1.0,
        bundles=["qwen-image-2512-fp8", "qwen-image-2512-lightning-lora"],
        hint="Professional studio lighting, clean backdrop, sharp product detail",
    ),
    "machine_bg_swap": Preset(
        id="machine_bg_swap",
        label="Replace machine photo background",
        when_to_use="Swap background of a product/machine photo while keeping subject",
        mode="edit",
        steps=20,
        cfg=4.0,
        bundles=["qwen-image-2512-fp8", "qwen-image-edit-2511-fp8"],
    ),
    "machine_cutaway": Preset(
        id="machine_cutaway",
        label="Technical cutaway / exploded view",
        when_to_use="Technical cutaway, exploded view, engineering diagrams",
        mode="painterly",
        width=1024,
        height=1024,
        steps=30,
        cfg=4.0,
        bundles=["chroma1-hd"],
        hint="Technical cutaway illustration, labeled components, engineering blueprint style",
    ),
    "scene_pose": Preset(
        id="scene_pose",
        label="Cinematic scene — pose control",
        when_to_use="Movie still guided by reference pose/skeleton from uploaded photo",
        mode="control",
        control_type="pose",
        width=1328,
        height=768,
        steps=30,
        cfg=4.0,
        control_strength=0.85,
        bundles=["qwen-image-2512-fp8", "qwen-controlnet-2512-fun-union"],
        hint="Cinematic movie still, film grain, dramatic lighting",
    ),
    "scene_depth": Preset(
        id="scene_depth",
        label="Cinematic scene — depth control",
        when_to_use="Movie still guided by depth map from reference photo",
        mode="control",
        control_type="depth",
        width=1328,
        height=768,
        steps=30,
        cfg=4.0,
        control_strength=0.85,
        bundles=["qwen-image-2512-fp8", "qwen-controlnet-2512-fun-union"],
    ),
    "scene_canny": Preset(
        id="scene_canny",
        label="Cinematic scene — edge control",
        when_to_use="Movie still guided by edge/layout from reference photo",
        mode="control",
        control_type="canny",
        width=1328,
        height=768,
        steps=30,
        cfg=4.0,
        control_strength=0.85,
        bundles=["qwen-image-2512-fp8", "qwen-controlnet-2512-fun-union"],
    ),
    "scene_bg_replace": Preset(
        id="scene_bg_replace",
        label="Background replace (inpaint)",
        when_to_use="Replace background via inpaint mask while keeping foreground subject",
        mode="bg_replace",
        steps=4,
        cfg=1.0,
        bundles=["qwen-image-2512-fp8", "qwen-controlnet-diffsynth", "qwen-image-2512-lightning-lora"],
    ),
    "scene_pose_styled": Preset(
        id="scene_pose_styled",
        label="Pose + realism LoRA cinematic",
        when_to_use="Film still with pose control plus photorealism LoRA",
        mode="control",
        control_type="pose",
        loras=[LoraSpec(LORA["realism"], 0.6)],
        width=1328,
        height=768,
        steps=30,
        cfg=4.0,
        control_strength=0.85,
        bundles=["qwen-image-2512-fp8", "qwen-controlnet-2512-fun-union", "qwen-lora-realism"],
    ),
    "upscale_only": Preset(
        id="upscale_only",
        label="Upscale only",
        when_to_use="RealESRGAN 4x upscale of an existing image, no generation",
        mode="upscale",
        bundles=["upscalers-esrgan"],
    ),
    "character_master_draft": Preset(
        id="character_master_draft",
        label="Character master (DRAFT)",
        when_to_use="Fast character-identity candidates with Lightning 4-step; never the locked MASTER sheet",
        mode="txt2img",
        loras=[LoraSpec(LORA["lightning"], 1.0)],
        width=1080,
        height=1920,
        steps=4,
        cfg=1.0,
        bundles=["qwen-image-2512-fp8", "qwen-image-2512-lightning-lora"],
        hint="Single consistent character portrait, identity lock sheet, no text",
    ),
    "character_master_master": Preset(
        id="character_master_master",
        label="Character master (MASTER)",
        when_to_use="Full-quality Qwen 2512 character identity sheet; no Lightning or Turbo LoRA",
        mode="txt2img",
        width=1080,
        height=1920,
        steps=30,
        cfg=4.0,
        bundles=["qwen-image-2512-fp8"],
        hint="Single consistent character portrait, identity lock sheet, no text",
    ),
    "general": Preset(
        id="general",
        label="General purpose",
        when_to_use="Default fallback for general image generation requests",
        mode="txt2img",
        loras=[LoraSpec(LORA["lightning"], 1.0)],
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        steps=4,
        cfg=1.0,
        bundles=["qwen-image-2512-fp8", "qwen-image-2512-lightning-lora"],
    ),
}


def list_presets() -> list[dict[str, Any]]:
    return [
        {
            "id": p.id,
            "label": p.label,
            "when_to_use": p.when_to_use,
            "mode": p.mode,
            "bundles": p.bundles,
            "defaults": {
                "width": p.width,
                "height": p.height,
                "steps": p.steps,
                "cfg": p.cfg,
                "upscale": p.upscale,
                "control_type": p.control_type,
                "control_strength": p.control_strength,
                "loras": [{"name": l.name, "strength": l.strength} for l in p.loras],
            },
        }
        for p in PRESETS.values()
    ]


def get_preset(preset_id: str) -> Preset | None:
    return PRESETS.get(preset_id)


def _coerce_dimension(
    explicit: int | None,
    override: Any,
    default: int,
    *,
    min_dim: int = 256,
    max_dim: int = 4096,
) -> int:
    for candidate in (explicit, override):
        if candidate is None:
            continue
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        if min_dim <= value <= max_dim:
            return value
    return default


def resolve(
    preset_id: str,
    *,
    refined_prompt: str,
    overrides: dict[str, Any] | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    image_name: str | None = None,
) -> dict[str, Any]:
    """Merge preset + planner overrides into an executable plan."""
    preset = PRESETS.get(preset_id) or PRESETS["general"]
    ov = overrides or {}

    plan: dict[str, Any] = {
        "preset": preset.id,
        "mode": ov.get("mode", preset.mode),
        "prompt": refined_prompt,
        "negative_prompt": ov.get("negative_prompt", preset.negative_prompt),
        "width": _coerce_dimension(width, ov.get("width"), preset.width),
        "height": _coerce_dimension(height, ov.get("height"), preset.height),
        "steps": ov.get("steps", preset.steps),
        "cfg": ov.get("cfg", preset.cfg),
        "upscale": ov.get("upscale", preset.upscale),
        "loras": deepcopy(ov.get("loras") or [{"name": l.name, "strength": l.strength} for l in preset.loras]),
        "control_type": ov.get("control_type", preset.control_type),
        "control_strength": ov.get("control_strength", preset.control_strength),
        "bundles": preset.bundles,
        "image_name": image_name,
    }

    seed_val = seed if seed is not None else ov.get("seed")
    if seed_val is not None:
        plan["seed"] = seed_val

    if "denoise" in ov:
        plan["denoise"] = ov["denoise"]

    return plan
