from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .camera import camera_catalog
from .operations import (
    DEFAULT_LIVE_DURATION,
    DEFAULT_LIVE_FPS,
    LIVE_WALLPAPER_TARGETS,
    OperationSpec,
)

SHOT_SUFFIX = (
    "cinematic motion, continuous shot, rich material detail, "
    "no text, no watermark, no UI overlay, no morph, no identity change"
)
SHOT_NEGATIVE = (
    "blurry, jitter, still frame, hard cuts, watermark, text, logo, morphing, warping, flicker"
)
ACTION_SUFFIXES: dict[str, str] = {
    "battle": "intense battle action, physical impact, debris, kinetic energy",
    "destruction": "structural destruction, collapsing material, dust and debris",
    "magic": "visible magical energy, glowing particles, controlled spectacle",
    "particles": "dense particle motion, embers, sparks, atmospheric volume",
    "weather": "dramatic weather, wind-driven atmosphere, rain or storm motion",
}


@dataclass(frozen=True)
class InputSlot:
    id: str
    label: str
    kind: str
    required: bool = False
    multiple: bool = False


@dataclass(frozen=True)
class Guide:
    key: str
    frame_idx: int
    strength: float = 1.0
    kind: str = "image"
    mask_key: str | None = None
    signal: str | None = None


@dataclass(frozen=True)
class ControlSpec:
    kind: str
    lora: str
    strength: float = 1.0
    source_key: str | None = None


@dataclass(frozen=True)
class FlowSpec:
    id: str
    label: str
    description: str
    slots: tuple[InputSlot, ...]
    stages: tuple[str, ...]
    required_bundles: list[str] = field(default_factory=list)
    prompt_required: bool = True
    camera: bool = False
    control: bool = False
    keyframes: bool = False
    parent_job: bool = False
    clips: bool = False
    action: bool = False
    min_images: int = 0
    max_images: int = 0

    def to_operation_spec(self) -> OperationSpec:
        return OperationSpec(
            id=self.id,
            label=self.label,
            description=self.description,
            required_bundles=list(self.required_bundles),
            prompt_required=self.prompt_required,
            min_images=self.min_images,
            max_images=self.max_images,
            requires_image=any(slot.id == "image" and slot.required for slot in self.slots),
            reference_slots=tuple(slot.id for slot in self.slots),
            default_width=1216,
            default_height=704,
            steps=8,
            cfg=1.0,
            edit_steps=8,
            edit_cfg=1.0,
        )


_CORE = ["ltx-2.5-distilled", "ltx-2.5-studio"]
_COMPOSE = ["qwen-image-edit-2511-fp8", *_CORE]
_CONTROL = ["ltx-2.5-distilled", "ltx-2.5-studio", "ltx-2.5-control"]

FLOWS: dict[str, FlowSpec] = {
    "shot_single_image": FlowSpec(
        id="shot_single_image",
        label="Single image I2V",
        description="Animate a start frame into a continuous shot.",
        slots=(InputSlot("image", "Start frame", "image", required=True),),
        stages=("shot",),
        required_bundles=list(_CORE),
        min_images=1,
        max_images=1,
    ),
    "shot_start_end": FlowSpec(
        id="shot_start_end",
        label="Start + end frame",
        description="Transition from an exact start composition to a desired ending.",
        slots=(
            InputSlot("image", "Start frame", "image", required=True),
            InputSlot("end_image", "End frame", "image", required=True),
        ),
        stages=("shot",),
        required_bundles=list(_CORE),
        min_images=2,
        max_images=2,
    ),
    "shot_multi_reference": FlowSpec(
        id="shot_multi_reference",
        label="Multi-reference shot",
        description="Compose a start frame from character, location, and prop refs, then animate.",
        slots=(
            InputSlot("image", "Start frame", "image"),
            InputSlot("character", "Character", "image"),
            InputSlot("location", "Location", "image"),
            InputSlot("prop", "Prop", "image"),
        ),
        stages=("compose", "shot"),
        required_bundles=list(_COMPOSE),
        min_images=1,
        max_images=4,
    ),
    "shot_multi_keyframe": FlowSpec(
        id="shot_multi_keyframe",
        label="Multi-keyframe shot",
        description="Progress through ordered visual states.",
        slots=(
            InputSlot("image", "Start frame", "image", required=True),
            InputSlot("keyframes", "Keyframes", "images", multiple=True),
        ),
        stages=("shot",),
        required_bundles=list(_CORE),
        keyframes=True,
        min_images=2,
        max_images=12,
    ),
    "shot_continuation": FlowSpec(
        id="shot_continuation",
        label="Previous shot continuation",
        description="Seamless next shot from the previous clip's trailing motion, plus optional refs.",
        slots=(
            InputSlot("video", "Previous clip", "video", required=True),
            InputSlot("image", "Optional start override", "image"),
            InputSlot("character", "Character ref", "image"),
        ),
        stages=("extract", "shot"),
        required_bundles=list(_CORE),
    ),
    "shot_motion_reference": FlowSpec(
        id="shot_motion_reference",
        label="Image + motion reference",
        description="Transfer camera, body, or action motion from a reference video.",
        slots=(
            InputSlot("image", "Start frame", "image", required=True),
            InputSlot("video", "Motion reference", "video", required=True),
        ),
        stages=("control", "shot"),
        required_bundles=list(_CONTROL),
        control=True,
        min_images=1,
        max_images=1,
    ),
    "shot_character_interaction": FlowSpec(
        id="shot_character_interaction",
        label="Character interaction",
        description="Dialogue, walking together, fighting — scene plus multiple character refs.",
        slots=(
            InputSlot("image", "Scene", "image"),
            InputSlot("character", "Character A", "image", required=True),
            InputSlot("character_b", "Character B", "image"),
            InputSlot("location", "Location", "image"),
        ),
        stages=("compose", "shot"),
        required_bundles=list(_COMPOSE),
        min_images=1,
        max_images=4,
    ),
    "shot_camera": FlowSpec(
        id="shot_camera",
        label="Camera motion shot",
        description="Dolly, orbit, crane, tracking, handheld, or push-in from a start frame.",
        slots=(InputSlot("image", "Start frame", "image", required=True),),
        stages=("control", "shot"),
        required_bundles=list(_CONTROL),
        camera=True,
        min_images=1,
        max_images=1,
    ),
    "shot_action": FlowSpec(
        id="shot_action",
        label="Action / VFX shot",
        description="Battles, destruction, magic, particles, weather.",
        slots=(
            InputSlot("image", "Start frame", "image", required=True),
            InputSlot("end_image", "Optional end frame", "image"),
        ),
        stages=("shot",),
        required_bundles=list(_CORE),
        action=True,
        min_images=1,
        max_images=2,
    ),
    "shot_regenerate": FlowSpec(
        id="shot_regenerate",
        label="Regeneration / repair",
        description="Retry a previous shot with a new seed while keeping approved inputs.",
        slots=(InputSlot("parent_job_id", "Parent job", "job", required=True),),
        stages=("shot",),
        required_bundles=list(_CORE),
        parent_job=True,
        prompt_required=False,
    ),
    "shot_extend": FlowSpec(
        id="shot_extend",
        label="Extend shot",
        description="Add another 5–10 seconds from the last frames of an existing clip.",
        slots=(InputSlot("video", "Existing clip", "video", required=True),),
        stages=("extract", "shot"),
        required_bundles=list(_CORE),
    ),
    "shot_stitch": FlowSpec(
        id="shot_stitch",
        label="Shot stitch / post",
        description="Join, crossfade, interpolate, upscale, and keep audio from generated clips.",
        slots=(InputSlot("clips", "Clips", "videos", required=True, multiple=True),),
        stages=("post",),
        required_bundles=["ltx-2.5-distilled", "upscalers-esrgan"],
        prompt_required=False,
        clips=True,
    ),
}

SHOT_OPERATIONS: dict[str, OperationSpec] = {flow.id: flow.to_operation_spec() for flow in FLOWS.values()}


def profile_for_flow(flow_id: str) -> str:
    flow = FLOWS.get(flow_id)
    if flow and flow.stages and flow.stages[0] == "compose":
        return "comfyui"
    return "comfy-ltx"


def shot_flow_catalog() -> list[dict[str, Any]]:
    items = []
    for flow in FLOWS.values():
        items.append(
            {
                "id": flow.id,
                "label": flow.label,
                "description": flow.description,
                "kind": "shot",
                "requiredBundles": list(flow.required_bundles),
                "promptRequired": flow.prompt_required,
                "referenceSlots": [slot.id for slot in flow.slots],
                "slots": [
                    {
                        "id": slot.id,
                        "label": slot.label,
                        "kind": slot.kind,
                        "required": slot.required,
                        "multiple": slot.multiple,
                    }
                    for slot in flow.slots
                ],
                "stages": list(flow.stages),
                "camera": flow.camera,
                "control": flow.control,
                "keyframes": flow.keyframes,
                "parentJob": flow.parent_job,
                "clips": flow.clips,
                "action": flow.action,
                "defaultWidth": 1216,
                "defaultHeight": 704,
                "defaultSteps": 8,
                "defaultCfg": 1.0,
                "defaultDuration": DEFAULT_LIVE_DURATION,
                "defaultFps": DEFAULT_LIVE_FPS,
                "defaultRefine": True,
                "profile": profile_for_flow(flow.id),
            }
        )
    return items


def shot_flow_capabilities() -> dict[str, Any]:
    return {
        "targets": list(LIVE_WALLPAPER_TARGETS),
        "defaultDuration": DEFAULT_LIVE_DURATION,
        "defaultFps": DEFAULT_LIVE_FPS,
        "defaultRefine": True,
        "maxDuration": 30,
        "cameras": camera_catalog(),
        "controls": ["canny", "depth", "pose", "motion_track"],
        "actions": list(ACTION_SUFFIXES),
        "preferNvfp4": True,
    }
