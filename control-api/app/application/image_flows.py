from __future__ import annotations

from typing import Any

from .. import presets as legacy_presets
from ..domain.errors import DomainError, ErrorCode

FLOWS = ("t2i", "text_edit", "merge", "semantic_edit", "layered", "control", "lightning")
CONTROL_TYPES = ("pose", "depth", "canny")
LIGHTNING_LORA = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"

FLOW_META: dict[str, dict[str, Any]] = {
    "t2i": {
        "label": "Text-to-Image",
        "mode": "txt2img",
        "operation": "image.generate",
        "min_images": 0,
        "max_images": 1,
        "bundles": ["qwen-image-2512-fp8"],
        "steps": 30,
        "cfg": 4.0,
        "prompt_required": True,
    },
    "text_edit": {
        "label": "In-image text edit",
        "mode": "edit",
        "operation": "image.edit",
        "min_images": 1,
        "max_images": 1,
        "bundles": ["qwen-image-edit-2511-fp8"],
        "steps": 20,
        "cfg": 4.0,
        "prompt_required": True,
    },
    "merge": {
        "label": "Multi-image merge",
        "mode": "edit",
        "operation": "image.edit",
        "min_images": 2,
        "max_images": 3,
        "bundles": ["qwen-image-edit-2511-fp8"],
        "steps": 24,
        "cfg": 4.0,
        "prompt_required": True,
    },
    "semantic_edit": {
        "label": "Semantic edit",
        "mode": "edit",
        "operation": "image.edit",
        "min_images": 1,
        "max_images": 1,
        "bundles": ["qwen-image-edit-2511-fp8"],
        "steps": 20,
        "cfg": 4.0,
        "prompt_required": True,
    },
    "layered": {
        "label": "Layered decomposition",
        "mode": "layered",
        "operation": "image.layered",
        "min_images": 1,
        "max_images": 1,
        "bundles": ["qwen-image-layered", "qwen-image-2512-fp8"],
        "steps": 50,
        "cfg": 4.0,
        "prompt_required": False,
        "width": 640,
        "height": 640,
    },
    "control": {
        "label": "Union ControlNet",
        "mode": "control",
        "operation": "image.controlled",
        "min_images": 1,
        "max_images": 1,
        "bundles": ["qwen-image-2512-fp8", "qwen-controlnet-2512-fun-union"],
        "steps": 30,
        "cfg": 4.0,
        "prompt_required": True,
        "requires_control_type": True,
    },
    "lightning": {
        "label": "4-step Lightning",
        "mode": "txt2img",
        "operation": "image.generate",
        "min_images": 0,
        "max_images": 1,
        "bundles": ["qwen-image-2512-fp8", "qwen-image-2512-lightning-lora"],
        "steps": 4,
        "cfg": 1.0,
        "prompt_required": True,
        "loras": [{"name": LIGHTNING_LORA, "strength": 1.0}],
    },
}


class FlowError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def flow_catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": flow_id,
            "label": spec["label"],
            "operation": spec["operation"],
            "mode": spec["mode"],
            "minImages": spec["min_images"],
            "maxImages": spec["max_images"],
            "bundles": list(spec["bundles"]),
            "promptRequired": spec["prompt_required"],
        }
        for flow_id, spec in FLOW_META.items()
    ]


def resolve_flow_plan(
    flow: str,
    *,
    prompt: str = "",
    image_count: int = 0,
    control_type: str | None = None,
    control_strength: float | None = None,
    layers: int | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    upscale: bool | None = None,
) -> dict[str, Any]:
    if flow not in FLOW_META:
        raise FlowError(f"Unknown flow: {flow}")
    spec = FLOW_META[flow]
    prompt = str(prompt or "").strip()
    if spec["prompt_required"] and not prompt:
        raise FlowError(f"prompt is required for {flow}")
    if image_count < spec["min_images"]:
        raise FlowError(f"{flow} requires at least {spec['min_images']} image(s)")
    if image_count > spec["max_images"]:
        raise FlowError(f"{flow} accepts at most {spec['max_images']} image(s)")

    resolved_control = None
    if spec.get("requires_control_type"):
        resolved_control = (control_type or "").strip() or "pose"
        if resolved_control not in CONTROL_TYPES:
            raise FlowError("controlType must be pose, depth, or canny")

    layer_count = 3
    if flow == "layered":
        layer_count = 3 if layers is None else int(layers)
        if layer_count < 1 or layer_count > 8:
            raise FlowError("layers must be 1-8")

    plan: dict[str, Any] = {
        "flow": flow,
        "mode": spec["mode"],
        "prompt": prompt,
        "negative_prompt": "",
        "width": int(width or spec.get("width") or legacy_presets.DEFAULT_WIDTH),
        "height": int(height or spec.get("height") or legacy_presets.DEFAULT_HEIGHT),
        "steps": int(spec["steps"]),
        "cfg": float(spec["cfg"]),
        "seed": seed,
        "loras": list(spec.get("loras") or []),
        "upscale": bool(upscale) if upscale is not None else False,
        "bundles": list(spec["bundles"]),
        "filename_prefix": "ComfyUI",
    }
    if resolved_control:
        plan["control_type"] = resolved_control
        strength = 0.85 if control_strength is None else float(control_strength)
        if strength < 0 or strength > 2:
            raise FlowError("controlStrength must be between 0 and 2")
        plan["control_strength"] = strength
    if flow == "layered":
        plan["layers"] = layer_count
        plan["width"] = int(width or 640)
        plan["height"] = int(height or 640)
    if flow == "lightning":
        plan["upscale"] = False if upscale is None else bool(upscale)
    return plan


def domain_flow_error(exc: FlowError) -> DomainError:
    return DomainError(ErrorCode.INVALID_REQUEST, str(exc))
