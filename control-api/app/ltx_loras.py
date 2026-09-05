"""Official LTX camera / IC-LoRA catalog and prompt-based selection."""

from __future__ import annotations

from typing import Any

CAMERA_BUNDLE = "ltx-camera-loras"
IC_UNION_BUNDLE = "ltx-iclora-union"
IC_DETAILER_BUNDLE = "ltx-iclora-detailer"
IC_LIPDUB_BUNDLE = "ltx-iclora-lipdub"
IC_MOTION_BUNDLE = "ltx-iclora-motion-track"

CAMERA_MOTIONS: dict[str, dict[str, Any]] = {
    "dolly_in": {
        "id": "dolly_in",
        "label": "Dolly in",
        "file": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
        "bundle": CAMERA_BUNDLE,
        "keywords": (
            "dolly in",
            "dolly-in",
            "dolly forward",
            "dolly towards",
            "push in",
            "push-in",
            "push forward",
            "pushes in",
            "camera pushes",
            "zoom in",
            "zooms in",
            "moves closer",
            "moving closer",
            "moves toward",
            "moving toward",
        ),
    },
    "dolly_out": {
        "id": "dolly_out",
        "label": "Dolly out",
        "file": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
        "bundle": CAMERA_BUNDLE,
        "keywords": (
            "dolly out",
            "dolly-out",
            "dolly back",
            "pull back",
            "pulls back",
            "pulling back",
            "push out",
            "zoom out",
            "zooms out",
            "moves away",
            "moving away",
            "reveals the",
        ),
    },
    "dolly_left": {
        "id": "dolly_left",
        "label": "Dolly left",
        "file": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
        "bundle": CAMERA_BUNDLE,
        "keywords": (
            "dolly left",
            "dolly-left",
            "truck left",
            "track left",
            "tracking left",
            "slides left",
            "slide left",
            "moves left",
            "moving left",
        ),
    },
    "dolly_right": {
        "id": "dolly_right",
        "label": "Dolly right",
        "file": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
        "bundle": CAMERA_BUNDLE,
        "keywords": (
            "dolly right",
            "dolly-right",
            "truck right",
            "track right",
            "tracking right",
            "slides right",
            "slide right",
            "moves right",
            "moving right",
        ),
    },
    "jib_up": {
        "id": "jib_up",
        "label": "Jib up",
        "file": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
        "bundle": CAMERA_BUNDLE,
        "keywords": (
            "jib up",
            "jib-up",
            "crane up",
            "crane-up",
            "tilt up",
            "tilts up",
            "rises up",
            "camera rises",
            "lifts up",
        ),
    },
    "jib_down": {
        "id": "jib_down",
        "label": "Jib down",
        "file": "ltx-2-19b-lora-camera-control-jib-down.safetensors",
        "bundle": CAMERA_BUNDLE,
        "keywords": (
            "jib down",
            "jib-down",
            "crane down",
            "crane-down",
            "tilt down",
            "tilts down",
            "lowers",
            "camera descends",
        ),
    },
    "static": {
        "id": "static",
        "label": "Static",
        "file": "ltx-2-19b-lora-camera-control-static.safetensors",
        "bundle": CAMERA_BUNDLE,
        "keywords": (
            "static camera",
            "locked off",
            "locked-off",
            "locked off camera",
            "tripod",
            "no camera movement",
            "camera locked",
            "fixed camera",
        ),
    },
}

IC_LORAS: dict[str, dict[str, Any]] = {
    "union": {
        "id": "union",
        "label": "Union Control (depth / canny / pose)",
        "file": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "bundle": IC_UNION_BUNDLE,
        "needs_video": True,
    },
    "detailer": {
        "id": "detailer",
        "label": "Detailer",
        "file": "ltx-2-19b-ic-lora-detailer.safetensors",
        "bundle": IC_DETAILER_BUNDLE,
        "needs_video": True,
    },
    "lipdub": {
        "id": "lipdub",
        "label": "LipDub",
        "file": "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors",
        "bundle": IC_LIPDUB_BUNDLE,
        "needs_video": False,
        "explicit_only": True,
    },
    "motion_track": {
        "id": "motion_track",
        "label": "Motion Track",
        "file": "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        "bundle": IC_MOTION_BUNDLE,
        "needs_video": True,
        "explicit_only": True,
    },
}

CONTROL_TYPES = ("depth", "canny", "pose")

POSE_KEYWORDS = (
    "dance",
    "dancing",
    "choreograph",
    "pose",
    "skeleton",
    "walk cycle",
    "walking",
    "running",
    "body motion",
    "human motion",
)
CANNY_KEYWORDS = (
    "architecture",
    "edges",
    "canny",
    "product shot",
    "geometric",
    "cityscape",
    "mechanical",
    "building facade",
)
DETAILER_KEYWORDS = (
    "more detail",
    "sharper",
    "detailer",
    "enhance details",
    "fine detail",
    "micro detail",
)

CAMERA_MOTION_VALUES = ("auto", "none", *CAMERA_MOTIONS)
IC_LORA_VALUES = ("auto", "none", *IC_LORAS)
CONTROL_TYPE_VALUES = ("auto", *CONTROL_TYPES)


def list_camera_motions() -> list[dict[str, Any]]:
    return [
        {"id": "auto", "label": "Auto (from prompt)"},
        {"id": "none", "label": "None"},
        *[
            {"id": spec["id"], "label": spec["label"], "file": spec["file"], "bundle": spec["bundle"]}
            for spec in CAMERA_MOTIONS.values()
        ],
    ]


def list_ic_loras() -> list[dict[str, Any]]:
    return [
        {"id": "auto", "label": "Auto (from prompt + reference video)"},
        {"id": "none", "label": "None"},
        *[
            {
                "id": spec["id"],
                "label": spec["label"],
                "file": spec["file"],
                "bundle": spec["bundle"],
                "needsVideo": bool(spec.get("needs_video", True)),
                "explicitOnly": bool(spec.get("explicit_only")),
            }
            for spec in IC_LORAS.values()
        ],
    ]


def explicit_ic_lora(ic_id: str, strength: float = 1.0) -> dict[str, Any]:
    spec = IC_LORAS[ic_id]
    return {
        "name": spec["file"],
        "strength": max(0.0, min(1.0, float(strength))),
        "id": spec["id"],
        "kind": "iclora",
        "bundle": spec["bundle"],
    }


def list_control_types() -> list[dict[str, str]]:
    return [
        {"id": "auto", "label": "Auto (from prompt)"},
        {"id": "depth", "label": "Depth"},
        {"id": "canny", "label": "Canny edges"},
        {"id": "pose", "label": "Pose (DWPose)"},
    ]


def _normalize_choice(value: str | None, allowed: tuple[str, ...], default: str) -> str:
    raw = (value or default).strip().lower().replace("-", "_").replace(" ", "_")
    return raw if raw in allowed else default


def _first_keyword_hit(text: str, keywords: tuple[str, ...]) -> str | None:
    for key in keywords:
        if key in text:
            return key
    return None


def _match_camera(prompt: str) -> str | None:
    text = prompt.lower()
    for motion_id, spec in CAMERA_MOTIONS.items():
        if _first_keyword_hit(text, spec["keywords"]):
            return motion_id
    return None


def _match_control(prompt: str) -> str:
    text = prompt.lower()
    if _first_keyword_hit(text, POSE_KEYWORDS):
        return "pose"
    if _first_keyword_hit(text, CANNY_KEYWORDS):
        return "canny"
    return "depth"


def _wants_detailer(prompt: str, detailer: bool) -> bool:
    if detailer:
        return True
    return bool(_first_keyword_hit(prompt.lower(), DETAILER_KEYWORDS))


def decide(
    prompt: str,
    *,
    camera_motion: str | None = "auto",
    ic_lora: str | None = "auto",
    control_type: str | None = "auto",
    detailer: bool = False,
    has_reference_video: bool = False,
    lora_strength: float = 1.0,
    ic_lora_strength: float = 1.0,
) -> dict[str, Any]:
    """Pick at most one camera LoRA and optional IC-LoRA(s) from prompt + overrides."""
    camera_choice = _normalize_choice(camera_motion, CAMERA_MOTION_VALUES, "auto")
    ic_choice = _normalize_choice(ic_lora, IC_LORA_VALUES, "auto")
    control_choice = _normalize_choice(control_type, CONTROL_TYPE_VALUES, "auto")
    strength = max(0.0, min(2.0, float(lora_strength)))
    ic_strength = max(0.0, min(1.0, float(ic_lora_strength)))

    camera_id: str | None = None
    camera_source = "none"
    if camera_choice == "none":
        camera_source = "override"
    elif camera_choice in CAMERA_MOTIONS:
        camera_id = camera_choice
        camera_source = "override"
    else:
        camera_id = _match_camera(prompt)
        camera_source = "prompt" if camera_id else "none"

    loras: list[dict[str, Any]] = []
    bundles: list[str] = []
    if camera_id:
        spec = CAMERA_MOTIONS[camera_id]
        loras.append({"name": spec["file"], "strength": strength, "id": camera_id, "kind": "camera"})
        bundles.append(spec["bundle"])

    ic_ids: list[str] = []
    ic_skip: str | None = None
    resolved_control: str | None = None
    ic_source = "none"

    wants_detailer = _wants_detailer(prompt, detailer)
    if ic_choice == "none" and not wants_detailer:
        ic_source = "override"
    elif not has_reference_video:
        if ic_choice != "none" or wants_detailer:
            ic_skip = "reference video required"
        ic_source = "skipped"
    else:
        if ic_choice == "union":
            ic_ids.append("union")
            ic_source = "override"
        elif ic_choice == "detailer":
            ic_ids.append("detailer")
            ic_source = "override"
        elif ic_choice == "auto":
            if wants_detailer and not (
                _first_keyword_hit(prompt.lower(), POSE_KEYWORDS)
                or _first_keyword_hit(prompt.lower(), CANNY_KEYWORDS)
            ):
                ic_ids.append("detailer")
            else:
                ic_ids.append("union")
            ic_source = "prompt"
        if wants_detailer and "detailer" not in ic_ids and ic_choice != "none":
            ic_ids.append("detailer")
            if ic_source == "none":
                ic_source = "override" if detailer else "prompt"
        if "union" in ic_ids:
            resolved_control = control_choice if control_choice != "auto" else _match_control(prompt)

    ic_loras: list[dict[str, Any]] = []
    for ic_id in ic_ids:
        spec = IC_LORAS[ic_id]
        ic_loras.append(
            {
                "name": spec["file"],
                "strength": ic_strength,
                "id": ic_id,
                "kind": "iclora",
            }
        )
        bundles.append(spec["bundle"])

    return {
        "camera_motion": camera_id,
        "camera_motion_source": camera_source,
        "loras": loras,
        "ic_lora": ic_ids[0] if ic_ids else None,
        "ic_loras": ic_loras,
        "ic_enabled": bool(ic_loras),
        "control_type": resolved_control,
        "ic_skip": ic_skip,
        "ic_source": ic_source,
        "lora_strength": strength,
        "ic_lora_strength": ic_strength,
        "bundles": list(dict.fromkeys(bundles)),
        "selected": {
            "camera": camera_id,
            "ic": ic_ids,
            "controlType": resolved_control,
        },
    }
