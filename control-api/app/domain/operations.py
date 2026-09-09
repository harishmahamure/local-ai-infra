from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import DomainError, ErrorCode


SAMPLERS: tuple[str, ...] = (
    "euler",
    "euler_ancestral",
    "res_multistep",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
    "ddim",
    "uni_pc",
)
SCHEDULERS: tuple[str, ...] = (
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
)
ASPECTS: tuple[dict[str, int | str], ...] = (
    {"id": "1:1", "width": 1328, "height": 1328},
    {"id": "16:9", "width": 1664, "height": 928},
    {"id": "9:16", "width": 928, "height": 1664},
    {"id": "4:3", "width": 1472, "height": 1104},
    {"id": "3:4", "width": 1104, "height": 1472},
    {"id": "3:2", "width": 1584, "height": 1056},
    {"id": "2:3", "width": 1056, "height": 1584},
)
DEFAULT_SHIFT = 3.1
DEFAULT_NEGATIVE = "blurry, low quality, watermark, text, logo, deformed, ugly, bad anatomy"
DEFAULT_STYLE_PRESET = "cinematic_naturalism"
STYLED_OPERATIONS = frozenset(
    {
        "generate_character",
        "generate_character_turnaround",
        "generate_keyframe",
        "generate_shot_reference",
    }
)
SPECTACLE_NEGATIVE = "halo, aura, glowing skin, glowing eyes, divine rays, magical particles, text, watermark"


@dataclass(frozen=True)
class StylePreset:
    id: str
    label: str
    suffix: str = ""
    negative: str = ""


STYLE_PRESETS: dict[str, StylePreset] = {
    "cinematic_naturalism": StylePreset(
        id="cinematic_naturalism",
        label="Cinematic Naturalism",
        suffix=(
            "Cinematic Naturalism, premium 3D feature-animation quality, "
            "sophisticated stylized realism, physically grounded materials, "
            "naturalistic illumination, subtle surface detail, restrained color, "
            "believable human anatomy, animated character not photoreal live-action"
        ),
        negative=(
            "photoreal live-action, photograph, DSLR, documentary still, "
            "cartoon, anime, chibi, oversized eyes, pixar cute, "
            "elderly saint cliché, heavy blanket sadhu, ornamental guru, "
            "fashion-model face, heroic physique, "
            f"{SPECTACLE_NEGATIVE}"
        ),
    ),
    "painterly_concept": StylePreset(
        id="painterly_concept",
        label="Painterly concept",
        suffix=(
            "painted concept frame, visible brush structure, film-previs finish, "
            "hand-painted cinematic illustration"
        ),
        negative=f"photoreal photograph, 3D render plastic, UI overlay, {SPECTACLE_NEGATIVE}",
    ),
    "graphic_still": StylePreset(
        id="graphic_still",
        label="Graphic still",
        suffix=(
            "graphic illustrated still, designed color, clean shapes, "
            "editorial animation frame"
        ),
        negative=f"photograph, painterly oil texture, photoreal pores, {SPECTACLE_NEGATIVE}",
    ),
    "photoreal_cinematic": StylePreset(
        id="photoreal_cinematic",
        label="Photoreal cinematic",
        suffix="live-action feature still, grounded photography, cinematic film lighting",
        negative=f"cartoon, anime, 3D animation, plastic skin, {SPECTACLE_NEGATIVE}",
    ),
    "documentary": StylePreset(
        id="documentary",
        label="Documentary",
        suffix="observational documentary still, unstyled natural light, no hero lighting",
        negative=f"cinematic grade, fashion beauty, fantasy glow, animation, {SPECTACLE_NEGATIVE}",
    ),
    "off": StylePreset(id="off", label="Off"),
}


@dataclass
class OperationSpec:
    id: str
    label: str
    description: str
    required_bundles: list[str]
    prompt_required: bool = True
    min_images: int = 0
    max_images: int = 0
    requires_mask: bool = False
    requires_image: bool = False
    reference_slots: tuple[str, ...] = ()
    default_width: int = 1328
    default_height: int = 1328
    steps: int = 50
    cfg: float = 4.0
    edit_steps: int = 40
    edit_cfg: float = 3.0


OPERATIONS: dict[str, OperationSpec] = {
    "generate_character": OperationSpec(
        id="generate_character",
        label="Generate character",
        description="Full-body character from a prompt. Optional face reference locks identity.",
        required_bundles=["qwen-image-2512-fp8"],
        max_images=1,
        reference_slots=("image",),
    ),
    "generate_character_turnaround": OperationSpec(
        id="generate_character_turnaround",
        label="Character turnaround",
        description="Front, three-quarter, side, and back views as four assets. Optional character image skips the base render.",
        required_bundles=["qwen-image-2512-fp8", "qwen-image-edit-2511-fp8"],
        max_images=1,
        reference_slots=("character", "image"),
    ),
    "generate_attire": OperationSpec(
        id="generate_attire",
        label="Generate attire",
        description="Standalone garment on a neutral background, or dress a character reference.",
        required_bundles=["qwen-image-2512-fp8", "qwen-image-edit-2511-fp8"],
        max_images=1,
        reference_slots=("character", "image"),
    ),
    "generate_location": OperationSpec(
        id="generate_location",
        label="Generate location",
        description="Wide establishing environment.",
        required_bundles=["qwen-image-2512-fp8"],
        default_width=1664,
        default_height=928,
    ),
    "generate_prop": OperationSpec(
        id="generate_prop",
        label="Generate prop",
        description="Single hero prop on a neutral studio background.",
        required_bundles=["qwen-image-2512-fp8"],
    ),
    "generate_keyframe": OperationSpec(
        id="generate_keyframe",
        label="Generate keyframe",
        description="Story beat still using up to three references: character, attire, location.",
        required_bundles=["qwen-image-edit-2511-fp8"],
        min_images=1,
        max_images=3,
        reference_slots=("character", "attire", "location", "image"),
        default_width=1664,
        default_height=928,
    ),
    "generate_shot_reference": OperationSpec(
        id="generate_shot_reference",
        label="Generate shot reference",
        description="Camera-aware still with framing, lens, and height, using the same references as a keyframe.",
        required_bundles=["qwen-image-edit-2511-fp8"],
        min_images=1,
        max_images=3,
        reference_slots=("character", "attire", "location", "image"),
        default_width=1664,
        default_height=928,
    ),
    "inpaint_asset": OperationSpec(
        id="inpaint_asset",
        label="Inpaint asset",
        description="Regenerate the masked region of an image.",
        required_bundles=["qwen-image-2512-fp8", "qwen-controlnet-diffsynth"],
        prompt_required=True,
        min_images=1,
        max_images=1,
        requires_image=True,
        requires_mask=True,
        reference_slots=("image", "mask"),
    ),
    "outpaint_asset": OperationSpec(
        id="outpaint_asset",
        label="Outpaint asset",
        description="Expand an image with per-edge padding.",
        required_bundles=["qwen-image-2512-fp8", "qwen-controlnet-diffsynth"],
        prompt_required=True,
        min_images=1,
        max_images=1,
        requires_image=True,
        reference_slots=("image",),
    ),
    "upscale_asset": OperationSpec(
        id="upscale_asset",
        label="Upscale asset",
        description="RealESRGAN 2x or 4x upscale.",
        required_bundles=["upscalers-esrgan"],
        prompt_required=False,
        min_images=1,
        max_images=1,
        requires_image=True,
        reference_slots=("image",),
        steps=0,
        cfg=0.0,
        edit_steps=0,
        edit_cfg=0.0,
    ),
}

TURNAROUND_VIEWS: tuple[tuple[str, str], ...] = (
    ("front", "front view, facing camera, full body, orthographic character-sheet pose, even studio lighting, neutral background"),
    ("three_quarter", "three-quarter view, 3/4 angle, full body, same character identity and costume, even studio lighting, neutral background"),
    ("side", "side view, strict profile, full body, same character identity and costume, even studio lighting, neutral background"),
    ("back", "back view, from behind, full body, same character identity and costume, even studio lighting, neutral background"),
)

FACE_LOCK_PREFIX = (
    "Generate the character of the person in image 1. "
)
FACE_LOCK_SUFFIX = (
    "Preserve the exact face, bone structure, age, skin, and likeness from the reference."
)
FACE_LOCK_NEGATIVE = (
    "different face, identity change, extra people, beauty-filter morph, face swap, "
    "changed bone structure, different person"
)

ATTIRE_STANDALONE = (
    "product photography of the garment only, on a dress form / invisible mannequin, "
    "centered, studio lighting, seamless neutral background, no person, no face"
)
PROP_STANDALONE = (
    "hero product shot of a single prop, centered, studio lighting, seamless neutral background, "
    "no people, no text, no watermark"
)
SHOT_GRAMMAR = (
    "cinematic production still, {framing} shot, {lens}mm lens, camera at {height}, "
    "consistent character identity from the references, film lighting"
)


def compose_face_lock(description: str, negative: str = "") -> tuple[str, str]:
    desc = str(description or "").strip()
    if desc:
        positive = f"{FACE_LOCK_PREFIX}{desc}. {FACE_LOCK_SUFFIX}"
    else:
        positive = f"{FACE_LOCK_PREFIX}{FACE_LOCK_SUFFIX}"
    extra = str(negative or "").strip()
    neg = FACE_LOCK_NEGATIVE if not extra else f"{FACE_LOCK_NEGATIVE}, {extra}"
    return positive, neg


def resolve_style_preset(value: Any) -> StylePreset:
    if value is None or str(value).strip() == "":
        key = DEFAULT_STYLE_PRESET
    else:
        key = str(value).strip()
    preset = STYLE_PRESETS.get(key)
    if preset is None:
        raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unknown style preset {key}")
    return preset


def operation_capabilities() -> dict[str, Any]:
    return {
        "samplers": list(SAMPLERS),
        "schedulers": list(SCHEDULERS),
        "aspects": [dict(item) for item in ASPECTS],
        "stylePresets": [{"id": item.id, "label": item.label} for item in STYLE_PRESETS.values()],
    }


def operation_catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "description": spec.description,
            "requiredBundles": list(spec.required_bundles),
            "promptRequired": spec.prompt_required,
            "minImages": spec.min_images,
            "maxImages": spec.max_images,
            "requiresMask": spec.requires_mask,
            "requiresImage": spec.requires_image,
            "referenceSlots": list(spec.reference_slots),
            "defaultWidth": spec.default_width,
            "defaultHeight": spec.default_height,
            "defaultSteps": spec.edit_steps if spec.required_bundles == ["qwen-image-edit-2511-fp8"] else spec.steps,
            "defaultCfg": spec.edit_cfg if spec.required_bundles == ["qwen-image-edit-2511-fp8"] else spec.cfg,
            "defaultShift": DEFAULT_SHIFT,
        }
        for spec in OPERATIONS.values()
    ]
