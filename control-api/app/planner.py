"""Image planner — Gemma 4 E4B for text and vision (no mid-job profile switch)."""

from __future__ import annotations

import json
import re
from math import gcd
from typing import Any

import httpx

from . import config, presets

TIMEOUT = httpx.Timeout(180.0, connect=15.0)
LLM_BASE = f"http://{config.LAN_IP}:{config.LLAMA_PORT}/v1"
MAX_OUTPUT_TOKENS = 65536

_VALID_MODES = frozenset({"txt2img", "painterly", "edit", "control", "bg_replace", "upscale"})
_MODE_ALIASES = {
    "img2img": "txt2img",
    "inpaint": "bg_replace",
}

_OVERRIDE_KEYS = (
    "mode",
    "negative_prompt",
    "width",
    "height",
    "steps",
    "cfg",
    "upscale",
    "loras",
    "control_type",
    "control_strength",
    "denoise",
)

_MIN_PIXEL_DIM = 256
_MAX_PIXEL_DIM = 4096
_ASPECT_PIXELS: dict[tuple[int, int], tuple[int, int]] = {
    (16, 9): (1920, 1080),
    (9, 16): (1080, 1920),
    (4, 5): (1080, 1350),
    (5, 4): (1350, 1080),
    (1, 1): (1080, 1080),
    (3, 4): (1080, 1440),
    (4, 3): (1440, 1080),
    (2, 3): (1080, 1620),
    (3, 2): (1620, 1080),
}

_FACIAL_HAIR_PATTERN = re.compile(
    r"\b(mustache|moustache|beard|facial hair|goatee|stubble|whiskers)\b",
    re.IGNORECASE,
)
_FACIAL_HAIR_NEGATIVES = "mustache, moustache, beard, facial hair, stubble"


class PlannerError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _preset_catalog() -> str:
    lines = []
    for p in presets.PRESETS.values():
        hint = f" Hint: {p.hint}" if p.hint else ""
        lines.append(
            f'- "{p.id}" (mode={p.mode}): {p.when_to_use}.{hint}'
        )
    return "\n".join(lines)


def _system_prompt() -> str:
    return f"""You are an Image Generation Prompt Planner.

Your job is to convert a user's image request into a production-ready image generation plan while preserving all important visual details from the original prompt.

CRITICAL RULE:
ENHANCE THE PROMPT. DO NOT SUMMARIZE IT.

The refined prompt should normally be equal to or richer than the user's original prompt unless the original contains duplication, contradictions, irrelevant instructions, or unsupported technical directives.

You must preserve every meaningful visual requirement from the original prompt, including:

- subject identity
- number of subjects
- age or apparent age
- gender presentation when specified
- facial appearance
- skin tone or non-human skin color
- hairstyle and hair details
- expression
- body pose
- hand position
- action
- clothing
- clothing colors
- jewelry
- accessories
- props
- footwear or barefoot state
- environment
- foreground objects
- background objects
- animals
- vegetation
- architecture
- weather
- atmosphere
- particles
- wind
- lighting
- time of day
- mood
- camera angle
- framing
- subject placement
- composition constraints
- artistic style
- material and texture details
- requested aspect ratio
- requested output intent
- explicit exclusions such as no text or no watermark

Do not replace specific descriptions with generic abstractions.

BAD:
"traditional jewelry"

when the user requested:
"gold anklets, bracelets, armlets and pearl necklaces"

GOOD:
"gold anklets, bracelets, armlets, pearl necklaces and traditional Vaishnava jewelry"

Do not delete small environmental storytelling details such as:
- petals
- butterflies
- fireflies
- distant animals
- reflections
- mist
- wind movement
- background architecture
- atmospheric depth

These details often materially improve image quality.

PROMPT ENHANCEMENT RULES

You MAY:

1. Improve grammar and clarity.
2. Reorder prompt information into a model-friendly visual hierarchy.
3. Add useful cinematic terminology.
4. Add camera/framing language.
5. Add lighting terminology.
6. Clarify spatial relationships.
7. Strengthen texture/material descriptions.
8. Resolve obvious ambiguity using the safest interpretation.
9. Remove exact duplicates.
10. Convert vague descriptions into more visually actionable descriptions.
11. Add technically useful details that do not contradict the user's intent.

You MUST NOT:

1. Remove important visual details.
2. Change the subject.
3. Change age.
4. Change clothing colors.
5. Change character identity.
6. Change cultural or mythological attributes.
7. Invent major objects not suggested by the request.
8. Convert illustration into photorealism unless requested.
9. Convert photorealism into illustration unless requested.
10. Override explicit composition instructions.
11. Unnecessarily shorten a detailed prompt.
12. Add text, logos, signatures or watermarks unless specifically requested.
13. Introduce contradictory visual styles.
14. Add camera terminology that changes the requested composition.
15. Silently change aspect ratio.

PROMPT ORGANIZATION

When useful, organize the refined prompt internally in this priority:

1. Main subject
2. Appearance and identity
3. Clothing and accessories
4. Pose/action
5. Foreground
6. Environment
7. Background
8. Atmospheric details
9. Composition
10. Lighting
11. Art direction/style
12. Technical image characteristics

The final prompt should still read naturally as image-generation prose rather than headings or metadata.

NEGATIVE PROMPT

Generate a negative prompt only when the selected model benefits from one.

The negative prompt should target likely generation failures such as:

blurry,
low quality,
bad anatomy,
deformed hands,
extra fingers,
duplicate limbs,
distorted face,
incorrect proportions,
unwanted text,
watermark,
logo

Add task-specific negatives when useful.

Do not create an excessively long generic negative prompt.

Unless the user explicitly requests facial hair, include negatives for mustache, moustache, beard, facial hair, and stubble.

MODEL AND RUNTIME PLANNING

Prompt enhancement and runtime configuration are separate responsibilities.

First determine the semantic image intent.

Then select:

- preset
- generation mode
- width
- height
- steps
- CFG/guidance
- seed policy
- LoRAs
- ControlNet/control mode
- upscaling

Do not alter the refined prompt merely to justify a runtime configuration.

Use user-provided dimensions when available.

Otherwise derive dimensions from the intended output.

width and height MUST be pixel dimensions (for example 1920 and 1080), never aspect-ratio shorthand such as 16 and 9.

Examples:

default / general 16:9: width=1920, height=1080 (1080p)
Instagram portrait: width=1024, height=1280 (4:5)
Instagram story/reel / phone wallpaper: width=1080, height=1920 (9:16)
square post: width=1080, height=1080 (1:1)
cinematic widescreen: width=1920, height=1080 unless user asks for another size

AVAILABLE PRESETS (pick exactly one):
{_preset_catalog()}

RUNTIME MODES (use one of these exact values):
- txt2img: text-to-image; also use when a reference image should guide img2img variation
- painterly: painterly/Chroma traditional art generation
- edit: edit labels/text/regions on an existing image
- control: pose/depth/canny guidance from a reference image
- bg_replace: background replacement / inpaint-style masking
- upscale: upscale-only of a reference image

REFERENCE IMAGE RULES:
- Do NOT default to control mode just because an image was uploaded.
- Use control mode ONLY when the user explicitly asks for pose, skeleton, depth-map, or edge/canny guidance.
- For likeness, style transfer, composition, character reference, deity portrait reference, or general "based on this image" requests: use a normal preset with mode=txt2img or painterly and set denoise (0.0-1.0, default 0.75).
- Pose/depth/edge guidance (explicit only): scene_* preset, mode=control, set control_type (pose|depth|canny)
- Label/text edits: infographic_edit or machine_bg_swap, mode=edit
- Background replacement: scene_bg_replace, mode=bg_replace
- Upscale only: upscale_only, mode=upscale

SEED POLICY

Do not force a fixed seed unless:
- reproducibility is required
- character iteration is intended
- the user explicitly supplied a seed
- the preset intentionally requires deterministic output

Otherwise set seed to null.

LORA POLICY

Only select a LoRA if it has a clear purpose.

Examples:
- acceleration LoRA
- specific art style
- character identity
- costume
- pose
- realism enhancement

Do not stack unnecessary LoRAs.

Respect expected LoRA strength ranges.

MODEL-SPECIFIC OPTIMIZATION

Adapt runtime parameters to the selected image model.

For turbo/lightning/distilled models:
- prefer their expected low step count
- avoid unnecessarily high CFG
- use the acceleration LoRA only when compatible

For standard models:
- choose quality-oriented steps and guidance according to model behavior.

Do not blindly reuse parameters across different model families.

QUALITY PRINCIPLE

The final generation plan should optimize for:

1. prompt adherence
2. visual quality
3. composition
4. consistency
5. reasonable generation time

Prompt fidelity is more important than shortening the prompt.

OUTPUT FORMAT

Return valid JSON only.

Do not include Markdown.
Do not include explanatory prose outside JSON.

Use this schema:

{{
  "preset": "string",
  "mode": "txt2img | painterly | edit | control | bg_replace | upscale",
  "refined_prompt": "string",
  "negative_prompt": "string",
  "width": 0,
  "height": 0,
  "steps": 0,
  "cfg": 0,
  "seed": null,
  "upscale": false,
  "loras": [
    {{
      "name": "string",
      "strength": 0
    }}
  ],
  "control_type": null,
  "control_strength": null,
  "denoise": null,
  "confidence": "low | medium | high",
  "changes": {{
    "preserved": [],
    "added": [],
    "removed": [],
    "rewritten": []
  }}
}}

CHANGES FIELD

The changes field is important for observability.

"preserved": important user requirements explicitly retained.
"added": significant additions made by the planner.
"removed": only content actually removed. Important visual details should almost never appear here.
"rewritten": concepts whose wording was substantially improved without changing meaning.

If nothing was removed, return "removed": [].

FINAL VALIDATION

Before returning the response, verify:

- Did I preserve every important object?
- Did I preserve clothing and colors?
- Did I preserve accessories and jewelry?
- Did I preserve pose and action?
- Did I preserve background details?
- Did I preserve atmosphere?
- Did I preserve composition?
- Did I preserve style?
- Did I accidentally shorten useful detail?
- Did I introduce contradictions?
- Are model parameters compatible with the selected model?

If refinement caused loss of detail, restore the missing details before returning the JSON.

Use "general" only when no preset fits well."""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # Strip markdown fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    # Find first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _normalize_mode(mode: str | None) -> str | None:
    if not mode:
        return None
    mode = _MODE_ALIASES.get(str(mode).strip().lower(), str(mode).strip().lower())
    if mode not in _VALID_MODES:
        return None
    return mode


def _planner_overrides(parsed: dict[str, Any]) -> dict[str, Any]:
    """Map top-level planner JSON into preset override fields."""
    overrides: dict[str, Any] = {}

    # Backward compatibility with older overrides object.
    legacy = parsed.get("overrides")
    if isinstance(legacy, dict):
        overrides.update(legacy)

    for key in _OVERRIDE_KEYS:
        if key in parsed and parsed[key] is not None:
            overrides[key] = parsed[key]

    mode = _normalize_mode(overrides.get("mode"))
    if mode:
        overrides["mode"] = mode
    elif "mode" in overrides:
        overrides.pop("mode", None)

    return overrides


def _is_valid_pixel_dim(value: Any) -> bool:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return False
    return _MIN_PIXEL_DIM <= n <= _MAX_PIXEL_DIM


def _aspect_ratio_key(width: int, height: int) -> tuple[int, int]:
    g = gcd(width, height)
    return width // g, height // g


def _resolve_pixel_dimensions(
    width: Any,
    height: Any,
    preset: presets.Preset,
) -> tuple[int, int]:
    """Convert planner width/height into valid pixel dimensions."""
    try:
        w = int(width) if width is not None else None
        h = int(height) if height is not None else None
    except (TypeError, ValueError):
        w = h = None

    if w and h and _is_valid_pixel_dim(w) and _is_valid_pixel_dim(h):
        return w, h

    if w and h and w > 0 and h > 0 and w < _MIN_PIXEL_DIM and h < _MIN_PIXEL_DIM:
        mapped = _ASPECT_PIXELS.get(_aspect_ratio_key(w, h))
        if mapped:
            return mapped
        short_side = 1080
        scale = short_side / min(w, h)
        return int(round(w * scale)), int(round(h * scale))

    return preset.width, preset.height


def _uses_lightning_lora(overrides: dict[str, Any], preset: presets.Preset) -> bool:
    loras = overrides.get("loras")
    if loras is None:
        loras = [{"name": l.name, "strength": l.strength} for l in preset.loras]
    for lora in loras or []:
        name = str(lora.get("name", "")).lower()
        if "lightning" in name or "turbo" in name:
            return True
    return False


def _sanitize_planner_overrides(
    overrides: dict[str, Any],
    preset_id: str,
    *,
    has_image: bool,
) -> dict[str, Any]:
    preset = presets.PRESETS.get(preset_id, presets.PRESETS["general"])
    cleaned = dict(overrides)

    width, height = _resolve_pixel_dimensions(
        cleaned.pop("width", None),
        cleaned.pop("height", None),
        preset,
    )
    cleaned["width"] = width
    cleaned["height"] = height

    if _uses_lightning_lora(cleaned, preset):
        steps = cleaned.get("steps")
        if steps is not None:
            try:
                if int(steps) > 8:
                    cleaned.pop("steps", None)
            except (TypeError, ValueError):
                cleaned.pop("steps", None)
        cfg = cleaned.get("cfg")
        if cfg is not None:
            try:
                if float(cfg) > 2.0:
                    cleaned.pop("cfg", None)
            except (TypeError, ValueError):
                cleaned.pop("cfg", None)

    if has_image and cleaned.get("mode") in (None, "txt2img", "painterly"):
        cleaned.setdefault("denoise", 0.75)

    return cleaned


def _apply_facial_hair_negatives(
    *,
    prompt: str,
    refined: str,
    preset_id: str,
    overrides: dict[str, Any],
) -> None:
    if _FACIAL_HAIR_PATTERN.search(prompt) or _FACIAL_HAIR_PATTERN.search(refined):
        return

    current_neg = overrides.get("negative_prompt")
    if current_neg is None:
        preset_obj = presets.PRESETS.get(preset_id, presets.PRESETS["general"])
        current_neg = preset_obj.negative_prompt

    if current_neg:
        overrides["negative_prompt"] = f"{current_neg.rstrip(',')}, {_FACIAL_HAIR_NEGATIVES}"
    else:
        overrides["negative_prompt"] = _FACIAL_HAIR_NEGATIVES


PLANNER_MODELS = {
    "gemma": config.PLANNER_MODEL_TEXT,
    "llama-fast": config.PLANNER_MODEL_VISION,
}


def planner_profile(*, has_image: bool) -> str:
    return config.PLANNER_PROFILE_VISION if has_image else config.PLANNER_PROFILE_TEXT


def start_planner_profile(*, has_image: bool, profile: str | None = None) -> str:
    """Load the LLM profile for planning (Gemma for text and vision)."""
    from . import runtime

    resolved = profile or planner_profile(has_image=has_image)
    status = runtime.get_status()
    if status.get("profile") == resolved and status.get("loadState") == "LOADED":
        return resolved
    runtime.start_profile(resolved)
    return resolved


def _planner_model_name(*, has_image: bool, profile: str | None = None) -> str:
    if profile and profile in PLANNER_MODELS:
        return PLANNER_MODELS[profile]
    return config.PLANNER_MODEL_VISION if has_image else config.PLANNER_MODEL_TEXT


def _chat(
    messages: list[dict[str, Any]],
    *,
    has_image: bool = False,
    profile: str | None = None,
) -> str:
    payload = {
        "model": _planner_model_name(has_image=has_image, profile=profile),
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    with httpx.Client(base_url=LLM_BASE, timeout=TIMEOUT) as client:
        r = client.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise PlannerError("PLANNER_EMPTY", "LLM returned no choices")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise PlannerError("PLANNER_EMPTY", "LLM returned empty content")
    return content


def plan(
    *,
    prompt: str,
    image_data_url: str | None = None,
    fast_gen: bool = False,
    profile: str | None = None,
) -> dict[str, Any]:
    """Call the planner LLM and return a resolved execution plan."""
    has_image = bool(image_data_url)
    user_text = prompt
    if fast_gen:
        user_text = (
            f"{prompt}\n\n"
            "[Runtime: fast generation enabled — prefer Lightning-compatible presets, "
            "4 steps, cfg 1.0 when compatible; omit steps to let the preset decide.]"
        )
    user_content: list[dict[str, Any]] | str
    if image_data_url:
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    else:
        user_content = user_text

    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_content},
    ]

    try:
        raw = _chat(messages, has_image=has_image, profile=profile)
        parsed = _extract_json(raw)
    except (json.JSONDecodeError, httpx.HTTPError) as exc:
        # Fallback to general preset with original prompt
        return presets.resolve(
            "general",
            refined_prompt=prompt,
            overrides={},
        )

    preset_id = parsed.get("preset", "general")
    if preset_id not in presets.PRESETS:
        preset_id = "general"

    confidence = parsed.get("confidence", "medium")
    if confidence == "low":
        preset_id = "general"

    refined = parsed.get("refined_prompt") or prompt
    overrides = _planner_overrides(parsed)
    overrides = _sanitize_planner_overrides(
        overrides,
        preset_id,
        has_image=bool(image_data_url),
    )
    seed = parsed.get("seed")

    _apply_facial_hair_negatives(
        prompt=prompt,
        refined=refined,
        preset_id=preset_id,
        overrides=overrides,
    )

    resolved = presets.resolve(
        preset_id,
        refined_prompt=refined,
        overrides=overrides,
        seed=seed,
        width=parsed.get("width"),
        height=parsed.get("height"),
    )
    resolved["planner"] = {
        "preset": preset_id,
        "confidence": confidence,
        "changes": parsed.get("changes"),
        "raw_response": parsed,
    }
    return resolved
