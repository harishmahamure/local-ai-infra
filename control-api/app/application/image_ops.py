from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..domain.errors import DomainError, ErrorCode
from ..domain.operations import (
    ATTIRE_STANDALONE,
    DEFAULT_NEGATIVE,
    DEFAULT_SHIFT,
    OPERATIONS,
    PROP_STANDALONE,
    SAMPLERS,
    SCHEDULERS,
    SHOT_GRAMMAR,
    STYLED_OPERATIONS,
    StylePreset,
    TURNAROUND_VIEWS,
    compose_face_lock,
    resolve_style_preset,
)
from ..infrastructure.comfy.graphs import MODELS

LIGHTNING_LORA = MODELS["qwen_lora_lightning"]
LIGHTNING_EDIT_LORA = MODELS["qwen_edit_lora_lightning"]


@dataclass
class Advanced:
    sampler_name: str = "euler"
    scheduler: str = "simple"
    shift: float = DEFAULT_SHIFT
    lora_strength: float = 1.0
    control_strength: float = 1.0


@dataclass
class RenderStep:
    mode: str
    prompt: str
    negative_prompt: str = ""
    width: int = 1328
    height: int = 1328
    steps: int = 50
    cfg: float = 4.0
    seed: int = 0
    denoise: float | None = None
    image_keys: list[str] = field(default_factory=list)
    mask_key: str | None = None
    loras: list[dict[str, Any]] = field(default_factory=list)
    pad: dict[str, int] | None = None
    upscale_scale: int | None = None
    filename_prefix: str = "engine"
    required_nodes: tuple[str, ...] = ()
    sampler_name: str = "euler"
    scheduler: str = "simple"
    shift: float = DEFAULT_SHIFT
    lora_strength: float = 1.0
    control_strength: float = 1.0


def _seed(body: dict[str, Any]) -> int:
    value = body.get("seed")
    if value is None:
        return random.randint(0, 2**31 - 1)
    return int(value)


def _dims(spec, body: dict[str, Any]) -> tuple[int, int]:
    width = int(body.get("width") or spec.default_width)
    height = int(body.get("height") or spec.default_height)
    if width < 64 or height < 64 or width > 4096 or height > 4096:
        raise DomainError(ErrorCode.INVALID_PARAMETER, "width and height must be between 64 and 4096")
    return width, height


def _clamp(value: float, lo: float, hi: float, name: str) -> float:
    if value < lo or value > hi:
        raise DomainError(ErrorCode.INVALID_PARAMETER, f"{name} must be between {lo} and {hi}")
    return value


def _advanced(body: dict[str, Any]) -> Advanced:
    sampler = str(body.get("sampler_name") or "euler")
    if sampler not in SAMPLERS:
        raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unknown sampler {sampler}")
    scheduler = str(body.get("scheduler") or "simple")
    if scheduler not in SCHEDULERS:
        raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unknown scheduler {scheduler}")
    shift_raw = body.get("shift")
    shift = float(shift_raw) if shift_raw is not None else DEFAULT_SHIFT
    _clamp(shift, 0.0, 100.0, "shift")
    lora_raw = body.get("lora_strength")
    lora_strength = float(lora_raw) if lora_raw is not None else 1.0
    _clamp(lora_strength, 0.0, 2.0, "lora_strength")
    control_raw = body.get("control_strength")
    control_strength = float(control_raw) if control_raw is not None else 1.0
    _clamp(control_strength, 0.0, 2.0, "control_strength")
    return Advanced(
        sampler_name=sampler,
        scheduler=scheduler,
        shift=shift,
        lora_strength=lora_strength,
        control_strength=control_strength,
    )


def _loras(body: dict[str, Any], *, mode: str, strength: float) -> list[dict[str, Any]]:
    if not body.get("fast"):
        return []
    name = LIGHTNING_EDIT_LORA if mode == "edit" else LIGHTNING_LORA
    return [{"name": name, "strength": strength}]


def _steps_cfg(spec, body: dict[str, Any], *, mode: str) -> tuple[int, float]:
    if body.get("fast") and spec.steps:
        return 4, 1.0
    if body.get("steps") is not None:
        steps = int(body["steps"])
    elif mode == "edit":
        steps = spec.edit_steps
    else:
        steps = spec.steps or 50
    if body.get("cfg") is not None:
        cfg = float(body["cfg"])
    elif mode == "edit":
        cfg = spec.edit_cfg
    else:
        cfg = spec.cfg
    return steps, cfg


def _style_prompt(prompt: str, preset: StylePreset) -> str:
    if not preset.suffix:
        return prompt
    return f"{prompt}. {preset.suffix}"


def _negative(body: dict[str, Any], cfg: float, extra: str = "", style_extra: str = "") -> str:
    user = str(body.get("negative_prompt") or "").strip()
    if cfg <= 1.0 and not user:
        return ""
    parts: list[str] = []
    if user:
        parts.append(user)
    else:
        parts.append(DEFAULT_NEGATIVE)
        if extra:
            parts.append(extra)
    if style_extra:
        parts.append(style_extra)
    return ", ".join(part for part in parts if part)


def _asset_id(value: Any) -> str | None:
    if isinstance(value, dict) and value.get("asset_id"):
        return str(value["asset_id"])
    if isinstance(value, str) and value.startswith("ast_"):
        return value
    return None


def collect_input_refs(body: dict[str, Any]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for key in ("image", "mask", "character", "attire", "location"):
        asset_id = _asset_id(body.get(key))
        if asset_id:
            refs[key] = asset_id
    extras = body.get("references") or []
    if isinstance(extras, list):
        for index, item in enumerate(extras[:3]):
            asset_id = _asset_id(item)
            if asset_id:
                refs[f"reference_{index}"] = asset_id
    return refs


def _ref_count(refs: dict[str, str], spec) -> int:
    keys = [k for k in refs if k != "mask"]
    return len(keys)


def plan_operation(operation: str, body: dict[str, Any]) -> tuple[list[RenderStep], dict[str, str], list[str]]:
    spec = OPERATIONS.get(operation)
    if spec is None:
        raise DomainError(ErrorCode.UNSUPPORTED_OPERATION, f"Unknown operation {operation}")
    prompt = str(body.get("prompt") or "").strip()
    if spec.prompt_required and not prompt:
        raise DomainError(ErrorCode.INVALID_REQUEST, f"prompt is required for {operation}")
    refs = collect_input_refs(body)
    image_count = _ref_count(refs, spec)
    if spec.requires_image and "image" not in refs and "character" not in refs:
        raise DomainError(ErrorCode.INVALID_REQUEST, f"{operation} requires an image asset")
    if spec.requires_mask and "mask" not in refs:
        raise DomainError(ErrorCode.INVALID_REQUEST, f"{operation} requires a mask asset")
    if image_count < spec.min_images:
        raise DomainError(ErrorCode.INVALID_REQUEST, f"{operation} requires at least {spec.min_images} image(s)")
    if spec.max_images and image_count > spec.max_images:
        raise DomainError(ErrorCode.INVALID_REQUEST, f"{operation} accepts at most {spec.max_images} image(s)")

    seed = _seed(body)
    width, height = _dims(spec, body)
    adv = _advanced(body)
    bundles = list(spec.required_bundles)
    style = resolve_style_preset(body.get("style_preset") if operation in STYLED_OPERATIONS else "off")

    def step(*, mode: str, **kwargs: Any) -> RenderStep:
        steps, cfg = _steps_cfg(spec, body, mode=mode)
        loras = _loras(body, mode=mode, strength=adv.lora_strength)
        kwargs.setdefault("steps", steps)
        kwargs.setdefault("cfg", cfg)
        kwargs.setdefault("loras", loras)
        kwargs.setdefault("negative_prompt", _negative(body, cfg, style_extra=style.negative))
        return RenderStep(
            mode=mode,
            width=width,
            height=height,
            seed=seed,
            sampler_name=adv.sampler_name,
            scheduler=adv.scheduler,
            shift=adv.shift,
            lora_strength=adv.lora_strength,
            control_strength=adv.control_strength,
            **kwargs,
        )

    if operation == "generate_character":
        if "image" in refs:
            pos, face_neg = compose_face_lock(prompt, str(body.get("negative_prompt") or ""))
            cfg = _steps_cfg(spec, body, mode="edit")[1]
            if cfg <= 1.0 and not str(body.get("negative_prompt") or "").strip():
                face_lock_neg = ""
            else:
                face_lock_neg = ", ".join(part for part in (face_neg, style.negative) if part)
            return (
                [
                    step(
                        mode="edit",
                        prompt=_style_prompt(pos, style),
                        negative_prompt=face_lock_neg,
                        image_keys=["image"],
                        filename_prefix="character",
                        required_nodes=("TextEncodeQwenImageEditPlus", "UNETLoader", "KSampler", "SaveImage"),
                    )
                ],
                refs,
                ["qwen-image-edit-2511-fp8"],
            )
        return (
            [
                step(
                    mode="txt2img",
                    prompt=_style_prompt(prompt, style),
                    filename_prefix="character",
                    required_nodes=("UNETLoader", "CLIPLoader", "VAELoader", "KSampler", "SaveImage"),
                )
            ],
            refs,
            bundles,
        )

    if operation == "generate_character_turnaround":
        planned: list[RenderStep] = []
        if "character" not in refs and "image" not in refs:
            bundles = ["qwen-image-2512-fp8", "qwen-image-edit-2511-fp8"]
            planned.append(
                step(
                    mode="txt2img",
                    prompt=_style_prompt(
                        f"{prompt}, full body character sheet, T-pose-friendly standing pose, even studio lighting, neutral background",
                        style,
                    ),
                    filename_prefix="turnaround_base",
                    required_nodes=("UNETLoader", "CLIPLoader", "VAELoader", "KSampler", "SaveImage"),
                )
            )
        base_keys = ["character"] if "character" in refs else (["image"] if "image" in refs else ["__base__"])
        extra_neg = "different person, extra limbs, cropped, text, watermark"
        for view, view_prompt in TURNAROUND_VIEWS:
            planned.append(
                step(
                    mode="edit",
                    prompt=_style_prompt(
                        f"{prompt}. {view_prompt}. Keep identity, costume, colors, and proportions identical.",
                        style,
                    ),
                    negative_prompt=_negative(
                        body,
                        _steps_cfg(spec, body, mode="edit")[1],
                        extra_neg,
                        style_extra=style.negative,
                    ),
                    image_keys=list(base_keys),
                    filename_prefix=f"turnaround_{view}",
                    required_nodes=("TextEncodeQwenImageEditPlus", "UNETLoader", "KSampler", "SaveImage"),
                )
            )
        return planned, refs, bundles

    if operation == "generate_attire":
        if "character" in refs or "image" in refs:
            key = "character" if "character" in refs else "image"
            return (
                [
                    step(
                        mode="edit",
                        prompt=f"Dress this character in: {prompt}. Keep the same face, body, and pose.",
                        image_keys=[key],
                        filename_prefix="attire",
                        required_nodes=("TextEncodeQwenImageEditPlus", "KSampler", "SaveImage"),
                    )
                ],
                refs,
                ["qwen-image-edit-2511-fp8"],
            )
        return (
            [
                step(
                    mode="txt2img",
                    prompt=f"{prompt}. {ATTIRE_STANDALONE}",
                    negative_prompt=_negative(body, _steps_cfg(spec, body, mode="txt2img")[1], "person, face, hands, mannequin head"),
                    filename_prefix="attire",
                    required_nodes=("UNETLoader", "KSampler", "SaveImage"),
                )
            ],
            refs,
            bundles,
        )

    if operation == "generate_location":
        return (
            [
                step(
                    mode="txt2img",
                    prompt=prompt,
                    filename_prefix="location",
                    required_nodes=("UNETLoader", "KSampler", "SaveImage"),
                )
            ],
            refs,
            bundles,
        )

    if operation == "generate_prop":
        return (
            [
                step(
                    mode="txt2img",
                    prompt=f"{prompt}. {PROP_STANDALONE}",
                    negative_prompt=_negative(body, _steps_cfg(spec, body, mode="txt2img")[1], "people, hands, text, watermark"),
                    filename_prefix="prop",
                    required_nodes=("UNETLoader", "KSampler", "SaveImage"),
                )
            ],
            refs,
            bundles,
        )

    if operation in {"generate_keyframe", "generate_shot_reference"}:
        keys: list[str] = []
        for slot in ("character", "attire", "location", "image"):
            if slot in refs and slot not in keys:
                keys.append(slot)
        for extra in sorted(k for k in refs if k.startswith("reference_")):
            keys.append(extra)
        keys = keys[:3]
        if operation == "generate_shot_reference":
            framing = str(body.get("framing") or "medium")
            lens = str(body.get("lens") or "35")
            height_desc = str(body.get("camera_height") or "eye level")
            shot_prompt = f"{prompt}. {SHOT_GRAMMAR.format(framing=framing, lens=lens, height=height_desc)}"
        else:
            shot_prompt = prompt
        return (
            [
                step(
                    mode="edit",
                    prompt=_style_prompt(shot_prompt, style),
                    negative_prompt=_negative(
                        body,
                        _steps_cfg(spec, body, mode="edit")[1],
                        "identity change, extra people, text, watermark",
                        style_extra=style.negative,
                    ),
                    image_keys=keys,
                    filename_prefix="keyframe" if operation == "generate_keyframe" else "shot_reference",
                    required_nodes=("TextEncodeQwenImageEditPlus", "KSampler", "SaveImage"),
                )
            ],
            refs,
            bundles,
        )

    if operation == "inpaint_asset":
        return (
            [
                step(
                    mode="inpaint",
                    prompt=prompt,
                    image_keys=["image"],
                    mask_key="mask",
                    filename_prefix="inpaint",
                    required_nodes=("ModelPatchLoader", "QwenImageDiffsynthControlnet", "KSampler", "SaveImage"),
                )
            ],
            refs,
            bundles,
        )

    if operation == "outpaint_asset":
        pad = {
            "left": int(body.get("left") or 0),
            "right": int(body.get("right") or 0),
            "top": int(body.get("top") or 0),
            "bottom": int(body.get("bottom") or 0),
            "feathering": int(body.get("feathering") or 40),
        }
        if pad["left"] + pad["right"] + pad["top"] + pad["bottom"] <= 0:
            raise DomainError(ErrorCode.INVALID_REQUEST, "outpaint_asset requires left/right/top/bottom padding")
        return (
            [
                step(
                    mode="outpaint",
                    prompt=prompt,
                    image_keys=["image"],
                    pad=pad,
                    filename_prefix="outpaint",
                    required_nodes=("ImagePadForOutpaint", "QwenImageDiffsynthControlnet", "KSampler", "SaveImage"),
                )
            ],
            refs,
            bundles,
        )

    if operation == "upscale_asset":
        scale = int(body.get("scale") or 4)
        if scale not in {2, 4}:
            raise DomainError(ErrorCode.INVALID_PARAMETER, "scale must be 2 or 4")
        return (
            [
                step(
                    mode="upscale",
                    prompt=prompt,
                    image_keys=["image"],
                    upscale_scale=scale,
                    filename_prefix="upscale",
                    required_nodes=("UpscaleModelLoader", "ImageUpscaleWithModel", "SaveImage"),
                )
            ],
            refs,
            bundles,
        )

    raise DomainError(ErrorCode.UNSUPPORTED_OPERATION, f"Unknown operation {operation}")


def step_to_plan(
    step: RenderStep,
    *,
    uploaded: dict[str, str],
    previous_name: str | None,
    base_name: str | None = None,
) -> dict[str, Any]:
    names: list[str] = []
    for key in step.image_keys:
        if key == "__previous__":
            if not previous_name:
                raise DomainError(ErrorCode.INVALID_REQUEST, "Turnaround previous image is missing")
            names.append(previous_name)
        elif key == "__base__":
            if not base_name:
                raise DomainError(ErrorCode.INVALID_REQUEST, "Turnaround base image is missing")
            names.append(base_name)
        elif key in uploaded:
            names.append(uploaded[key])
    plan: dict[str, Any] = {
        "mode": step.mode,
        "prompt": step.prompt,
        "negative_prompt": step.negative_prompt,
        "width": step.width,
        "height": step.height,
        "steps": step.steps,
        "cfg": step.cfg,
        "seed": step.seed,
        "loras": list(step.loras),
        "filename_prefix": step.filename_prefix,
        "sampler_name": step.sampler_name,
        "scheduler": step.scheduler,
        "shift": step.shift,
        "lora_strength": step.lora_strength,
        "control_strength": step.control_strength,
    }
    if names:
        plan["image_name"] = names[0]
        plan["image_names"] = names
    if step.denoise is not None:
        plan["denoise"] = step.denoise
    if step.mask_key and step.mask_key in uploaded:
        plan["mask_name"] = uploaded[step.mask_key]
    if step.pad:
        plan["pad"] = dict(step.pad)
    if step.upscale_scale:
        plan["upscale_scale"] = step.upscale_scale
    return plan
