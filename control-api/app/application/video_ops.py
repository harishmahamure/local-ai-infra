from __future__ import annotations

from typing import Any

from ..domain.errors import DomainError, ErrorCode
from ..domain.operations import (
    DEFAULT_LIVE_DURATION,
    DEFAULT_LIVE_FPS,
    LIVE_WALLPAPER_NEGATIVE,
    LIVE_WALLPAPER_SUFFIX,
    LIVE_WALLPAPER_TARGETS,
    LTX_VIDEO_NEGATIVE,
    LTX_VIDEO_SUFFIX,
    VIDEO_OPERATIONS,
)
from .image_ops import RenderStep, _clamp, _seed, collect_input_refs

I2V_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "LoadImage",
    "LTXVPreprocess",
    "EmptyLTXVLatentVideo",
    "LTXVImgToVideoInplace",
    "LTXVConditioning",
    "LTXVEmptyLatentAudio",
    "LTXVScheduler",
    "LTXVConcatAVLatent",
    "LTXVDualCFGGuider",
    "SamplerCustomAdvanced",
    "LTXVSeparateAVLatent",
    "CreateVideo",
    "SaveVideo",
)
T2V_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "EmptyLTXVLatentVideo",
    "LTXVConditioning",
    "LTXVEmptyLatentAudio",
    "LTXVScheduler",
    "LTXVConcatAVLatent",
    "LTXVDualCFGGuider",
    "SamplerCustomAdvanced",
    "LTXVSeparateAVLatent",
    "CreateVideo",
    "SaveVideo",
)


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _video_dims(spec, body: dict[str, Any], *, default_target: str) -> tuple[int, int]:
    if body.get("width") is not None or body.get("height") is not None:
        width = int(body.get("width") or spec.default_width)
        height = int(body.get("height") or spec.default_height)
    else:
        target = str(body.get("target") or default_target).strip()
        if target not in LIVE_WALLPAPER_TARGETS:
            raise DomainError(ErrorCode.INVALID_PARAMETER, "target must be mobile or video")
        width, height = LIVE_WALLPAPER_TARGETS[target]
    if width < 64 or height < 64 or width > 4096 or height > 4096:
        raise DomainError(ErrorCode.INVALID_PARAMETER, "width and height must be between 64 and 4096")
    return width, height


def _length(duration: float, fps: int) -> int:
    raw = int(round(duration * fps)) + 1
    n = max(1, round((raw - 1) / 8))
    length = n * 8 + 1
    return min(length, 897)


def plan_video_operation(operation: str, body: dict[str, Any]) -> tuple[list[RenderStep], dict[str, str], list[str]]:
    spec = VIDEO_OPERATIONS.get(operation)
    if spec is None:
        raise DomainError(ErrorCode.UNSUPPORTED_OPERATION, f"Unknown operation {operation}")
    prompt = str(body.get("prompt") or "").strip()
    if spec.prompt_required and not prompt:
        raise DomainError(ErrorCode.INVALID_REQUEST, f"prompt is required for {operation}")
    refs = collect_input_refs(body)
    t2v = operation == "generate_video"
    if not t2v and "image" not in refs:
        raise DomainError(ErrorCode.INVALID_REQUEST, f"{operation} requires an image asset")

    seed = _seed(body)
    width, height = _video_dims(spec, body, default_target="video" if t2v else "mobile")
    duration = float(body.get("duration") if body.get("duration") is not None else DEFAULT_LIVE_DURATION)
    _clamp(duration, 1.0, 30.0, "duration")
    fps = int(body.get("fps") if body.get("fps") is not None else DEFAULT_LIVE_FPS)
    if fps < 8 or fps > 60:
        raise DomainError(ErrorCode.INVALID_PARAMETER, "fps must be between 8 and 60")
    steps = int(body.get("steps") if body.get("steps") is not None else spec.steps)
    if steps < 1 or steps > 40:
        raise DomainError(ErrorCode.INVALID_PARAMETER, "steps must be between 1 and 40")
    video_cfg = float(body.get("video_cfg") if body.get("video_cfg") is not None else body.get("cfg") if body.get("cfg") is not None else spec.cfg)
    _clamp(video_cfg, 0.0, 15.0, "video_cfg")
    refine = _bool(body.get("refine"), True)
    user_neg = str(body.get("negative_prompt") or "").strip()
    if t2v:
        suffix = LTX_VIDEO_SUFFIX
        negative = user_neg or LTX_VIDEO_NEGATIVE
        mode = "t2v"
        prefix = "ltx_video"
        nodes = T2V_NODES
        image_keys: list[str] = []
    else:
        suffix = LIVE_WALLPAPER_SUFFIX
        negative = user_neg or LIVE_WALLPAPER_NEGATIVE
        mode = "i2v"
        prefix = "live_wallpaper"
        nodes = I2V_NODES
        image_keys = ["image"]
    bundles = ["ltx-2.5-distilled"]
    if refine:
        bundles.append("ltx-2.5-studio")

    step = RenderStep(
        mode=mode,
        prompt=f"{prompt}. {suffix}",
        negative_prompt=negative,
        width=width,
        height=height,
        steps=steps,
        cfg=video_cfg,
        seed=seed,
        image_keys=image_keys,
        filename_prefix=prefix,
        required_nodes=nodes,
        sampler_name=str(body.get("sampler_name") or "euler_ancestral"),
        profile="comfy-ltx",
        fps=fps,
        duration=duration,
        refine=refine,
        video_cfg=video_cfg,
        length=_length(duration, fps),
    )
    return [step], refs if image_keys else {}, bundles
