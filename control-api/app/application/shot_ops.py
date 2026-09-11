from __future__ import annotations

from typing import Any

from ..domain.camera import resolve_camera
from ..domain.errors import DomainError, ErrorCode
from ..domain.operations import (
    DEFAULT_LIVE_DURATION,
    DEFAULT_LIVE_FPS,
    LIVE_WALLPAPER_TARGETS,
)
from ..domain.quality import (
    QUALITY_LORA,
    estimate_budget,
    ic_lora_for,
    quality_defaults,
    select_unet,
)
from ..domain.shots import (
    ACTION_SUFFIXES,
    FLOWS,
    SHOT_NEGATIVE,
    SHOT_OPERATIONS,
    SHOT_SUFFIX,
    ControlSpec,
    FlowSpec,
    Guide,
)
from .image_ops import RenderStep, _clamp, _seed, collect_input_refs
from .video_ops import _bool, _length

SHOT_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "EmptyLTXVLatentVideo",
    "LTXVConditioning",
    "LTXVAddGuide",
    "LTXVCropGuides",
    "LTXVEmptyLatentAudio",
    "LTXVConcatAVLatent",
    "LTXVDualCFGGuider",
    "SamplerCustomAdvanced",
    "CreateVideo",
    "SaveVideo",
)
POST_NODES = ("LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo")
COMPOSE_NODES = ("UNETLoader", "CLIPLoader", "VAELoader", "TextEncodeQwenImageEditPlus", "SaveImage")


def _dims(spec, body: dict[str, Any]) -> tuple[int, int]:
    if body.get("width") is not None or body.get("height") is not None:
        width = int(body.get("width") or spec.default_width)
        height = int(body.get("height") or spec.default_height)
    else:
        target = str(body.get("target") or "video").strip()
        if target not in LIVE_WALLPAPER_TARGETS:
            raise DomainError(ErrorCode.INVALID_PARAMETER, "target must be mobile or video")
        width, height = LIVE_WALLPAPER_TARGETS[target]
    if width < 64 or height < 64 or width > 4096 or height > 4096:
        raise DomainError(ErrorCode.INVALID_PARAMETER, "width and height must be between 64 and 4096")
    return width, height


def align_length(duration: float, fps: int) -> int:
    return _length(duration, fps)


def snap_frame_idx(idx: int, length: int, *, video_guide: bool = False) -> int:
    if length < 1:
        raise DomainError(ErrorCode.INVALID_PARAMETER, "length must be at least 1")
    value = int(idx)
    if value < 0:
        value = length + value
    value = max(0, min(value, length - 1))
    if video_guide:
        value = (value // 8) * 8
    return value


def validate_guides(guides: list[Guide], length: int) -> list[Guide]:
    seen: set[int] = set()
    last = -1
    resolved: list[Guide] = []
    for guide in guides:
        video = guide.kind == "video"
        idx = snap_frame_idx(guide.frame_idx, length, video_guide=video)
        if idx in seen:
            raise DomainError(ErrorCode.INVALID_PARAMETER, f"duplicate guide frame_idx {idx}")
        if idx < last:
            raise DomainError(ErrorCode.INVALID_PARAMETER, "guide frame indices must be non-decreasing")
        seen.add(idx)
        last = idx
        resolved.append(
            Guide(
                key=guide.key,
                frame_idx=idx,
                strength=guide.strength,
                kind=guide.kind,
                mask_key=guide.mask_key,
                signal=guide.signal,
            )
        )
    return resolved


def _video_params(spec, body: dict[str, Any]) -> tuple[int, int, float, int, int, float, bool, int]:
    seed = _seed(body)
    width, height = _dims(spec, body)
    duration = float(body.get("duration") if body.get("duration") is not None else DEFAULT_LIVE_DURATION)
    _clamp(duration, 1.0, 30.0, "duration")
    fps = int(body.get("fps") if body.get("fps") is not None else DEFAULT_LIVE_FPS)
    if fps < 8 or fps > 60:
        raise DomainError(ErrorCode.INVALID_PARAMETER, "fps must be between 8 and 60")
    steps = int(body.get("steps") if body.get("steps") is not None else spec.steps)
    if steps < 1 or steps > 40:
        raise DomainError(ErrorCode.INVALID_PARAMETER, "steps must be between 1 and 40")
    video_cfg = float(
        body.get("video_cfg")
        if body.get("video_cfg") is not None
        else body.get("cfg")
        if body.get("cfg") is not None
        else spec.cfg
    )
    _clamp(video_cfg, 0.0, 15.0, "video_cfg")
    refine = _bool(body.get("refine"), True)
    return seed, width, height, duration, fps, video_cfg, refine, steps


def _prompt(flow: FlowSpec, body: dict[str, Any]) -> tuple[str, str]:
    prompt = str(body.get("prompt") or "").strip()
    if flow.prompt_required and not prompt:
        raise DomainError(ErrorCode.INVALID_REQUEST, f"prompt is required for {flow.id}")
    user_neg = str(body.get("negative_prompt") or "").strip()
    suffix = SHOT_SUFFIX
    if flow.camera and body.get("camera"):
        suffix = f"{resolve_camera(body.get('camera')).prompt}. {suffix}"
    if flow.action:
        action = str(body.get("action") or "battle").strip()
        extra = ACTION_SUFFIXES.get(action)
        if extra is None:
            raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unknown action {action}")
        suffix = f"{extra}. {suffix}"
    text = f"{prompt}. {suffix}" if prompt else suffix
    return text, user_neg or SHOT_NEGATIVE


def _require(refs: dict[str, str], key: str, flow_id: str) -> None:
    if key not in refs:
        raise DomainError(ErrorCode.INVALID_REQUEST, f"{flow_id} requires {key}")


def _guides_for(flow: FlowSpec, refs: dict[str, str], body: dict[str, Any], length: int) -> list[Guide]:
    guides: list[Guide] = []
    if flow.id in {"shot_single_image", "shot_camera", "shot_motion_reference", "shot_action"}:
        if "image" in refs:
            guides.append(Guide(key="image", frame_idx=0, strength=1.0))
        if flow.id in {"shot_action", "shot_start_end"} and "end_image" in refs:
            guides.append(Guide(key="end_image", frame_idx=-1, strength=1.0))
    if flow.id == "shot_start_end":
        _require(refs, "image", flow.id)
        _require(refs, "end_image", flow.id)
        guides = [
            Guide(key="image", frame_idx=0, strength=1.0),
            Guide(key="end_image", frame_idx=-1, strength=1.0),
        ]
    if flow.id in {"shot_multi_reference", "shot_character_interaction"}:
        start_key = "__previous__" if "compose" in flow.stages else "image"
        if start_key == "image" and "image" not in refs:
            raise DomainError(ErrorCode.INVALID_REQUEST, f"{flow.id} needs a start frame or references")
        guides.append(Guide(key=start_key, frame_idx=0, strength=1.0))
    if flow.id == "shot_multi_keyframe":
        _require(refs, "image", flow.id)
        guides.append(Guide(key="image", frame_idx=0, strength=1.0))
        raw = body.get("keyframes") or []
        if not isinstance(raw, list) or not raw:
            raise DomainError(ErrorCode.INVALID_REQUEST, "shot_multi_keyframe requires keyframes")
        for index, item in enumerate(raw[:12]):
            key = f"keyframe_{index}"
            if key not in refs:
                continue
            payload = item if isinstance(item, dict) else {}
            idx = int(payload.get("frame_idx") if payload.get("frame_idx") is not None else (index + 1) * 8)
            strength = float(payload.get("strength") if payload.get("strength") is not None else 1.0)
            guides.append(Guide(key=key, frame_idx=idx, strength=strength))
    if flow.id in {"shot_continuation", "shot_extend"}:
        _require(refs, "video", flow.id)
        guides.append(Guide(key="video", frame_idx=0, strength=0.95, kind="video"))
        if "image" in refs:
            guides.append(Guide(key="image", frame_idx=0, strength=0.6))
    if flow.id == "shot_regenerate":
        if "image" in refs:
            guides.append(Guide(key="image", frame_idx=0, strength=1.0))
        if "end_image" in refs:
            guides.append(Guide(key="end_image", frame_idx=-1, strength=1.0))
        if "video" in refs:
            guides.append(Guide(key="video", frame_idx=0, strength=0.95, kind="video"))
    return validate_guides(guides, length)


def _control_for(flow: FlowSpec, refs: dict[str, str], body: dict[str, Any]) -> ControlSpec | None:
    if flow.id == "shot_motion_reference":
        _require(refs, "video", flow.id)
        kind = str(body.get("control") or "canny").strip()
        lora = ic_lora_for(kind)
        if lora is None:
            raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unknown control {kind}")
        return ControlSpec(kind=kind, lora=lora, strength=float(body.get("control_strength") or 1.0), source_key="video")
    if flow.id == "shot_camera":
        resolve_camera(body.get("camera"))
        return ControlSpec(
            kind="motion_track",
            lora=ic_lora_for("motion_track") or "",
            strength=float(body.get("control_strength") or 1.0),
            source_key="image",
        )
    return None


def _compose_keys(refs: dict[str, str]) -> list[str]:
    order = ("image", "character", "character_b", "location", "prop")
    return [key for key in order if key in refs][:3]


def _shot_step(
    *,
    flow: FlowSpec,
    body: dict[str, Any],
    refs: dict[str, str],
    seed: int,
    width: int,
    height: int,
    duration: float,
    fps: int,
    video_cfg: float,
    refine: bool,
    steps: int,
    prompt: str,
    negative: str,
    guides: list[Guide],
    control: ControlSpec | None,
    quality: dict[str, Any],
) -> RenderStep:
    use_lora = control is not None or bool(quality.get("quality_lora"))
    prefer = bool(quality.get("prefer_nvfp4"))
    unet = select_unet(use_lora=use_lora, prefer_nvfp4=prefer)
    loras: list[dict[str, Any]] = []
    if quality.get("quality_lora"):
        loras.append({"name": QUALITY_LORA, "strength": 0.75})
    video_guides = sum(1 for guide in guides if guide.kind == "video")
    budget = estimate_budget(width, height, _length(duration, fps), len(guides), video_guides=video_guides)
    warnings = [budget.warning] if budget.warning else []
    camera = None
    if flow.camera:
        preset = resolve_camera(body.get("camera"))
        camera = {"id": preset.id, "prompt": preset.prompt}
    control_dict = None
    if control:
        control_dict = {
            "kind": control.kind,
            "lora": control.lora,
            "strength": control.strength,
            "source_key": control.source_key,
        }
    image_keys = [guide.key for guide in guides if guide.kind != "video" and not guide.key.startswith("__")]
    video_keys = [guide.key for guide in guides if guide.kind == "video"]
    if control and control.source_key == "video" and "video" not in video_keys:
        video_keys.append("video")
    return RenderStep(
        mode="shot",
        prompt=prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        steps=steps,
        cfg=video_cfg,
        seed=seed,
        image_keys=image_keys,
        filename_prefix=flow.id,
        required_nodes=SHOT_NODES,
        sampler_name=str(body.get("sampler_name") or "euler_ancestral"),
        profile="comfy-ltx",
        fps=fps,
        duration=duration,
        refine=refine,
        video_cfg=video_cfg,
        length=_length(duration, fps),
        guides=[
            {
                "key": guide.key,
                "frame_idx": guide.frame_idx,
                "strength": guide.strength,
                "kind": guide.kind,
                "mask_key": guide.mask_key,
                "signal": guide.signal,
            }
            for guide in guides
        ],
        control=control_dict,
        camera=camera,
        video_keys=video_keys,
        windowed=budget.windowed,
        img_compression=int(quality["img_compression"]),
        temporal_refine=bool(quality["temporal_refine"]) and refine,
        clip_device=str(quality["clip_device"]),
        unet_name=unet,
        warnings=warnings,
        loras=loras,
        parent_job_id=str(body.get("parent_job_id") or "") or None,
    )


def plan_shot_flow(operation: str, body: dict[str, Any]) -> tuple[list[RenderStep], dict[str, str], list[str]]:
    flow = FLOWS.get(operation)
    spec = SHOT_OPERATIONS.get(operation)
    if flow is None or spec is None:
        raise DomainError(ErrorCode.UNSUPPORTED_OPERATION, f"Unknown shot flow {operation}")
    refs = collect_input_refs(body)
    quality = quality_defaults(body)
    seed, width, height, duration, fps, video_cfg, refine, steps = _video_params(spec, body)
    prompt, negative = _prompt(flow, body)
    bundles = list(flow.required_bundles)
    steps_out: list[RenderStep] = []

    if flow.id == "shot_stitch":
        clip_keys = [key for key in refs if key.startswith("clip_")]
        if len(clip_keys) < 2:
            raise DomainError(ErrorCode.INVALID_REQUEST, "shot_stitch requires at least two clips")
        post = {
            "mode": str(body.get("post_mode") or "join"),
            "crossfade": int(body.get("crossfade") or 0),
            "upscale": _bool(body.get("upscale"), False),
            "interpolate": _bool(body.get("interpolate"), False),
        }
        steps_out.append(
            RenderStep(
                mode="post",
                prompt=prompt,
                width=width,
                height=height,
                seed=seed,
                video_keys=clip_keys,
                filename_prefix="shot_stitch",
                required_nodes=POST_NODES,
                profile="comfy-ltx",
                fps=fps,
                duration=duration,
                post=post,
            )
        )
        return steps_out, refs, bundles

    if "compose" in flow.stages:
        keys = _compose_keys(refs)
        if not keys:
            raise DomainError(ErrorCode.INVALID_REQUEST, f"{flow.id} requires character, location, or scene refs")
        compose_prompt = (
            f"Compose a single cinematic start frame from the references. "
            f"Preserve identity and wardrobe. {prompt}"
        )
        steps_out.append(
            RenderStep(
                mode="edit",
                prompt=compose_prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                steps=int(body.get("compose_steps") or 40),
                cfg=float(body.get("compose_cfg") or 3.0),
                seed=seed,
                image_keys=keys,
                filename_prefix=f"{flow.id}_compose",
                required_nodes=COMPOSE_NODES,
                profile="comfyui",
            )
        )

    guides = _guides_for(flow, refs, body, _length(duration, fps))
    if "compose" in flow.stages:
        guides = validate_guides([Guide(key="__previous__", frame_idx=0, strength=1.0), *guides[1:]], _length(duration, fps))
    control = _control_for(flow, refs, body)
    shot = _shot_step(
        flow=flow,
        body=body,
        refs=refs,
        seed=seed,
        width=width,
        height=height,
        duration=duration,
        fps=fps,
        video_cfg=video_cfg,
        refine=refine,
        steps=steps,
        prompt=prompt,
        negative=negative,
        guides=guides,
        control=control,
        quality=quality,
    )
    if shot.unet_name and "nvfp4" in shot.unet_name:
        if "ltx-2.5-nvfp4" not in bundles:
            bundles.append("ltx-2.5-nvfp4")
    steps_out.append(shot)
    return steps_out, refs, bundles
