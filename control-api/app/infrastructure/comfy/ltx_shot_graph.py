from __future__ import annotations

from typing import Any

from ...domain.camera import resolve_camera, tracks_inputs
from ...domain.quality import (
    CLIP_INT8,
    SPATIAL_UPSCALER,
    TEMPORAL_UPSCALER,
    UNET_INT8,
    WINDOW_LENGTH,
    WINDOW_OVERLAP,
)
from .graphs import GraphBuilder
from .ltx_graph import (
    MODELS,
    REFINE_SIGMAS,
    STAGE1_SIGMAS,
    _align,
    _concat_av,
    _sample,
)

VIDEO_VAE = MODELS["vae"]
AUDIO_VAE = MODELS["audio_vae"]


def _clip(gb: GraphBuilder, device: str) -> str:
    inputs: dict[str, Any] = {"clip_name": CLIP_INT8, "type": "ltxv"}
    if device:
        inputs["device"] = device
    return gb.add("CLIPLoader", inputs)


def _condition(gb: GraphBuilder, clip: str, prompt: str, negative: str, fps: float) -> str:
    pos = gb.add("CLIPTextEncode", {"text": prompt, "clip": gb.ref(clip)})
    neg = gb.add("CLIPTextEncode", {"text": negative, "clip": gb.ref(clip)})
    return gb.add("LTXVConditioning", {"positive": gb.ref(pos), "negative": gb.ref(neg), "frame_rate": fps})


def _load_image(gb: GraphBuilder, name: str, compression: int) -> str:
    load = gb.add("LoadImage", {"image": name})
    return gb.add("LTXVPreprocess", {"image": gb.ref(load), "img_compression": compression})


def _load_video_frames(gb: GraphBuilder, name: str, *, tail_seconds: float, compression: int) -> str:
    video = gb.add("LoadVideo", {"file": name})
    sliced = gb.add(
        "Video Slice",
        {"video": gb.ref(video), "start_time": -abs(tail_seconds), "duration": abs(tail_seconds), "strict_duration": False},
    )
    frames = gb.add("GetVideoComponents", {"video": gb.ref(sliced)})
    return gb.add("LTXVPreprocess", {"image": gb.ref(frames), "img_compression": compression})


def _control_image(
    gb: GraphBuilder,
    source: str,
    kind: str,
    *,
    camera: dict[str, Any] | None,
    width: int,
    height: int,
    length: int,
) -> str:
    if kind == "canny":
        return gb.add("Canny", {"image": gb.ref(source), "low_threshold": 0.4, "high_threshold": 0.8})
    if kind == "motion_track" and camera:
        preset = resolve_camera(camera.get("id"))
        tracks = gb.add("GenerateTracks", tracks_inputs(preset, width=width, height=height, length=length))
        repeated = gb.add("RepeatImageBatch", {"image": gb.ref(source), "amount": length})
        return gb.add(
            "WanMoveVisualizeTracks",
            {
                "images": gb.ref(repeated),
                "tracks": gb.ref(tracks),
                "line_resolution": 16,
                "circle_size": 4,
                "opacity": 0.85,
                "line_width": 2,
            },
        )
    return source


def _apply_guides(
    gb: GraphBuilder,
    *,
    pos_neg: str,
    vae: str,
    latent: str,
    uploaded: dict[str, str],
    guides: list[dict[str, Any]],
    compression: int,
    iclora: str | None,
    control: dict[str, Any] | None,
    camera: dict[str, Any] | None,
    width: int,
    height: int,
    length: int,
    fps: float,
) -> tuple[str, str, int]:
    current_cond = pos_neg
    current_latent = latent
    latent_slot = 0
    for guide in guides:
        key = str(guide.get("key") or "")
        kind = str(guide.get("kind") or "image")
        name = uploaded.get(key)
        if key == "__previous__":
            name = uploaded.get("__previous__") or uploaded.get("image")
        if not name:
            continue
        if kind == "video":
            image = _load_video_frames(gb, name, tail_seconds=max(1.0, length / max(fps, 1.0)), compression=compression)
        else:
            image = _load_image(gb, name, compression)
            if control and control.get("source_key") == key:
                image = _control_image(
                    gb,
                    image,
                    str(control.get("kind") or "canny"),
                    camera=camera,
                    width=width,
                    height=height,
                    length=length,
                )
        inputs: dict[str, Any] = {
            "positive": gb.ref(current_cond, 0),
            "negative": gb.ref(current_cond, 1),
            "vae": gb.ref(vae),
            "latent": gb.ref(current_latent, latent_slot),
            "image": gb.ref(image),
            "frame_idx": int(guide.get("frame_idx") or 0),
            "strength": float(guide.get("strength") if guide.get("strength") is not None else 1.0),
        }
        if iclora:
            inputs["iclora_parameters"] = gb.ref(iclora)
        node = gb.add("LTXVAddGuide", inputs)
        current_cond = node
        current_latent = node
        latent_slot = 2
    return current_cond, current_latent, latent_slot


def _upsample(gb: GraphBuilder, samples: str, vae: str, model_name: str) -> str:
    loader = gb.add("LatentUpscaleModelLoader", {"model_name": model_name})
    return gb.add(
        "LTXVLatentUpsampler",
        {"samples": gb.ref(samples), "upscale_model": gb.ref(loader), "vae": gb.ref(vae)},
    )


def build(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    width = _align(int(plan.get("width") or 1216))
    height = _align(int(plan.get("height") or 704))
    length = int(plan.get("length") or 97)
    fps = float(plan.get("fps") or 24)
    refine = bool(plan.get("refine"))
    temporal = bool(plan.get("temporal_refine"))
    compression = int(plan.get("img_compression") if plan.get("img_compression") is not None else 2)
    seed = int(plan.get("seed") or 0)
    steps = int(plan.get("steps") or 8)
    video_cfg = float(plan.get("video_cfg") if plan.get("video_cfg") is not None else plan.get("cfg") or 1.0)
    sampler_name = str(plan.get("sampler_name") or "euler_ancestral")
    unet_name = str(plan.get("unet_name") or UNET_INT8)
    clip_device = str(plan.get("clip_device") or "cpu")
    control = plan.get("control") if isinstance(plan.get("control"), dict) else None
    camera = plan.get("camera") if isinstance(plan.get("camera"), dict) else None
    guides = list(plan.get("guides") or [])

    uploaded: dict[str, str] = {}
    if plan.get("image_name"):
        uploaded["image"] = str(plan["image_name"])
    for index, name in enumerate(plan.get("image_names") or []):
        uploaded.setdefault("image" if index == 0 else f"ref_{index}", str(name))
    for key, name in (plan.get("uploaded") or {}).items():
        uploaded[str(key)] = str(name)
    names = list(plan.get("image_names") or [])
    for guide in guides:
        key = str(guide.get("key") or "")
        if key in {"image", "__previous__"} and plan.get("image_name"):
            uploaded[key] = str(plan["image_name"])
        elif key == "end_image" and len(names) > 1:
            uploaded[key] = names[1]
        elif key.startswith("keyframe_"):
            try:
                idx = int(key.split("_")[1])
            except (IndexError, ValueError):
                idx = -1
            if 0 <= idx + 1 < len(names):
                uploaded[key] = names[idx + 1]
    for index, name in enumerate(plan.get("video_names") or []):
        uploaded.setdefault("video" if index == 0 else f"clip_{index}", str(name))

    video_vae = gb.add("VAELoader", {"vae_name": VIDEO_VAE})
    audio_vae = gb.add("VAELoader", {"vae_name": AUDIO_VAE})
    clip = _clip(gb, clip_device)
    unet = gb.add("UNETLoader", {"unet_name": unet_name, "weight_dtype": "default"})
    iclora = None
    if control and control.get("lora"):
        unet = gb.add(
            "LoraLoaderModelOnly",
            {"model": gb.ref(unet), "lora_name": control["lora"], "strength_model": float(control.get("strength") or 1.0)},
        )
        iclora = gb.add("GetICLoRAParameters", {"iclora_model": gb.ref(unet)})
    for lora in plan.get("loras") or []:
        if lora.get("name"):
            unet = gb.add(
                "LoraLoaderModelOnly",
                {"model": gb.ref(unet), "lora_name": lora["name"], "strength_model": float(lora.get("strength") or 1.0)},
            )
    if plan.get("windowed"):
        unet = gb.add(
            "LTXVContextWindows",
            {
                "model": gb.ref(unet),
                "context_length": WINDOW_LENGTH,
                "context_overlap": WINDOW_OVERLAP,
                "context_schedule": "standard_uniform",
                "context_stride": 1,
                "closed_loop": False,
                "fuse_method": "pyramid",
                "freenoise": True,
                "retain_first_frame": True,
                "split_conds_to_windows": False,
            },
        )
    if str(plan.get("filename_prefix") or "").startswith("shot_action") or plan.get("spatio_temporal"):
        unet = gb.add(
            "LTXVSpatioTemporalGuidance",
            {"model": gb.ref(unet), "scale": 1.15, "blocks": "29", "start_percent": 0.0, "end_percent": 1.0},
        )

    latent = gb.add("EmptyLTXVLatentVideo", {"width": width, "height": height, "length": length, "batch_size": 1})
    cond = _condition(gb, clip, str(plan.get("prompt") or ""), str(plan.get("negative_prompt") or ""), fps)
    cond, latent, latent_slot = _apply_guides(
        gb,
        pos_neg=cond,
        vae=video_vae,
        latent=latent,
        uploaded=uploaded,
        guides=guides,
        compression=compression,
        iclora=iclora,
        control=control,
        camera=camera,
        width=width,
        height=height,
        length=length,
        fps=fps,
    )
    if control and control.get("source_key") == "video" and "video" in uploaded:
        motion = _load_video_frames(gb, uploaded["video"], tail_seconds=max(1.0, length / max(fps, 1.0)), compression=compression)
        motion = _control_image(gb, motion, str(control.get("kind") or "canny"), camera=camera, width=width, height=height, length=length)
        inputs: dict[str, Any] = {
            "positive": gb.ref(cond, 0),
            "negative": gb.ref(cond, 1),
            "vae": gb.ref(video_vae),
            "latent": gb.ref(latent, latent_slot),
            "image": gb.ref(motion),
            "frame_idx": 0,
            "strength": float(control.get("strength") or 1.0),
        }
        if iclora:
            inputs["iclora_parameters"] = gb.ref(iclora)
        cond = gb.add("LTXVAddGuide", inputs)
        latent = cond
        latent_slot = 2

    av = _concat_av(gb, gb.ref(latent, latent_slot), audio_vae, length, fps)
    sampled = _sample(
        gb,
        model=unet,
        positive=gb.ref(cond, 0),
        negative=gb.ref(cond, 1),
        latent=gb.ref(av),
        sigmas=gb.ref(gb.add("ManualSigmas", {"sigmas": STAGE1_SIGMAS}) if refine else gb.add(
            "LTXVScheduler",
            {
                "steps": steps,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": gb.ref(av),
            },
        )),
        seed=seed,
        sampler_name=sampler_name,
        video_cfg=video_cfg,
        audio_cfg=1.0,
    )
    if refine:
        separated = gb.add("LTXVSeparateAVLatent", {"av_latent": gb.ref(sampled)})
        video_for_refine = _upsample(gb, separated, video_vae, SPATIAL_UPSCALER)
        av_refine = gb.add("LTXVConcatAVLatent", {"video_latent": gb.ref(video_for_refine), "audio_latent": gb.ref(separated, 1)})
        sampled = _sample(
            gb,
            model=unet,
            positive=gb.ref(cond, 0),
            negative=gb.ref(cond, 1),
            latent=gb.ref(av_refine),
            sigmas=gb.ref(gb.add("ManualSigmas", {"sigmas": REFINE_SIGMAS})),
            seed=seed,
            sampler_name=sampler_name,
            video_cfg=video_cfg,
            audio_cfg=1.0,
        )
        if temporal:
            separated = gb.add("LTXVSeparateAVLatent", {"av_latent": gb.ref(sampled)})
            video_temporal = _upsample(gb, separated, video_vae, TEMPORAL_UPSCALER)
            av_temporal = gb.add("LTXVConcatAVLatent", {"video_latent": gb.ref(video_temporal), "audio_latent": gb.ref(separated, 1)})
            sampled = _sample(
                gb,
                model=unet,
                positive=gb.ref(cond, 0),
                negative=gb.ref(cond, 1),
                latent=gb.ref(av_temporal),
                sigmas=gb.ref(gb.add("ManualSigmas", {"sigmas": REFINE_SIGMAS})),
                seed=seed,
                sampler_name=sampler_name,
                video_cfg=video_cfg,
                audio_cfg=1.0,
            )
    cropped = gb.add(
        "LTXVCropGuides",
        {"positive": gb.ref(cond, 0), "negative": gb.ref(cond, 1), "latent": gb.ref(sampled)},
    )
    separated = gb.add("LTXVSeparateAVLatent", {"av_latent": gb.ref(cropped, 2)})
    decode = gb.add(
        "VAEDecodeTiled",
        {
            "samples": gb.ref(separated, 0),
            "vae": gb.ref(video_vae),
            "tile_size": 768,
            "overlap": 96,
            "temporal_size": 64,
            "temporal_overlap": 16,
        },
    )
    audio = gb.add("LTXVAudioVAEDecode", {"samples": gb.ref(separated, 1), "audio_vae": gb.ref(audio_vae)})
    video = gb.add("CreateVideo", {"images": gb.ref(decode), "fps": fps, "audio": gb.ref(audio)})
    gb.add(
        "SaveVideo",
        {
            "video": gb.ref(video),
            "filename_prefix": str(plan.get("filename_prefix") or "ltx_shot"),
            "format": "mp4",
            "codec": "h264",
        },
    )
    return gb.build()
