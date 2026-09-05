"""ComfyUI graph builder for LTX-2.5 22B distilled video flows."""

from __future__ import annotations

from typing import Any

M = {
    "unet": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "text_encoder": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "prompt_enhancer": "gemma4_e2b_it_int8_convrot.safetensors",
    "video_vae": "ltx-2.5-video-vae-bf16.safetensors",
    "loader_ckpt": "ltx-2.5-distilled-transformer-comfy-int8-convrot.safetensors",
    "audio_vae_ckpt": "ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscaler": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "depth_anything": "video_depth_anything_vits.pth",
}

DEFAULT_STEPS = 20
DEFAULT_VIDEO_CFG = 1.0
DEFAULT_AUDIO_CFG = 1.0
DEFAULT_REFINE_SEED = 42
DEFAULT_FPS = 24.0
DEFAULT_SAMPLER = "euler_ancestral"
DEFAULT_MAX_SHIFT = 2.05
DEFAULT_BASE_SHIFT = 0.95
DEFAULT_TERMINAL = 0.1
DEFAULT_REFINE_STEPS = 3
DEFAULT_REFINE_DENOISE = 0.4
# ComfyUI LTX-2.5-Quality-T2V.json manual sigma schedules (sharper than generic LTXVScheduler).
STUDIO_STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
STUDIO_REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
NEG_DEFAULT = (
    "blurry, low quality, still frame, frames, watermark, overlay, titles, "
    "has blurbox, has subtitles"
)


class GraphBuilder:
    def __init__(self) -> None:
        self._nid = 0
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_meta: dict[str, dict[str, str]] = {}

    def add(
        self,
        class_type: str,
        inputs: dict[str, Any],
        *,
        stage: str | None = None,
        label: str | None = None,
    ) -> str:
        self._nid += 1
        nid = str(self._nid)
        self.nodes[nid] = {"class_type": class_type, "inputs": inputs}
        if stage:
            self.node_meta[nid] = {"stage": stage, "label": label or class_type}
        return nid

    def ref(self, nid: str, slot: int = 0) -> list[Any]:
        return [nid, slot]

    def build(self) -> dict[str, Any]:
        return self.nodes


def sanitize_pixels(value: int, *, minimum: int = 256, maximum: int = 2048, align: int = 32) -> int:
    v = max(minimum, min(maximum, int(value)))
    return max(minimum, (v // align) * align)


def sanitize_length(value: int, *, minimum: int = 17, maximum: int = 241) -> int:
    """LTX video latents require length == 8k + 1."""
    v = max(minimum, min(maximum, int(value)))
    return ((v - 1) // 8) * 8 + 1


def duration_to_length(duration: float, fps: float) -> int:
    # Matches the ComfyUI workflow expression: duration * frame_rate + 1
    return sanitize_length(int(round(float(duration) * float(fps))) + 1)


def combine_prompt(video_prompt: str, audio_prompt: str) -> str:
    """Merge video + audio into one LTX-style paragraph (better prompt adherence)."""
    video = (video_prompt or "").strip()
    audio = (audio_prompt or "").strip()
    silent_markers = ("silent", "no sound", "no music", "no audio")
    if not audio or any(m in audio.lower() for m in silent_markers):
        audio = "silent, no sound, no music, no background audio"
    if not video:
        return audio
    lower = video.lower()
    if "audio:" in lower or any(m in lower for m in silent_markers):
        return video
    return f"{video} Audio: {audio}."


def _plan_float(plan: dict[str, Any], key: str, default: float) -> float:
    val = plan.get(key)
    return float(val) if val is not None else default


def _plan_int(plan: dict[str, Any], key: str, default: int) -> int:
    val = plan.get(key)
    return int(val) if val is not None else default


def _plan_bool(plan: dict[str, Any], key: str, default: bool = False) -> bool:
    val = plan.get(key)
    if val is None:
        return default
    return bool(val)


def _normalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    refine = _plan_bool(plan, "refine")
    align = 64 if refine else 32
    width = sanitize_pixels(plan["width"], align=align)
    height = sanitize_pixels(plan["height"], align=align)
    length = sanitize_length(int(plan.get("length", 97)))
    fps = _plan_float(plan, "fps", DEFAULT_FPS)
    return {**plan, "width": width, "height": height, "length": length, "fps": fps}


def _manual_sigmas(gb: GraphBuilder, sigmas: str, *, stage: str, label: str) -> list[Any]:
    nid = gb.add(
        "ManualSigmas",
        {"sigmas": sigmas},
        stage=stage,
        label=label,
    )
    return gb.ref(nid)


def _stage1_sigmas(gb: GraphBuilder, plan: dict[str, Any], av_latent_ref: list[Any], steps: int) -> list[Any]:
    if _plan_bool(plan, "refine") and not _plan_bool(plan, "use_ltx_scheduler", False):
        sigmas = str(plan.get("stage1_sigmas") or STUDIO_STAGE1_SIGMAS)
        return _manual_sigmas(gb, sigmas, stage="sample", label="Stage 1 sigmas (studio)")
    stretch = plan.get("stretch")
    if stretch is None:
        stretch = True
    terminal = _plan_float(plan, "terminal", DEFAULT_TERMINAL)
    max_shift = _plan_float(plan, "max_shift", DEFAULT_MAX_SHIFT)
    base_shift = _plan_float(plan, "base_shift", DEFAULT_BASE_SHIFT)
    sigmas_nid = gb.add(
        "LTXVScheduler",
        {
            "steps": steps,
            "max_shift": max_shift,
            "base_shift": base_shift,
            "stretch": bool(stretch),
            "terminal": terminal,
            "latent": av_latent_ref,
        },
        stage="sample",
        label="LTX scheduler",
    )
    return gb.ref(sigmas_nid)


def _refine_sigmas(gb: GraphBuilder, plan: dict[str, Any], av_latent_ref: list[Any], refine_steps: int) -> list[Any]:
    if not _plan_bool(plan, "use_ltx_scheduler", False):
        sigmas = str(plan.get("refine_sigmas") or STUDIO_REFINE_SIGMAS)
        return _manual_sigmas(gb, sigmas, stage="refine", label="Refine sigmas (studio)")
    stretch = plan.get("stretch")
    if stretch is None:
        stretch = True
    terminal = _plan_float(plan, "terminal", DEFAULT_TERMINAL)
    max_shift = _plan_float(plan, "max_shift", DEFAULT_MAX_SHIFT)
    base_shift = _plan_float(plan, "base_shift", DEFAULT_BASE_SHIFT)
    refine_denoise = _plan_float(plan, "refine_denoise", DEFAULT_REFINE_DENOISE)
    refine_sigmas_full = gb.add(
        "LTXVScheduler",
        {
            "steps": refine_steps,
            "max_shift": max_shift,
            "base_shift": base_shift,
            "stretch": bool(stretch),
            "terminal": terminal,
            "latent": av_latent_ref,
        },
        stage="refine",
        label="Refine scheduler",
    )
    refine_sigmas = gb.add(
        "SplitSigmasDenoise",
        {"sigmas": gb.ref(refine_sigmas_full), "denoise": refine_denoise},
        stage="refine",
        label="Refine denoise split",
    )
    return gb.ref(refine_sigmas)


def _apply_standard_loras(gb: GraphBuilder, model_id: str, plan: dict[str, Any]) -> str:
    current = model_id
    for lora in plan.get("loras") or []:
        name = lora.get("name")
        if not name:
            continue
        strength = float(lora.get("strength", 1.0))
        current = gb.add(
            "LoraLoaderModelOnly",
            {"model": gb.ref(current), "lora_name": name, "strength_model": strength},
            stage="load_models",
            label=f"Camera LoRA {name}",
        )
    return current


def _apply_ic_loras(
    gb: GraphBuilder, model_id: str, plan: dict[str, Any]
) -> tuple[str, list[Any] | None]:
    current = model_id
    downscale_ref: list[Any] | None = None
    for lora in plan.get("ic_loras") or []:
        name = lora.get("name")
        if not name:
            continue
        strength = float(lora.get("strength", 1.0))
        current = gb.add(
            "LTXICLoRALoaderModelOnly",
            {"model": gb.ref(current), "lora_name": name, "strength_model": strength},
            stage="load_models",
            label=f"IC-LoRA {name}",
        )
        downscale_ref = gb.ref(current, 1)
    return current, downscale_ref


def _load_model(gb: GraphBuilder, plan: dict[str, Any]) -> tuple[str, list[Any] | None]:
    weight_dtype = str(plan.get("weight_dtype") or "default")
    unet = gb.add(
        "UNETLoader",
        {"unet_name": M["unet"], "weight_dtype": weight_dtype},
        stage="load_models",
        label="Load LTX UNet",
    )
    unet = _apply_standard_loras(gb, unet, plan)
    return _apply_ic_loras(gb, unet, plan)


def middle_frame_idx(length: int) -> int:
    """Snap the midpoint to a multiple of 8 (LTX guide requirement for 9+ frame clips)."""
    mid = max(0, (int(length) - 1) // 2)
    return (mid // 8) * 8


def _load_still(gb: GraphBuilder, image_name: str, *, label: str) -> list[Any]:
    loaded = gb.add("LoadImage", {"image": image_name}, stage="encode", label=label)
    pre = gb.add(
        "LTXVPreprocess",
        {"image": gb.ref(loaded), "img_compression": 18},
        stage="encode",
        label=f"Preprocess {label}",
    )
    return gb.ref(pre)


def _i2v_inplace(
    gb: GraphBuilder,
    *,
    vae_ref: list[Any],
    image_ref: list[Any],
    latent_ref: list[Any],
    strength: float,
    stage: str,
    label: str,
    bypass: bool = False,
) -> list[Any]:
    nid = gb.add(
        "LTXVImgToVideoInplace",
        {
            "vae": vae_ref,
            "image": image_ref,
            "latent": latent_ref,
            "strength": strength,
            "bypass": bypass,
        },
        stage=stage,
        label=label,
    )
    return gb.ref(nid)


def _add_frame_guide(
    gb: GraphBuilder,
    *,
    cond_pos_ref: list[Any],
    cond_neg_ref: list[Any],
    vae_ref: list[Any],
    latent_ref: list[Any],
    image_ref: list[Any],
    frame_idx: int,
    strength: float,
    stage: str,
    label: str,
) -> tuple[list[Any], list[Any], list[Any]]:
    nid = gb.add(
        "LTXVAddGuide",
        {
            "positive": cond_pos_ref,
            "negative": cond_neg_ref,
            "vae": vae_ref,
            "latent": latent_ref,
            "image": image_ref,
            "frame_idx": int(frame_idx),
            "strength": float(strength),
        },
        stage=stage,
        label=label,
    )
    return gb.ref(nid, 0), gb.ref(nid, 1), gb.ref(nid, 2)


def _empty_video(gb: GraphBuilder, width: int, height: int, length: int) -> list[Any]:
    nid = gb.add(
        "EmptyLTXVLatentVideo",
        {"width": width, "height": height, "length": length, "batch_size": 1},
        stage="encode",
        label="Empty video latent",
    )
    return gb.ref(nid)


def _video_vae(gb: GraphBuilder) -> list[Any]:
    nid = gb.add(
        "VAELoader",
        {"vae_name": M["video_vae"]},
        stage="load_models",
        label="Load video VAE",
    )
    return gb.ref(nid)


def _annotate_reference(gb: GraphBuilder, images_ref: list[Any], plan: dict[str, Any]) -> list[Any]:
    control = str(plan.get("control_type") or "depth")
    if control == "canny":
        nid = gb.add(
            "CannyEdgePreprocessor",
            {
                "image": images_ref,
                "low_threshold": _plan_int(plan, "canny_low", 92),
                "high_threshold": _plan_int(plan, "canny_high", 200),
                "resolution": _plan_int(plan, "annotator_resolution", 512),
            },
            stage="ic_guide",
            label="Canny edges",
        )
        return gb.ref(nid)
    if control == "pose":
        nid = gb.add(
            "DWPreprocessor",
            {
                "image": images_ref,
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": _plan_int(plan, "annotator_resolution", 512),
                "bbox_detector": "yolox_l.onnx",
                "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
                "scale_stick_for_xinsr_cn": "disable",
            },
            stage="ic_guide",
            label="DWPose",
        )
        return gb.ref(nid)
    model = gb.add(
        "LoadVideoDepthAnythingModel",
        {"model": M["depth_anything"]},
        stage="ic_guide",
        label="Load Video Depth Anything",
    )
    depths = gb.add(
        "VideoDepthAnythingProcess",
        {
            "vda_model": gb.ref(model),
            "images": images_ref,
            "input_size": 518,
            "max_res": 960,
            "precision": "fp32",
        },
        stage="ic_guide",
        label="Depth annotate",
    )
    out = gb.add(
        "VideoDepthAnythingOutput",
        {"depths": gb.ref(depths), "colormap": "gray"},
        stage="ic_guide",
        label="Depth map images",
    )
    return gb.ref(out)


def _inject_ic_guides(
    gb: GraphBuilder,
    *,
    plan: dict[str, Any],
    cond_pos_ref: list[Any],
    cond_neg_ref: list[Any],
    video_latent_ref: list[Any],
    video_vae_ref: list[Any],
    downscale_ref: list[Any] | None,
) -> tuple[list[Any], list[Any], list[Any]]:
    still_ref = plan.get("ic_guide_image_ref")
    video_name = str(plan.get("video_name") or "")
    if still_ref and not video_name:
        guide_images = still_ref
    else:
        load_video = gb.add(
            "LoadVideo",
            {"file": video_name},
            stage="ic_guide",
            label="Load reference video",
        )
        components = gb.add(
            "GetVideoComponents",
            {"video": gb.ref(load_video)},
            stage="ic_guide",
            label="Video frames",
        )
        frame_ref = gb.ref(components)
        if _plan_bool(plan, "ic_guide_raw"):
            guide_images = frame_ref
        else:
            guide_images = _annotate_reference(gb, frame_ref, plan)
    strength = _plan_float(plan, "ic_lora_strength", 1.0)
    guide_inputs: dict[str, Any] = {
        "positive": cond_pos_ref,
        "negative": cond_neg_ref,
        "vae": video_vae_ref,
        "latent": video_latent_ref,
        "image": guide_images,
        "frame_idx": 0,
        "strength": strength,
        "crop": "disabled",
        "use_tiled_encode": False,
        "tile_size": 256,
        "tile_overlap": 64,
    }
    if downscale_ref is not None:
        guide_inputs["latent_downscale_factor"] = downscale_ref
    else:
        guide_inputs["latent_downscale_factor"] = 2.0
    guide = gb.add(
        "LTXAddVideoICLoRAGuide",
        guide_inputs,
        stage="ic_guide",
        label="IC-LoRA guide",
    )
    return gb.ref(guide, 0), gb.ref(guide, 1), gb.ref(guide, 2)


def _sampling_pass(
    gb: GraphBuilder,
    *,
    plan: dict[str, Any],
    model_ref: list[Any],
    av_latent_ref: list[Any],
    cond_pos_ref: list[Any],
    cond_neg_ref: list[Any],
    steps: int,
    sigmas_ref: list[Any] | None = None,
    stage: str = "sample",
    label: str = "Sample AV latent",
) -> str:
    if stage == "refine":
        seed = _plan_int(plan, "refine_seed", DEFAULT_REFINE_SEED)
    else:
        seed = _plan_int(plan, "seed", 42)
    video_cfg = _plan_float(plan, "video_cfg", DEFAULT_VIDEO_CFG)
    audio_cfg = _plan_float(plan, "audio_cfg", DEFAULT_AUDIO_CFG)
    sampler_name = str(plan.get("sampler_name") or DEFAULT_SAMPLER)

    noise = gb.add("RandomNoise", {"noise_seed": seed}, stage=stage, label="Random noise")
    sampler = gb.add(
        "KSamplerSelect",
        {"sampler_name": sampler_name},
        stage=stage,
        label=f"Sampler ({sampler_name})",
    )
    if sigmas_ref is None:
        sigmas_ref = _stage1_sigmas(gb, plan, av_latent_ref, steps)

    guider = gb.add(
        "LTXVDualCFGGuider",
        {
            "model": model_ref,
            "positive": cond_pos_ref,
            "negative": cond_neg_ref,
            "video_cfg": video_cfg,
            "audio_cfg": audio_cfg,
        },
        stage=stage,
        label="Dual CFG guider",
    )
    return gb.add(
        "SamplerCustomAdvanced",
        {
            "noise": gb.ref(noise),
            "guider": gb.ref(guider),
            "sampler": gb.ref(sampler),
            "sigmas": sigmas_ref,
            "latent_image": av_latent_ref,
        },
        stage=stage,
        label=label,
    )


def _decode_and_export(
    gb: GraphBuilder,
    *,
    plan: dict[str, Any],
    sampled_av_ref: list[Any],
    video_vae_ref: list[Any],
    audio_vae_ref: list[Any],
) -> None:
    fps = _plan_float(plan, "fps", DEFAULT_FPS)
    separated = gb.add(
        "LTXVSeparateAVLatent",
        {"av_latent": sampled_av_ref},
        stage="decode",
        label="Separate AV latent",
    )
    if _plan_bool(plan, "tiled_decode"):
        decoded_video = gb.add(
            "VAEDecodeTiled",
            {
                "samples": gb.ref(separated, 0),
                "vae": video_vae_ref,
                "tile_size": _plan_int(plan, "tile_size", 512),
                "overlap": _plan_int(plan, "tile_overlap", 64),
                "temporal_size": _plan_int(plan, "temporal_size", 64),
                "temporal_overlap": _plan_int(plan, "temporal_overlap", 16),
            },
            stage="decode",
            label="Tiled VAE decode",
        )
    else:
        decoded_video = gb.add(
            "VAEDecode",
            {"samples": gb.ref(separated, 0), "vae": video_vae_ref},
            stage="decode",
            label="VAE decode",
        )
    remux_audio_ref = plan.get("remux_audio_ref")
    if remux_audio_ref is None:
        decoded_audio = gb.add(
            "LTXVAudioVAEDecode",
            {"samples": gb.ref(separated, 1), "audio_vae": audio_vae_ref},
            stage="decode",
            label="Audio VAE decode",
        )
        audio_ref = gb.ref(decoded_audio)
    else:
        audio_ref = remux_audio_ref
    video = gb.add(
        "CreateVideo",
        {"images": gb.ref(decoded_video), "fps": fps, "audio": audio_ref},
        stage="export",
        label="Create video",
    )
    gb.add(
        "SaveVideo",
        {
            "video": gb.ref(video),
            "filename_prefix": plan.get("filename_prefix", "ltx_video"),
            "format": "mp4",
            "codec": "h264",
        },
        stage="export",
        label="Save MP4",
    )


def _append_sampling_tail(
    gb: GraphBuilder,
    *,
    plan: dict[str, Any],
    video_latent_ref: list[Any],
    cond_pos_ref: list[Any],
    cond_neg_ref: list[Any],
    video_vae_ref: list[Any] | None = None,
    start_image_ref: list[Any] | None = None,
    frame_guides: list[dict[str, Any]] | None = None,
    crop_guides: bool = False,
) -> None:
    refine = _plan_bool(plan, "refine")
    steps = _plan_int(plan, "steps", DEFAULT_STEPS)
    refine_steps = _plan_int(plan, "refine_steps", DEFAULT_REFINE_STEPS)
    fps = _plan_float(plan, "fps", DEFAULT_FPS)
    length = _plan_int(plan, "length", 97)
    ic_enabled = _plan_bool(plan, "ic_enabled") and (
        bool(plan.get("video_name")) or plan.get("ic_guide_image_ref") is not None
    )
    frame_guides = list(frame_guides or [])

    if video_vae_ref is None:
        video_vae_nid = gb.add(
            "VAELoader",
            {"vae_name": M["video_vae"]},
            stage="load_models",
            label="Load video VAE",
        )
        video_vae_ref = gb.ref(video_vae_nid)

    model_nid, ic_downscale = _load_model(gb, plan)
    model_ref = gb.ref(model_nid)

    if ic_enabled:
        cond_pos_ref, cond_neg_ref, video_latent_ref = _inject_ic_guides(
            gb,
            plan=plan,
            cond_pos_ref=cond_pos_ref,
            cond_neg_ref=cond_neg_ref,
            video_latent_ref=video_latent_ref,
            video_vae_ref=video_vae_ref,
            downscale_ref=ic_downscale,
        )

    audio_vae = gb.add(
        "LTXVAudioVAELoader",
        {"ckpt_name": M["audio_vae_ckpt"]},
        stage="load_models",
        label="Load audio VAE",
    )
    audio_vae_ref = gb.ref(audio_vae)

    audio_name = str(plan.get("audio_name") or "")
    if audio_name:
        load_audio = gb.add(
            "LoadAudio",
            {"audio": audio_name},
            stage="encode",
            label="Load source audio",
        )
        encoded_audio = gb.add(
            "LTXVAudioVAEEncode",
            {"audio": gb.ref(load_audio), "audio_vae": audio_vae_ref},
            stage="encode",
            label="Encode source audio",
        )
        tokens = gb.add(
            "LTXVSetAudioRefTokens",
            {
                "positive": cond_pos_ref,
                "negative": cond_neg_ref,
                "audio_latent": gb.ref(encoded_audio),
            },
            stage="encode",
            label="Freeze audio tokens",
        )
        cond_pos_ref = gb.ref(tokens, 0)
        cond_neg_ref = gb.ref(tokens, 1)
        audio_latent_ref = gb.ref(tokens, 2)
        plan["remux_audio_ref"] = gb.ref(load_audio)
    else:
        empty_audio = gb.add(
            "LTXVEmptyLatentAudio",
            {
                "frames_number": length,
                "frame_rate": fps,
                "batch_size": 1,
                "audio_vae": audio_vae_ref,
            },
            stage="encode",
            label="Empty audio latent",
        )
        audio_latent_ref = gb.ref(empty_audio)
    av_latent = gb.add(
        "LTXVConcatAVLatent",
        {"video_latent": video_latent_ref, "audio_latent": audio_latent_ref},
        stage="encode",
        label="Concat AV latent",
    )

    sampled_stage1 = _sampling_pass(
        gb,
        plan=plan,
        model_ref=model_ref,
        av_latent_ref=gb.ref(av_latent),
        cond_pos_ref=cond_pos_ref,
        cond_neg_ref=cond_neg_ref,
        steps=steps,
        stage="sample",
        label="Stage 1 sample",
    )

    if ic_enabled or crop_guides or frame_guides:
        cropped = gb.add(
            "LTXVCropGuides",
            {
                "positive": cond_pos_ref,
                "negative": cond_neg_ref,
                "latent": gb.ref(sampled_stage1),
            },
            stage="refine" if refine else "sample",
            label="Crop guides",
        )
        cond_pos_ref = gb.ref(cropped, 0)
        cond_neg_ref = gb.ref(cropped, 1)
        sampled_stage1 = cropped
        sampled_stage1_ref = gb.ref(cropped, 2)
    else:
        sampled_stage1_ref = gb.ref(sampled_stage1)

    if not refine:
        _decode_and_export(
            gb,
            plan=plan,
            sampled_av_ref=sampled_stage1_ref,
            video_vae_ref=video_vae_ref,
            audio_vae_ref=audio_vae_ref,
        )
        return

    separated1 = gb.add(
        "LTXVSeparateAVLatent",
        {"av_latent": sampled_stage1_ref},
        stage="refine",
        label="Separate stage-1 AV",
    )
    upscaler = gb.add(
        "LatentUpscaleModelLoader",
        {"model_name": M["latent_upscaler"]},
        stage="refine",
        label="Load latent upscaler",
    )
    upscaled_video = gb.add(
        "LTXVLatentUpsampler",
        {
            "samples": gb.ref(separated1, 0),
            "upscale_model": gb.ref(upscaler),
            "vae": video_vae_ref,
        },
        stage="refine",
        label="Latent 2x upscale",
    )
    video_for_refine = gb.ref(upscaled_video)
    if start_image_ref is not None:
        video_for_refine = _i2v_inplace(
            gb,
            vae_ref=video_vae_ref,
            image_ref=start_image_ref,
            latent_ref=video_for_refine,
            strength=_plan_float(plan, "refine_strength", 1.0),
            stage="refine",
            label="Re-inject start image",
        )
    refine_pos, refine_neg = cond_pos_ref, cond_neg_ref
    if frame_guides:
        for guide in frame_guides:
            refine_pos, refine_neg, video_for_refine = _add_frame_guide(
                gb,
                cond_pos_ref=refine_pos,
                cond_neg_ref=refine_neg,
                vae_ref=video_vae_ref,
                latent_ref=video_for_refine,
                image_ref=guide["image_ref"],
                frame_idx=int(guide["frame_idx"]),
                strength=float(guide.get("strength", 0.7)),
                stage="refine",
                label=str(guide.get("label") or "Re-apply frame guide"),
            )
        crop_guides = True
    av_latent2 = gb.add(
        "LTXVConcatAVLatent",
        {"video_latent": video_for_refine, "audio_latent": gb.ref(separated1, 1)},
        stage="refine",
        label="Rejoin AV for refine",
    )

    refine_sigmas_ref = _refine_sigmas(gb, plan, gb.ref(av_latent2), refine_steps)

    sampled_stage2 = _sampling_pass(
        gb,
        plan=plan,
        model_ref=model_ref,
        av_latent_ref=gb.ref(av_latent2),
        cond_pos_ref=refine_pos,
        cond_neg_ref=refine_neg,
        steps=refine_steps,
        sigmas_ref=refine_sigmas_ref,
        stage="refine",
        label="Stage 2 refine sample",
    )
    if crop_guides or frame_guides:
        cropped2 = gb.add(
            "LTXVCropGuides",
            {
                "positive": refine_pos,
                "negative": refine_neg,
                "latent": gb.ref(sampled_stage2),
            },
            stage="refine",
            label="Crop refine guides",
        )
        sampled_stage2 = cropped2
    _decode_and_export(
        gb,
        plan=plan,
        sampled_av_ref=gb.ref(sampled_stage2),
        video_vae_ref=video_vae_ref,
        audio_vae_ref=audio_vae_ref,
    )


def _prompt_enhance_sampling(plan: dict[str, Any]) -> dict[str, Any]:
    """Dynamic combo payload for TextGenerateLTX2Prompt (matches ComfyUI LTX Studio workflow)."""
    return {
        "sampling_mode": "on",
        "temperature": float(plan.get("prompt_enhance_temperature", 0.7)),
        "top_k": int(plan.get("prompt_enhance_top_k", 64)),
        "top_p": float(plan.get("prompt_enhance_top_p", 0.95)),
        "min_p": float(plan.get("prompt_enhance_min_p", 0.05)),
        "repetition_penalty": float(plan.get("prompt_enhance_repetition_penalty", 1.15)),
        "presence_penalty": float(plan.get("prompt_enhance_presence_penalty", 0.0)),
        "seed": int(plan.get("prompt_enhance_seed", 0)),
    }


def _resolve_positive_text(
    gb: GraphBuilder,
    plan: dict[str, Any],
    *,
    image_ref: list[Any] | None = None,
) -> Any:
    raw = combine_prompt(plan.get("prompt", ""), plan.get("audio_prompt", ""))
    if not _plan_bool(plan, "prompt_enhance"):
        return raw

    enhancer_clip = gb.add(
        "CLIPLoader",
        {
            "clip_name": M["prompt_enhancer"],
            "type": "ltxv",
            "device": "default",
        },
        stage="prompt_enhance",
        label="Load prompt enhancer",
    )
    enhance_inputs: dict[str, Any] = {
        "clip": gb.ref(enhancer_clip),
        "prompt": raw,
        "max_length": _plan_int(plan, "prompt_enhance_max_length", 600),
        "sampling_mode": _prompt_enhance_sampling(plan),
        "thinking": bool(plan.get("prompt_enhance_thinking", False)),
        "use_default_template": bool(plan.get("prompt_enhance_use_template", True)),
    }
    if image_ref is not None:
        enhance_inputs["image"] = image_ref
    enhanced = gb.add(
        "TextGenerateLTX2Prompt",
        enhance_inputs,
        stage="prompt_enhance",
        label="Enhance prompt (LTX Studio)",
    )
    return gb.ref(enhanced)


def _encode_text(
    gb: GraphBuilder,
    plan: dict[str, Any],
    fps: float,
    *,
    image_ref: list[Any] | None = None,
) -> str:
    pos_text = _resolve_positive_text(gb, plan, image_ref=image_ref)
    neg_text = plan.get("negative_prompt", NEG_DEFAULT)
    clip = gb.add(
        "LTXAVTextEncoderLoader",
        {
            "text_encoder": M["text_encoder"],
            "ckpt_name": M["loader_ckpt"],
            "device": "default",
        },
        stage="load_models",
        label="Load LTX text encoder",
    )
    pos = gb.add(
        "CLIPTextEncode",
        {"text": pos_text, "clip": gb.ref(clip)},
        stage="encode",
        label="Encode positive",
    )
    neg = gb.add(
        "CLIPTextEncode",
        {"text": neg_text, "clip": gb.ref(clip)},
        stage="encode",
        label="Encode negative",
    )
    return gb.add(
        "LTXVConditioning",
        {"positive": gb.ref(pos), "negative": gb.ref(neg), "frame_rate": fps},
        stage="encode",
        label="LTX conditioning",
    )


def _apply_start_image(
    gb: GraphBuilder,
    merged: dict[str, Any],
    *,
    video_vae_ref: list[Any],
    video_latent_ref: list[Any],
    image_name: str,
    refine: bool,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    start_ref = _load_still(gb, image_name, label="Load start image")
    cond = _encode_text(gb, merged, merged["fps"], image_ref=start_ref)
    stage1_strength = 0.7 if refine else _plan_float(merged, "strength", 1.0)
    video_latent_ref = _i2v_inplace(
        gb,
        vae_ref=video_vae_ref,
        image_ref=start_ref,
        latent_ref=video_latent_ref,
        strength=stage1_strength,
        stage="encode",
        label="Image to video encode",
    )
    return start_ref, video_latent_ref, gb.ref(cond, 0), gb.ref(cond, 1)


def _apply_end_middle_guides(
    gb: GraphBuilder,
    merged: dict[str, Any],
    *,
    video_vae_ref: list[Any],
    video_latent_ref: list[Any],
    cond_pos_ref: list[Any],
    cond_neg_ref: list[Any],
) -> tuple[list[Any], list[Any], list[Any], list[dict[str, Any]]]:
    guides: list[dict[str, Any]] = []
    strength = _plan_float(merged, "guide_strength", 0.7)
    if merged.get("end_image_name"):
        end_ref = _load_still(gb, str(merged["end_image_name"]), label="Load last frame")
        cond_pos_ref, cond_neg_ref, video_latent_ref = _add_frame_guide(
            gb,
            cond_pos_ref=cond_pos_ref,
            cond_neg_ref=cond_neg_ref,
            vae_ref=video_vae_ref,
            latent_ref=video_latent_ref,
            image_ref=end_ref,
            frame_idx=-1,
            strength=strength,
            stage="encode",
            label="Last-frame guide",
        )
        guides.append({"image_ref": end_ref, "frame_idx": -1, "strength": strength, "label": "Last-frame refine"})
    if merged.get("middle_image_name"):
        mid_ref = _load_still(gb, str(merged["middle_image_name"]), label="Load middle frame")
        mid_idx = middle_frame_idx(merged["length"])
        cond_pos_ref, cond_neg_ref, video_latent_ref = _add_frame_guide(
            gb,
            cond_pos_ref=cond_pos_ref,
            cond_neg_ref=cond_neg_ref,
            vae_ref=video_vae_ref,
            latent_ref=video_latent_ref,
            image_ref=mid_ref,
            frame_idx=mid_idx,
            strength=strength,
            stage="encode",
            label="Middle-frame guide",
        )
        guides.append({"image_ref": mid_ref, "frame_idx": mid_idx, "strength": strength, "label": "Middle-frame refine"})
    return cond_pos_ref, cond_neg_ref, video_latent_ref, guides


def _build_graph(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    mode = plan.get("mode", "t2v")
    merged = _normalize_plan(plan)
    refine = _plan_bool(merged, "refine")
    gen_width = merged["width"] // 2 if refine else merged["width"]
    gen_height = merged["height"] // 2 if refine else merged["height"]
    fps = merged["fps"]
    length = merged["length"]

    gb = GraphBuilder()
    video_vae_ref = _video_vae(gb)
    video_latent_ref = _empty_video(gb, gen_width, gen_height, length)
    start_image_ref: list[Any] | None = None
    frame_guides: list[dict[str, Any]] = []
    crop_guides = False

    if mode == "t2v":
        cond = _encode_text(gb, merged, fps)
        cond_pos_ref, cond_neg_ref = gb.ref(cond, 0), gb.ref(cond, 1)
    elif mode in {"i2v", "a2v", "flf2v", "lipsync", "motion_transfer"}:
        start_name = merged.get("image_name")
        if mode in {"i2v", "flf2v", "lipsync"} and not start_name:
            raise ValueError(f"image_name is required for {mode}")
        if mode == "flf2v" and not merged.get("end_image_name"):
            raise ValueError("end_image_name is required for flf2v")
        if mode == "a2v" and not merged.get("audio_name"):
            raise ValueError("audio_name is required for a2v")
        if mode == "lipsync" and not merged.get("audio_name"):
            raise ValueError("audio_name is required for lipsync")
        if mode == "motion_transfer" and not merged.get("video_name"):
            raise ValueError("video_name is required for motion_transfer")

        cond = None
        cond_pos_ref: list[Any]
        cond_neg_ref: list[Any]
        if start_name:
            start_image_ref, video_latent_ref, cond_pos_ref, cond_neg_ref = _apply_start_image(
                gb,
                merged,
                video_vae_ref=video_vae_ref,
                video_latent_ref=video_latent_ref,
                image_name=str(start_name),
                refine=refine,
            )
        else:
            cond = _encode_text(gb, merged, fps)
            cond_pos_ref, cond_neg_ref = gb.ref(cond, 0), gb.ref(cond, 1)

        if mode == "lipsync" and start_image_ref is not None and not merged.get("video_name"):
            merged["ic_guide_image_ref"] = start_image_ref
            merged["ic_enabled"] = True

        if mode == "motion_transfer":
            merged["ic_guide_raw"] = True
            merged["ic_enabled"] = True

        cond_pos_ref, cond_neg_ref, video_latent_ref, frame_guides = _apply_end_middle_guides(
            gb,
            merged,
            video_vae_ref=video_vae_ref,
            video_latent_ref=video_latent_ref,
            cond_pos_ref=cond_pos_ref,
            cond_neg_ref=cond_neg_ref,
        )
        crop_guides = bool(frame_guides)
        if mode == "flf2v":
            # Official FLF is single-stage unless quality re-apply is requested.
            if not refine:
                frame_guides = []
    else:
        raise ValueError(f"Unknown LTX mode: {mode}")

    _append_sampling_tail(
        gb,
        plan=merged,
        video_latent_ref=video_latent_ref,
        cond_pos_ref=cond_pos_ref,
        cond_neg_ref=cond_neg_ref,
        video_vae_ref=video_vae_ref,
        start_image_ref=start_image_ref if mode != "t2v" else None,
        frame_guides=frame_guides,
        crop_guides=crop_guides,
    )
    return gb.build(), dict(gb.node_meta)


def build_t2v(plan: dict[str, Any]) -> dict[str, Any]:
    graph, _ = _build_graph({**plan, "mode": "t2v"})
    return graph


def build_i2v(plan: dict[str, Any]) -> dict[str, Any]:
    graph, _ = _build_graph({**plan, "mode": "i2v"})
    return graph


def build(plan: dict[str, Any]) -> dict[str, Any]:
    graph, _ = _build_graph(plan)
    return graph


def build_with_meta(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    return _build_graph(plan)
