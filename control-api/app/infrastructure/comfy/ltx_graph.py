from __future__ import annotations

from typing import Any

from .graphs import GraphBuilder

MODELS = {
    "distilled": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "clip": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "vae": "ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscaler": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}

# ComfyUI LTX-2.5-Quality distilled schedules.
STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"


def _align(value: int, step: int = 32) -> int:
    return max(step, (int(value) // step) * step)


def _sample(
    gb: GraphBuilder,
    *,
    model: str,
    positive: list[Any],
    negative: list[Any],
    latent: list[Any],
    sigmas: list[Any],
    seed: int,
    sampler_name: str,
    video_cfg: float,
    audio_cfg: float,
) -> str:
    noise = gb.add("RandomNoise", {"noise_seed": seed})
    sampler = gb.add("KSamplerSelect", {"sampler_name": sampler_name})
    guider = gb.add(
        "LTXVDualCFGGuider",
        {
            "model": gb.ref(model),
            "positive": positive,
            "negative": negative,
            "video_cfg": video_cfg,
            "audio_cfg": audio_cfg,
        },
    )
    return gb.add(
        "SamplerCustomAdvanced",
        {
            "noise": gb.ref(noise),
            "guider": gb.ref(guider),
            "sampler": gb.ref(sampler),
            "sigmas": sigmas,
            "latent_image": latent,
        },
    )


def _condition(gb: GraphBuilder, clip: str, prompt: str, negative: str, fps: float) -> str:
    pos = gb.add("CLIPTextEncode", {"text": prompt, "clip": gb.ref(clip)})
    neg = gb.add("CLIPTextEncode", {"text": negative, "clip": gb.ref(clip)})
    return gb.add(
        "LTXVConditioning",
        {"positive": gb.ref(pos), "negative": gb.ref(neg), "frame_rate": fps},
    )


def _concat_av(gb: GraphBuilder, video_latent: str | list[Any], audio_vae: str, length: int, fps: float) -> str:
    empty_audio = gb.add(
        "LTXVEmptyLatentAudio",
        {
            "frames_number": length,
            "frame_rate": fps,
            "batch_size": 1,
            "audio_vae": gb.ref(audio_vae),
        },
    )
    video_ref = video_latent if isinstance(video_latent, list) else gb.ref(video_latent)
    return gb.add(
        "LTXVConcatAVLatent",
        {"video_latent": video_ref, "audio_latent": gb.ref(empty_audio)},
    )


def _sigmas(gb: GraphBuilder, *, refine: bool, steps: int, latent: str) -> str:
    if refine:
        return gb.add("ManualSigmas", {"sigmas": STAGE1_SIGMAS})
    return gb.add(
        "LTXVScheduler",
        {
            "steps": steps,
            "max_shift": 2.05,
            "base_shift": 0.95,
            "stretch": True,
            "terminal": 0.1,
            "latent": gb.ref(latent),
        },
    )


def _sample_and_export(
    gb: GraphBuilder,
    *,
    plan: dict[str, Any],
    video_latent: str,
    video_vae: str,
    audio_vae: str,
    cond: str,
    unet: str,
    start_image: str | None = None,
) -> None:
    refine = bool(plan.get("refine"))
    length = int(plan.get("length") or 97)
    fps = float(plan.get("fps") or 24)
    steps = int(plan.get("steps") or 8)
    seed = int(plan.get("seed") or 0)
    video_cfg = float(plan.get("video_cfg") if plan.get("video_cfg") is not None else plan.get("cfg") or 1.0)
    sampler_name = str(plan.get("sampler_name") or "euler_ancestral")

    av = _concat_av(gb, video_latent, audio_vae, length, fps)
    sampled = _sample(
        gb,
        model=unet,
        positive=gb.ref(cond, 0),
        negative=gb.ref(cond, 1),
        latent=gb.ref(av),
        sigmas=gb.ref(_sigmas(gb, refine=refine, steps=steps, latent=av)),
        seed=seed,
        sampler_name=sampler_name,
        video_cfg=video_cfg,
        audio_cfg=1.0,
    )
    if refine:
        separated = gb.add("LTXVSeparateAVLatent", {"av_latent": gb.ref(sampled)})
        upscaler = gb.add("LatentUpscaleModelLoader", {"model_name": MODELS["latent_upscaler"]})
        video_for_refine = gb.add(
            "LTXVLatentUpsampler",
            {
                "samples": gb.ref(separated, 0),
                "upscale_model": gb.ref(upscaler),
                "vae": gb.ref(video_vae),
            },
        )
        if start_image is not None:
            video_for_refine = gb.add(
                "LTXVImgToVideoInplace",
                {
                    "vae": gb.ref(video_vae),
                    "image": gb.ref(start_image),
                    "latent": gb.ref(video_for_refine),
                    "strength": 1.0,
                    "bypass": False,
                },
            )
        av_refine = gb.add(
            "LTXVConcatAVLatent",
            {"video_latent": gb.ref(video_for_refine), "audio_latent": gb.ref(separated, 1)},
        )
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
    separated = gb.add("LTXVSeparateAVLatent", {"av_latent": gb.ref(sampled)})
    decode = gb.add(
        "VAEDecodeTiled",
        {
            "samples": gb.ref(separated, 0),
            "vae": gb.ref(video_vae),
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 64,
            "temporal_overlap": 16,
        },
    )
    audio = gb.add(
        "LTXVAudioVAEDecode",
        {"samples": gb.ref(separated, 1), "audio_vae": gb.ref(audio_vae)},
    )
    video = gb.add("CreateVideo", {"images": gb.ref(decode), "fps": fps, "audio": gb.ref(audio)})
    gb.add(
        "SaveVideo",
        {
            "video": gb.ref(video),
            "filename_prefix": str(plan.get("filename_prefix") or "ltx_video"),
            "format": "mp4",
            "codec": "h264",
        },
    )


def _common_models(gb: GraphBuilder) -> tuple[str, str, str, str]:
    video_vae = gb.add("VAELoader", {"vae_name": MODELS["vae"]})
    clip = gb.add("CLIPLoader", {"clip_name": MODELS["clip"], "type": "ltxv"})
    unet = gb.add("UNETLoader", {"unet_name": MODELS["distilled"], "weight_dtype": "default"})
    audio_vae = gb.add("VAELoader", {"vae_name": MODELS["audio_vae"]})
    return video_vae, clip, unet, audio_vae


def _stage_size(plan: dict[str, Any]) -> tuple[int, int, int, float, bool]:
    refine = bool(plan.get("refine"))
    width = _align(int(plan.get("width") or 704))
    height = _align(int(plan.get("height") or 1216))
    length = int(plan.get("length") or 97)
    fps = float(plan.get("fps") or 24)
    stage_w = _align(width // 2) if refine else width
    stage_h = _align(height // 2) if refine else height
    return stage_w, stage_h, length, fps, refine


def build_i2v(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    stage_w, stage_h, length, fps, refine = _stage_size(plan)
    image_name = plan.get("image_name") or "input.png"
    video_vae, clip, unet, audio_vae = _common_models(gb)
    empty = gb.add(
        "EmptyLTXVLatentVideo",
        {"width": stage_w, "height": stage_h, "length": length, "batch_size": 1},
    )
    load = gb.add("LoadImage", {"image": image_name})
    pre = gb.add("LTXVPreprocess", {"image": gb.ref(load), "img_compression": 18})
    video_latent = gb.add(
        "LTXVImgToVideoInplace",
        {
            "vae": gb.ref(video_vae),
            "image": gb.ref(pre),
            "latent": gb.ref(empty),
            "strength": 0.7 if refine else 1.0,
            "bypass": False,
        },
    )
    cond = _condition(gb, clip, str(plan.get("prompt") or ""), str(plan.get("negative_prompt") or ""), fps)
    _sample_and_export(
        gb,
        plan=plan,
        video_latent=video_latent,
        video_vae=video_vae,
        audio_vae=audio_vae,
        cond=cond,
        unet=unet,
        start_image=pre,
    )
    return gb.build()


def build_t2v(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    stage_w, stage_h, length, fps, _refine = _stage_size(plan)
    video_vae, clip, unet, audio_vae = _common_models(gb)
    video_latent = gb.add(
        "EmptyLTXVLatentVideo",
        {"width": stage_w, "height": stage_h, "length": length, "batch_size": 1},
    )
    cond = _condition(gb, clip, str(plan.get("prompt") or ""), str(plan.get("negative_prompt") or ""), fps)
    _sample_and_export(
        gb,
        plan=plan,
        video_latent=video_latent,
        video_vae=video_vae,
        audio_vae=audio_vae,
        cond=cond,
        unet=unet,
    )
    return gb.build()


def build(plan: dict[str, Any]) -> dict[str, Any]:
    mode = plan.get("mode", "i2v")
    if mode == "i2v":
        return build_i2v(plan)
    if mode == "t2v":
        return build_t2v(plan)
    raise ValueError(f"Unknown LTX mode: {mode}")
