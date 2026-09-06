"""Dynamic ComfyUI API-format graph builder for Qwen-Image (+ Chroma painterly)."""

from __future__ import annotations

from typing import Any


M = {
    "qwen_unet": "qwen_image_2512_fp8_e4m3fn.safetensors",
    "qwen_edit_unet": "qwen_image_edit_2511_fp8mixed.safetensors",
    "qwen_clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "qwen_vae": "qwen_image_vae.safetensors",
    "qwen_lora_lightning": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
    "chroma_unet": "Chroma1-HD.safetensors",
    "chroma_t5": "t5xxl_fp8_e4m3fn.safetensors",
    "flux_vae": "flux_ae.safetensors",
    "upscale_x4": "RealESRGAN_x4plus.pth",
    "upscale_x2": "RealESRGAN_x2plus.pth",
    "qwen_controlnet_union": "Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors",
    "qwen_inpaint_patch": "qwen_image_inpaint_diffsynth_controlnet.safetensors",
    "qwen_layered_unet": "qwen_image_layered_fp8mixed.safetensors",
    "qwen_layered_vae": "qwen_image_layered_vae.safetensors",
}

LAYERED_REQUIRED_NODES = (
    "EmptyQwenImageLayeredLatentImage",
    "LatentCut",
    "LatentCutToBatch",
)


class GraphBuilder:
    def __init__(self) -> None:
        self._nid = 0
        self.nodes: dict[str, dict[str, Any]] = {}

    def add(self, class_type: str, inputs: dict[str, Any]) -> str:
        self._nid += 1
        nid = str(self._nid)
        self.nodes[nid] = {"class_type": class_type, "inputs": inputs}
        return nid

    def ref(self, nid: str, slot: int = 0) -> list[Any]:
        return [nid, slot]

    def build(self) -> dict[str, Any]:
        return self.nodes


def _apply_loras(gb: GraphBuilder, model_id: str, loras: list[dict[str, Any]]) -> str:
    current = model_id
    for lora in loras or []:
        name = lora.get("name")
        strength = float(lora.get("strength", 1.0))
        if not name:
            continue
        current = gb.add(
            "LoraLoaderModelOnly",
            {"model": gb.ref(current), "lora_name": name, "strength_model": strength},
        )
    return current


def _upscale_model_name(plan: dict[str, Any] | None = None, scale: int | None = None) -> str:
    if plan is not None:
        scale = int(plan.get("upscale_scale", 4))
    elif scale is None:
        scale = 4
    return M["upscale_x2"] if int(scale) <= 2 else M["upscale_x4"]


def _add_upscale(gb: GraphBuilder, image_id: str, *, scale: int | None = None, plan: dict[str, Any] | None = None) -> str:
    loader = gb.add("UpscaleModelLoader", {"model_name": _upscale_model_name(plan=plan, scale=scale)})
    upscale = gb.add(
        "ImageUpscaleWithModel",
        {"upscale_model": gb.ref(loader), "image": gb.ref(image_id)},
    )
    return upscale


def _scale_image_to_megapixels(gb: GraphBuilder, image_id: str, megapixels: float) -> str:
    return gb.add(
        "ImageScaleToTotalPixels",
        {
            "image": gb.ref(image_id),
            "upscale_method": "lanczos",
            "megapixels": megapixels,
            "resolution_steps": 1,
        },
    )


def build_txt2img(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    loras = plan.get("loras") or []
    steps = int(plan.get("steps", 30))
    cfg = float(plan.get("cfg", 4.0))
    seed = int(plan.get("seed", 42))

    unet = gb.add("UNETLoader", {"unet_name": M["qwen_unet"], "weight_dtype": "default"})
    clip = gb.add(
        "CLIPLoader",
        {"clip_name": M["qwen_clip"], "type": "qwen_image", "weight_dtype": "default"},
    )
    vae = gb.add("VAELoader", {"vae_name": M["qwen_vae"]})

    image_name = plan.get("image_name")
    if image_name:
        load = gb.add("LoadImage", {"image": image_name})
        scale = _scale_image_to_megapixels(
            gb,
            load,
            (plan["width"] * plan["height"]) / 1_000_000.0,
        )
        latent = gb.add("VAEEncode", {"pixels": gb.ref(scale), "vae": gb.ref(vae)})
        denoise = float(plan.get("denoise", 0.75))
    else:
        latent = gb.add(
            "EmptySD3LatentImage",
            {"width": plan["width"], "height": plan["height"], "batch_size": 1},
        )
        denoise = 1.0

    model = _apply_loras(gb, unet, loras)

    pos = gb.add("CLIPTextEncode", {"clip": gb.ref(clip), "text": plan["prompt"]})
    neg = gb.add(
        "CLIPTextEncode",
        {"clip": gb.ref(clip), "text": plan.get("negative_prompt", "")},
    )

    neg_out = neg
    if steps <= 4 and loras:
        neg_out = gb.add("ConditioningZeroOut", {"conditioning": gb.ref(pos)})

    sampler = gb.add(
        "KSampler",
        {
            "model": gb.ref(model),
            "positive": gb.ref(pos),
            "negative": gb.ref(neg_out),
            "latent_image": gb.ref(latent),
            "seed": seed,
            "control_after_generate": "fixed",
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": denoise,
        },
    )
    decode = gb.add("VAEDecode", {"samples": gb.ref(sampler), "vae": gb.ref(vae)})
    image = decode
    if plan.get("upscale"):
        image = _add_upscale(gb, decode, plan=plan)
    prefix = plan.get("filename_prefix") or "ComfyUI"
    gb.add("SaveImage", {"images": gb.ref(image), "filename_prefix": prefix})
    return gb.build()


def build_painterly(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    steps = int(plan.get("steps", 30))
    cfg = float(plan.get("cfg", 4.0))
    seed = int(plan.get("seed", 42))

    unet = gb.add("UNETLoader", {"unet_name": M["chroma_unet"], "weight_dtype": "default"})
    clip = gb.add(
        "CLIPLoader",
        {"clip_name": M["chroma_t5"], "type": "chroma", "weight_dtype": "default"},
    )
    vae = gb.add("VAELoader", {"vae_name": M["flux_vae"]})

    image_name = plan.get("image_name")
    if image_name:
        load = gb.add("LoadImage", {"image": image_name})
        scale = _scale_image_to_megapixels(
            gb,
            load,
            (plan["width"] * plan["height"]) / 1_000_000.0,
        )
        latent = gb.add("VAEEncode", {"pixels": gb.ref(scale), "vae": gb.ref(vae)})
        denoise = float(plan.get("denoise", 0.75))
    else:
        latent = gb.add(
            "EmptySD3LatentImage",
            {"width": plan["width"], "height": plan["height"], "batch_size": 1},
        )
        denoise = 1.0

    pos = gb.add("CLIPTextEncode", {"clip": gb.ref(clip), "text": plan["prompt"]})
    neg = gb.add(
        "CLIPTextEncode",
        {"clip": gb.ref(clip), "text": plan.get("negative_prompt", "")},
    )

    sampler = gb.add(
        "KSampler",
        {
            "model": gb.ref(unet),
            "positive": gb.ref(pos),
            "negative": gb.ref(neg),
            "latent_image": gb.ref(latent),
            "seed": seed,
            "control_after_generate": "fixed",
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": denoise,
        },
    )
    decode = gb.add("VAEDecode", {"samples": gb.ref(sampler), "vae": gb.ref(vae)})
    image = decode
    if plan.get("upscale"):
        image = _add_upscale(gb, decode, plan=plan)
    gb.add("SaveImage", {"images": gb.ref(image), "filename_prefix": "ComfyUI"})
    return gb.build()


def _edit_image_names(plan: dict[str, Any]) -> list[str]:
    names = [str(n) for n in (plan.get("image_names") or []) if n]
    if not names and plan.get("image_name"):
        names = [str(plan["image_name"])]
    if not names:
        names = ["input.png"]
    if len(names) > 3:
        raise ValueError("edit mode accepts at most 3 reference images")
    return names


def build_edit(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    steps = int(plan.get("steps", 20))
    cfg = float(plan.get("cfg", 4.0))
    seed = int(plan.get("seed", 42))
    image_names = _edit_image_names(plan)
    loads = [gb.add("LoadImage", {"image": name}) for name in image_names]

    unet = gb.add("UNETLoader", {"unet_name": M["qwen_edit_unet"], "weight_dtype": "default"})
    clip = gb.add(
        "CLIPLoader",
        {"clip_name": M["qwen_clip"], "type": "qwen_image", "weight_dtype": "default"},
    )
    vae = gb.add("VAELoader", {"vae_name": M["qwen_vae"]})

    if len(loads) == 1:
        pos = gb.add(
            "TextEncodeQwenImageEdit",
            {
                "clip": gb.ref(clip),
                "vae": gb.ref(vae),
                "image": gb.ref(loads[0]),
                "prompt": plan["prompt"],
            },
        )
    else:
        encode_inputs: dict[str, Any] = {
            "clip": gb.ref(clip),
            "vae": gb.ref(vae),
            "image": gb.ref(loads[0]),
            "prompt": plan["prompt"],
        }
        for idx, load_id in enumerate(loads[1:], start=2):
            encode_inputs[f"image{idx}"] = gb.ref(load_id)
        pos = gb.add("TextEncodeQwenImageEditPlus", encode_inputs)
    neg = gb.add(
        "CLIPTextEncode",
        {"clip": gb.ref(clip), "text": plan.get("negative_prompt", "blurry, low quality, watermark")},
    )
    vae_enc = gb.add("VAEEncode", {"pixels": gb.ref(loads[0]), "vae": gb.ref(vae)})

    sampler = gb.add(
        "KSampler",
        {
            "model": gb.ref(unet),
            "positive": gb.ref(pos),
            "negative": gb.ref(neg),
            "latent_image": gb.ref(vae_enc),
            "seed": seed,
            "control_after_generate": "fixed",
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    )
    decode = gb.add("VAEDecode", {"samples": gb.ref(sampler), "vae": gb.ref(vae)})
    gb.add("SaveImage", {"images": gb.ref(decode), "filename_prefix": "ComfyUI"})
    return gb.build()


def build_control(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    loras = plan.get("loras") or []
    control_type = plan.get("control_type") or "pose"
    steps = int(plan.get("steps", 30))
    cfg = float(plan.get("cfg", 4.0))
    seed = int(plan.get("seed", 42))
    strength = float(plan.get("control_strength", 0.85))
    image_name = plan.get("image_name") or "control_reference.png"

    load = gb.add("LoadImage", {"image": image_name})
    scale = _scale_image_to_megapixels(gb, load, 1.6)

    if control_type == "canny":
        preproc = gb.add("Canny", {"image": gb.ref(scale), "low_threshold": 0.33, "high_threshold": 0.35})
    elif control_type == "pose":
        preproc = gb.add(
            "DWPreprocessor",
            {
                "image": gb.ref(scale),
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": 512,
                "bbox_detector": "yolox_l.onnx",
                "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
                "scale_stick_for_xinsr_cn": "disable",
            },
        )
    elif control_type in ("depth", "depth_midas"):
        if control_type == "depth_midas":
            preproc = gb.add(
                "MiDaS Depth Approximation",
                {
                    "image": gb.ref(scale),
                    "use_cpu": "false",
                    "midas_type": "DPT_Large",
                    "invert_depth": "false",
                },
            )
        else:
            preproc = gb.add(
                "DepthAnythingV2Preprocessor",
                {"image": gb.ref(scale), "ckpt_name": "depth_anything_v2_vitl.pth", "resolution": 512},
            )
    else:
        raise ValueError(f"Unknown control_type: {control_type}")

    unet = gb.add("UNETLoader", {"unet_name": M["qwen_unet"], "weight_dtype": "default"})
    clip = gb.add(
        "CLIPLoader",
        {"clip_name": M["qwen_clip"], "type": "qwen_image", "weight_dtype": "default"},
    )
    vae = gb.add("VAELoader", {"vae_name": M["qwen_vae"]})
    controlnet = gb.add("ControlNetLoader", {"control_net_name": M["qwen_controlnet_union"]})
    latent = gb.add(
        "EmptySD3LatentImage",
        {"width": plan["width"], "height": plan["height"], "batch_size": 1},
    )

    model = _apply_loras(gb, unet, loras)
    ms = gb.add("ModelSamplingAuraFlow", {"model": gb.ref(model), "shift": 3.1})

    pos = gb.add("CLIPTextEncode", {"clip": gb.ref(clip), "text": plan["prompt"]})
    neg = gb.add(
        "CLIPTextEncode",
        {"clip": gb.ref(clip), "text": plan.get("negative_prompt", "")},
    )

    cn_apply = gb.add(
        "ControlNetApplyAdvanced",
        {
            "positive": gb.ref(pos),
            "negative": gb.ref(neg),
            "control_net": gb.ref(controlnet),
            "image": gb.ref(preproc),
            "vae": gb.ref(vae),
            "strength": strength,
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
    )

    sampler = gb.add(
        "KSampler",
        {
            "model": gb.ref(ms),
            "positive": gb.ref(cn_apply, 0),
            "negative": gb.ref(cn_apply, 1),
            "latent_image": gb.ref(latent),
            "seed": seed,
            "control_after_generate": "fixed",
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    )
    decode = gb.add("VAEDecode", {"samples": gb.ref(sampler), "vae": gb.ref(vae)})
    gb.add("SaveImage", {"images": gb.ref(decode), "filename_prefix": "ComfyUI"})
    return gb.build()


def build_bg_replace(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    steps = int(plan.get("steps", 4))
    cfg = float(plan.get("cfg", 1.0))
    seed = int(plan.get("seed", 42))
    image_name = plan.get("image_name") or "subject_with_mask.png"

    load = gb.add("LoadImage", {"image": image_name})
    scale = _scale_image_to_megapixels(gb, load, 1.68)

    unet = gb.add("UNETLoader", {"unet_name": M["qwen_unet"], "weight_dtype": "default"})
    clip = gb.add(
        "CLIPLoader",
        {"clip_name": M["qwen_clip"], "type": "qwen_image", "weight_dtype": "default"},
    )
    vae = gb.add("VAELoader", {"vae_name": M["qwen_vae"]})
    patch = gb.add("ModelPatchLoader", {"name": M["qwen_inpaint_patch"]})

    lora = gb.add(
        "LoraLoaderModelOnly",
        {
            "model": gb.ref(unet),
            "lora_name": M["qwen_lora_lightning"],
            "strength_model": 1.0,
        },
    )
    ms = gb.add("ModelSamplingAuraFlow", {"model": gb.ref(lora), "shift": 3.1})

    pos = gb.add("CLIPTextEncode", {"clip": gb.ref(clip), "text": plan["prompt"]})
    neg = gb.add(
        "CLIPTextEncode",
        {"clip": gb.ref(clip), "text": plan.get("negative_prompt", "")},
    )

    cn_patch = gb.add(
        "QwenImageDiffsynthControlnet",
        {
            "model": gb.ref(ms),
            "model_patch": gb.ref(patch),
            "vae": gb.ref(vae),
            "image": gb.ref(scale),
            "mask": gb.ref(load, 1),
            "strength": float(plan.get("control_strength", 1.0)),
        },
    )
    vae_enc = gb.add("VAEEncode", {"pixels": gb.ref(scale), "vae": gb.ref(vae)})

    sampler = gb.add(
        "KSampler",
        {
            "model": gb.ref(cn_patch),
            "positive": gb.ref(pos),
            "negative": gb.ref(neg),
            "latent_image": gb.ref(vae_enc),
            "seed": seed,
            "control_after_generate": "fixed",
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    )
    decode = gb.add("VAEDecode", {"samples": gb.ref(sampler), "vae": gb.ref(vae)})
    gb.add("SaveImage", {"images": gb.ref(decode), "filename_prefix": "ComfyUI"})
    return gb.build()


def build_layered(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    steps = int(plan.get("steps", 50))
    cfg = float(plan.get("cfg", 4.0))
    seed = int(plan.get("seed", 42))
    layers = max(1, min(8, int(plan.get("layers", 3))))
    width = int(plan.get("width", 640))
    height = int(plan.get("height", 640))
    image_name = plan.get("image_name") or "input.png"
    prompt = str(plan.get("prompt") or "Decompose this image into clean RGBA layers.")

    load = gb.add("LoadImage", {"image": image_name})
    scaled = gb.add(
        "ImageScale",
        {
            "image": gb.ref(load),
            "upscale_method": "lanczos",
            "width": width,
            "height": height,
            "crop": "center",
        },
    )
    unet = gb.add("UNETLoader", {"unet_name": M["qwen_layered_unet"], "weight_dtype": "default"})
    clip = gb.add(
        "CLIPLoader",
        {"clip_name": M["qwen_clip"], "type": "qwen_image", "weight_dtype": "default"},
    )
    vae = gb.add("VAELoader", {"vae_name": M["qwen_layered_vae"]})
    pos = gb.add(
        "TextEncodeQwenImageEdit",
        {
            "clip": gb.ref(clip),
            "vae": gb.ref(vae),
            "image": gb.ref(scaled),
            "prompt": prompt,
        },
    )
    neg = gb.add("CLIPTextEncode", {"clip": gb.ref(clip), "text": plan.get("negative_prompt", "")})
    latent = gb.add(
        "EmptyQwenImageLayeredLatentImage",
        {"width": width, "height": height, "layers": layers, "batch_size": 1},
    )
    sampler = gb.add(
        "KSampler",
        {
            "model": gb.ref(unet),
            "positive": gb.ref(pos),
            "negative": gb.ref(neg),
            "latent_image": gb.ref(latent),
            "seed": seed,
            "control_after_generate": "fixed",
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    )
    cut = gb.add("LatentCut", {"samples": gb.ref(sampler), "dim": "t", "index": 1})
    batch = gb.add("LatentCutToBatch", {"samples": gb.ref(cut), "dim": "t"})
    decode = gb.add("VAEDecode", {"samples": gb.ref(batch), "vae": gb.ref(vae)})
    prefix = plan.get("filename_prefix") or "ComfyUI"
    gb.add("SaveImage", {"images": gb.ref(decode), "filename_prefix": prefix})
    return gb.build()


def build_upscale(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    image_name = plan.get("image_name") or "input.png"
    load = gb.add("LoadImage", {"image": image_name})
    image = _add_upscale(gb, load, plan=plan)
    gb.add("SaveImage", {"images": gb.ref(image), "filename_prefix": "ComfyUI"})
    return gb.build()


def build(plan: dict[str, Any]) -> dict[str, Any]:
    mode = plan.get("mode", "txt2img")
    if mode == "txt2img":
        return build_txt2img(plan)
    if mode == "painterly":
        return build_painterly(plan)
    if mode == "edit":
        if not plan.get("image_name") and not plan.get("image_names"):
            raise ValueError("image is required for edit mode")
        return build_edit(plan)
    if mode == "layered":
        if not plan.get("image_name"):
            raise ValueError("image is required for layered mode")
        return build_layered(plan)
    if mode == "control":
        if not plan.get("image_name"):
            raise ValueError("image is required for control mode")
        return build_control(plan)
    if mode == "bg_replace":
        if not plan.get("image_name"):
            raise ValueError("image is required for bg_replace mode")
        return build_bg_replace(plan)
    if mode == "upscale":
        if not plan.get("image_name"):
            raise ValueError("image is required for upscale mode")
        return build_upscale(plan)
    raise ValueError(f"Unknown mode: {mode}")
