from __future__ import annotations

from typing import Any


MODELS = {
    "qwen_unet": "qwen_image_2512_fp8_e4m3fn.safetensors",
    "qwen_edit_unet": "qwen_image_edit_2511_fp8mixed.safetensors",
    "qwen_clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "qwen_vae": "qwen_image_vae.safetensors",
    "qwen_lora_lightning": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
    "qwen_edit_lora_lightning": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "qwen_inpaint_patch": "qwen_image_inpaint_diffsynth_controlnet.safetensors",
    "upscale_x4": "RealESRGAN_x4plus.pth",
    "upscale_x2": "RealESRGAN_x2plus.pth",
}

DEFAULT_SHIFT = 3.1
INPAINT_MEGAPIXELS = 1.68


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


def _qwen_model(
    gb: GraphBuilder,
    unet: str,
    loras: list[dict[str, Any]] | None = None,
    *,
    shift: float = DEFAULT_SHIFT,
) -> str:
    model = _apply_loras(gb, unet, loras or [])
    return gb.add("ModelSamplingAuraFlow", {"model": gb.ref(model), "shift": float(shift)})


def _qwen_clip_vae(gb: GraphBuilder) -> tuple[str, str]:
    clip = gb.add(
        "CLIPLoader",
        {"clip_name": MODELS["qwen_clip"], "type": "qwen_image"},
    )
    vae = gb.add("VAELoader", {"vae_name": MODELS["qwen_vae"]})
    return clip, vae


def _scale_image(gb: GraphBuilder, image: str, megapixels: float) -> str:
    return gb.add(
        "ImageScaleToTotalPixels",
        {
            "image": gb.ref(image),
            "upscale_method": "lanczos",
            "megapixels": megapixels,
            "resolution_steps": 1,
        },
    )


def _scale_mask(gb: GraphBuilder, mask_ref: str, megapixels: float, *, mask_slot: int = 0) -> str:
    mask_img = gb.add("MaskToImage", {"mask": gb.ref(mask_ref, mask_slot)})
    scaled = _scale_image(gb, mask_img, megapixels)
    return gb.add("ImageToMask", {"image": gb.ref(scaled), "channel": "red"})


def _sampler(
    gb: GraphBuilder,
    *,
    model: str,
    positive: str,
    negative: str,
    latent: str,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    sampler_name: str = "euler",
    scheduler: str = "simple",
) -> str:
    return gb.add(
        "KSampler",
        {
            "model": gb.ref(model),
            "positive": gb.ref(positive),
            "negative": gb.ref(negative),
            "latent_image": gb.ref(latent),
            "seed": seed,
            "control_after_generate": "fixed",
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
        },
    )


def _save(gb: GraphBuilder, image: str, prefix: str) -> None:
    gb.add("SaveImage", {"images": gb.ref(image), "filename_prefix": prefix})


def _plan_sampler(plan: dict[str, Any]) -> tuple[str, str]:
    return str(plan.get("sampler_name") or "euler"), str(plan.get("scheduler") or "simple")


def build_txt2img(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    loras = list(plan.get("loras") or [])
    steps = int(plan.get("steps", 50))
    cfg = float(plan.get("cfg", 4.0))
    seed = int(plan.get("seed", 42))
    prefix = str(plan.get("filename_prefix") or "ComfyUI")
    sampler_name, scheduler = _plan_sampler(plan)
    shift = float(plan.get("shift") if plan.get("shift") is not None else DEFAULT_SHIFT)

    unet = gb.add("UNETLoader", {"unet_name": MODELS["qwen_unet"], "weight_dtype": "default"})
    clip, vae = _qwen_clip_vae(gb)
    latent = gb.add(
        "EmptySD3LatentImage",
        {"width": int(plan["width"]), "height": int(plan["height"]), "batch_size": 1},
    )
    model = _qwen_model(gb, unet, loras, shift=shift)
    pos = gb.add("CLIPTextEncode", {"clip": gb.ref(clip), "text": plan["prompt"]})
    if cfg <= 1.0:
        neg_out = gb.add("ConditioningZeroOut", {"conditioning": gb.ref(pos)})
    else:
        neg_out = gb.add("CLIPTextEncode", {"clip": gb.ref(clip), "text": plan.get("negative_prompt", "")})
    decode = gb.add(
        "VAEDecode",
        {
            "samples": gb.ref(
                _sampler(
                    gb,
                    model=model,
                    positive=pos,
                    negative=neg_out,
                    latent=latent,
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    denoise=1.0,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                )
            ),
            "vae": gb.ref(vae),
        },
    )
    _save(gb, decode, prefix)
    return gb.build()


def build_edit(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    steps = int(plan.get("steps", 40))
    cfg = float(plan.get("cfg", 3.0))
    seed = int(plan.get("seed", 42))
    names = list(plan.get("image_names") or ([plan["image_name"]] if plan.get("image_name") else []))
    if not names:
        raise ValueError("image is required for edit mode")
    prefix = str(plan.get("filename_prefix") or "ComfyUI")
    sampler_name, scheduler = _plan_sampler(plan)
    shift = float(plan.get("shift") if plan.get("shift") is not None else DEFAULT_SHIFT)
    loras = list(plan.get("loras") or [])

    unet = gb.add("UNETLoader", {"unet_name": MODELS["qwen_edit_unet"], "weight_dtype": "default"})
    clip, vae = _qwen_clip_vae(gb)
    loads = [gb.add("LoadImage", {"image": name}) for name in names[:3]]
    encode_inputs: dict[str, Any] = {
        "clip": gb.ref(clip),
        "vae": gb.ref(vae),
        "image1": gb.ref(loads[0]),
        "prompt": plan["prompt"],
    }
    for idx, load_id in enumerate(loads[1:], start=2):
        encode_inputs[f"image{idx}"] = gb.ref(load_id)
    pos = gb.add("TextEncodeQwenImageEditPlus", encode_inputs)
    if cfg <= 1.0:
        neg = gb.add("ConditioningZeroOut", {"conditioning": gb.ref(pos)})
    else:
        neg_inputs = {**encode_inputs, "prompt": ""}
        neg = gb.add("TextEncodeQwenImageEditPlus", neg_inputs)
    if len(loads) >= 2:
        pos = gb.add(
            "FluxKontextMultiReferenceLatentMethod",
            {"conditioning": gb.ref(pos), "reference_latents_method": "index_timestep_zero"},
        )
        if cfg > 1.0:
            neg = gb.add(
                "FluxKontextMultiReferenceLatentMethod",
                {"conditioning": gb.ref(neg), "reference_latents_method": "index_timestep_zero"},
            )
    model = _qwen_model(gb, unet, loras, shift=shift)
    model = gb.add("CFGNorm", {"model": gb.ref(model), "strength": 1.0})
    if plan.get("width") and plan.get("height"):
        latent = gb.add(
            "EmptySD3LatentImage",
            {"width": int(plan["width"]), "height": int(plan["height"]), "batch_size": 1},
        )
    else:
        scaled = gb.add("FluxKontextImageScale", {"image": gb.ref(loads[0])})
        latent = gb.add("VAEEncode", {"pixels": gb.ref(scaled), "vae": gb.ref(vae)})
    decode = gb.add(
        "VAEDecode",
        {
            "samples": gb.ref(
                _sampler(
                    gb,
                    model=model,
                    positive=pos,
                    negative=neg,
                    latent=latent,
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    denoise=1.0,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                )
            ),
            "vae": gb.ref(vae),
        },
    )
    _save(gb, decode, prefix)
    return gb.build()


def _inpaint_core(gb: GraphBuilder, plan: dict[str, Any], image_ref: str, mask_ref: str, *, mask_slot: int = 0) -> str:
    steps = int(plan.get("steps", 4))
    cfg = float(plan.get("cfg", 1.0))
    seed = int(plan.get("seed", 42))
    sampler_name, scheduler = _plan_sampler(plan)
    shift = float(plan.get("shift") if plan.get("shift") is not None else DEFAULT_SHIFT)
    strength = float(plan.get("lora_strength") if plan.get("lora_strength") is not None else 1.0)
    unet = gb.add("UNETLoader", {"unet_name": MODELS["qwen_unet"], "weight_dtype": "default"})
    clip, vae = _qwen_clip_vae(gb)
    patch = gb.add("ModelPatchLoader", {"name": MODELS["qwen_inpaint_patch"]})
    loras: list[dict[str, Any]] = []
    if cfg <= 1.0:
        loras.append({"name": MODELS["qwen_lora_lightning"], "strength": strength})
    model = _qwen_model(gb, unet, loras, shift=shift)
    pos = gb.add("CLIPTextEncode", {"clip": gb.ref(clip), "text": plan["prompt"]})
    if cfg <= 1.0:
        neg = gb.add("ConditioningZeroOut", {"conditioning": gb.ref(pos)})
    else:
        neg = gb.add("CLIPTextEncode", {"clip": gb.ref(clip), "text": plan.get("negative_prompt", "")})
    scaled = _scale_image(gb, image_ref, INPAINT_MEGAPIXELS)
    mask = _scale_mask(gb, mask_ref, INPAINT_MEGAPIXELS, mask_slot=mask_slot)
    patched = gb.add(
        "QwenImageDiffsynthControlnet",
        {
            "model": gb.ref(model),
            "model_patch": gb.ref(patch),
            "vae": gb.ref(vae),
            "image": gb.ref(scaled),
            "mask": gb.ref(mask),
            "strength": float(plan.get("control_strength", 1.0)),
        },
    )
    vae_enc = gb.add("VAEEncode", {"pixels": gb.ref(scaled), "vae": gb.ref(vae)})
    return gb.add(
        "VAEDecode",
        {
            "samples": gb.ref(
                _sampler(
                    gb,
                    model=patched,
                    positive=pos,
                    negative=neg,
                    latent=vae_enc,
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    denoise=1.0,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                )
            ),
            "vae": gb.ref(vae),
        },
    )


def build_inpaint(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    image_name = plan.get("image_name") or "input.png"
    mask_name = plan.get("mask_name")
    load = gb.add("LoadImage", {"image": image_name})
    if mask_name:
        mask_load = gb.add("LoadImage", {"image": mask_name})
        mask = gb.add("ImageToMask", {"image": gb.ref(mask_load), "channel": "red"})
        decode = _inpaint_core(gb, plan, load, mask)
    else:
        decode = _inpaint_core(gb, plan, load, load, mask_slot=1)
    _save(gb, decode, str(plan.get("filename_prefix") or "ComfyUI"))
    return gb.build()


def build_outpaint(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    image_name = plan.get("image_name") or "input.png"
    pad = plan.get("pad") or {}
    load = gb.add("LoadImage", {"image": image_name})
    padded = gb.add(
        "ImagePadForOutpaint",
        {
            "image": gb.ref(load),
            "left": int(pad.get("left") or 0),
            "top": int(pad.get("top") or 0),
            "right": int(pad.get("right") or 0),
            "bottom": int(pad.get("bottom") or 0),
            "feathering": int(pad.get("feathering") or 40),
        },
    )
    decode = _inpaint_core(gb, plan, padded, padded, mask_slot=1)
    _save(gb, decode, str(plan.get("filename_prefix") or "ComfyUI"))
    return gb.build()


def build_upscale(plan: dict[str, Any]) -> dict[str, Any]:
    gb = GraphBuilder()
    image_name = plan.get("image_name") or "input.png"
    scale = int(plan.get("upscale_scale") or 4)
    model_name = MODELS["upscale_x2"] if scale <= 2 else MODELS["upscale_x4"]
    load = gb.add("LoadImage", {"image": image_name})
    loader = gb.add("UpscaleModelLoader", {"model_name": model_name})
    upscale = gb.add("ImageUpscaleWithModel", {"upscale_model": gb.ref(loader), "image": gb.ref(load)})
    _save(gb, upscale, str(plan.get("filename_prefix") or "ComfyUI"))
    return gb.build()


def build(plan: dict[str, Any]) -> dict[str, Any]:
    mode = plan.get("mode", "txt2img")
    if mode == "txt2img":
        return build_txt2img(plan)
    if mode == "edit":
        return build_edit(plan)
    if mode == "inpaint":
        return build_inpaint(plan)
    if mode == "outpaint":
        return build_outpaint(plan)
    if mode == "upscale":
        return build_upscale(plan)
    raise ValueError(f"Unknown mode: {mode}")
