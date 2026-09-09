# Image ops — model fallbacks

Use the catalog on disk first. Change graphs only after a smoke op fails or quality is unusable.

## Too slow (50-step Qwen-Image)

1. Submit with `"fast": true` (Lightning 4-step LoRA, cfg 1.0). Bundle: `qwen-image-2512-lightning-lora` for t2i, or the Edit-2511 Lightning LoRA bundled with `qwen-image-edit-2511-fp8`.
2. If Lightning artifacts show, keep 50-step (edit: 40-step) and lower resolution (1024²).

## Photoreal character looks plastic

Apply `qwen-realism-lora` (`loras/qwen-realism-lora.safetensors`, already on `/data/model_backup/comfyui/loras`). Not in the default graph; add it as a LoRA on `generate_character` if needed.

## Inpaint / outpaint looks wrong

DiffSynth patch (`qwen_image_inpaint_diffsynth_controlnet`) is the default.
Fallback: native `InpaintModelConditioning` + `SetLatentNoiseMask` on the Qwen-Image 2512 UNet (same checkpoint, no patch). Keep the same mask upload path.

## Upscale looks soft

Default is RealESRGAN x2/x4 (`upscalers-esrgan`).
Fallback already on disk: `seedvr2_7b_nvfp4.safetensors` in `diffusion_models/`. Wire a SeedVR2 graph only if ESRGAN fails a visual check.

## Pose / ControlNet

Not used by the 10 operations. If a later pose lock is needed, install `comfyui_controlnet_aux` into `~/ComfyUI/custom_nodes` and use Fun-Union `Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors` (already on disk). Native `DWPreprocessor` is not in ComfyUI 0.34.

## VRAM thrash on turnaround

Turnaround switches Qwen-Image → Qwen-Image-Edit after the optional base render. If the 5090 pages, skip the base t2i and pass an existing `character` asset so every step stays on the edit checkpoint.
