# License allowlist

Downloader **refuses** any model whose `license` field is not listed here.

## Allowed (commercial use)

| License | Notes |
|---------|-------|
| Apache-2.0 | Unrestricted commercial use with attribution |
| MIT | Unrestricted commercial use |
| BSD-3-Clause | Unrestricted commercial use |

## Conditional (commercial with obligations)

| Model / component | License basis | Obligation |
|-------------------|---------------|------------|
| `gemma-4-e4b` | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | Commercial use permitted; prohibited-use policy applies; not Apache-2.0 |
| LTX-2.5 (`ltx-2.5-distilled`) | LTX-2 Community License | Free for orgs under **$10M annual revenue**; gated HuggingFace repo `Lightricks/LTX-2.5`; requires HF token with gated-repo scope. ComfyUI ≥ 0.32.0 on port 8189 (`comfy-ltx` profile). |

## Blocked (not in catalog)

| License / family | Reason |
|------------------|--------|
| FLUX.1-dev / FLUX.2-dev NC | Non-commercial without paid BFL license |
| FLUX.2 klein 9B NC | Non-commercial |
| Qwen3.8-Max (2.4T) | Revenue / MaaS clauses |
| Tencent Hunyuan Community | Territorial limits, MAU thresholds, attribution |
| Stability Community (SD3.5) | Revenue cap |
| OpenRAIL / CreativeML NC variants | Restricted commercial |
| 4x-UltraSharp (Kim2091/UltraSharp) | CC-BY-NC-SA-4.0 — non-commercial; use `RealESRGAN_x4plus` instead |

## v1 catalog models

All entries in `models.yaml` are Apache-2.0 or MIT and `commercial: true`, except `gemma-4-e4b` (Gemma Terms).

### Image model families

| Bundle | Family | License |
|--------|--------|---------|
| `qwen-image-2512-fp8` | Qwen-Image 2512 | Apache-2.0 |
| `chroma1-hd` | Chroma1-HD 8.9B (painterly mode) | Apache-2.0 |
| `upscalers-esrgan` | RealESRGAN x2/x4 | MIT (via uwg/upscaler) |

### Qwen style LoRAs

| Bundle | Source | License | Use for |
|--------|--------|---------|---------|
| `qwen-lora-advertisement` | RayyanAhmed9477/qwen-image-2512-lora-advertisement | Apache-2.0 | Corporate ad posters |
| `qwen-lora-poster` | Robbierayrob/qwen_image_lora_poster | MIT | Typography-first poster layouts |
| `qwen-lora-flat-cartoon` | furaidosu/flat-cartoon-qwen-image-2512 | Apache-2.0 | Flat icons, maps, diagram panels |
| `qwen-lora-eligen-poster` | DiffSynth-Studio/Qwen-Image-EliGen-Poster | Apache-2.0 | Multi-block text region posters |
| `qwen-lora-realism` | flymy-ai/qwen-image-realism-lora | Apache-2.0 | Photorealistic subjects in layouts |

### Scene control

| Bundle | Source | License | Use for |
|--------|--------|---------|---------|
| `qwen-controlnet-2512-fun-union` | alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union | Apache-2.0 | Pose, depth, canny control |
| `qwen-controlnet-diffsynth` | Comfy-Org/Qwen-Image-DiffSynth-ControlNets | Apache-2.0 | Background replace via inpaint |

### Video + audio (LTX-2.5)

| Bundle | Family | License | Use for |
|--------|--------|---------|---------|
| `ltx-2.5-distilled` | LTX-2.5 22B distilled int8 + Gemma4 TE + AV VAEs | LTX-2-Community | Standard/fast LTX generation |
| `ltx-2.5-studio` | Distilled + spatial upscaler + duration head (optional extras) | LTX-2-Community | Quality refine, ComfyUI LTX Studio workflow |
| `ltx-2.5-prompt-enhancer` | Gemma4 E2B int8 (ComfyUI TextGenerateLTX2Prompt) | Gemma Terms | Optional prompt_enhance in ComfyUI workflows |
| `ltx-camera-loras` | LTX-2 19B camera-control LoRAs (dolly / jib / static) | LTX-2-Community | Prompt or `cameraMotion` selection; do not stack |
| `ltx-iclora-union` | LTX-2.3 22B Union Control IC-LoRA | LTX-2-Community | Depth / canny / pose from a reference video |
| `ltx-iclora-detailer` | LTX-2 19B Detailer IC-LoRA | LTX-2-Community | Video-to-video detail pass (reference video) |
| `ltx-iclora-lipdub` | LTX-2.3 22B LipDub IC-LoRA | LTX-2-Community | Image/keyframe + uploaded-audio lip sync |
| `ltx-iclora-motion-track` | LTX-2.3 22B Motion Track IC-LoRA | LTX-2-Community | Motion transfer from a reference video |

### LLM profiles

| Bundle | Role | License |
|--------|------|---------|
| `gemma-4-e4b` | Small text + vision LLM / planner | Gemma Terms |
| `qwen36-35b-a3b-rq` | Large text + vision LLM (RotorQuant Q4_K_M) | Apache-2.0 |
