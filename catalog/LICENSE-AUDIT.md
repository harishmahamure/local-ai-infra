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
| `ltx-2.5-distilled`, `ltx-2.5-studio`, `ltx-2.5-nvfp4`, `ltx-2.5-control` | [LTX-2 Community License](https://huggingface.co/Lightricks/LTX-2.5) | Commercial use permitted under LTX-2-Community terms |

## Blocked (not in catalog)

| License / family | Reason |
|------------------|--------|
| FLUX.1-dev / FLUX.2-dev NC | Non-commercial without paid BFL license |
| Qwen3.8-Max (2.4T) | Revenue / MaaS clauses |
| Tencent Hunyuan Community | Territorial limits, MAU thresholds, attribution |
| Stability Community (SD3.5) | Revenue cap |
| OpenRAIL / CreativeML NC variants | Restricted commercial |

## Current catalog

Image, video, and speech weights on `/data/model_backup` that are not listed in the table stay unused until those runtimes are planned.

| Bundle | Family | License | Use for |
|--------|--------|---------|---------|
| `gemma-4-e4b` | Gemma 4 E4B Q4_K_M + mmproj | Gemma Terms | text + vision chat (128K) |
| `qwen36-35b-a3b-rq` | Qwen3.6-35B-A3B RotorQuant Q4 + mmproj | Apache-2.0 | text + vision chat (262K native, 1M YaRN) |
| `qwen-image-2512-fp8` | Qwen-Image 2512 fp8 + VL encoder + VAE | Apache-2.0 | text-to-image character / location / prop / attire |
| `qwen-image-2512-lightning-lora` | 4-step Lightning LoRA | Apache-2.0 | fast drafts (`fast=true`) |
| `qwen-image-edit-2511-fp8` | Qwen-Image-Edit 2511 fp8 + Lightning 4-step LoRA | Apache-2.0 | turnaround, keyframe, shot reference, dress-on, face-lock character, `fast=true` |
| `qwen-controlnet-diffsynth` | DiffSynth inpaint patch + Lightning LoRA | Apache-2.0 | inpaint / outpaint |
| `upscalers-esrgan` | RealESRGAN x2 / x4 | MIT | `upscale_asset`, devotion wallpaper 2x |
| `ltx-2.5-distilled` | LTX 2.5 distilled | LTX-2-Community | T2V + live wallpaper I2V |
| `ltx-2.5-studio` | LTX 2.5 distilled + spatial refine | LTX-2-Community | live wallpaper quality refine |
| `ltx-2.5-nvfp4` | LTX 2.5 distilled nvfp4 transformer | LTX-2-Community | high-quality shot flows without LoRA |
| `ltx-2.5-control` | LTX 2.3 IC-LoRAs (union / motion-track / dubit) | LTX-2-Community | F06 motion transfer, F08 camera tracks |
