# Models and LoRAs (Wan excluded)

Source of truth: [catalog/models.yaml](../catalog/models.yaml). Excluded: `wan2.2-i2v-fp8` and `wan2.2-t2v-fp8`.

Download any bundle with `./bin/ai download <id>`.

## Image models (ComfyUI)

| Bundle | Role | VRAM | License | Local files |
|--------|------|------|---------|-------------|
| `qwen-image-2512-fp8` | Base T2I (Qwen-Image 2512) | 20 GB | Apache-2.0 | `diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors`, `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`, `vae/qwen_image_vae.safetensors` |
| `qwen-image-edit-2511-fp8` | Image edit (depends on Qwen 2512) | 21 GB | Apache-2.0 | `diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors` |
| `chroma1-hd` | Painterly T2I (Chroma1-HD) | 22 GB | Apache-2.0 | `diffusion_models/Chroma1-HD.safetensors`, `text_encoders/t5xxl_fp8_e4m3fn.safetensors`, `vae/flux_ae.safetensors` |
| `qwen-controlnet-2512-fun-union` | Pose / depth / canny | 4 GB | Apache-2.0 | `controlnet/Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors` |
| `qwen-controlnet-diffsynth` | Background replace / inpaint | 3 GB | Apache-2.0 | `loras/qwen_image_union_diffsynth_lora.safetensors`, `model_patches/qwen_image_inpaint_diffsynth_controlnet.safetensors` |
| `upscalers-esrgan` | x2 / x4 upscale | 0 | MIT | `upscale_models/RealESRGAN_x4plus.pth`, `upscale_models/RealESRGAN_x2plus.pth` |

## Qwen LoRAs (ComfyUI `loras/`)

Wired in [control-api/app/presets.py](../control-api/app/presets.py) `LORA` map. All depend on `qwen-image-2512-fp8`.

| Bundle | File | Use |
|--------|------|-----|
| `qwen-image-2512-lightning-lora` | `Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors` | 4-step speed (CFG 1.0) |
| `qwen-image-2512-turbo-lora` | `Wuli-Qwen-Image-2512-Turbo-LoRA-2steps-V1.0-bf16.safetensors` | 2-step turbo |
| `qwen-lora-advertisement` | `qwen-advertisement-lora.safetensors` | Corporate ad posters |
| `qwen-lora-poster` | `qwen-poster-lora.safetensors` | Typography-first posters |
| `qwen-lora-flat-cartoon` | `qwen-flat-cartoon-lora.safetensors` | Flat icons / diagrams |
| `qwen-lora-eligen-poster` | `qwen-eligen-poster-lora.safetensors` | Multi-block text posters |
| `qwen-lora-realism` | `qwen-realism-lora.safetensors` | Photoreal subjects in layouts |

Also shipped with DiffSynth (control, not a style LoRA): `qwen_image_union_diffsynth_lora.safetensors`.

## LTX-2.5 video models (ComfyUI-ltx)

Gated HuggingFace repo `Lightricks/LTX-2.5`. License: LTX-2-Community (orgs under $10M revenue).

| Bundle | Role | VRAM | Local files |
|--------|------|------|-------------|
| `ltx-2.5-distilled` | Fast / standard T2V+I2V | 28 GB | Distilled transformer, Gemma4 TE, video VAE, audio VAE; optional spatial upscaler |
| `ltx-2.5-studio` | Quality refine (superset of distilled) | 32 GB | Above + **required** spatial upscaler; optional duration head, temporal upscaler, distilled LoRA |
| `ltx-2.5-prompt-enhancer` | ComfyUI `prompt_enhance` | 2 GB | `text_encoders/gemma4_e2b_it_int8_convrot.safetensors` (Gemma Terms) |

Do **not** apply `ltx-2.5-22b-distilled-lora-450-bf16` on the distilled transformer.

## LTX LoRAs

Defined in [control-api/app/ltx_loras.py](../control-api/app/ltx_loras.py). Official LTX-2.5 card: most LTX-2.3 LoRAs run on 2.5. LTX-2 **19B** camera and Detailer adapters do **not** — camera is prompt-only (`cameraMotion`).

### IC-LoRAs (LTX-2.3, compatible with 2.5)

Union needs a reference video. LipDub is explicit on lip-sync ops (start image + audio). Motion Track is explicit on `video.motion_transfer`.

| Bundle | ID | File |
|--------|----|------|
| `ltx-iclora-union` | `union` | `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` (depth / canny / pose) |
| `ltx-iclora-lipdub` | `lipdub` | `ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors` |
| `ltx-iclora-motion-track` | `motion_track` | `ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors` |

Studio optional: `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (do not use on distilled).

## LLM models (llama.cpp)

| Bundle | Role | VRAM | License | Files |
|--------|------|------|---------|-------|
| `gemma-4-e4b` | Fast text + vision / planner | 7 GB | Gemma Terms | `gemma-4-E4B-it-Q4_K_M.gguf`, `gemma-4-E4B-mmproj-BF16.gguf` |
| `qwen36-35b-a3b-rq` | Large text + vision (RotorQuant) | 22 GB | Apache-2.0 | `Qwen3.6-35B-A3B-Q4_K_M.gguf`, `Qwen3.6-35B-A3B-mmproj-F16.gguf` |

## Counts (Wan excluded)

- **21** catalog bundles: 9 image/video/utility (6 image/control/upscale + 3 LTX cores) + 7 Qwen LoRAs + 3 LTX-2.3 IC-LoRA bundles + 2 llama.cpp LLMs
- **3** LTX-2.3 IC-LoRA files (union, lipdub, motion-track)
- Omitted: `wan2.2-i2v-fp8`, `wan2.2-t2v-fp8`; LTX-2 19B camera/detailer adapters

## GPU status (2026-09-05)

Checked against local [catalog/models.yaml](../catalog/models.yaml), then `GET /api/v1/models` and SSH probes on `gpu-box` (`~/ComfyUI/models/`). Retired: `qwen38-27b-fast`, `qwen38-27b-slow`. Added: `qwen36-35b-a3b-rq`.

The GPU control catalog (`~/ai-inference/control/catalog/models.yaml`) can lag this repo. Sync before IC-LoRA downloads will work:

```bash
./bin/ai sync
./bin/ai download ltx-2.5-prompt-enhancer ltx-iclora-union ltx-iclora-lipdub ltx-iclora-motion-track
```

`catalog/installed.json` on the GPU still lists retired leftovers (`ernie-image-turbo`, `flux1-schnell-fp8`, `hidream-i1-fast`, `z-image-turbo`, `qwen-image-lightning-lora`). Do not add them here.

### Complete on disk

| Bundle | Size |
|--------|------|
| `qwen-image-2512-fp8` | 28.0 GB |
| `qwen-image-edit-2511-fp8` | 19.1 GB |
| `chroma1-hd` | 21.4 GB |
| `qwen-controlnet-2512-fun-union` | 3.3 GB |
| `qwen-controlnet-diffsynth` | 3.0 GB |
| `upscalers-esrgan` | 127.9 MB |
| `qwen-image-2512-lightning-lora` | 810.2 MB |
| `qwen-image-2512-turbo-lora` | 2.2 GB |
| `qwen-lora-advertisement` | 1.6 GB |
| `qwen-lora-poster` | 45.1 MB |
| `qwen-lora-flat-cartoon` | 225.2 MB |
| `qwen-lora-eligen-poster` | 450.2 MB |
| `qwen-lora-realism` | 90.1 MB |
| `ltx-2.5-distilled` | 37.0 GB (4/4 required) |
| `ltx-2.5-studio` | 37.0 GB required files; spatial upscaler 996 MB present |
| `gemma-4-e4b` | 5.9 GB |

### Missing

| Bundle | Missing files |
|--------|----------------|
| `qwen36-35b-a3b-rq` | `Qwen3.6-35B-A3B-Q4_K_M.gguf`, `Qwen3.6-35B-A3B-mmproj-F16.gguf` |
| `ltx-2.5-prompt-enhancer` | `text_encoders/gemma4_e2b_it_int8_convrot.safetensors` |
| `ltx-iclora-union` | `loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` |
| `ltx-iclora-lipdub` | `loras/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors` |
| `ltx-iclora-motion-track` | `loras/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors` |

Studio optionals also missing (not required): `ltx-2.5-duration-head-bf16.safetensors`, `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`, `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (do not apply the distilled LoRA on the distilled transformer).
