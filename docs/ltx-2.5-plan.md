# LTX-2.5 video + audio pipeline

Second ComfyUI install (`~/ComfyUI-ltx`, port **8189**) running native LTX-2.5 nodes from ComfyUI **≥ v0.32.0** (pinned to **v0.34.3**). Shares the primary model tree at `~/ComfyUI/models/` — no duplicate weight downloads.

The primary **`comfy`** profile also runs `~/ComfyUI-ltx` on port **8188** (Qwen-Image). The legacy `~/ComfyUI` tree holds models only and is not used as a ComfyUI runtime.

## Profiles

| Profile | Port | Use |
|---------|------|-----|
| `comfy` | 8188 | Qwen-Image (ComfyUI-ltx runtime) |
| `comfy-ltx` | 8189 | LTX-2.5 video (T2V / I2V / A2V / FLF / lip sync / motion transfer) |

Starting either Comfy profile stops the other (same GPU mutex as llama/gemma).

## ComfyUI manual workflows (optional)

This repo includes a ready-made **high quality** workflow:

| File | Description |
|------|-------------|
| [comfyui/workflows/LTX-2.5-Quality-T2V.json](../comfyui/workflows/LTX-2.5-Quality-T2V.json) | Two-stage distilled T2V with latent upscale refine (1216×704 default) |
| [comfyui/workflows/LTX-2.5-Quality-I2V.json](../comfyui/workflows/LTX-2.5-Quality-I2V.json) | Official two-stage I2V (start-image re-inject on refine) |
| [comfyui/workflows/LTX-2.5-Quality-A2V.json](../comfyui/workflows/LTX-2.5-Quality-A2V.json) | Audio-to-video; remuxes the uploaded track |
| [comfyui/workflows/LTX-2.5-Quality-FLF2V.json](../comfyui/workflows/LTX-2.5-Quality-FLF2V.json) | First + last frame guides (official single-stage) |
| [comfyui/workflows/LTX-2.5-Quality-I2V-LipSync.json](../comfyui/workflows/LTX-2.5-Quality-I2V-LipSync.json) | Image + audio lip sync (LipDub IC-LoRA) |
| [comfyui/workflows/LTX-2.5-Quality-FLF-LipSync.json](../comfyui/workflows/LTX-2.5-Quality-FLF-LipSync.json) | First/last + audio lip sync |
| [comfyui/workflows/LTX-2.5-Quality-FML-LipSync.json](../comfyui/workflows/LTX-2.5-Quality-FML-LipSync.json) | First/middle/last + audio lip sync |
| [comfyui/workflows/LTX-2.5-Quality-MotionTransfer.json](../comfyui/workflows/LTX-2.5-Quality-MotionTransfer.json) | Motion Track IC-LoRA from a reference video |

After `./bin/ai sync`, it is copied to `~/ComfyUI-ltx/blueprints/` on the GPU box.

### Load in ComfyUI

1. Start the profile: `./bin/ai start comfy-ltx`
2. Open **http://192.168.50.100:8189**
3. **Workflow → Open** (or drag the JSON) → select `LTX-2.5-Quality-T2V.json` from blueprints
4. Edit the prompt, click **Queue Prompt**

The Control UI **LTX video** page with **Speed: Quality** runs the same pipeline via API — no manual workflow needed.

### Other templates

ComfyUI also ships built-in templates: **Workflow → Browse Templates** → search **LTX 2.5**.

Templates live in the ComfyUI venv:
`~/ComfyUI-ltx/venv/lib/python3.12/site-packages/comfyui_workflow_templates_json/templates/`

Older blueprints under `~/ComfyUI-ltx/blueprints/` tagged LTX 2.0/2.3 are obsolete — use **2.5** workflows only.

## Setup on GPU box

1. Sync control repo: `./bin/ai sync`
2. Bootstrap (installs ComfyUI-ltx venv + systemd units):
   ```bash
   ./bin/ai bootstrap
   ```
3. Accept [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) license on HuggingFace (same account as `HF_TOKEN`).
4. Download weights:
   ```bash
   # Minimum (standard/fast generation)
   ./bin/ai download ltx-2.5-distilled

   # Full LTX Studio stack (quality refine + ComfyUI workflow) — recommended
   ./bin/ai download ltx-2.5-studio

   # Optional: ComfyUI built-in prompt enhancer (enable prompt_enhance in workflow)
   ./bin/ai download ltx-2.5-prompt-enhancer

   # Official LTX-2.3 IC-LoRAs (documented compatible with 2.5)
   ./bin/ai download ltx-iclora-union
   ./bin/ai download ltx-iclora-lipdub
   ./bin/ai download ltx-iclora-motion-track
   ```
5. Optional — if you only downloaded `ltx-2.5-distilled`, add the spatial upscaler separately (included in `ltx-2.5-studio`):
   ```bash
   ./bin/ai download ltx-2.5-studio
   ```
6. Symlinks for split-weight loaders are created automatically by `link_ltx_checkpoints.sh` during install.

## API

Prompts are sent **verbatim** — there is no LLM planning phase for video jobs.

```bash
# Text-to-video (LTX Studio quality — same as ComfyUI workflow)
curl -X POST http://127.0.0.1:8090/api/v1/ltx-video \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "t2v",
    "speed": "quality",
    "orientation": "landscape",
    "prompt": "A rainy street at night, neon reflections",
    "audioPrompt": "soft rain on pavement, distant traffic hum",
    "duration": 4
  }'

# With optional prompt enhancer + camera language from the prompt
curl -X POST http://127.0.0.1:8090/api/v1/ltx-video \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "t2v",
    "preset": "ltx_studio",
    "prompt": "cat on a windowsill, slow dolly forward",
    "audioPrompt": "silent, no sound",
    "duration": 4,
    "promptEnhance": true,
    "cameraMotion": "auto"
  }'

# Image-to-video with explicit parameters
curl -X POST http://127.0.0.1:8090/api/v1/ltx-video \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "i2v",
    "prompt": "Slow camera push-in, fabric ripples in the breeze",
    "audioPrompt": "gentle wind, no music",
    "image": "<base64 or data URL>",
    "width": 704,
    "height": 1216,
    "steps": 8,
    "videoCfg": 1.0,
    "audioCfg": 1.0,
    "strength": 0.85
  }'

# Poll
curl http://127.0.0.1:8090/api/v1/ltx-video/{jobId}
```

**Audio prompt is strongly recommended.** Without an explicit sound description, LTX tends to invent background music. Use `"silent, no sound, no music"` for silence.

### Request parameters

| Field | Description |
|-------|-------------|
| `mode` | `t2v` (default), `i2v`, `a2v`, `flf2v`, `lipsync`, `motion_transfer` |
| `prompt` | Scene / motion description (verbatim) |
| `audioPrompt` | Sound design (verbatim) |
| `preset` | `ltx_reel`, `ltx_landscape`, `ltx_square`, `ltx_quality`, `ltx_studio`, `ltx_fast` |
| `duration` | Seconds (1–30); converted to `length = duration × fps + 1`, then snapped to 8k+1 |
| `width`, `height` | Pixels (256–2048); aligned to 64 when `refine=true` |
| `length` | Frame count (17–897); must be 8k+1. 30s at 24fps is 721 frames. |
| `steps`, `videoCfg`, `audioCfg` | Distilled model uses CFG 1.0 / 1.0 — higher values over-saturate and blur |
| `samplerName`, `maxShift`, `baseShift`, `terminal`, `stretch` | LTX scheduler tuning |
| `strength` | I2V image conditioning strength (0–1) |
| `refine` | Enable 2-stage latent upscale refine pass |
| `refineSteps`, `refineDenoise` | Second-stage sampling |
| `tiledDecode` | Use tiled VAE decode (lower VRAM) |
| `promptEnhance` | LTX Studio prompt expansion (same as ComfyUI `prompt_enhance`) |
| `cameraMotion` | `auto` / `none` / `dolly_in` / `dolly_out` / `dolly_left` / `dolly_right` / `jib_up` / `jib_down` / `static` (prompt-only; no 19B camera LoRA) |
| `referenceVideo` | Base64 / data URL; required for IC-LoRA. Size/length still follow the request (reference is resized). |
| `icLora` | `auto` / `none` / `union` |
| `controlType` | `auto` / `depth` / `canny` / `pose` |
| `icLoraStrength` | Union / LipDub / Motion Track strength (default 1.0) |
| `negativePrompt`, `seed` | Standard overrides |
| `plan` | Explicit plan object (bypasses preset merge) |

## Job flow

1. **switching** — start `comfy-ltx` profile
2. **load_models** — load LTX weights on :8189
3. **encode** — text/audio/image conditioning
4. **sample** — primary denoise pass
5. **refine** — optional latent upscaler + second pass
6. **decode** — VAE decode (tiled or full)
7. **export** — MP4 with audio under `jobs/{jobId}/videos/`

## Presets

| ID | Aspect | Default size | Notes |
|----|--------|--------------|-------|
| `ltx_reel` | 9:16 portrait | 704×1216 | Social vertical |
| `ltx_landscape` | 16:9 | 1216×704 | Widescreen |
| `ltx_square` | 1:1 | 896×896 | Square |
| `ltx_quality` | 9:16 portrait | 704×1216 | Higher steps + `refine=true` (LTX Studio) |
| `ltx_studio` | 16:9 landscape | 1216×704 | Same as ComfyUI LTX-2.5-Quality-T2V.json |

**Speed: Standard** and **Quality** (or preset **`ltx_studio`**) run the same 2-stage refine pipeline as the ComfyUI blueprint: half-res sample, latent 2× upscale, 3-step refine, tiled decode. Dual CFG is **1.0 / 1.0** (the distilled transformer is CFG-distilled; raising it degrades quality and makes the negative prompt inert). Set **`promptEnhance: true`** (or enable in UI) for LTX Studio prompt expansion (requires `ltx-2.5-prompt-enhancer`).

Refine requires the **ltx-2.5-studio** bundle (spatial upscaler). Fast is single-pass preview only.

## Model files

| Bundle | Contents |
|--------|----------|
| `ltx-2.5-distilled` | UNet, Gemma4 TE, video VAE, audio VAE |
| `ltx-2.5-studio` | Above + **spatial upscaler** (required for refine) + optional duration head, temporal upscaler, distilled LoRA |
| `ltx-2.5-prompt-enhancer` | `gemma4_e2b_it_int8_convrot.safetensors` for ComfyUI `prompt_enhance` |
| `ltx-iclora-union` | LTX-2.3 Union Control IC-LoRA (depth / canny / pose from a reference video) |
| `ltx-iclora-lipdub` | LipDub IC-LoRA (image/keyframe + uploaded-audio lip sync) |
| `ltx-iclora-motion-track` | Motion Track IC-LoRA (motion transfer from a reference video) |

Do **not** apply `ltx-2.5-22b-distilled-lora-450-bf16` on the distilled transformer.

| Path | File |
|------|------|
| `diffusion_models/` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` |
| `text_encoders/` | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` |
| `text_encoders/` | `gemma4_e2b_it_int8_convrot.safetensors` (prompt enhancer bundle) |
| `vae/` | `ltx-2.5-video-vae-bf16.safetensors` |
| `vae/` | `ltx-2.5-audio-vae-bf16.safetensors` |
| `latent_upscale_models/` | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` (studio bundle) |
| `model_patches/` | `ltx-2.5-duration-head-bf16.safetensors` (studio, optional) |
| `latent_upscale_models/` | `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` (studio, optional) |
| `loras/` | `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (studio, optional) |

## Out of scope

- Native multishot / ID-LoRA
- Motion-track editor UI, Ingredients sheets, water / shave / HDR specialty IC-LoRAs
- 4K / HDR / temporal upscaler
- Full-precision (non-distilled) transformer
- Reading the enhanced prompt back from ComfyUI to pick LoRAs (decision uses the original prompt)

Implemented on the engine + Control UI + `comfyui/workflows/` blueprints: T2V, A2V, I2V, FLF2V, image/FLF/FML lip sync (LipDub IC-LoRA), and motion transfer (Motion Track IC-LoRA). Uploaded audio is encoded and remuxed; text `audioPrompt` is only used when no audio asset is present.
