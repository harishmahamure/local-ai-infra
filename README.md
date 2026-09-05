# ai-project — Remote inference base

Control plane for a LAN GPU box (5090): **ComfyUI** (Qwen-Image dynamic graphs) + **llama.cpp** (Gemma 4 E4B text+vision planner + Qwen3.6-35B-A3B RotorQuant vision). Commercial-use models (Apache-2.0/MIT + Gemma Terms for planner).

## Quick start (Mac)

One-time setup from the terminal:

```bash
cp .env.example .env
chmod +x bin/ai scripts/*.sh scripts/remote/*.sh

./scripts/setup_ssh_key.sh   # enter password once
./bin/ai bootstrap           # sync, GPU build, control API + Mac proxy
```

Add to PATH: `export PATH="$(pwd)/bin:$PATH"`

After bootstrap, use the **Control UI** for everything else (downloads, status, switching models). You do not need to keep a terminal open.

## Control UI (primary)

Open in your browser:

| Where | URL |
| ----- | --- |
| This Mac | http://127.0.0.1:8090 |
| LAN (other devices) | http://\<this-mac-lan-ip\>:8090 |
| GPU box (direct) | http://192.168.50.100:8090 |

Or run `./bin/ai ui` to open it.

From the UI you can:

- **Dashboard** — overall progress bar, bytes on disk, active download file
- **Generate** — natural-language image generation (Gemma plans → Qwen executes)
- **Downloads** — every catalog file with status (Downloaded / Missing), current file, log
- **Runtime** — start/stop GPU profiles (llama-fast, gemma, comfy)
- **Models** — bundles with per-file sizes and status

API docs (same host): http://127.0.0.1:8090/docs

## Endpoints after a profile is loaded

| Service | URL | Profile |
| ------- | --- | ------- |
| LLM + vision | http://192.168.50.100:8080/v1 | llama-fast (Qwen3.6) or gemma |
| Image planner | http://192.168.50.100:8080/v1 | gemma (text + vision) |
| ComfyUI | http://192.168.50.100:8188 | comfy |
| Control UI | http://127.0.0.1:8090 | always (status, downloads, generate) |

llama-fast and gemma share port **8080**; switching reloads the process.

## Status: loaded vs downloaded

Use the Control UI, or the API:

| State | Meaning |
| ----- | ------- |
| `LOADED` | Profile running and API responds |
| `STARTING` | Service up but API not ready |
| `STOPPED` | GPU idle — nothing in VRAM |
| `CONFLICT` | More than one service — use **Stop all**, then start one profile |

**Download status** (disk): `OK` / `PARTIAL` / `MISSING` per catalog bundle. Shown in the UI catalog table and `GET /api/v1/models`.

## Control API (for other services)

Mac proxy: `http://<this-mac-lan-ip>:8090` · GPU direct: `http://192.168.50.100:8090`

| Method | Path | Action |
| ------ | ---- | ------ |
| GET | `/api/v1/status` | Runtime + GPU |
| GET | `/api/v1/capabilities` | Presets, modes, LoRAs (+ readiness) |
| POST | `/api/v1/generate` | Queue LLM-planned image generation (202) |
| GET | `/api/v1/generate/{job_id}` | Poll job status; completed jobs include inline `images[].data` (base64 data URLs) |
| GET | `/api/v1/downloads` | Background download job |
| POST | `/api/v1/downloads` | Start download (`{"ids":[]}` = all) |
| PUT | `/api/v1/profile` | Switch model `{"id":"llama-fast"}` |
| DELETE | `/api/v1/profile` | Stop all |

Example — start download from another service:

```bash
curl -X POST http://127.0.0.1:8090/api/v1/downloads \
  -H 'Content-Type: application/json' \
  -d '{"ids":[]}'
```

Example — switch to ComfyUI:

```bash
curl -X PUT http://127.0.0.1:8090/api/v1/profile \
  -H 'Content-Type: application/json' \
  -d '{"id":"comfy"}'
```

### Image generation API (LLM-planned)

Send a natural-language prompt. The control-api automatically:

1. Starts **gemma** profile → Gemma 4 E4B classifies request, picks a curated preset, enhances prompt
2. Starts **comfy** profile → builds dynamic Qwen-Image graph from the plan → queues ComfyUI

```bash
curl -X POST http://127.0.0.1:8090/api/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Lord Shiva on Kailash, golden aura, phone wallpaper"}'

curl http://127.0.0.1:8090/api/v1/generate/{job_id}
```

**Request body** (`POST /api/v1/generate`):

Two modes (if both are sent, `prompts[]` takes precedence):

| Mode | Body | Behavior |
| ---- | ---- | -------- |
| Multi-prompt | `{ "prompts": [ {...}, ... ] }` | Gemma plans each item (unless item has explicit `plan`). Gemma loads once, Comfy loads once. |
| Same-prompt batch | `{ "prompt": "...", "count": 5, "seed": 42 }` | Expands to 5 items with seeds 42..46 |

| Field | Required | Notes |
| ----- | -------- | ----- |
| `prompts` | no | Array of 1–10 prompt items (see below) |
| `prompt` | yes* | Natural language image request (shorthand mode) |
| `image` | no | Base64 or data URL reference (edit/control modes) |
| `plan` | no | Explicit plan object — bypasses Gemma planner |
| `seed` | no | Integer seed (`count` mode uses seed, seed+1, …) |
| `width`, `height` | no | Override preset dimensions |
| `count` | no | Same-prompt batch size 1–10 (default 1) |

**Prompt item** (each element of `prompts[]`):

| Field | Required | Notes |
| ----- | -------- | ----- |
| `prompt` | yes* | Per-item natural language request |
| `image` | no | Optional reference image for this item |
| `plan` | no | Bypass Gemma for this item only |
| `seed`, `width`, `height` | no | Per-item overrides |

\*Not required if `plan` is provided.

**Job phases**: `planning` (Gemma, all items) → `switching` → `running` (Comfy, sequential) → `completed` / `failed`

**Error policy**: If one item fails, remaining items continue. Job `status` is `completed` if at least one item succeeded; `failed` only if all items failed.

Images are written to disk as each item finishes:

```
~/ai-inference/logs/jobs/{job_id}/images/0000_00_ComfyUI_....png
```

Poll the job for metadata; fetch files via URL (survives Wi‑Fi drops):

```json
{
  "jobId": "...",
  "total": 3,
  "completedCount": 2,
  "failedCount": 1,
  "status": "completed",
  "phase": "done",
  "items": [
    {
      "index": 0,
      "prompt": "Lord Shiva wallpaper",
      "status": "completed",
      "plan": { "mode": "txt2img", "..." : "..." },
      "images": [
        {
          "index": 0,
          "filename": "0000_00_ComfyUI_00001_.png",
          "mime": "image/png",
          "url": "/api/v1/generate/{job_id}/images/0000_00_ComfyUI_00001_.png"
        }
      ],
      "error": null
    }
  ],
  "images": ["...flattened from all items..."],
  "storageDir": "/home/.../ai-inference/logs/jobs/{job_id}/images"
}
```

```bash
curl -O "http://127.0.0.1:8090/api/v1/generate/{job_id}/images/0000_00_ComfyUI_00001_.png"
```

Multi-prompt batch:

```bash
curl -X POST http://127.0.0.1:8090/api/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompts": [
      {"prompt": "Lord Shiva on Kailash, phone wallpaper"},
      {"prompt": "Corporate cloud infrastructure poster"},
      {"prompt": "Flat cartoon mascot for fintech app"}
    ]
  }'
```

Same-prompt variations:

```bash
curl -X POST http://127.0.0.1:8090/api/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Lord Shiva wallpaper variations","count":10,"seed":42}'
```

**Error codes**: `409 COMFY_NOT_LOADED`, `409 MODELS_MISSING`, `422 VALIDATION_ERROR`, `404 JOB_NOT_FOUND`.

List presets and readiness:

```bash
curl http://127.0.0.1:8090/api/v1/capabilities
```

Example — pose-controlled cinematic still with reference photo:

```bash
curl -X POST http://127.0.0.1:8090/api/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Cinematic movie still, freight train at golden hour, film grain",
    "image": "<base64 reference photo>"
  }'
```

Example — bypass planner with explicit plan:

```bash
curl -X POST http://127.0.0.1:8090/api/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Corporate infrastructure poster",
    "plan": {
      "mode": "txt2img",
      "prompt": "Professional corporate poster...",
      "loras": [{"name": "qwen-advertisement-lora.safetensors", "strength": 0.85}],
      "steps": 30,
      "cfg": 4.5,
      "width": 1024,
      "height": 1280,
      "bundles": ["qwen-image-2512-fp8", "qwen-lora-advertisement"]
    }
  }'
```

## Exclusive GPU (one model)

The 5090 runs **exactly one** profile. Starting a new one stops the others automatically.

| Allowed | Not allowed |
| ------- | ----------- |
| llama-fast **or** gemma **or** comfy **or** comfy-ltx **or** idle | Two profiles at once |
| One GGUF (+ vision projector) per llama/gemma profile | Fast MTP draft loaded by default |

## Profiles

| Profile | What runs |
| ------- | --------- |
| llama-fast | Qwen3.6-35B-A3B RotorQuant Q4 + mmproj, 262K ctx (1M YaRN env flip), iso3 KV |
| gemma | Gemma 4 E4B Q4 + mmproj, 128K ctx — text + vision planner |
| comfy | ComfyUI-ltx on :8188 — Qwen-Image |
| comfy-ltx | ComfyUI LTX-2.5 on :8189 — text/image-to-video with synchronized audio ([plan](docs/ltx-2.5-plan.md)) |

## LLM + vision (llama.cpp)

Both LLM profiles start **llama-server with `--mmproj`**. Prefer the engine text API (`POST /v1/text/chat`) so images go through assets, not filesystem paths.

```bash
./bin/ai start gemma
./bin/ai start llama-fast
```

```bash
curl http://127.0.0.1:8090/v1/text/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-e4b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }]
  }'
```

Without `--mmproj`, the server loads text-only and rejects images.

### KV cache (iso3) — scrya-com/rotorquant

Bootstrap clones **[scrya-com/rotorquant](https://github.com/scrya-com/rotorquant)** to `~/ai-inference/rotorquant`. That repo is Python/Triton (no `llama-server`). Its Quick Start builds CUDA `iso3` from [johndpope/llama-cpp-turboquant](https://github.com/johndpope/llama-cpp-turboquant) `feature/planarquant-kv-cache`, so bootstrap also clones that into `~/ai-inference/llama.cpp` (replaces ggml-org).

| Path | Origin |
| ---- | ------ |
| `~/ai-inference/rotorquant` | [scrya-com/rotorquant](https://github.com/scrya-com/rotorquant) |
| `~/ai-inference/llama.cpp` | johndpope fork (iso3 `llama-server`) |
| `qwen36-35b-a3b-rq` GGUF | RotorQuant **weights** (`MODEL_PATH`) |

Profiles request `CACHE_TYPE_K=iso3` / `CACHE_TYPE_V=iso3`. [`scripts/remote/llama-server.sh`](scripts/remote/llama-server.sh) falls back to `q8_0` if `--help` lacks `iso3` (never shrinks `-c`). Qwen3.6 / Gemma 4 may not load on this older C++ tree.

```bash
./bin/ai sync
./bin/ai bootstrap
```

```bash
ssh gpu-box 'git -C ~/ai-inference/rotorquant remote -v'
ssh gpu-box 'git -C ~/ai-inference/llama.cpp remote -v && git -C ~/ai-inference/llama.cpp branch --show-current'
ssh gpu-box '~/ai-inference/llama.cpp/build/bin/llama-server --help 2>&1 | grep -Ei "iso3|cache-type-k"'
```

## CLI (setup and bootstrap only)

The `ai` CLI is for one-time setup and opening the UI. Day-to-day control is via the UI or API above.

| Command | Use |
| ------- | --- |
| `ai bootstrap` | GPU setup: clone scrya-com/rotorquant + rebuild iso3 llama-server (johndpope fork) |
| `ai sync` | Push control files to GPU box |
| `ai ui` | Open Control UI in browser |
| `ai inventory` | Remote hardware inventory |

## Models (catalog)

Complete list of models and LoRAs: [docs/models-loras.md](docs/models-loras.md).

Machine catalog: [catalog/models.yaml](catalog/models.yaml). Licenses: [catalog/LICENSE-AUDIT.md](catalog/LICENSE-AUDIT.md).

Download any bundle: `./bin/ai download <id>`.

### Presets (curated recipes)

Presets live in [control-api/app/presets.py](control-api/app/presets.py). The Gemma planner picks one and supplies tuned steps/cfg/LoRA strengths. Examples:

| Preset ID | Mode | Use case |
| --------- | ---- | -------- |
| `deity_wallpaper` | txt2img | Deity phone wallpaper + Lightning LoRA |
| `painterly` | painterly | Chroma1-HD traditional art |
| `infographic_corp_ad` | txt2img | Corporate poster + Advertisement LoRA |
| `scene_pose` | control | Cinematic still + DWPose |
| `scene_bg_replace` | bg_replace | Background inpaint replace |
| `general` | txt2img | Fallback daily driver |

Download example:

```bash
./bin/ai download qwen-image-2512-fp8
```

Large downloads: add `--prune-cache` on GPU to drop HF cache blobs after copy:

```bash
ssh gpu-box 'source ~/ai-inference/venv/bin/activate && cd ~/ai-inference/control && python3 scripts/download_models.py --id qwen-image-2512-fp8 --prune-cache'
```

### Custom nodes

Scene pose/depth preprocessors require custom nodes. Declared in [comfyui/custom_nodes.yaml](comfyui/custom_nodes.yaml) and installed by `ai bootstrap`.

| Pack | Purpose |
| ---- | ------- |
| `comfyui_controlnet_aux` | DWPose, DepthAnythingV2 preprocessors |

`scene_canny` uses native ComfyUI Canny only — no custom pack required.

### LTX-2.5 video + audio

Active pipeline on profile `comfy-ltx` (ComfyUI **v0.34.3** on port **8189**). Prompts are sent **verbatim** — no LLM planning. See [docs/ltx-2.5-plan.md](docs/ltx-2.5-plan.md) for setup, parameters, optional 2-stage refine, and model download.

```bash
./bin/ai download ltx-2.5-distilled          # minimum
./bin/ai download ltx-2.5-studio             # quality refine + ComfyUI LTX Studio workflow
./bin/ai download ltx-2.5-prompt-enhancer    # optional ComfyUI prompt_enhance
```

```bash
curl -X POST http://127.0.0.1:8090/api/v1/ltx-video \
  -H 'Content-Type: application/json' \
  -d '{"mode":"t2v","preset":"ltx_quality","prompt":"…","audioPrompt":"…","refine":true}'
```

## SSH

Host alias `gpu-box` → `harishmahamure@192.168.50.100` with `~/.ssh/id_ed25519_gpu`.

Public key (if setup_ssh_key fails):

```bash
cat ~/.ssh/id_ed25519_gpu.pub
# paste into ~/.ssh/authorized_keys on the GPU box
```
