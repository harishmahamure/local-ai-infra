# ai-project — GPU control plane

Control plane for a LAN GPU box (RTX 5090). **One profile at a time:**

- `gemma` — Gemma 4 E4B Q4_K_M + mmproj (text + vision, 128K) on **:8080**
- `llama-fast` — Qwen3.6-35B-A3B RotorQuant Q4 + mmproj (text + vision, 262K) on **:8080**
- `comfyui` — Qwen-Image 2512 + Edit (image jobs) on **:8188**

Submitting an image job auto-switches to ComfyUI (chat returns 409 `GPU_BUSY` until the queue drains and you reload a chat profile).

## Quick start (Mac)

```bash
cp .env.example .env
chmod +x bin/ai scripts/*.sh scripts/remote/*.sh

./scripts/setup_ssh_key.sh   # enter password once
./bin/ai bootstrap           # sync, units, control API + Mac proxy
```

Add to PATH: `export PATH="$(pwd)/bin:$PATH"`

Bootstrap reuses:

- `~/llama-cpp-turboquant` → `~/ai-inference/llama.cpp` (llama-fast)
- `~/llama.cpp` (ggml-org master, Gemma 4)
- `/data/model_backup/llamacpp` → `~/ai-inference/models/llamacpp`
- `~/ComfyUI` + `/data/model_backup/comfyui` (Qwen-Image weights)

## Dashboard

Open **http://192.168.50.100:8090/** (or `./bin/ai ui`) for the React control UI: Overview, Runtime, Images, Jobs, Models, and Downloads. Text chat stays API-only (`POST /v1/text/chat`); the Runtime page shows a copyable curl.

Local hot reload (proxies `/v1` to the Mac control proxy):

```bash
cd web && npm install && npm run dev   # http://127.0.0.1:5173
```

Production build writes into `control-api/static/`. Run before `ai sync` when the UI changes:

```bash
cd web && npm run build
./bin/ai sync
```

## Control API

| Where | URL |
| ----- | --- |
| GPU dashboard | http://192.168.50.100:8090/ |
| GPU API | http://192.168.50.100:8090 |
| Swagger | http://192.168.50.100:8090/docs |
| Spec | `control-api/openapi.yaml` |

| Method | Path | Action |
| ------ | ---- | ------ |
| GET | `/health` | Liveness |
| GET | `/ready` | 200 if a profile is `LOADED` |
| GET | `/v1/status` | Runtime + GPU + image queue |
| GET | `/v1/models` | Catalog disk status + runtime state |
| POST | `/v1/models/{id}/load` | Start `llama-fast`, `gemma`, or `comfyui` |
| POST | `/v1/models/{id}/unload` | Stop all GPU profiles |
| GET | `/v1/text/models` | Chat-capable models |
| POST | `/v1/text/chat` | Sync JSON, or SSE when `stream=true` |
| GET | `/v1/image/operations` | The 10 Qwen image ops |
| POST | `/v1/image/jobs` | Queue one image job (202). Header `Idempotency-Key` optional |
| GET | `/v1/jobs` | Job list (`status`, cursor pagination) |
| DELETE | `/v1/jobs` | Delete jobs (`status` optional; skips running). `delete_assets=true` removes files |
| GET | `/v1/jobs/{id}` | Status, phase, progress, assets |
| GET | `/v1/jobs/{id}/events` | SSE progress |
| POST | `/v1/jobs/{id}/cancel` | Cancel queued/running job |
| DELETE | `/v1/jobs/{id}` | Delete job (`delete_assets=true` also removes generated files) |
| DELETE | `/v1/jobs/{id}/assets` | Delete all images for one job |
| POST | `/v1/assets` | Multipart upload |
| DELETE | `/v1/assets` | Delete all images on disk (skips files on a running job) |
| GET | `/v1/assets/{id}` | Asset metadata |
| GET | `/v1/assets/{id}/content` | Image bytes |
| DELETE | `/v1/assets/{id}` | Delete image metadata and file on the GPU box |
| GET | `/v1/downloads` | Download state |
| POST | `/v1/downloads` | Queue a catalog download |

Image operations: `generate_character`, `generate_character_turnaround`, `generate_attire`, `generate_location`, `generate_prop`, `generate_keyframe`, `generate_shot_reference`, `inpaint_asset`, `outpaint_asset`, `upscale_asset`.

## CLI

```bash
ai ui
ai status
ai start gemma
ai start llama-fast
ai start comfyui
ai stop
ai download gemma-4-e4b
```

Chat example (after `ai start gemma`):

```bash
curl -X POST http://127.0.0.1:8090/v1/text/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-e4b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Image example:

```bash
curl -X POST http://192.168.50.100:8090/v1/image/jobs \
  -H 'Content-Type: application/json' \
  -d '{"operation":"generate_prop","prompt":"a brass lantern","fast":true}'
```

Vision chat uses OpenAI-style `image_url` parts (`data:` or `https://` only).

## Status

| State | Meaning |
| ----- | ------- |
| `LOADED` | Profile running and its HTTP API responds |
| `STARTING` | Service up but API not ready |
| `STOPPED` | GPU idle — nothing in VRAM |
| `CONFLICT` | More than one service — `ai stop` then start one profile |

Job status: `QUEUED` / `RUNNING` / `SUCCEEDED` / `FAILED` / `CANCELLED`.

Download status (disk): `OK` / `PARTIAL` / `MISSING` per catalog bundle.

## Layout on the GPU box

```
~/ai-inference/control     synced repo
~/ai-inference/venv        control-api venv
~/ai-inference/state/jobs.db
~/ai-inference/assets/
~/ai-inference/models/llamacpp  → /data/model_backup/llamacpp
~/ai-inference/llama.cpp        → ~/llama-cpp-turboquant
~/llama.cpp                     ggml-org (Gemma 4)
~/ComfyUI                       ComfyUI 0.34 + .venv
~/model_backup/comfyui          → /data/model_backup/comfyui
```

## Tests

```bash
.venv/bin/pytest control-api/tests -q
./scripts/smoke_image_ops.sh    # live GPU box
```

Fallbacks if a model underperforms: `docs/image-ops-fallbacks.md`.
