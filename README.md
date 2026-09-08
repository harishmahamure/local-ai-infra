# ai-project — llama.cpp control plane

Control plane for a LAN GPU box (RTX 5090): **llama.cpp** only. Two exclusive profiles share port **8080**:

- `gemma` — Gemma 4 E4B Q4_K_M + mmproj (text + vision, 128K)
- `llama-fast` — Qwen3.6-35B-A3B RotorQuant Q4 + mmproj (text + vision, 262K native / 1M YaRN)

Image, video, and speech are out of this service. Weights for those remain on `/data/model_backup` until those runtimes are re-planned.

## Quick start (Mac)

```bash
cp .env.example .env
chmod +x bin/ai scripts/*.sh scripts/remote/*.sh

./scripts/setup_ssh_key.sh   # enter password once
./bin/ai bootstrap           # sync, link existing llama.cpp trees, control API + Mac proxy
```

Add to PATH: `export PATH="$(pwd)/bin:$PATH"`

Bootstrap does **not** rebuild llama.cpp or download GGUFs. It reuses:

- `~/llama-cpp-turboquant` → `~/ai-inference/llama.cpp` (llama-fast / turbo2)
- `~/llama.cpp` (ggml-org master, for Gemma 4)
- `/data/model_backup/llamacpp` → `~/ai-inference/models/llamacpp`

## Dashboard

Open **http://192.168.50.100:8090/** (or `./bin/ai ui`) for GPU status, exclusive Gemma / Qwen load-unload, and GGUF downloads.

Mac `http://127.0.0.1:8090` only works if the Mac proxy owns that port. Another local agent on 8090 will hide this UI — use the GPU URL instead.

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
| GET | `/v1/status` | Runtime + GPU |
| GET | `/v1/models` | GGUF disk status + runtime state |
| POST | `/v1/models/{id}/load` | Start `llama-fast` or `gemma` |
| POST | `/v1/models/{id}/unload` | Stop all GPU profiles |
| GET | `/v1/text/models` | Chat-capable models |
| POST | `/v1/text/chat` | Sync JSON, or SSE when `stream=true` |
| GET | `/v1/downloads` | GGUF download state |
| POST | `/v1/downloads` | Queue a GGUF download |

## CLI

```bash
ai ui
ai status
ai start gemma
ai start llama-fast
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

Vision uses OpenAI-style `image_url` parts (`data:` or `https://` only — no filesystem paths).

## Status

| State | Meaning |
| ----- | ------- |
| `LOADED` | Profile running and llama-server responds |
| `STARTING` | Service up but API not ready |
| `STOPPED` | GPU idle — nothing in VRAM |
| `CONFLICT` | More than one service — `ai stop` then start one profile |

Download status (disk): `OK` / `PARTIAL` / `MISSING` per catalog bundle.

## Layout on the GPU box

```
~/ai-inference/control     synced repo
~/ai-inference/venv        control-api venv
~/ai-inference/models/llamacpp  → /data/model_backup/llamacpp
~/ai-inference/llama.cpp        → ~/llama-cpp-turboquant
~/llama.cpp                     ggml-org (Gemma 4)
```

## Tests

```bash
.venv/bin/pytest control-api/tests -q
```
