# Architecture Gap Analysis

Brownfield audit of this repository as a reusable **media inference engine**.
The future movie/product application is out of scope.

## What already exists

A LAN GPU control plane for one RTX 5090 with exclusive systemd profiles
(`comfy` :8188, `comfy-ltx` :8189, `gemma` / `llama-*` :8080).

Working inference paths:

- Qwen Image 2512 / Edit / ControlNet / Chroma / RealESRGAN via
  `control-api/app/qwen_graph.py`
- LTX 2.5 distilled/studio T2V+I2V via `control-api/app/ltx_graph.py`
- Curated image presets in `control-api/app/presets.py`
- LTX presets and camera/IC-LoRA selection in `ltx_presets.py` / `ltx_loras.py`
- WF_01 Character Master (candidates + select/promote)
- ComfyUI REST client in `comfy_client.py`
- Catalog-gated downloader: `catalog/models.yaml` + `scripts/download_models.py`
- GPU profile switching in `runtime.py`
- Legacy ops UI in `control-api/static/`

Catalog today (21 commercial-gated bundles): Qwen 2512/Edit, Lightning/Turbo and
style LoRAs, Chroma1-HD, Fun Union + DiffSynth controls, RealESRGAN, LTX 2.5
distilled/studio/prompt-enhancer, LTX-2.3 IC-LoRAs, Gemma 4 E4B, Qwen3.6-35B.

Not in catalog / not wired: SAM 2.1, Depth Anything V2 (node-side only), DINOv2,
official LTX 2.5 DFR, SeedVR2, FlashVSR, LatentSync, Chatterbox, ACE-Step,
Stable Audio Open, RIFE.

## Reuse vs extract vs leave alone

**Reuse as adapters**

- `ComfyClient` (add interrupt)
- `qwen_graph` / `ltx_graph` builders
- `scripts/download_models.py`
- `runtime.start_profile` / `stop_profile` / `get_status`
- `downloads.py` systemd trigger
- Existing character-master and LTX LoRA tests

**Extract**

- Unified Job / Artifact / Operation / Preset / Workflow / Model / LoRA
- Single worker + queue
- Asset IDs instead of filesystem paths / base64 / `sourceJobId`+filename
- Catalog-driven resolution
- Public job states (`QUEUED`…`CANCELLED`)

**Leave as legacy `/api/v1`**

- LLM planner on `/api/v1/generate`
- Character-master select/promote
- Prompt-inferred camera language on `/api/v1/ltx-video`
- Static Control UI

## Tight coupling that `/v1` must not leak

| Coupling | New rule |
|---|---|
| Filesystem paths in responses | Asset IDs only |
| Base64 / source job filename | `inputs.*.asset_id` |
| Hardcoded checkpoint filenames | Catalog files; graphs stay adapters |
| Node class names in app code | Stay inside ComfyUI adapter |
| Planner on every generate | INPUT PROMPT → EXECUTE PROMPT |
| Auto camera language from prompt | Caller or preset must opt in |
| Four JSON job stores | One `JobRepository` |
| Character select/promote | Future app chooses artifacts |

## API policy

- Keep `/api/v1/*` as deprecated compatibility
- Add stable `/v1/*` (snake_case)
- Do not add `/projects`, `/movies`, `/scenes`, `/users`

## Missing integrations (later phases)

Operation **contracts** for the film assembly path are frozen in
[movie-production-ops.md](movie-production-ops.md) and
`GET /v1/operations`. Executors and weights are not wired:

SAM / depth-as-operation, DINO QC, DFR, SeedVR2, FlashVSR, RIFE, LatentSync
(LTX LipDub covers `video.lipsync` today), commercial-safe TTS (Chatterbox /
Kokoro-class), ACE-Step-class music, FFmpeg mix / concat / finalize.

Do **not** add Stable Audio Open (blocked license). LTX-2.3 IC-LoRAs are
`compatibility_test_required: true` and must not be auto-selected for LTX 2.5
MASTER. LTX-2 19B camera/detailer adapters are not in the catalog.
