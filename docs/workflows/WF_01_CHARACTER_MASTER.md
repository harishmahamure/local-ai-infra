# WF_01 Character Master

Locked character-identity sheet for the movie/reel pipeline. Later shots edit this image (Qwen Image Edit) instead of regenerating the person from text. Generic identity LoRAs are not used.

This is the only implemented movie workflow. WF_02–WF_15 are deferred.

## Production purpose

Create a reusable `CHARACTER_MASTER` asset: multiple candidates, human selection, and a reproducibility JSON sidecar. MASTER quality never uses Lightning or Turbo LoRAs.

## Required catalog bundles

| Quality | Bundles | On-disk files |
|---------|---------|---------------|
| MASTER | `qwen-image-2512-fp8` | `diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors`, `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`, `vae/qwen_image_vae.safetensors` |
| DRAFT | above + `qwen-image-2512-lightning-lora` | `loras/Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors` |
| Optional | `qwen-lora-realism` | `loras/qwen-realism-lora.safetensors` — only if `realismLora` is set |
| Planner | `gemma-4-e4b` (text + vision) | Gemma 4 E4B Q4 + mmproj |

Download: `./bin/ai download qwen-image-2512-fp8` (and lightning LoRA for DRAFT).

## Required custom nodes

None beyond stock ComfyUI nodes already used by `build_txt2img`:

- `UNETLoader`
- `CLIPLoader` (`type=qwen_image`)
- `VAELoader`
- `EmptySD3LatentImage` or `VAEEncode` (reference image)
- `CLIPTextEncode`
- `ConditioningZeroOut` (DRAFT Lightning path, steps ≤ 4)
- `KSampler` (`euler` / `simple`)
- `VAEDecode`
- `SaveImage` (`filename_prefix=wf01_character_master`)
- `LoraLoaderModelOnly` (DRAFT Lightning, or optional realism)

## Compatibility

Verified against [control-api/app/qwen_graph.py](../../control-api/app/qwen_graph.py) and [catalog/models.yaml](../../catalog/models.yaml). Qwen 2512 FP8 is catalog-complete on the GPU box. No new node pack is required.

## Inputs

`POST /api/v1/workflows/character-master`

| Field | Type | Notes |
|-------|------|--------|
| `characterDescription` | string | required |
| `style` | string | optional |
| `quality` | `draft` \| `master` | default `master` |
| `aspectRatio` | `9:16` \| `16:9` \| `1:1` \| `4:5` | default `9:16` → 1080×1920 |
| `width` / `height` | int | optional override (256–4096) |
| `seed` | int | optional; candidates use `seed+i` |
| `candidateCount` | int | DRAFT default 2, MASTER default 4, max 8 |
| `image` | base64 / data URL | optional img2img reference |
| `denoise` | float | with `image` only; default 0.65 |
| `realismLora` | float 0–1 or false | default **off** |
| `upscale` | bool | default false |
| `characterId` | string | optional reuse id |

## Outputs

Job fields:

- `candidates[]` — index, seed, status, url
- `selected` — null until select
- `metadata` — reproducibility block
- `assetDir` — `{AI_LOGS}/assets/characters/{characterId}/`

After `POST /api/v1/workflows/character-master/{jobId}/select` with `{ "candidateIndex": 0 }`:

- `{AI_LOGS}/assets/characters/{characterId}/CHARACTER_MASTER.png`
- `selected.assetId`, `selected.masterUrl`

## Presets

| ID | Quality | Steps | CFG | LoRAs |
|----|---------|-------|-----|-------|
| `character_master_draft` | DRAFT | 4 | 1.0 | Lightning only |
| `character_master_master` | MASTER | 30 | 4.0 | none |

The wallpaper planner may suggest other presets. WF_01 **discards** those picks and locks the preset above. Prompt prose is the only planner output that is kept.

## API examples

```bash
# MASTER — 4 candidates, 9:16
curl -X POST http://127.0.0.1:8090/api/v1/workflows/character-master \
  -H 'Content-Type: application/json' \
  -d '{
    "characterDescription": "Maratha cavalry officer, early 40s, weathered face, short beard, rust-red turban, steel chest plate",
    "style": "cinematic historical, natural skin, film still",
    "quality": "master",
    "aspectRatio": "9:16",
    "seed": 12001
  }'

# Poll
curl http://127.0.0.1:8090/api/v1/workflows/character-master/{jobId}

# Promote candidate 1 to CHARACTER_MASTER
curl -X POST http://127.0.0.1:8090/api/v1/workflows/character-master/{jobId}/select \
  -H 'Content-Type: application/json' \
  -d '{"candidateIndex": 1}'
```

DRAFT (Lightning, 2 candidates):

```bash
curl -X POST http://127.0.0.1:8090/api/v1/workflows/character-master \
  -H 'Content-Type: application/json' \
  -d '{
    "characterDescription": "same officer, identity exploration",
    "quality": "draft",
    "candidateCount": 1,
    "seed": 7
  }'
```

## Errors

| Code | HTTP | When |
|------|------|------|
| `VALIDATION_ERROR` | 422 | empty description, bad quality/aspect/count, denoise without image |
| `MASTER_LORA_FORBIDDEN` | 422 | Lightning/Turbo present on a MASTER plan |
| `MODELS_MISSING` | 409 | required catalog bundle not complete |
| `COMFY_NOT_LOADED` | 409 | ComfyUI :8188 not loaded |
| `JOB_NOT_FOUND` | 404 | unknown job |
| `CANDIDATE_NOT_FOUND` | 404 | select index missing or not completed |

## VRAM (RTX 5090 32 GB)

| Phase | Resident | Approx |
|-------|----------|--------|
| DRAFT plan | Gemma 4 E4B | ~6 GB |
| MASTER plan | Gemma 4 E4B text + vision | ~7 GB |
| Generate | Qwen 2512 FP8 + text encoder + VAE | ~20 GB (catalog) |
| Candidates | sequential, same weights | no extra |

Planner and Comfy are never coresident. The existing GPU profile mutex unloads the planner before `comfy` starts. Do not further-quantize Qwen for MASTER.

## Retry

- Re-POST the same `characterId` and `seed` to regenerate the candidate set.
- Sequential candidates: a failed index does not discard completed siblings. Select any completed candidate.
- Job images remain under `{AI_LOGS}/jobs/{jobId}/images/`.

## Tests

```bash
cd control-api && python3 tests/test_character_master.py
```

Live GPU smoke (optional): POST DRAFT with `candidateCount: 1`, poll until `completed`, confirm PNG + `metadata.json`.
