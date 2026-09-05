# `/v1` Media Engine Contract

Stable operation API for the future application. This service owns model
execution, GPU, workflows, temporary artifacts, and reproducibility — not
projects, users, or creative planning.

## Jobs

`POST /v1/jobs`

```json
{
  "operation": "video.image_to_video",
  "preset": "master",
  "inputs": {
    "start_image": { "asset_id": "ast_01KABC" },
    "prompt": "Medium cinematic tracking shot...",
    "duration_seconds": 6,
    "aspect_ratio": "9:16"
  },
  "parameters": { "seed": 482917 },
  "client_context": {
    "project_id": "external-project-123",
    "scene_id": "scene-4",
    "shot_id": "shot-12"
  }
}
```

Response `202`:

```json
{
  "job_id": "job_01KXYZ",
  "status": "queued",
  "operation": "video.image_to_video",
  "preset": "master"
}
```

`client_context` is opaque tracing metadata. It is never interpreted as domain
entities.

Header `Idempotency-Key` is required for safe HTTP retries. A repeated key with
the same body returns the original job. A repeated key with a different body
returns `409`.

## Job states

`QUEUED`, `VALIDATING`, `RESOLVING`, `WAITING_FOR_RESOURCE`, `RUNNING`,
`POST_PROCESSING`, `QC`, `COMPLETED`, `FAILED`, `CANCELLED`.

Internal `phase` is separate (`ltx_sampling`, `model_loading`, …).

## Retry

`POST /v1/jobs/{job_id}/retry`

- `same` — clone inputs and parameters
- `new_seed` — clone with a new seed
- `override` — merge `parameters`
- `rerun_stage` — Phase 1 treats as a full rerun (single stage)

Infrastructure failures may auto-retry once. Creative failures never auto-spend GPU.

## Assets

External callers never send filesystem paths. Upload via `POST /v1/assets`.
Default TTL is 24 hours. This engine is not the permanent DAM.

## Errors

```json
{
  "error": {
    "code": "MODEL_NOT_AVAILABLE",
    "message": "SeedVR2 7B is not downloaded.",
    "details": {},
    "retryable": false,
    "request_id": "..."
  }
}
```

## Phase 1 executable operations

- `image.generate` — Qwen; no planner
- `image.upscale` — RealESRGAN
- `video.generate` / `video.image_to_video` / `video.audio_to_video` / `video.first_last_frames` / `video.lipsync` / `video.motion_transfer` — LTX-2.5; no auto camera LoRA
- `text.chat` — sync/SSE via `POST /v1/text/chat` (Gemma or Qwen3.6); not a Comfy job

`video.lipsync` needs `ltx-iclora-lipdub`. `video.motion_transfer` needs `ltx-iclora-motion-track`. Other operations stay discoverable but `available: false`.

## Text models

`GET /v1/text/models` lists `gemma-4-e4b` and `qwen36-35b-a3b-rq` with `modalities`, `vision`, `profile`, and `context_length`.

`POST /v1/models/{id}/start` and `/stop` alias load/unload. Images in chat use `image_asset.asset_id` or `image_url` data/HTTP URLs — never filesystem paths.

## Prompt policy

Default: execute the caller prompt as written. Model-native enhancers run only
when a workflow/preset explicitly enables them.
