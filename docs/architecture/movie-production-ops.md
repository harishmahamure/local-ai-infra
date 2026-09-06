# Movie-production operations (API inventory)

This engine is a reusable **media inference platform**. A movie or reel app
composes `POST /v1/jobs` calls. It does **not** own projects, scenes, scripts,
users, or a permanent DAM. `client_context` is opaque tracing only.

Discover the live contract:

- `GET /v1/operations` — every id, `implemented`, `available`, schemas
- `GET /v1/operations/{operation_id}` — plus `workflow_ids`

Submitting an unimplemented operation returns `UNSUPPORTED_OPERATION`.
`available` is true only when `implemented` is true and required catalog
files are on disk.

## Shot graph (recommended)

```
still (image.*) → speech (audio.tts) → picture lock (video.lipsync)
                                      ↘ stems (audio.music / ambience / sfx / foley)
                                         → audio.mix → media.concat → media.finalize
```

## Live (callable today)

| Operation | Role |
|---|---|
| `image.generate` / `image.edit` / `image.controlled` / `image.layered` | Character / location stills, identity-locked keyframes |
| `image.upscale` | Still delivery |
| `video.generate` / `video.image_to_video` / `video.audio_to_video` / `video.first_last_frames` | Shots (1–30s) |
| `video.lipsync` | Dialogue / narration picture lock (`ltx25-lipsync`) |
| `video.motion_transfer` | Performance / camera from a reference clip |
| `audio.tts` | Hindi / English speech (`chatterbox-multilingual` + `chatterbox-hi`) |
| `audio.music` / `audio.sfx` / `audio.ambience` / `audio.foley` | Score and stems via ACE-Step 1.5 (prompt-only foley) |
| `audio.mix` / `audio.normalize` / `audio.inspect` | FFmpeg mix (−16 LUFS cinematic), loudnorm, ffprobe |
| `media.concat` / `media.finalize` | Cut or short crossfade; mux H.264/AAC mp4 |
| `text.chat` | Caller-owned prompt / script assist — engine does not plan movies |

`video.lipsync` needs uploaded `audio` + `start_image` asset ids and catalog
bundles `ltx-2.5-distilled` + `ltx-iclora-lipdub`.

`audio.tts` needs Chatterbox weights. Hindi uses `language=hi` (presets
`narrator_hindi` / `dialogue_hindi`) and optional `speaker_ref` for clone.
TTS is in-process CUDA: profile `tts` stops llama/comfy and leaves the GPU
free. Output is WAV 48 kHz.

`audio.music` / stems use native Comfy ACE-Step 1.5 on profile `comfy`.
`cinematic_master` sets `no_vocals`. Duration 0.25–180s. `audio.foley` is
prompt-only in Phase A — a `video` input is ignored and recorded as a job
warning. Picture-timed foley is later.

FFmpeg jobs (`audio.mix`, `audio.normalize`, `audio.inspect`, `media.concat`,
`media.finalize`) do not take the GPU profile.

After sync, download Phase A weights:

```bash
./bin/ai download chatterbox-multilingual chatterbox-hi ace-step-1.5
```

WF_01 Character Master remains on legacy `/api/v1/workflows/character-master`.
The engine equivalent is `image.generate` + preset `character_master`.

## Contract only (`implemented: false`) — Phase B / C

| Operation | Film job | Later executor (not wired) |
|---|---|---|
| `video.continue` | Next shot from the previous clip | LTX i2v / flf |
| `video.enhance` / `video.upscale` / `video.interpolate` / `video.color_grade` | Restore, scale, smoothness, look | License-checked later |
| `video.extract_frame` / `video.inspect` / `video.mask` | Still pull, metadata, tracking | FFmpeg / SAM-class |
| `image.mask` / `image.depth` / `image.inspect` | Prep for control / QC | SAM / Depth Anything |
| `qc.image` / `qc.video` / `qc.audio` / `qc.media` | Gate artifacts | Heuristics / DINO later |

Picture-locked Foley and a dedicated LatentSync path stay later; LTX LipDub
already covers `video.lipsync`.

## App-layer only (never this engine)

Script, shot list, casting, subtitles-as-story, users, billing, permanent DAM,
`/projects`, `/movies`, `/scenes`.

## Internal WF_01–WF_15 → operations

Clients never `POST /workflows/WF_14`. These names are pipeline documentation.

| Internal | Operation(s) | Status |
|---|---|---|
| WF_01 Character Master | `image.generate` + `character_master` | Live (legacy select/promote) |
| WF_02 Location still | `image.generate` | Live |
| WF_03 Costume / prop | `image.generate` / `image.edit` | Live |
| WF_04 Identity-locked keyframe | `image.edit` + `identity_strict` | Live generate; strict is policy |
| WF_05 Controlled keyframe | `image.controlled` | Live |
| WF_06 Shot | `video.generate` / `video.image_to_video` / `video.first_last_frames` | Live |
| WF_07 Continuation | `video.continue` | Contract (Phase B) |
| WF_08 Lip sync | `audio.tts` then `video.lipsync` | Live |
| WF_09 Motion transfer | `video.motion_transfer` | Live |
| WF_10 Enhance / upscale | `video.enhance` / `video.upscale` / `video.interpolate` | Contract (Phase C) |
| WF_11 Narration | `audio.tts` + `narrator_hindi` | Live |
| WF_12 Dialogue | `audio.tts` + `dialogue_hindi` | Live |
| WF_13 Stems | `audio.ambience` / `audio.sfx` / `audio.foley` | Live (prompt-only foley) |
| WF_14 Score | `audio.music` + `cinematic_master` | Live |
| WF_15 Assembly | `audio.mix` → `media.concat` → `media.finalize` | Live |

## License

Stay on [LICENSE-AUDIT.md](../../catalog/LICENSE-AUDIT.md). Do not add
Stable Audio Open, MusicGen NC, or other blocked families.
