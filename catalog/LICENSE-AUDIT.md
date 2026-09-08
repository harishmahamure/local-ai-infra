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

## Blocked (not in catalog)

| License / family | Reason |
|------------------|--------|
| FLUX.1-dev / FLUX.2-dev NC | Non-commercial without paid BFL license |
| Qwen3.8-Max (2.4T) | Revenue / MaaS clauses |
| Tencent Hunyuan Community | Territorial limits, MAU thresholds, attribution |
| Stability Community (SD3.5) | Revenue cap |
| OpenRAIL / CreativeML NC variants | Restricted commercial |

## Current catalog

All entries in `models.yaml` are `commercial: true`. Image, video, and speech weights on `/data/model_backup` are not in this catalog until those runtimes are re-planned.

| Bundle | Family | License | Use for |
|--------|--------|---------|---------|
| `gemma-4-e4b` | Gemma 4 E4B Q4_K_M + mmproj | Gemma Terms | text + vision chat (128K) |
| `qwen36-35b-a3b-rq` | Qwen3.6-35B-A3B RotorQuant Q4 + mmproj | Apache-2.0 | text + vision chat (262K native, 1M YaRN) |
