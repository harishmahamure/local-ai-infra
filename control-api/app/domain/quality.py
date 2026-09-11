"""Quality posture for LTX shot flows.

Spike (2026-09-10): the nvfp4 distilled transformer is not on this box yet, so
LoRA patching onto nvfp4 weights could not be verified. IC-LoRAs and the
quality LoRA therefore attach to the on-disk int8-convrot transformer. Shots
that do not load a LoRA prefer nvfp4. Over-budget shots enable context windows
instead of rejecting the request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

UNET_NVFP4 = "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
UNET_INT8 = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
CLIP_INT8 = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
QUALITY_LORA = "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
SPATIAL_UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
TEMPORAL_UPSCALER = "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"

IC_LORA_UNION = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
IC_LORA_MOTION = "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"
IC_LORA_DUBIT = "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors"

CLIP_DEVICE = "cpu"
GUIDE_COMPRESSION = 2
TOKEN_ENVELOPE = 14000
WINDOW_LENGTH = 65
WINDOW_OVERLAP = 24


@dataclass(frozen=True)
class TokenBudget:
    tokens: int
    envelope: int
    windowed: bool
    warning: str | None = None


def select_unet(*, use_lora: bool, prefer_nvfp4: bool = True) -> str:
    """Prefer nvfp4 for quality; keep LoRA on int8 until the nvfp4 spike lands."""
    if use_lora:
        return UNET_INT8
    return UNET_NVFP4 if prefer_nvfp4 else UNET_INT8


def latent_tokens(width: int, height: int, length: int, guide_count: int, *, video_guides: int = 0) -> int:
    spatial = max(1, (int(width) // 32) * (int(height) // 32))
    temporal = max(1, (int(length) - 1) // 8 + 1)
    return spatial * temporal + guide_count * spatial + video_guides * spatial * max(1, temporal // 2)


def estimate_budget(
    width: int,
    height: int,
    length: int,
    guide_count: int,
    *,
    video_guides: int = 0,
    envelope: int = TOKEN_ENVELOPE,
) -> TokenBudget:
    tokens = latent_tokens(width, height, length, guide_count, video_guides=video_guides)
    if tokens <= envelope:
        return TokenBudget(tokens=tokens, envelope=envelope, windowed=False)
    return TokenBudget(
        tokens=tokens,
        envelope=envelope,
        windowed=True,
        warning=f"Token budget {tokens} exceeds {envelope}; LTXVContextWindows enabled.",
    )


def ic_lora_for(kind: str) -> str | None:
    mapping = {
        "canny": IC_LORA_UNION,
        "depth": IC_LORA_UNION,
        "pose": IC_LORA_UNION,
        "union": IC_LORA_UNION,
        "motion_track": IC_LORA_MOTION,
        "motion": IC_LORA_MOTION,
        "dubit": IC_LORA_DUBIT,
    }
    return mapping.get(str(kind or "").strip())


def quality_defaults(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "prefer_nvfp4": body.get("prefer_nvfp4") is not False,
        "img_compression": int(body.get("img_compression") if body.get("img_compression") is not None else GUIDE_COMPRESSION),
        "temporal_refine": body.get("temporal_refine") is not False,
        "clip_device": str(body.get("clip_device") or CLIP_DEVICE),
        "quality_lora": bool(body.get("quality_lora")),
    }
