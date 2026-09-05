"""Shared types and reproducibility metadata for movie-pipeline workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

WORKFLOW_ID = "WF_01_CHARACTER_MASTER"
WORKFLOW_VERSION = "1.0.0"

Quality = Literal["draft", "master"]
AspectRatio = Literal["9:16", "16:9", "1:1", "4:5"]

QUALITY_VALUES: tuple[str, ...] = ("draft", "master")
ASPECT_VALUES: tuple[str, ...] = ("9:16", "16:9", "1:1", "4:5")

ASPECT_PIXELS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}

DEFAULT_ASPECT: AspectRatio = "9:16"
DEFAULT_QUALITY: Quality = "master"
DEFAULT_DENOISE = 0.65
MAX_CANDIDATES = 8
DEFAULT_CANDIDATES: dict[str, int] = {"draft": 2, "master": 4}

PRESET_BY_QUALITY: dict[str, str] = {
    "draft": "character_master_draft",
    "master": "character_master_master",
}

FORBIDDEN_MASTER_LORA_MARKERS = ("lightning", "turbo")

SAMPLER_NAME = "euler"
SCHEDULER_NAME = "simple"

METADATA_KEYS: tuple[str, ...] = (
    "job_id",
    "workflow_id",
    "workflow_version",
    "quality",
    "model_ids",
    "model_hashes",
    "lora_ids",
    "lora_weights",
    "seed",
    "prompt",
    "negative_prompt",
    "resolution",
    "sampler",
    "scheduler",
    "steps",
    "cfg",
    "denoise",
    "reference_asset_id",
    "parent_shot",
    "start_frame_id",
    "end_frame_id",
    "created_at",
    "character_id",
    "aspect_ratio",
    "filename_prefix",
    "selected_candidate_index",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dimensions_for_aspect(aspect: str) -> tuple[int, int]:
    return ASPECT_PIXELS[aspect]


def lora_is_forbidden_on_master(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in FORBIDDEN_MASTER_LORA_MARKERS)


def generation_metadata(
    *,
    job_id: str,
    quality: str,
    model_ids: list[str],
    model_hashes: dict[str, str | None],
    lora_ids: list[str],
    lora_weights: list[float],
    seed: int | None,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    denoise: float,
    character_id: str,
    aspect_ratio: str,
    reference_asset_id: str | None = None,
    selected_candidate_index: int | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "workflow_id": WORKFLOW_ID,
        "workflow_version": WORKFLOW_VERSION,
        "quality": quality,
        "model_ids": list(model_ids),
        "model_hashes": dict(model_hashes),
        "lora_ids": list(lora_ids),
        "lora_weights": list(lora_weights),
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": {"width": width, "height": height},
        "sampler": SAMPLER_NAME,
        "scheduler": SCHEDULER_NAME,
        "steps": steps,
        "cfg": cfg,
        "denoise": denoise,
        "reference_asset_id": reference_asset_id,
        "parent_shot": None,
        "start_frame_id": None,
        "end_frame_id": None,
        "created_at": created_at or utc_now(),
        "character_id": character_id,
        "aspect_ratio": aspect_ratio,
        "filename_prefix": "wf01_character_master",
        "selected_candidate_index": selected_candidate_index,
    }
