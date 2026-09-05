from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ArtifactType(str, Enum):
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"
    MASK = "MASK"
    DEPTH_MAP = "DEPTH_MAP"
    POSE_MAP = "POSE_MAP"
    CONTROL_MAP = "CONTROL_MAP"
    SUBTITLE = "SUBTITLE"
    METADATA = "METADATA"
    JSON = "JSON"
    FRAME = "FRAME"
    TEMPORARY = "TEMPORARY"


@dataclass
class Lineage:
    source_asset_ids: list[str] = field(default_factory=list)
    parent_job_id: str | None = None
    model_id: str | None = None
    workflow_id: str | None = None
    workflow_version: str | None = None
    seed: int | None = None
    prompt: str | None = None
    control_inputs: dict[str, Any] = field(default_factory=dict)
    loras: list[dict[str, Any]] = field(default_factory=list)
    generation_settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReproducibilityManifest:
    job_id: str
    operation: str
    preset: str
    workflow_id: str | None = None
    workflow_version: str | None = None
    model_ids: list[str] = field(default_factory=list)
    lora_ids: list[str] = field(default_factory=list)
    lora_strengths: list[float] = field(default_factory=list)
    seed: int | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    sampler: str | None = None
    scheduler: str | None = None
    steps: int | None = None
    cfg: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    precision: str | None = None
    execution_duration_ms: int | None = None
    output_checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "operation": self.operation,
            "preset": self.preset,
            "workflow_id": self.workflow_id,
            "workflow_version": self.workflow_version,
            "model_ids": list(self.model_ids),
            "lora_ids": list(self.lora_ids),
            "lora_strengths": list(self.lora_strengths),
            "seed": self.seed,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "steps": self.steps,
            "cfg": self.cfg,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "precision": self.precision,
            "execution_duration_ms": self.execution_duration_ms,
            "output_checksum": self.output_checksum,
        }


@dataclass
class Artifact:
    asset_id: str
    type: ArtifactType
    mime_type: str
    checksum: str
    size_bytes: int
    created_at: str
    expires_at: str
    retention_policy: str = "ttl"
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    lineage: Lineage = field(default_factory=Lineage)
    manifest: dict[str, Any] = field(default_factory=dict)
    deleted: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "type": self.type.value.lower(),
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "expires_at": self.expires_at,
            "retention_policy": self.retention_policy,
            "lineage": {
                "source_asset_ids": list(self.lineage.source_asset_ids),
                "parent_job_id": self.lineage.parent_job_id,
                "model_id": self.lineage.model_id,
                "workflow_id": self.lineage.workflow_id,
                "workflow_version": self.lineage.workflow_version,
                "seed": self.lineage.seed,
                "prompt": self.lineage.prompt,
            },
            "manifest": dict(self.manifest),
        }
