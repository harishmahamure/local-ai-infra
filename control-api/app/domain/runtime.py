from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TimeoutClass(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    ENHANCEMENT = "enhancement"
    MUSIC = "music"
    DOWNLOAD = "download"
    CPU = "cpu"


TIMEOUT_SECONDS: dict[TimeoutClass, int] = {
    TimeoutClass.IMAGE: 600,
    TimeoutClass.VIDEO: 1800,
    TimeoutClass.ENHANCEMENT: 1800,
    TimeoutClass.MUSIC: 900,
    TimeoutClass.DOWNLOAD: 7200,
    TimeoutClass.CPU: 180,
}

STAGE_TIMEOUT_SECONDS: dict[TimeoutClass, dict[str, int]] = {
    TimeoutClass.IMAGE: {
        "queue_wait": 3600,
        "runtime_prep": 120,
        "model_load": 180,
        "inference": 600,
        "post_processing": 120,
        "artifact_store": 120,
    },
    TimeoutClass.VIDEO: {
        "queue_wait": 7200,
        "runtime_prep": 180,
        "model_load": 300,
        "inference": 1800,
        "post_processing": 300,
        "artifact_store": 180,
    },
    TimeoutClass.ENHANCEMENT: {
        "queue_wait": 7200,
        "runtime_prep": 180,
        "model_load": 300,
        "inference": 1800,
        "post_processing": 300,
        "artifact_store": 180,
    },
    TimeoutClass.MUSIC: {
        "queue_wait": 3600,
        "runtime_prep": 90,
        "model_load": 180,
        "inference": 900,
        "post_processing": 120,
        "artifact_store": 120,
    },
    TimeoutClass.DOWNLOAD: {
        "queue_wait": 7200,
        "runtime_prep": 30,
        "model_load": 30,
        "inference": 7200,
        "post_processing": 60,
        "artifact_store": 60,
    },
    TimeoutClass.CPU: {
        "queue_wait": 1800,
        "runtime_prep": 30,
        "model_load": 30,
        "inference": 180,
        "post_processing": 60,
        "artifact_store": 60,
    },
}


def stage_timeout(timeout_class: TimeoutClass, stage: str) -> int:
    return STAGE_TIMEOUT_SECONDS.get(timeout_class, STAGE_TIMEOUT_SECONDS[TimeoutClass.IMAGE]).get(stage, TIMEOUT_SECONDS[timeout_class])


@dataclass
class ResourceRequirement:
    estimated_vram_gb: float = 0.0
    profile: str = "comfy"
    timeout_class: TimeoutClass = TimeoutClass.IMAGE
    gpu_required: bool = True

    @property
    def timeout_seconds(self) -> int:
        return TIMEOUT_SECONDS[self.timeout_class]


@dataclass
class ResourceLease:
    job_id: str
    profile: str
    estimated_vram_gb: float
