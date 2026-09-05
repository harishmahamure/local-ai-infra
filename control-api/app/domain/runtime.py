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
