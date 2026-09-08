from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelState(str, Enum):
    NOT_DOWNLOADED = "NOT_DOWNLOADED"
    DOWNLOADING = "DOWNLOADING"
    AVAILABLE = "AVAILABLE"
    LOADING = "LOADING"
    READY = "READY"
    UNLOADING = "UNLOADING"
    ERROR = "ERROR"
    DISABLED = "DISABLED"
    INCOMPATIBLE = "INCOMPATIBLE"


@dataclass
class ModelFile:
    repo: str
    path: str
    local: str
    optional: bool = False


@dataclass
class ModelRecord:
    id: str
    license: str
    commercial: bool
    dest: str
    files: list[ModelFile]
    family: str = ""
    role: str = ""
    version: str = ""
    provider: str = ""
    noncommercial_only: bool = False
    estimated_vram: float = 0.0
    supported_operations: list[str] = field(default_factory=list)
    compatible_controls: list[str] = field(default_factory=list)
    runtime: str = "llamacpp"
    enabled: bool = True
    gated: bool = False
    supported_context: int = 0
    context_native: int = 0
    context_max_yarn: int = 0
    modalities: list[str] = field(default_factory=list)

    def to_public_dict(self, *, state: ModelState, disk_status: str | None = None) -> dict[str, Any]:
        modalities = list(self.modalities)
        if not modalities:
            modalities = ["text"]
            if "vision" in self.compatible_controls:
                modalities.append("vision")
        payload: dict[str, Any] = {
            "id": self.id,
            "family": self.family,
            "role": self.role,
            "version": self.version,
            "provider": self.provider,
            "license": self.license,
            "commercial_use": self.commercial,
            "noncommercial_only": self.noncommercial_only,
            "estimated_vram": self.estimated_vram,
            "supported_operations": list(self.supported_operations),
            "compatible_controls": list(self.compatible_controls),
            "runtime": self.runtime,
            "enabled": self.enabled,
            "state": state.value,
            "disk_status": disk_status,
            "modalities": modalities,
            "vision": "vision" in modalities or "vision" in self.compatible_controls,
        }
        if self.supported_context:
            payload["context_length"] = self.supported_context
        if self.context_native:
            payload["context_native"] = self.context_native
        if self.context_max_yarn:
            payload["context_max_yarn"] = self.context_max_yarn
        return payload
