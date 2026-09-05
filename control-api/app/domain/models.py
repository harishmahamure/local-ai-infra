from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelState(str, Enum):
    NOT_DOWNLOADED = "NOT_DOWNLOADED"
    DOWNLOADING = "DOWNLOADING"
    AVAILABLE = "AVAILABLE"
    VERIFYING = "VERIFYING"
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
    dependencies: list[str] = field(default_factory=list)
    estimated_vram: float = 0.0
    supported_operations: list[str] = field(default_factory=list)
    supported_precisions: list[str] = field(default_factory=list)
    compatible_loras: list[str] = field(default_factory=list)
    compatible_controls: list[str] = field(default_factory=list)
    runtime: str = "comfyui"
    recommended_presets: list[str] = field(default_factory=list)
    enabled: bool = True
    gated: bool = False
    supported_context: int = 0
    context_native: int = 0
    context_max_yarn: int = 0
    modalities: list[str] = field(default_factory=list)

    def to_public_dict(self, *, state: ModelState, disk_status: str | None = None) -> dict[str, Any]:
        modalities = list(self.modalities)
        if not modalities and self.dest == "llamacpp":
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
            "dependencies": list(self.dependencies),
            "estimated_vram": self.estimated_vram,
            "supported_operations": list(self.supported_operations),
            "supported_precisions": list(self.supported_precisions),
            "compatible_loras": list(self.compatible_loras),
            "compatible_controls": list(self.compatible_controls),
            "runtime": self.runtime,
            "recommended_presets": list(self.recommended_presets),
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
