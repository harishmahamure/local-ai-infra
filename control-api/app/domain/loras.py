from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExclusiveGroup:
    id: str
    max_active: int = 1


@dataclass
class LoraRecord:
    id: str
    family: str
    role: str
    file: str
    compatible_models: list[str] = field(default_factory=list)
    incompatible_models: list[str] = field(default_factory=list)
    recommended_strength: float = 1.0
    allowed_strength_range: tuple[float, float] = (0.0, 2.0)
    exclusive_group: str | None = None
    license: str = ""
    commercial_use: bool = True
    noncommercial_only: bool = False
    compatibility_status: str = "validated"
    compatibility_test_required: bool = False
    bundle: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "role": self.role,
            "file": self.file,
            "compatible_models": list(self.compatible_models),
            "incompatible_models": list(self.incompatible_models),
            "recommended_strength": self.recommended_strength,
            "allowed_strength_range": list(self.allowed_strength_range),
            "exclusive_group": self.exclusive_group,
            "license": self.license,
            "commercial_use": self.commercial_use,
            "compatibility_status": self.compatibility_status,
            "compatibility_test_required": self.compatibility_test_required,
        }
