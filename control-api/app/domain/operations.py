from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Operation:
    id: str
    description: str
    presets: list[str]
    input_schema: dict[str, Any]
    parameter_schema: dict[str, Any]
    implemented: bool = False
    required_models: list[str] = field(default_factory=list)
    capability_path: tuple[str, ...] = ()

    def public_dict(self, *, available: bool, reason: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "description": self.description,
            "presets": list(self.presets),
            "input_schema": dict(self.input_schema),
            "parameter_schema": dict(self.parameter_schema),
            "available": available,
        }
        if reason:
            payload["reason"] = reason
        return payload
