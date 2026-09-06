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
    output_schema: dict[str, Any] = field(default_factory=dict)

    def public_dict(self, *, available: bool, reason: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "description": self.description,
            "presets": list(self.presets),
            "input_schema": dict(self.input_schema),
            "parameter_schema": dict(self.parameter_schema),
            "output_schema": dict(self.output_schema),
            "implemented": self.implemented,
            "required_models": list(self.required_models),
            "available": available,
        }
        if reason:
            payload["reason"] = reason
        return payload
