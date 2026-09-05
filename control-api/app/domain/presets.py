from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Preset:
    id: str
    label: str
    description: str
    operations: list[str]
    parameters: dict[str, Any] = field(default_factory=dict)
    forbidden_loras: list[str] = field(default_factory=list)
    required_models: list[str] = field(default_factory=list)
    workflow_id: str | None = None
    allow_prompt_enhance: bool = False

    def merged_parameters(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(self.parameters)
        for key, value in (overrides or {}).items():
            if value is not None:
                merged[key] = value
        return merged
