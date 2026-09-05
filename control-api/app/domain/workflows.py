from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .runtime import ResourceRequirement


@dataclass
class WorkflowDefinition:
    id: str
    version: str
    operation: str
    description: str
    executor: str
    builder: str
    required_models: list[str] = field(default_factory=list)
    optional_models: list[str] = field(default_factory=list)
    required_loras: list[str] = field(default_factory=list)
    optional_loras: list[str] = field(default_factory=list)
    required_nodes: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    supported_presets: list[str] = field(default_factory=list)
    resource: ResourceRequirement = field(default_factory=ResourceRequirement)
    bindings: dict[str, Any] = field(default_factory=dict)
    template_path: str | None = None
    supports_seed: bool = True
    supports_cancel: bool = True
    supports_batch: bool = False

    @property
    def qualified_id(self) -> str:
        return f"{self.id}:{self.version}"
