from __future__ import annotations

from dataclasses import dataclass

from ..domain.errors import DomainError, ErrorCode
from ..domain.loras import ExclusiveGroup, LoraRecord
from ..domain.models import ModelRecord
from ..domain.operations import Operation
from ..domain.presets import Preset
from ..domain.workflows import WorkflowDefinition


@dataclass
class CatalogRegistry:
    models: dict[str, ModelRecord]
    loras: dict[str, LoraRecord]
    groups: dict[str, ExclusiveGroup]
    presets: dict[str, Preset]
    workflows: dict[str, WorkflowDefinition]
    operations: dict[str, Operation]
    commercial_mode: bool = True

    def operation(self, operation_id: str) -> Operation:
        op = self.operations.get(operation_id)
        if not op:
            raise DomainError(ErrorCode.UNSUPPORTED_OPERATION, f"Unknown operation: {operation_id}")
        return op

    def preset(self, preset_id: str) -> Preset:
        preset = self.presets.get(preset_id)
        if not preset:
            raise DomainError(ErrorCode.UNSUPPORTED_PRESET, f"Unknown preset: {preset_id}")
        return preset

    def workflows_for(self, operation_id: str) -> list[WorkflowDefinition]:
        return [wf for wf in self.workflows.values() if wf.operation == operation_id]
