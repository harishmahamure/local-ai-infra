from __future__ import annotations

from typing import Any

from ..domain.errors import DomainError, ErrorCode
from ..domain.operations import Operation
from ..domain.presets import Preset
from ..domain.workflows import WorkflowDefinition
from .catalog import CatalogRegistry

ALLOWED_OVERRIDES = {
    "seed",
    "steps",
    "cfg",
    "candidate_count",
    "scale",
    "width",
    "height",
    "duration_seconds",
    "loras",
    "negative_prompt",
    "aspect_ratio",
}

ASPECT_PIXELS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "2:3": (1080, 1620),
}


def resolve_operation(catalog: CatalogRegistry, operation_id: str) -> Operation:
    return catalog.operation(operation_id)


UNIVERSAL_PRESETS = {"draft", "balanced", "master"}


def resolve_preset(catalog: CatalogRegistry, operation: Operation, preset_id: str) -> Preset:
    preset = catalog.preset(preset_id)
    if preset.id in UNIVERSAL_PRESETS:
        return preset
    if preset.operations and operation.id not in preset.operations:
        raise DomainError(
            ErrorCode.UNSUPPORTED_PRESET,
            f"Preset {preset_id} is not valid for {operation.id}",
            {"preset": preset_id, "operation": operation.id},
        )
    return preset


def resolve_workflow(catalog: CatalogRegistry, operation: Operation, preset: Preset) -> WorkflowDefinition:
    if preset.workflow_id:
        wf = catalog.workflows.get(preset.workflow_id)
        if wf is None:
            raise DomainError(ErrorCode.WORKFLOW_NOT_AVAILABLE, f"Workflow {preset.workflow_id} is not registered")
        return wf
    matches = [wf for wf in catalog.workflows_for(operation.id) if preset.id in wf.supported_presets or not wf.supported_presets]
    if not matches:
        if not operation.implemented:
            raise DomainError(ErrorCode.UNSUPPORTED_OPERATION, f"Operation {operation.id} is not implemented yet")
        raise DomainError(ErrorCode.WORKFLOW_NOT_AVAILABLE, f"No workflow for {operation.id} + {preset.id}")
    return matches[0]


def validate_overrides(parameters: dict[str, Any], workflow: WorkflowDefinition) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in parameters.items():
        if key not in ALLOWED_OVERRIDES:
            raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unsupported parameter: {key}", {"field": key})
        if key == "seed" and not workflow.supports_seed:
            raise DomainError(ErrorCode.INVALID_PARAMETER, "Seed is not supported for this workflow")
        cleaned[key] = value
    return cleaned


def enforce_license(catalog: CatalogRegistry, model_ids: list[str]) -> None:
    if not catalog.commercial_mode:
        return
    for mid in model_ids:
        model = catalog.models.get(mid)
        if model and (model.noncommercial_only or not model.commercial):
            raise DomainError(
                ErrorCode.LICENSE_RESTRICTED,
                f"Model {mid} is not allowed in commercial mode",
                {"model_id": mid},
            )
        lora = catalog.loras.get(mid)
        if lora and (lora.noncommercial_only or not lora.commercial_use):
            raise DomainError(
                ErrorCode.LICENSE_RESTRICTED,
                f"LoRA {mid} is not allowed in commercial mode",
                {"lora_id": mid},
            )


def resolve_loras(catalog: CatalogRegistry, preset: Preset, parameters: dict[str, Any], model_ids: list[str]) -> list[dict[str, Any]]:
    requested = parameters.get("loras")
    if requested is None:
        requested = preset.parameters.get("loras") or []
    if not requested:
        return []

    resolved: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    for item in requested:
        if isinstance(item, str):
            lora_id, strength = item, None
        elif isinstance(item, dict):
            lora_id = str(item.get("id") or item.get("name") or "")
            strength = item.get("strength")
        else:
            raise DomainError(ErrorCode.INVALID_PARAMETER, "Invalid LoRA entry")
        rec = catalog.loras.get(lora_id)
        if rec is None:
            raise DomainError(ErrorCode.LORA_INCOMPATIBLE, f"Unknown LoRA: {lora_id}")
        if rec.id in preset.forbidden_loras:
            raise DomainError(ErrorCode.LORA_INCOMPATIBLE, f"LoRA {rec.id} is forbidden on preset {preset.id}")
        if rec.compatibility_test_required:
            raise DomainError(
                ErrorCode.LORA_INCOMPATIBLE,
                f"LoRA {rec.id} requires a compatibility test and is not used automatically",
                {"lora_id": rec.id},
            )
        if rec.noncommercial_only and catalog.commercial_mode:
            raise DomainError(ErrorCode.LICENSE_RESTRICTED, f"LoRA {rec.id} is license-restricted")
        if any(mid in rec.incompatible_models for mid in model_ids):
            raise DomainError(ErrorCode.LORA_INCOMPATIBLE, f"LoRA {rec.id} is incompatible with selected models")
        if rec.compatible_models and not any(mid in rec.compatible_models for mid in model_ids):
            raise DomainError(ErrorCode.LORA_INCOMPATIBLE, f"LoRA {rec.id} is not compatible with {model_ids}")
        value = rec.recommended_strength if strength is None else float(strength)
        lo, hi = rec.allowed_strength_range
        if value < lo or value > hi:
            raise DomainError(ErrorCode.INVALID_PARAMETER, f"LoRA {rec.id} strength {value} outside {lo}-{hi}")
        if rec.exclusive_group:
            used = group_counts.get(rec.exclusive_group, 0)
            max_active = catalog.groups.get(rec.exclusive_group).max_active if rec.exclusive_group in catalog.groups else 1
            if used >= max_active:
                raise DomainError(ErrorCode.LORA_INCOMPATIBLE, f"Exclusive group {rec.exclusive_group} allows {max_active}")
            group_counts[rec.exclusive_group] = used + 1
        resolved.append({"id": rec.id, "name": rec.file, "strength": value})
    return resolved


def resolve_dimensions(inputs: dict[str, Any], preset: Preset) -> tuple[int, int]:
    params = preset.parameters
    aspect = str(inputs.get("aspect_ratio") or params.get("aspect_ratio") or "")
    width = inputs.get("width") or params.get("width")
    height = inputs.get("height") or params.get("height")
    if width and height:
        return int(width), int(height)
    if aspect in ASPECT_PIXELS:
        return ASPECT_PIXELS[aspect]
    return int(params.get("width") or 1080), int(params.get("height") or 1920)


def availability_reason(
    catalog: CatalogRegistry,
    operation: Operation,
    disk_status: dict[str, str],
) -> str | None:
    if not operation.implemented:
        return "not_implemented"
    for mid in operation.required_models:
        model = catalog.models.get(mid)
        if model and not model.enabled:
            return "disabled"
        if catalog.commercial_mode and model and (model.noncommercial_only or not model.commercial):
            return "license_restricted"
        status = disk_status.get(mid)
        if status not in {"complete", "ok"}:
            return "model_not_downloaded"
    return None
