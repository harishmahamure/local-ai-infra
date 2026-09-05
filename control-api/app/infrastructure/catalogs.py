from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..application.catalog import CatalogRegistry
from ..domain.errors import DomainError, ErrorCode
from ..domain.loras import ExclusiveGroup, LoraRecord
from ..domain.models import ModelFile, ModelRecord
from ..domain.presets import Preset
from ..domain.runtime import ResourceRequirement, TimeoutClass
from ..domain.workflows import WorkflowDefinition
from .operations import all_operations


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DomainError(ErrorCode.INTERNAL_ERROR, f"Catalog missing: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise DomainError(ErrorCode.INTERNAL_ERROR, f"Catalog invalid: {path}")
    return data


def _range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return default


def load_catalogs(catalog_dir: Path, *, commercial_mode: bool = True) -> CatalogRegistry:
    models_raw = _load_yaml(catalog_dir / "models.yaml")
    loras_raw = _load_yaml(catalog_dir / "loras.yaml")
    presets_raw = _load_yaml(catalog_dir / "presets.yaml")
    workflows_raw = _load_yaml(catalog_dir / "workflows.yaml")

    models: dict[str, ModelRecord] = {}
    for item in models_raw.get("models") or []:
        files = [
            ModelFile(
                repo=str(f.get("repo", "")),
                path=str(f.get("path", "")),
                local=str(f.get("local", "")),
                optional=bool(f.get("optional", False)),
            )
            for f in item.get("files") or []
        ]
        record = ModelRecord(
            id=str(item["id"]),
            license=str(item.get("license", "")),
            commercial=bool(item.get("commercial", False)),
            dest=str(item.get("dest", "comfyui")),
            files=files,
            family=str(item.get("family", "")),
            role=str(item.get("role", "")),
            version=str(item.get("version", "")),
            provider=str(item.get("provider", "")),
            noncommercial_only=bool(item.get("noncommercial_only", False)),
            dependencies=list(item.get("depends_on") or item.get("dependencies") or []),
            estimated_vram=float(item.get("estimated_vram", item.get("vram_gb", 0) or 0)),
            supported_operations=list(item.get("supported_operations") or []),
            supported_precisions=list(item.get("supported_precisions") or []),
            compatible_loras=list(item.get("compatible_loras") or []),
            compatible_controls=list(item.get("compatible_controls") or []),
            runtime=str(item.get("runtime", "comfyui" if item.get("dest") == "comfyui" else "llamacpp")),
            recommended_presets=list(item.get("recommended_presets") or []),
            enabled=bool(item.get("enabled", True)),
            gated=bool(item.get("gated", False)),
            supported_context=int(item.get("supported_context") or 0),
            context_native=int(item.get("context_native") or 0),
            context_max_yarn=int(item.get("context_max_yarn") or 0),
            modalities=list(item.get("modalities") or []),
        )
        models[record.id] = record

    groups = {
        str(gid): ExclusiveGroup(id=str(gid), max_active=int((g or {}).get("max_active", 1)))
        for gid, g in (loras_raw.get("groups") or {}).items()
    }
    loras: dict[str, LoraRecord] = {}
    for item in loras_raw.get("loras") or []:
        rec = LoraRecord(
            id=str(item["id"]),
            family=str(item.get("family", "")),
            role=str(item.get("role", "")),
            file=str(item.get("file", "")),
            compatible_models=list(item.get("compatible_models") or []),
            incompatible_models=list(item.get("incompatible_models") or []),
            recommended_strength=float(item.get("recommended_strength", 1.0)),
            allowed_strength_range=_range(item.get("allowed_strength_range"), (0.0, 2.0)),
            exclusive_group=item.get("exclusive_group"),
            license=str(item.get("license", "")),
            commercial_use=bool(item.get("commercial_use", True)),
            noncommercial_only=bool(item.get("noncommercial_only", False)),
            compatibility_status=str(item.get("compatibility_status", "validated")),
            compatibility_test_required=bool(item.get("compatibility_test_required", False)),
            bundle=item.get("bundle"),
        )
        loras[rec.id] = rec

    presets: dict[str, Preset] = {}
    for item in presets_raw.get("presets") or []:
        preset = Preset(
            id=str(item["id"]),
            label=str(item.get("label", item["id"])),
            description=str(item.get("description", "")),
            operations=list(item.get("operations") or []),
            parameters=dict(item.get("parameters") or {}),
            forbidden_loras=list(item.get("forbidden_loras") or []),
            required_models=list(item.get("required_models") or []),
            workflow_id=item.get("workflow_id"),
            allow_prompt_enhance=bool(item.get("allow_prompt_enhance", False)),
        )
        presets[preset.id] = preset

    workflows: dict[str, WorkflowDefinition] = {}
    for item in workflows_raw.get("workflows") or []:
        res = item.get("resource") or {}
        timeout = str(res.get("timeout_class", "image"))
        wf = WorkflowDefinition(
            id=str(item["id"]),
            version=str(item.get("version", "v1")),
            operation=str(item["operation"]),
            description=str(item.get("description", "")),
            executor=str(item.get("executor", "comfyui")),
            builder=str(item.get("builder", "")),
            required_models=list(item.get("required_models") or []),
            optional_models=list(item.get("optional_models") or []),
            required_loras=list(item.get("required_loras") or []),
            optional_loras=list(item.get("optional_loras") or []),
            required_nodes=list(item.get("required_nodes") or []),
            input_schema=dict(item.get("input_schema") or {}),
            output_schema=dict(item.get("output_schema") or {}),
            supported_presets=list(item.get("supported_presets") or []),
            resource=ResourceRequirement(
                estimated_vram_gb=float(res.get("estimated_vram_gb", 0)),
                profile=str(res.get("profile", "comfy")),
                timeout_class=TimeoutClass(timeout),
                gpu_required=bool(res.get("gpu_required", True)),
            ),
            bindings=dict(item.get("bindings") or {}),
            template_path=item.get("template_path"),
            supports_seed=bool(item.get("supports_seed", True)),
            supports_cancel=bool(item.get("supports_cancel", True)),
            supports_batch=bool(item.get("supports_batch", False)),
        )
        workflows[wf.id] = wf

    operations = {op.id: op for op in all_operations()}
    return CatalogRegistry(
        models=models,
        loras=loras,
        groups=groups,
        presets=presets,
        workflows=workflows,
        operations=operations,
        commercial_mode=commercial_mode,
    )
