from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..domain.errors import DomainError, ErrorCode
from ..domain.models import ModelFile, ModelRecord


@dataclass
class CatalogRegistry:
    models: dict[str, ModelRecord]
    commercial_mode: bool = True


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DomainError(ErrorCode.INTERNAL_ERROR, f"Catalog missing: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise DomainError(ErrorCode.INTERNAL_ERROR, f"Catalog invalid: {path}")
    return data


def load_catalogs(catalog_dir: Path, *, commercial_mode: bool = True) -> CatalogRegistry:
    models_raw = _load_yaml(catalog_dir / "models.yaml")
    models: dict[str, ModelRecord] = {}
    for item in models_raw.get("models") or []:
        dest = str(item.get("dest", "llamacpp"))
        if dest not in {"llamacpp", "comfyui"}:
            continue
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
            dest=dest,
            files=files,
            family=str(item.get("family", "")),
            role=str(item.get("role", "")),
            version=str(item.get("version", "")),
            provider=str(item.get("provider", "")),
            noncommercial_only=bool(item.get("noncommercial_only", False)),
            estimated_vram=float(item.get("estimated_vram", item.get("vram_gb", 0) or 0)),
            supported_operations=list(item.get("supported_operations") or ["text.chat"]),
            compatible_controls=list(item.get("compatible_controls") or []),
            runtime=str(item.get("runtime", "llamacpp")),
            enabled=bool(item.get("enabled", True)),
            gated=bool(item.get("gated", False)),
            supported_context=int(item.get("supported_context") or 0),
            context_native=int(item.get("context_native") or 0),
            context_max_yarn=int(item.get("context_max_yarn") or 0),
            modalities=list(item.get("modalities") or []),
        )
        models[record.id] = record
    return CatalogRegistry(models=models, commercial_mode=commercial_mode)


def enforce_license(catalog: CatalogRegistry, model_ids: list[str]) -> None:
    if not catalog.commercial_mode:
        return
    for model_id in model_ids:
        model = catalog.models.get(model_id)
        if model is None:
            continue
        if model.noncommercial_only or not model.commercial:
            raise DomainError(
                ErrorCode.LICENSE_RESTRICTED,
                f"{model_id} is not allowed in commercial mode",
            )
