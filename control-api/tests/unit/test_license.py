from __future__ import annotations

from pathlib import Path

from app.infrastructure.catalogs import load_catalogs

REPO = Path(__file__).resolve().parents[3]


def test_catalog_only_llamacpp() -> None:
    catalog = load_catalogs(REPO / "catalog")
    assert set(catalog.models) == {"gemma-4-e4b", "qwen36-35b-a3b-rq"}
    assert catalog.models["gemma-4-e4b"].commercial is True
    assert catalog.models["gemma-4-e4b"].noncommercial_only is False
    assert catalog.models["qwen36-35b-a3b-rq"].commercial is True
    assert "text.chat" in catalog.models["gemma-4-e4b"].supported_operations
    assert "vision" in catalog.models["qwen36-35b-a3b-rq"].modalities
