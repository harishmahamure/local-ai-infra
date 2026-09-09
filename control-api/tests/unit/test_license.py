from __future__ import annotations

from pathlib import Path

from app.infrastructure.catalogs import load_catalogs

REPO = Path(__file__).resolve().parents[3]


def test_catalog_includes_llamacpp_and_comfyui() -> None:
    catalog = load_catalogs(REPO / "catalog")
    assert "gemma-4-e4b" in catalog.models
    assert "qwen36-35b-a3b-rq" in catalog.models
    assert "qwen-image-2512-fp8" in catalog.models
    assert "qwen-image-edit-2511-fp8" in catalog.models
    assert catalog.models["gemma-4-e4b"].commercial is True
    assert catalog.models["qwen-image-2512-fp8"].runtime == "comfyui"
    edit_files = [item.path for item in catalog.models["qwen-image-edit-2511-fp8"].files]
    assert any("Qwen-Image-Edit-2511-Lightning-4steps" in path for path in edit_files)
    assert "text.chat" in catalog.models["gemma-4-e4b"].supported_operations
    assert "vision" in catalog.models["qwen36-35b-a3b-rq"].modalities
