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
    assert "ltx-2.5-distilled" in catalog.models
    assert "ltx-2.5-studio" in catalog.models
    assert "ltx-2.5-nvfp4" in catalog.models
    assert "ltx-2.5-control" in catalog.models
    assert catalog.models["ltx-2.5-control"].license == "LTX-2-Community"
    assert catalog.models["gemma-4-e4b"].commercial is True
    assert catalog.models["qwen-image-2512-fp8"].runtime == "comfyui"
    assert catalog.models["ltx-2.5-distilled"].runtime == "comfy-ltx"
    assert catalog.models["ltx-2.5-distilled"].gated is True
    assert catalog.models["ltx-2.5-studio"].license == "LTX-2-Community"
    control_files = [item.local for item in catalog.models["ltx-2.5-control"].files]
    assert "loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" in control_files
    distilled_files = [item.local for item in catalog.models["ltx-2.5-distilled"].files]
    assert "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors" in distilled_files
    assert "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" in distilled_files
    assert "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors" in distilled_files
    assert "vae/ltx-2.5-video-vae-bf16.safetensors" in distilled_files
    edit_files = [item.path for item in catalog.models["qwen-image-edit-2511-fp8"].files]
    assert any("Qwen-Image-Edit-2511-Lightning-4steps" in path for path in edit_files)
    assert "text.chat" in catalog.models["gemma-4-e4b"].supported_operations
    assert "vision" in catalog.models["qwen36-35b-a3b-rq"].modalities
