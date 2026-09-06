from __future__ import annotations

from pathlib import Path

import pytest

from app.application.resolvers import (
    enforce_license,
    resolve_loras,
    resolve_operation,
    resolve_preset,
    resolve_workflow,
    validate_overrides,
)
from app.domain.errors import DomainError, ErrorCode
from app.infrastructure.catalogs import load_catalogs

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def catalog():
    return load_catalogs(REPO / "catalog", commercial_mode=True)


def test_resolve_image_generate_master(catalog) -> None:
    op = resolve_operation(catalog, "image.generate")
    preset = resolve_preset(catalog, op, "character_master")
    wf = resolve_workflow(catalog, op, preset)
    assert wf.id == "qwen-txt2img"
    assert "qwen-lightning" in preset.forbidden_loras


def test_unknown_operation(catalog) -> None:
    with pytest.raises(DomainError) as exc:
        resolve_operation(catalog, "movie.render")
    assert exc.value.code == ErrorCode.UNSUPPORTED_OPERATION


def test_identity_strict_not_for_generate(catalog) -> None:
    op = resolve_operation(catalog, "image.generate")
    with pytest.raises(DomainError) as exc:
        resolve_preset(catalog, op, "identity_strict")
    assert exc.value.code == ErrorCode.UNSUPPORTED_PRESET


def test_invalid_override(catalog) -> None:
    wf = catalog.workflows["qwen-txt2img"]
    with pytest.raises(DomainError) as exc:
        validate_overrides({"sampler_name": "euler"}, wf)
    assert exc.value.code == ErrorCode.INVALID_PARAMETER


def test_untested_ltx_iclora_rejected(catalog) -> None:
    preset = catalog.presets["master"]
    with pytest.raises(DomainError) as exc:
        resolve_loras(catalog, preset, {"loras": ["ltx-iclora-union"]}, ["ltx-2.5-distilled"])
    assert exc.value.code == ErrorCode.LORA_INCOMPATIBLE


def test_workflow_bindings_are_logical(catalog) -> None:
    wf = catalog.workflows["qwen-txt2img"]
    assert "prompt" in wf.bindings
    assert "seed" in wf.bindings
    assert wf.bindings["prompt"]["key"] == "prompt"


def test_exclusive_qwen_speed_group(catalog) -> None:
    preset = catalog.presets["draft"]
    with pytest.raises(DomainError) as exc:
        resolve_loras(
            catalog,
            preset,
            {"loras": ["qwen-lightning", "qwen-turbo"]},
            ["qwen-image-2512-fp8"],
        )
    assert exc.value.code == ErrorCode.LORA_INCOMPATIBLE


def test_license_restricted(catalog) -> None:
    catalog.models["qwen-image-2512-fp8"].noncommercial_only = True
    with pytest.raises(DomainError) as exc:
        enforce_license(catalog, ["qwen-image-2512-fp8"])
    assert exc.value.code == ErrorCode.LICENSE_RESTRICTED
