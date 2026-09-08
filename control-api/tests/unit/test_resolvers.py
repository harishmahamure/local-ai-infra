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


def test_resolve_audio_tts_narrator(catalog) -> None:
    op = resolve_operation(catalog, "audio.tts")
    preset = resolve_preset(catalog, op, "narrator_hindi")
    wf = resolve_workflow(catalog, op, preset)
    assert wf.id == "audio-tts"
    assert wf.executor == "tts"
    assert wf.builder == "tts_generate"
    cleaned = validate_overrides({"speaking_rate": 1.2, "seed": 1}, wf)
    assert cleaned["speaking_rate"] == 1.2


def test_resolve_image_generate_master(catalog) -> None:
    op = resolve_operation(catalog, "image.generate")
    preset = resolve_preset(catalog, op, "character_master")
    wf = resolve_workflow(catalog, op, preset)
    assert wf.id == "qwen-txt2img"
    assert "qwen-lightning" in preset.forbidden_loras


def test_resolve_face_lock_generate_preset(catalog) -> None:
    op = resolve_operation(catalog, "image.generate")
    preset = resolve_preset(catalog, op, "face_lock")
    wf = resolve_workflow(catalog, op, preset)
    assert wf.id == "qwen-txt2img"
    assert preset.parameters["denoise"] == 0.40
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


def test_qwen_txt2img_plan_sets_denoise_when_image_present() -> None:
    from app.domain.jobs import Job, JobStatus
    from app.domain.presets import Preset
    from app.infrastructure.comfyui.plans import _qwen_txt2img

    preset = Preset(
        id="master",
        label="",
        description="",
        operations=["image.generate"],
        parameters={"width": 1920, "height": 1080, "steps": 30, "cfg": 4.0},
    )
    job = Job(
        job_id="j1",
        operation="image.generate",
        preset="master",
        status=JobStatus.QUEUED,
        phase="queued",
        progress=0.0,
        inputs={"prompt": "same person", "image": {"asset_id": "a1"}, "denoise": 0.5},
        parameters={},
        client_context={},
        created_at="2026-01-01T00:00:00Z",
    )
    plan = _qwen_txt2img(job, preset, [])
    assert plan["denoise"] == 0.5
    no_image = Job(
        job_id="j2",
        operation="image.generate",
        preset="master",
        status=JobStatus.QUEUED,
        phase="queued",
        progress=0.0,
        inputs={"prompt": "a lantern"},
        parameters={},
        client_context={},
        created_at="2026-01-01T00:00:00Z",
    )
    plain = _qwen_txt2img(no_image, preset, [])
    assert "denoise" not in plain


def test_qwen_txt2img_face_lock_wraps_prompt_and_denoise() -> None:
    from app.application.image_flows import DEFAULT_CHARACTER_DENOISE, FACE_LOCK_PREFIX
    from app.domain.errors import DomainError, ErrorCode
    from app.domain.jobs import Job, JobStatus
    from app.domain.presets import Preset
    from app.infrastructure.comfyui.plans import _qwen_txt2img

    preset = Preset(
        id="face_lock",
        label="",
        description="",
        operations=["image.generate"],
        parameters={"width": 1920, "height": 1080, "steps": 30, "cfg": 4.0, "denoise": 0.40},
    )
    job = Job(
        job_id="j1",
        operation="image.generate",
        preset="face_lock",
        status=JobStatus.QUEUED,
        phase="queued",
        progress=0.0,
        inputs={"prompt": "same man in steel armor, dusk courtyard", "image": {"asset_id": "a1"}},
        parameters={},
        client_context={},
        created_at="2026-01-01T00:00:00Z",
    )
    plan = _qwen_txt2img(job, preset, [])
    assert plan["denoise"] == DEFAULT_CHARACTER_DENOISE
    assert FACE_LOCK_PREFIX in plan["prompt"]
    assert "steel armor, dusk courtyard" in plan["prompt"]
    missing = Job(
        job_id="j2",
        operation="image.generate",
        preset="face_lock",
        status=JobStatus.QUEUED,
        phase="queued",
        progress=0.0,
        inputs={"prompt": "same man in armor"},
        parameters={},
        client_context={},
        created_at="2026-01-01T00:00:00Z",
    )
    with pytest.raises(DomainError) as exc:
        _qwen_txt2img(missing, preset, [])
    assert exc.value.code == ErrorCode.INVALID_REQUEST


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
