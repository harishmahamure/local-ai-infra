from __future__ import annotations

from pathlib import Path

from app.infrastructure.catalogs import load_catalogs

REPO = Path(__file__).resolve().parents[3]


INVENTORY = {
    "image.generate",
    "image.edit",
    "image.controlled",
    "image.layered",
    "image.mask",
    "image.depth",
    "image.upscale",
    "image.inspect",
    "video.generate",
    "video.image_to_video",
    "video.audio_to_video",
    "video.first_last_frames",
    "video.continue",
    "video.enhance",
    "video.upscale",
    "video.interpolate",
    "video.color_grade",
    "video.lipsync",
    "video.motion_transfer",
    "video.extract_frame",
    "video.inspect",
    "video.mask",
    "audio.tts",
    "audio.sfx",
    "audio.ambience",
    "audio.foley",
    "audio.music",
    "audio.mix",
    "audio.normalize",
    "audio.inspect",
    "media.concat",
    "media.finalize",
    "qc.image",
    "qc.video",
    "qc.audio",
    "qc.media",
    "text.chat",
}

IMPLEMENTED = {
    "image.generate",
    "image.edit",
    "image.controlled",
    "image.layered",
    "image.upscale",
    "video.generate",
    "video.image_to_video",
    "video.audio_to_video",
    "video.first_last_frames",
    "video.lipsync",
    "video.motion_transfer",
    "audio.tts",
    "audio.sfx",
    "audio.ambience",
    "audio.foley",
    "audio.music",
    "audio.mix",
    "audio.normalize",
    "audio.inspect",
    "media.concat",
    "media.finalize",
    "text.chat",
}

STUB_REQUIRED = {
    "audio.tts": "text",
    "audio.music": "prompt",
    "audio.sfx": "prompt",
    "audio.ambience": "prompt",
    "audio.foley": "prompt",
    "audio.mix": "stems",
    "audio.normalize": "audio",
    "audio.inspect": "audio",
    "video.continue": "source_video",
    "video.enhance": "video",
    "video.upscale": "video",
    "video.interpolate": "video",
    "video.color_grade": "video",
    "video.extract_frame": "video",
    "video.inspect": "video",
    "video.mask": "video",
    "image.mask": "image",
    "image.depth": "image",
    "image.inspect": "image",
    "media.concat": "clips",
    "media.finalize": "video",
    "qc.image": "image",
    "qc.video": "video",
    "qc.audio": "audio",
    "qc.media": "video",
}

PHASE_A_WORKFLOWS = {
    "audio-tts": ("audio.tts", "tts_generate", "tts"),
    "audio-music": ("audio.music", "ace_step_music", "comfyui"),
    "audio-sfx": ("audio.sfx", "ace_step_sfx", "comfyui"),
    "audio-ambience": ("audio.ambience", "ace_step_ambience", "comfyui"),
    "audio-foley": ("audio.foley", "ace_step_foley", "comfyui"),
    "audio-mix": ("audio.mix", "ffmpeg_mix", "ffmpeg"),
    "audio-normalize": ("audio.normalize", "ffmpeg_normalize", "ffmpeg"),
    "audio-inspect": ("audio.inspect", "ffmpeg_inspect", "ffmpeg"),
    "media-concat": ("media.concat", "ffmpeg_concat", "ffmpeg"),
    "media-finalize": ("media.finalize", "ffmpeg_finalize", "ffmpeg"),
}

STUB_WORKFLOWS = {
    "video-continue": "video.continue",
    "video-enhance": "video.enhance",
    "video-upscale": "video.upscale",
    "video-interpolate": "video.interpolate",
}


def test_catalogs_load() -> None:
    catalog = load_catalogs(REPO / "catalog")
    assert "qwen-image-2512-fp8" in catalog.models
    assert "qwen-lightning" in catalog.loras
    assert "master" in catalog.presets
    assert "qwen-txt2img" in catalog.workflows
    assert "image.generate" in catalog.operations
    assert catalog.loras["ltx-iclora-union"].compatibility_test_required is True
    assert catalog.groups["qwen_speed"].max_active == 1
    assert "ltx-camera-loras" not in catalog.models
    assert "ltx-iclora-detailer" not in catalog.models
    assert "ltx-iclora-lipdub" in catalog.models
    assert "ltx-iclora-motion-track" in catalog.models
    assert "ltx-iclora-lipdub" in catalog.loras
    assert "ltx25-lipsync" in catalog.workflows
    assert catalog.operations["video.lipsync"].implemented is True
    assert catalog.operations["video.motion_transfer"].implemented is True
    assert catalog.operations["video.audio_to_video"].implemented is True
    assert catalog.operations["video.first_last_frames"].implemented is True
    assert "qwen-image-layered" in catalog.models
    assert "qwen-edit" in catalog.workflows
    assert "qwen-control" in catalog.workflows
    assert "qwen-layered" in catalog.workflows
    assert catalog.operations["image.edit"].implemented is True
    assert catalog.operations["image.controlled"].implemented is True
    assert catalog.operations["image.layered"].implemented is True
    assert set(catalog.operations) == INVENTORY
    for op_id, op in catalog.operations.items():
        assert op.input_schema, op_id
        assert op.output_schema, op_id
        assert op.implemented is (op_id in IMPLEMENTED)
    for op_id, field in STUB_REQUIRED.items():
        assert field in (catalog.operations[op_id].input_schema.get("required") or []), op_id
    for wf_id, operation in STUB_WORKFLOWS.items():
        assert catalog.workflows[wf_id].operation == operation
        assert catalog.workflows[wf_id].builder == ""
    for wf_id, (operation, builder, executor) in PHASE_A_WORKFLOWS.items():
        wf = catalog.workflows[wf_id]
        assert wf.operation == operation
        assert wf.builder == builder
        assert wf.executor == executor
    assert catalog.models["chatterbox-multilingual"].license == "MIT"
    assert catalog.models["chatterbox-hi"].license == "MIT"
    assert catalog.models["ace-step-1.5"].license == "Apache-2.0"
    assert catalog.models["chatterbox-multilingual"].commercial is True
    assert catalog.models["ace-step-1.5"].commercial is True
    assert "narrator_hindi" in catalog.presets
    assert "dialogue_hindi" in catalog.presets
    assert "cinematic_master" in catalog.presets
    assert "cinematic" in catalog.presets
    assert "audio.tts" in catalog.presets["master"].operations
    assert "audio.music" in catalog.presets["master"].operations


def test_text_model_catalog() -> None:
    catalog = load_catalogs(REPO / "catalog")
    assert "gemma-4-e4b" in catalog.models
    assert "qwen36-35b-a3b-rq" in catalog.models
    assert "qwen38-27b-fast" not in catalog.models
    assert "qwen38-27b-slow" not in catalog.models
    gemma = catalog.models["gemma-4-e4b"]
    qwen = catalog.models["qwen36-35b-a3b-rq"]
    assert gemma.supported_context == 131072
    assert "vision" in gemma.modalities
    assert "text.chat" in gemma.supported_operations
    assert qwen.supported_context == 262144
    assert qwen.context_max_yarn == 1010000
    assert qwen.license == "Apache-2.0"
    assert any("Qwen3.6-35B-A3B-Q4_K_M.gguf" == item.local for item in qwen.files)
    assert any("mmproj" in item.local for item in qwen.files)


def test_ltx_plan_passes_audio_prompt_and_lipsync_lora() -> None:
    from app.domain.jobs import Job, JobStatus
    from app.domain.presets import Preset
    from app.infrastructure.comfyui.plans import _ltx

    job = Job(
        job_id="job_test",
        operation="video.lipsync",
        preset="balanced",
        status=JobStatus.QUEUED,
        phase="queued",
        progress=0.0,
        inputs={"prompt": "she speaks", "audio_prompt": "clear speech", "duration_seconds": 4},
        parameters={"seed": 1},
        client_context={},
        created_at="2026-01-01T00:00:00Z",
    )
    preset = Preset(id="balanced", label="Balanced", description="", operations=["video.lipsync"], parameters={"steps": 20})
    plan = _ltx(job, preset, "ltx_lipsync")
    assert plan["mode"] == "lipsync"
    assert plan["audio_prompt"] == "clear speech"
    assert plan["ic_enabled"] is True
    assert plan["ic_loras"][0]["id"] == "lipdub"
