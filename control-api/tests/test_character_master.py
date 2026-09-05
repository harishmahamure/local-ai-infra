"""WF_01 Character Master: presets, graph lock, validation, metadata, select."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config, presets, qwen_graph  # noqa: E402
from app.workflows import assets, character_master, types  # noqa: E402


def _lora_names(graph: dict) -> list[str]:
    names = []
    for node in graph.values():
        if node.get("class_type") == "LoraLoaderModelOnly":
            names.append(node["inputs"]["lora_name"])
    return names


def _class_types(graph: dict) -> list[str]:
    return [node["class_type"] for node in graph.values()]


def _unet_name(graph: dict) -> str | None:
    for node in graph.values():
        if node.get("class_type") == "UNETLoader":
            return node["inputs"]["unet_name"]
    return None


def test_master_preset_has_no_lightning_or_turbo() -> None:
    preset = presets.get_preset("character_master_master")
    assert preset is not None
    names = [l.name.lower() for l in preset.loras]
    assert names == []
    assert not any("lightning" in n or "turbo" in n for n in names)
    assert preset.steps == 30
    assert preset.cfg == 4.0
    assert preset.bundles == ["qwen-image-2512-fp8"]


def test_draft_preset_uses_lightning() -> None:
    preset = presets.get_preset("character_master_draft")
    assert preset is not None
    names = [l.name for l in preset.loras]
    assert presets.LORA["lightning"] in names
    assert preset.steps == 4
    assert preset.cfg == 1.0


def test_master_graph_uses_qwen_2512_without_lora() -> None:
    plan = character_master.build_locked_plan(
        quality="master",
        prompt="a Rajput warrior, 17th century, weathered face",
        width=1080,
        height=1920,
        seed=7,
    )
    graph = qwen_graph.build(plan)
    assert _unet_name(graph) == qwen_graph.M["qwen_unet"]
    assert "LoraLoaderModelOnly" not in _class_types(graph)
    prefixes = [
        node["inputs"]["filename_prefix"]
        for node in graph.values()
        if node.get("class_type") == "SaveImage"
    ]
    assert prefixes == ["wf01_character_master"]
    clip = next(n for n in graph.values() if n["class_type"] == "CLIPLoader")
    assert clip["inputs"]["clip_name"] == qwen_graph.M["qwen_clip"]
    assert clip["inputs"]["type"] == "qwen_image"
    vae = next(n for n in graph.values() if n["class_type"] == "VAELoader")
    assert vae["inputs"]["vae_name"] == qwen_graph.M["qwen_vae"]


def test_draft_graph_includes_lightning_lora() -> None:
    plan = character_master.build_locked_plan(
        quality="draft",
        prompt="a Rajput warrior",
        width=1080,
        height=1920,
        seed=3,
    )
    graph = qwen_graph.build(plan)
    assert _lora_names(graph) == [presets.LORA["lightning"]]


def test_optional_realism_adds_only_that_lora() -> None:
    plan = character_master.build_locked_plan(
        quality="master",
        prompt="a Rajput warrior",
        width=1080,
        height=1920,
        realism_lora=0.4,
        seed=11,
    )
    graph = qwen_graph.build(plan)
    assert _lora_names(graph) == [presets.LORA["realism"]]
    assert "qwen-lora-realism" in plan["bundles"]


def test_master_rejects_lightning_lora() -> None:
    plan = character_master.build_locked_plan(
        quality="master",
        prompt="a warrior",
        width=1080,
        height=1920,
    )
    plan["loras"] = [{"name": presets.LORA["lightning"], "strength": 1.0}]
    try:
        character_master.assert_master_loras_allowed(plan, quality="master")
    except character_master.WorkflowError as exc:
        assert exc.code == "MASTER_LORA_FORBIDDEN"
        assert exc.status == 422
    else:
        raise AssertionError("expected MASTER_LORA_FORBIDDEN")


def test_validate_rejects_empty_description() -> None:
    try:
        character_master.validate_request({"characterDescription": "  "})
    except character_master.WorkflowError as exc:
        assert exc.code == "VALIDATION_ERROR"
        assert exc.status == 422
    else:
        raise AssertionError("expected VALIDATION_ERROR")


def test_validate_defaults_and_aspect() -> None:
    req = character_master.validate_request(
        {
            "characterDescription": "Maratha cavalry officer, 40s, short beard",
            "quality": "master",
        }
    )
    assert req["candidateCount"] == 4
    assert req["width"] == 1080
    assert req["height"] == 1920
    assert req["realismLora"] is None
    assert req["quality"] == "master"
    assert "identity lock sheet" in req["userPrompt"]


def test_validate_draft_candidate_default() -> None:
    req = character_master.validate_request(
        {"characterDescription": "a merchant", "quality": "draft"}
    )
    assert req["candidateCount"] == 2


def test_metadata_schema_contains_required_keys() -> None:
    meta = types.generation_metadata(
        job_id="job-1",
        quality="master",
        model_ids=["qwen-image-2512-fp8"],
        model_hashes={qwen_graph.M["qwen_unet"]: None},
        lora_ids=[],
        lora_weights=[],
        seed=42,
        prompt="a warrior",
        negative_prompt="blurry",
        width=1080,
        height=1920,
        steps=30,
        cfg=4.0,
        denoise=1.0,
        character_id="char-1",
        aspect_ratio="9:16",
    )
    for key in types.METADATA_KEYS:
        assert key in meta
    assert meta["workflow_id"] == "WF_01_CHARACTER_MASTER"
    assert meta["parent_shot"] is None
    assert meta["start_frame_id"] is None
    assert meta["end_frame_id"] is None
    assert meta["sampler"] == "euler"
    assert meta["scheduler"] == "simple"


def test_select_promotes_chosen_candidate() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        original = (character_master.JOBS_PATH, character_master.JOBS_ROOT, config.LOGS)
        character_master.JOBS_PATH = tmp_path / "character-master-jobs.json"
        character_master.JOBS_ROOT = tmp_path / "jobs"
        config.LOGS = tmp_path
        try:
            job_id = "job-select-1"
            character_id = "hero_01"
            images_dir = tmp_path / "jobs" / job_id / "images"
            images_dir.mkdir(parents=True)
            candidate = images_dir / "0000_00_wf01.png"
            candidate.write_bytes(b"fake-png")

            metadata = types.generation_metadata(
                job_id=job_id,
                quality="master",
                model_ids=["qwen-image-2512-fp8"],
                model_hashes={},
                lora_ids=[],
                lora_weights=[],
                seed=9,
                prompt="a warrior",
                negative_prompt="",
                width=1080,
                height=1920,
                steps=30,
                cfg=4.0,
                denoise=1.0,
                character_id=character_id,
                aspect_ratio="9:16",
            )
            job = {
                "jobId": job_id,
                "characterId": character_id,
                "status": "completed",
                "items": [
                    {
                        "index": 0,
                        "status": "completed",
                        "seed": 9,
                        "images": [
                            {
                                "filename": candidate.name,
                                "url": f"/api/v1/workflows/character-master/{job_id}/images/{candidate.name}",
                            }
                        ],
                    }
                ],
                "metadata": metadata,
                "selected": None,
            }
            (tmp_path / "character-master-jobs.json").write_text(json.dumps({"jobs": {job_id: job}}) + "\n")

            result = character_master.select_candidate(job_id, 0)
            assert result["selected"]["index"] == 0
            assert result["selected"]["assetId"] == character_id
            master = tmp_path / "assets" / "characters" / character_id / "CHARACTER_MASTER.png"
            assert master.is_file()
            assert master.read_bytes() == b"fake-png"
            stored = assets.load_metadata(character_id, root=tmp_path)
            assert stored is not None
            assert stored["selected_candidate_index"] == 0
        finally:
            character_master.JOBS_PATH, character_master.JOBS_ROOT, config.LOGS = original


def test_select_missing_candidate() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        original = (character_master.JOBS_PATH, character_master.JOBS_ROOT)
        character_master.JOBS_PATH = tmp_path / "character-master-jobs.json"
        character_master.JOBS_ROOT = tmp_path / "jobs"
        try:
            job_id = "job-missing"
            (tmp_path / "character-master-jobs.json").write_text(
                json.dumps(
                    {
                        "jobs": {
                            job_id: {
                                "jobId": job_id,
                                "characterId": "x",
                                "items": [],
                            }
                        }
                    }
                )
                + "\n"
            )
            try:
                character_master.select_candidate(job_id, 0)
            except character_master.WorkflowError as exc:
                assert exc.code == "CANDIDATE_NOT_FOUND"
            else:
                raise AssertionError("expected CANDIDATE_NOT_FOUND")
        finally:
            character_master.JOBS_PATH, character_master.JOBS_ROOT = original


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
