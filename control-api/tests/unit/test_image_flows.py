from __future__ import annotations

import pytest

from app.application.image_flows import (
    DEFAULT_CHARACTER_DENOISE,
    DEFAULT_IMG2IMG_DENOISE,
    FACE_LOCK_NEGATIVE,
    FACE_LOCK_PREFIX,
    FlowError,
    compose_face_lock_prompts,
    resolve_flow_plan,
)
from app.generate import (
    GenerateError,
    _control_aux_ready,
    _error_from_items,
    _needs_llm_planning,
    _normalize_items,
    _plan_item,
)
from app.qwen_graph import (
    build_control,
    build_edit,
    build_layered,
    build_txt2img,
    format_controlnet_comfy_error,
    select_control_preprocessor,
)


def test_t2i_defaults() -> None:
    plan = resolve_flow_plan("t2i", prompt="a lantern")
    assert plan["mode"] == "txt2img"
    assert plan["steps"] == 30
    assert plan["cfg"] == 4.0
    assert plan["loras"] == []


def test_lightning_locks_cfg_and_lora() -> None:
    plan = resolve_flow_plan("lightning", prompt="quick sketch")
    assert plan["steps"] == 4
    assert plan["cfg"] == 1.0
    assert plan["loras"][0]["name"].startswith("Qwen-Image-2512-Lightning")


def test_merge_requires_two_images() -> None:
    with pytest.raises(FlowError, match="at least 2"):
        resolve_flow_plan("merge", prompt="combine", image_count=1)
    plan = resolve_flow_plan("merge", prompt="combine", image_count=2)
    assert plan["mode"] == "edit"


def test_layered_bounds_and_optional_prompt() -> None:
    plan = resolve_flow_plan("layered", image_count=1, layers=3)
    assert plan["mode"] == "layered"
    assert plan["layers"] == 3
    assert plan["width"] == 640
    with pytest.raises(FlowError, match="1-8"):
        resolve_flow_plan("layered", image_count=1, layers=9)
    with pytest.raises(FlowError, match="at least 1"):
        resolve_flow_plan("layered", image_count=0)


def test_control_requires_type() -> None:
    with pytest.raises(FlowError, match="pose, depth, or canny"):
        resolve_flow_plan("control", prompt="pose hold", image_count=1, control_type="scribble")
    plan = resolve_flow_plan("control", prompt="pose hold", image_count=1, control_type="depth")
    assert plan["control_type"] == "depth"
    assert plan["control_strength"] == 0.85


def test_legacy_flow_skips_planner() -> None:
    items = _normalize_items(
        prompts=None,
        prompt="a river at dusk",
        image=None,
        plan=None,
        seed=1,
        width=1920,
        height=1080,
        count=1,
        flow="t2i",
    )
    assert _needs_llm_planning(items) is False
    plan = _plan_item(items[0])
    assert plan["flow"] == "t2i"
    assert plan["mode"] == "txt2img"
    assert plan["steps"] == 30


def test_build_edit_one_vs_three_images() -> None:
    one = build_edit({"prompt": "fix the sign", "image_name": "a.png", "seed": 1})
    types = [n["class_type"] for n in one.values()]
    assert types.count("LoadImage") == 1
    assert "TextEncodeQwenImageEdit" in types
    assert "TextEncodeQwenImageEditPlus" not in types

    three = build_edit(
        {
            "prompt": "merge these",
            "image_names": ["a.png", "b.png", "c.png"],
            "seed": 1,
        }
    )
    types3 = [n["class_type"] for n in three.values()]
    assert types3.count("LoadImage") == 3
    assert "TextEncodeQwenImageEditPlus" in types3
    encoder = next(n for n in three.values() if n["class_type"] == "TextEncodeQwenImageEditPlus")
    assert "image2" in encoder["inputs"]
    assert "image3" in encoder["inputs"]


def test_build_layered_nodes() -> None:
    graph = build_layered({"prompt": "", "image_name": "in.png", "layers": 4, "seed": 2})
    types = {n["class_type"] for n in graph.values()}
    assert "EmptyQwenImageLayeredLatentImage" in types
    assert "LatentCut" in types
    assert "LatentCutToBatch" in types
    latent = next(n for n in graph.values() if n["class_type"] == "EmptyQwenImageLayeredLatentImage")
    assert latent["inputs"]["layers"] == 4


def test_img2img_requires_image_and_defaults_denoise() -> None:
    with pytest.raises(FlowError, match="at least 1"):
        resolve_flow_plan("img2img", prompt="same person", image_count=0)
    plan = resolve_flow_plan("img2img", prompt="same person", image_count=1)
    assert plan["mode"] == "txt2img"
    assert plan["denoise"] == DEFAULT_IMG2IMG_DENOISE
    custom = resolve_flow_plan("img2img", prompt="same person", image_count=1, denoise=0.4)
    assert custom["denoise"] == 0.4
    with pytest.raises(FlowError, match="denoise"):
        resolve_flow_plan("img2img", prompt="same person", image_count=1, denoise=1.5)


def test_img2img_plan_item_skips_planner() -> None:
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    items = _normalize_items(
        prompts=None,
        prompt="same officer, new lighting",
        image=image,
        plan=None,
        seed=3,
        width=1080,
        height=1920,
        count=1,
        flow="img2img",
        denoise=0.5,
    )
    assert _needs_llm_planning(items) is False
    plan = _plan_item(items[0])
    assert plan["flow"] == "img2img"
    assert plan["denoise"] == 0.5
    with pytest.raises(GenerateError, match="at least 1"):
        _plan_item(
            _normalize_items(
                prompts=None,
                prompt="same officer",
                image=None,
                plan=None,
                seed=1,
                width=1080,
                height=1920,
                count=1,
                flow="img2img",
            )[0]
        )


def test_img2img_graph_encodes_reference() -> None:
    graph = build_txt2img(
        {
            "prompt": "same person, evening light",
            "image_name": "ref.png",
            "width": 1920,
            "height": 1080,
            "seed": 1,
            "denoise": 0.65,
        }
    )
    types = [n["class_type"] for n in graph.values()]
    assert "VAEEncode" in types
    assert "LoadImage" in types
    assert "EmptySD3LatentImage" not in types
    sampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["denoise"] == 0.65


def test_compose_face_lock_preserves_description() -> None:
    positive, negative = compose_face_lock_prompts(
        "Full-body still of the same man in steel armor, dusk courtyard",
        "blurry",
    )
    assert FACE_LOCK_PREFIX in positive
    assert "steel armor, dusk courtyard" in positive
    assert FACE_LOCK_NEGATIVE in negative
    assert "blurry" in negative


def test_character_requires_image_prompt_and_defaults_denoise() -> None:
    with pytest.raises(FlowError, match="prompt is required"):
        resolve_flow_plan("character", prompt="", image_count=1)
    with pytest.raises(FlowError, match="at least 1"):
        resolve_flow_plan("character", prompt="same man in armor", image_count=0)
    plan = resolve_flow_plan("character", prompt="same man in steel armor, dusk courtyard", image_count=1)
    assert plan["mode"] == "txt2img"
    assert plan["flow"] == "character"
    assert plan["denoise"] == DEFAULT_CHARACTER_DENOISE
    assert FACE_LOCK_PREFIX in plan["prompt"]
    assert "steel armor, dusk courtyard" in plan["prompt"]
    assert FACE_LOCK_NEGATIVE in plan["negative_prompt"]
    custom = resolve_flow_plan(
        "character",
        prompt="same man in armor",
        image_count=1,
        denoise=0.3,
    )
    assert custom["denoise"] == 0.3
    with pytest.raises(FlowError, match="denoise"):
        resolve_flow_plan("character", prompt="same man", image_count=1, denoise=1.5)


def test_character_plan_item_skips_planner() -> None:
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    items = _normalize_items(
        prompts=None,
        prompt="same man as a cavalry officer, dusk courtyard",
        image=image,
        plan=None,
        seed=3,
        width=1080,
        height=1920,
        count=1,
        flow="character",
    )
    assert _needs_llm_planning(items) is False
    plan = _plan_item(items[0])
    assert plan["flow"] == "character"
    assert plan["denoise"] == DEFAULT_CHARACTER_DENOISE
    with pytest.raises(GenerateError, match="at least 1"):
        _plan_item(
            _normalize_items(
                prompts=None,
                prompt="same man in armor",
                image=None,
                plan=None,
                seed=1,
                width=1080,
                height=1920,
                count=1,
                flow="character",
            )[0]
        )


def test_character_graph_encodes_reference_not_edit() -> None:
    plan = resolve_flow_plan("character", prompt="same man in steel armor", image_count=1, seed=1)
    plan["image_name"] = "harish.png"
    graph = build_txt2img(plan)
    types = [n["class_type"] for n in graph.values()]
    assert "VAEEncode" in types
    assert "LoadImage" in types
    assert "EmptySD3LatentImage" not in types
    assert "TextEncodeQwenImageEdit" not in types
    sampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["denoise"] == DEFAULT_CHARACTER_DENOISE


def test_select_control_preprocessor_falls_back_to_openpose() -> None:
    assert select_control_preprocessor("pose") == "DWPreprocessor"
    assert select_control_preprocessor("pose", {"OpenposePreprocessor", "Canny"}) == "OpenposePreprocessor"
    assert select_control_preprocessor("depth", {"MiDaS Depth Approximation"}) == "MiDaS Depth Approximation"
    with pytest.raises(ValueError, match="ComfyUI-ltx"):
        select_control_preprocessor("pose", {"Canny"})


def test_error_from_items_copies_first_failure() -> None:
    items = [
        {"status": "failed", "error": "Control preprocessor nodes are missing on ComfyUI"},
        {"status": "pending", "error": None},
    ]
    assert _error_from_items(items, completed=0) == "Control preprocessor nodes are missing on ComfyUI"
    items[0]["status"] = "completed"
    items[0]["error"] = None
    items[1] = {"status": "completed", "error": None}
    assert _error_from_items(items, completed=2) is None


def test_build_control_uses_selected_preprocessor() -> None:
    graph = build_control(
        {
            "prompt": "match this pose",
            "image_name": "guide.png",
            "width": 1920,
            "height": 1080,
            "seed": 1,
            "control_type": "pose",
            "preprocessor_node": "OpenposePreprocessor",
        }
    )
    types = [n["class_type"] for n in graph.values()]
    assert "OpenposePreprocessor" in types
    assert "DWPreprocessor" not in types
    assert "ControlNetLoader" in types
    assert "ControlNetApplyAdvanced" in types


def test_format_controlnet_load_error() -> None:
    msg = format_controlnet_comfy_error("ControlNetLoader failed: Fun ControlNet")
    assert "12359" in msg
    assert format_controlnet_comfy_error("unrelated sampler error") == "unrelated sampler error"


def test_control_aux_ready_requires_pose_or_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.generate.comfy_client.is_ready", lambda: True)
    monkeypatch.setattr("app.generate._available_comfy_nodes", lambda: {"Canny", "KSampler"})
    assert _control_aux_ready() is False
    monkeypatch.setattr("app.generate._available_comfy_nodes", lambda: {"OpenposePreprocessor", "Canny"})
    assert _control_aux_ready() is True
    monkeypatch.setattr("app.generate.comfy_client.is_ready", lambda: False)
    assert _control_aux_ready() is True
