from __future__ import annotations

import pytest

from app.application.image_flows import FlowError, resolve_flow_plan
from app.generate import _needs_llm_planning, _normalize_items, _plan_item
from app.qwen_graph import build_edit, build_layered


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
