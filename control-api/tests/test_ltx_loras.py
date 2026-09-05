"""Decision-node and graph wiring checks for LTX LoRAs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import ltx_graph, ltx_loras  # noqa: E402


def test_camera_auto_from_prompt() -> None:
    d = ltx_loras.decide("slow dolly forward through the alley", camera_motion="auto")
    assert d["camera_motion"] == "dolly_in"
    assert d["camera_motion_source"] == "prompt"
    assert d["loras"][0]["name"].endswith("dolly-in.safetensors")
    assert d["ic_enabled"] is False
    assert d["ic_skip"] == "reference video required" or d["ic_source"] in ("skipped", "none")


def test_camera_none_overrides_prompt() -> None:
    d = ltx_loras.decide("slow dolly forward", camera_motion="none")
    assert d["camera_motion"] is None
    assert d["loras"] == []
    assert d["camera_motion_source"] == "override"


def test_camera_explicit_jib() -> None:
    d = ltx_loras.decide("a quiet room", camera_motion="jib_up")
    assert d["camera_motion"] == "jib_up"
    assert d["camera_motion_source"] == "override"


def test_ic_skipped_without_video() -> None:
    d = ltx_loras.decide("dance in a hall", ic_lora="union", has_reference_video=False)
    assert d["ic_enabled"] is False
    assert d["ic_loras"] == []
    assert d["ic_skip"] == "reference video required"


def test_ic_union_pose_from_prompt() -> None:
    d = ltx_loras.decide(
        "a dancer walking across the stage",
        ic_lora="auto",
        has_reference_video=True,
    )
    assert d["ic_enabled"] is True
    assert d["ic_lora"] == "union"
    assert d["control_type"] == "pose"


def test_detailer_from_flag() -> None:
    d = ltx_loras.decide(
        "a rainy street",
        detailer=True,
        has_reference_video=True,
    )
    ids = [x["id"] for x in d["ic_loras"]]
    assert "detailer" in ids


def test_graph_camera_lora_only() -> None:
    graph = ltx_graph.build_t2v(
        {
            "width": 704,
            "height": 1216,
            "length": 97,
            "prompt": "slow dolly forward",
            "loras": [
                {
                    "name": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
                    "strength": 1.0,
                }
            ],
        }
    )
    types = [n["class_type"] for n in graph.values()]
    assert "LoraLoaderModelOnly" in types
    assert "LTXAddVideoICLoRAGuide" not in types
    assert "LTXICLoRALoaderModelOnly" not in types
    guider = next(n for n in graph.values() if n["class_type"] == "LTXVDualCFGGuider")
    loader = next(n for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly")
    assert guider["inputs"]["model"][0] != next(
        nid for nid, n in graph.items() if n["class_type"] == "UNETLoader"
    )
    assert loader["inputs"]["lora_name"].endswith("dolly-in.safetensors")


def test_graph_no_lora_without_plan() -> None:
    graph = ltx_graph.build_t2v(
        {"width": 704, "height": 1216, "length": 97, "prompt": "a quiet room"}
    )
    types = [n["class_type"] for n in graph.values()]
    assert "LoraLoaderModelOnly" not in types
    assert "LTXAddVideoICLoRAGuide" not in types


def _types(graph: dict) -> list[str]:
    return [n["class_type"] for n in graph.values()]


def test_graph_i2v_refine_reinjects_start() -> None:
    graph = ltx_graph.build(
        {
            "mode": "i2v",
            "width": 704,
            "height": 1216,
            "length": 97,
            "prompt": "the woman turns",
            "image_name": "start.png",
            "refine": True,
        }
    )
    types = _types(graph)
    assert types.count("LTXVImgToVideoInplace") == 2
    assert "LTXVPreprocess" in types
    assert "LTXVAudioVAEEncode" not in types
    assert "LTXVEmptyLatentAudio" in types


def test_graph_a2v_encodes_and_remuxes_audio() -> None:
    graph = ltx_graph.build(
        {
            "mode": "a2v",
            "width": 704,
            "height": 1216,
            "length": 97,
            "prompt": "a talking portrait",
            "audio_name": "speech.wav",
        }
    )
    types = _types(graph)
    assert "LoadAudio" in types
    assert "LTXVAudioVAEEncode" in types
    assert "LTXVSetAudioRefTokens" in types
    assert "LTXVEmptyLatentAudio" not in types
    assert "LTXVAudioVAEDecode" not in types
    create = next(n for n in graph.values() if n["class_type"] == "CreateVideo")
    load = next(nid for nid, n in graph.items() if n["class_type"] == "LoadAudio")
    assert create["inputs"]["audio"][0] == load


def test_graph_flf2v_guides() -> None:
    graph = ltx_graph.build(
        {
            "mode": "flf2v",
            "width": 768,
            "height": 512,
            "length": 97,
            "prompt": "the hand closes",
            "image_name": "start.png",
            "end_image_name": "end.png",
            "refine": False,
        }
    )
    types = _types(graph)
    assert types.count("LTXVAddGuide") == 1 or types.count("LTXVAddGuide") == 2
    guides = [n for n in graph.values() if n["class_type"] == "LTXVAddGuide"]
    assert any(n["inputs"]["frame_idx"] == -1 for n in guides)
    assert "LTXVCropGuides" in types


def test_graph_lipsync_fml() -> None:
    graph = ltx_graph.build(
        {
            "mode": "lipsync",
            "width": 704,
            "height": 1216,
            "length": 97,
            "prompt": "she speaks to camera",
            "image_name": "start.png",
            "end_image_name": "end.png",
            "middle_image_name": "mid.png",
            "audio_name": "speech.wav",
            "ic_loras": [ltx_loras.explicit_ic_lora("lipdub")],
            "ic_enabled": True,
        }
    )
    types = _types(graph)
    assert "LTXVAudioVAEEncode" in types
    assert "LTXICLoRALoaderModelOnly" in types
    assert "LTXAddVideoICLoRAGuide" in types
    assert types.count("LTXVAddGuide") == 2
    loader = next(n for n in graph.values() if n["class_type"] == "LTXICLoRALoaderModelOnly")
    assert "lipdub" in loader["inputs"]["lora_name"]


def test_graph_motion_transfer_raw_frames() -> None:
    graph = ltx_graph.build(
        {
            "mode": "motion_transfer",
            "width": 960,
            "height": 544,
            "length": 97,
            "prompt": "the object follows the path",
            "video_name": "ref.mp4",
            "image_name": "start.png",
            "ic_loras": [ltx_loras.explicit_ic_lora("motion_track")],
            "ic_enabled": True,
            "ic_guide_raw": True,
        }
    )
    types = _types(graph)
    assert "LoadVideo" in types
    assert "GetVideoComponents" in types
    assert "VideoDepthAnythingProcess" not in types
    assert "LTXAddVideoICLoRAGuide" in types
    loader = next(n for n in graph.values() if n["class_type"] == "LTXICLoRALoaderModelOnly")
    assert "motion-track" in loader["inputs"]["lora_name"]


def test_graph_ic_union_depth() -> None:
    graph = ltx_graph.build_t2v(
        {
            "width": 704,
            "height": 1216,
            "length": 97,
            "prompt": "a rainy street",
            "ic_enabled": True,
            "video_name": "ref.mp4",
            "control_type": "depth",
            "ic_loras": [
                {
                    "name": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
                    "strength": 1.0,
                }
            ],
        }
    )
    types = [n["class_type"] for n in graph.values()]
    for required in (
        "LoadVideo",
        "GetVideoComponents",
        "LoadVideoDepthAnythingModel",
        "VideoDepthAnythingProcess",
        "LTXICLoRALoaderModelOnly",
        "LTXAddVideoICLoRAGuide",
        "LTXVCropGuides",
    ):
        assert required in types, f"missing {required}"


if __name__ == "__main__":
    checks = [
        test_camera_auto_from_prompt,
        test_camera_none_overrides_prompt,
        test_camera_explicit_jib,
        test_ic_skipped_without_video,
        test_ic_union_pose_from_prompt,
        test_detailer_from_flag,
        test_graph_camera_lora_only,
        test_graph_no_lora_without_plan,
        test_graph_i2v_refine_reinjects_start,
        test_graph_a2v_encodes_and_remuxes_audio,
        test_graph_flf2v_guides,
        test_graph_lipsync_fml,
        test_graph_motion_transfer_raw_frames,
        test_graph_ic_union_depth,
    ]
    for fn in checks:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(checks)} checks passed")
