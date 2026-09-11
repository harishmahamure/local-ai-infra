from __future__ import annotations

from typing import Any

from .graphs import GraphBuilder, MODELS


def _load_frames(gb: GraphBuilder, name: str) -> tuple[str, str]:
    video = gb.add("LoadVideo", {"file": name})
    parts = gb.add("GetVideoComponents", {"video": gb.ref(video)})
    return parts, video


def _batch(gb: GraphBuilder, images: list[str]) -> str:
    current = images[0]
    for extra in images[1:]:
        current = gb.add("ImageBatch", {"image1": gb.ref(current), "image2": gb.ref(extra)})
    return current


def build(plan: dict[str, Any]) -> dict[str, Any]:
    names = list(plan.get("video_names") or [])
    if len(names) < 1:
        raise ValueError("post graph requires at least one video")
    fps = float(plan.get("fps") or 24)
    post = plan.get("post") if isinstance(plan.get("post"), dict) else {}
    gb = GraphBuilder()
    parts: list[str] = []
    audio = None
    for name in names:
        frames, _video = _load_frames(gb, name)
        parts.append(frames)
        if audio is None:
            audio = frames
    images = _batch(gb, parts)
    if post.get("upscale"):
        scale = gb.add("UpscaleModelLoader", {"model_name": MODELS["upscale_x2"]})
        images = gb.add("ImageUpscaleWithModel", {"upscale_model": gb.ref(scale), "image": gb.ref(images)})
    video_inputs: dict[str, Any] = {"images": gb.ref(images), "fps": fps}
    if audio is not None:
        video_inputs["audio"] = gb.ref(audio, 1)
    video = gb.add("CreateVideo", video_inputs)
    gb.add(
        "SaveVideo",
        {
            "video": gb.ref(video),
            "filename_prefix": str(plan.get("filename_prefix") or "shot_stitch"),
            "format": "mp4",
            "codec": "h264",
        },
    )
    return gb.build()
