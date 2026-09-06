from __future__ import annotations

from typing import Any

from ..domain.operations import Operation

_PROMPT = {"type": "string", "maxLength": 8000}
_ASSET = {"type": "object", "required": ["asset_id"], "properties": {"asset_id": {"type": "string"}}}
_SEED = {"type": "integer", "minimum": 0, "maximum": 2147483647}
_STEPS = {"type": "integer", "minimum": 1, "maximum": 80}
_DURATION_SHOT = {"type": "number", "minimum": 1, "maximum": 30}
_DURATION_STEM = {"type": "number", "minimum": 0.25, "maximum": 180}
_ASPECT = {"type": "string"}

_IMAGE_OUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "asset_ids": {"type": "array", "items": {"type": "string"}},
        "kind": {"type": "string", "const": "image"},
    },
}
_VIDEO_OUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "asset_ids": {"type": "array", "items": {"type": "string"}},
        "kind": {"type": "string", "const": "video"},
        "duration_seconds": {"type": "number"},
    },
}
_AUDIO_OUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "asset_ids": {"type": "array", "items": {"type": "string"}},
        "kind": {"type": "string", "const": "audio"},
        "duration_seconds": {"type": "number"},
        "sample_rate": {"type": "integer"},
    },
}
_INSPECT_OUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "duration_seconds": {"type": "number"},
        "sample_rate": {"type": "integer"},
        "codec": {"type": "string"},
        "loudness_lufs": {"type": "number"},
    },
}
_QC_OUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "score": {"type": "number"},
        "checks": {"type": "array", "items": {"type": "object"}},
    },
}
_SEED_STEPS = {"properties": {"seed": _SEED, "steps": _STEPS}}
_SEED_ONLY = {"properties": {"seed": _SEED}}

IMAGE_PRESETS = ["draft", "balanced", "master", "character_master"]
VIDEO_PRESETS = ["draft", "balanced", "master"]
GENERIC = ["draft", "balanced", "master"]
TTS_PRESETS = ["narrator_hindi", "dialogue_hindi", "draft", "master"]
MUSIC_PRESETS = ["draft", "balanced", "master", "cinematic_master"]
MIX_PRESETS = ["draft", "balanced", "master", "cinematic"]

_STEM = {
    "type": "object",
    "required": ["asset_id", "role"],
    "properties": {
        "asset_id": {"type": "string"},
        "role": {"type": "string", "enum": ["dialogue", "narration", "music", "sfx", "ambience", "foley"]},
        "start_seconds": {"type": "number", "minimum": 0},
        "gain_db": {"type": "number"},
    },
}
_CLIP = {
    "type": "object",
    "required": ["asset_id"],
    "properties": {
        "asset_id": {"type": "string"},
        "trim_start_seconds": {"type": "number", "minimum": 0},
        "trim_end_seconds": {"type": "number", "minimum": 0},
    },
}


def all_operations() -> list[Operation]:
    return [
        Operation(
            id="image.generate",
            description="Generate an image from a text prompt.",
            presets=IMAGE_PRESETS,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": _PROMPT,
                    "negative_prompt": _PROMPT,
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "aspect_ratio": _ASPECT,
                    "candidate_count": {"type": "integer", "minimum": 1, "maximum": 8},
                },
            },
            parameter_schema={"properties": {"seed": _SEED, "steps": _STEPS, "cfg": {"type": "number"}, "loras": {"type": "array"}}},
            output_schema=_IMAGE_OUT,
            implemented=True,
            required_models=["qwen-image-2512-fp8"],
            capability_path=("image", "generate"),
        ),
        Operation(
            id="image.edit",
            description="Edit an image with natural-language instructions or merge 2–3 references.",
            presets=["draft", "balanced", "master", "identity_strict"],
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": _PROMPT,
                    "reference_images": {"type": "array", "minItems": 1, "maxItems": 3, "items": _ASSET},
                    "image": _ASSET,
                    "negative_prompt": _PROMPT,
                },
            },
            parameter_schema={"properties": {"seed": _SEED, "steps": _STEPS, "cfg": {"type": "number"}}},
            output_schema=_IMAGE_OUT,
            implemented=True,
            required_models=["qwen-image-edit-2511-fp8"],
            capability_path=("image", "edit"),
        ),
        Operation(
            id="image.controlled",
            description="Controlled image generation (depth, pose, canny).",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["prompt", "control_image"],
                "properties": {
                    "prompt": _PROMPT,
                    "control_image": _ASSET,
                    "control_type": {"type": "string", "enum": ["pose", "depth", "canny"]},
                    "control_strength": {"type": "number", "minimum": 0, "maximum": 2},
                    "negative_prompt": _PROMPT,
                },
            },
            parameter_schema={"properties": {"seed": _SEED, "steps": _STEPS, "cfg": {"type": "number"}}},
            output_schema=_IMAGE_OUT,
            implemented=True,
            required_models=["qwen-image-2512-fp8", "qwen-controlnet-2512-fun-union"],
            capability_path=("image", "control"),
        ),
        Operation(
            id="image.layered",
            description="Decompose an image into RGBA layers (Qwen-Image-Layered).",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["image"],
                "properties": {
                    "image": _ASSET,
                    "prompt": _PROMPT,
                    "layers": {"type": "integer", "minimum": 1, "maximum": 8},
                },
            },
            parameter_schema={"properties": {"seed": _SEED, "steps": _STEPS, "cfg": {"type": "number"}}},
            output_schema=_IMAGE_OUT,
            implemented=True,
            required_models=["qwen-image-layered"],
            capability_path=("image", "layered"),
        ),
        Operation(
            id="image.mask",
            description="Generate a mask (SAM).",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["image"],
                "properties": {
                    "image": _ASSET,
                    "prompt": _PROMPT,
                    "point": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}},
                },
            },
            parameter_schema=_SEED_ONLY,
            output_schema=_IMAGE_OUT,
            capability_path=("image", "mask"),
        ),
        Operation(
            id="image.depth",
            description="Generate a depth map.",
            presets=GENERIC,
            input_schema={"type": "object", "required": ["image"], "properties": {"image": _ASSET}},
            parameter_schema={},
            output_schema=_IMAGE_OUT,
            capability_path=("image", "depth"),
        ),
        Operation(
            id="image.upscale",
            description="Conservative image upscale.",
            presets=["draft", "balanced", "master", "conservative_upscale"],
            input_schema={"type": "object", "required": ["image"], "properties": {"image": _ASSET, "scale": {"type": "integer", "enum": [2, 4]}}},
            parameter_schema={},
            output_schema=_IMAGE_OUT,
            implemented=True,
            required_models=["upscalers-esrgan"],
            capability_path=("image", "upscale"),
        ),
        Operation(
            id="image.inspect",
            description="Inspect image metadata.",
            presets=GENERIC,
            input_schema={"type": "object", "required": ["image"], "properties": {"image": _ASSET}},
            parameter_schema={},
            output_schema=_INSPECT_OUT,
            capability_path=("image", "inspect"),
        ),
        Operation(
            id="video.generate",
            description="Generate video from text.",
            presets=VIDEO_PRESETS,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": _PROMPT,
                    "negative_prompt": _PROMPT,
                    "audio_prompt": _PROMPT,
                    "duration_seconds": _DURATION_SHOT,
                    "aspect_ratio": _ASPECT,
                },
            },
            parameter_schema=_SEED_STEPS,
            output_schema=_VIDEO_OUT,
            implemented=True,
            required_models=["ltx-2.5-distilled"],
            capability_path=("video", "text_to_video"),
        ),
        Operation(
            id="video.image_to_video",
            description="Generate video from a start image.",
            presets=VIDEO_PRESETS,
            input_schema={
                "type": "object",
                "required": ["start_image", "prompt"],
                "properties": {
                    "start_image": _ASSET,
                    "prompt": _PROMPT,
                    "audio_prompt": _PROMPT,
                    "duration_seconds": _DURATION_SHOT,
                    "aspect_ratio": _ASPECT,
                },
            },
            parameter_schema=_SEED_STEPS,
            output_schema=_VIDEO_OUT,
            implemented=True,
            required_models=["ltx-2.5-distilled"],
            capability_path=("video", "image_to_video"),
        ),
        Operation(
            id="video.audio_to_video",
            description="Generate video conditioned on an uploaded audio clip.",
            presets=VIDEO_PRESETS,
            input_schema={
                "type": "object",
                "required": ["audio", "prompt"],
                "properties": {
                    "audio": _ASSET,
                    "prompt": _PROMPT,
                    "start_image": _ASSET,
                    "end_image": _ASSET,
                    "duration_seconds": _DURATION_SHOT,
                    "aspect_ratio": _ASPECT,
                },
            },
            parameter_schema=_SEED_STEPS,
            output_schema=_VIDEO_OUT,
            implemented=True,
            required_models=["ltx-2.5-distilled"],
            capability_path=("video", "audio_to_video"),
        ),
        Operation(
            id="video.first_last_frames",
            description="Generate video between a first and last frame.",
            presets=VIDEO_PRESETS,
            input_schema={
                "type": "object",
                "required": ["start_image", "end_image", "prompt"],
                "properties": {
                    "start_image": _ASSET,
                    "end_image": _ASSET,
                    "prompt": _PROMPT,
                    "audio_prompt": _PROMPT,
                    "duration_seconds": _DURATION_SHOT,
                    "aspect_ratio": _ASPECT,
                },
            },
            parameter_schema=_SEED_STEPS,
            output_schema=_VIDEO_OUT,
            implemented=True,
            required_models=["ltx-2.5-distilled"],
            capability_path=("video", "first_last_frames"),
        ),
        Operation(
            id="video.continue",
            description="Continue a video from the previous clip.",
            presets=VIDEO_PRESETS,
            input_schema={
                "type": "object",
                "required": ["source_video", "prompt"],
                "properties": {
                    "source_video": _ASSET,
                    "prompt": _PROMPT,
                    "audio_prompt": _PROMPT,
                    "duration_seconds": _DURATION_SHOT,
                    "overlap_frames": {"type": "integer", "minimum": 1, "maximum": 48},
                    "aspect_ratio": _ASPECT,
                },
            },
            parameter_schema=_SEED_STEPS,
            output_schema=_VIDEO_OUT,
            capability_path=("video", "continuation"),
        ),
        Operation(
            id="video.enhance",
            description="Restore or enhance a video.",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["video"],
                "properties": {
                    "video": _ASSET,
                    "strength": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            parameter_schema=_SEED_STEPS,
            output_schema=_VIDEO_OUT,
            capability_path=("video", "enhancement"),
        ),
        Operation(
            id="video.upscale",
            description="Generative video upscale (distinct from restore/enhance).",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["video"],
                "properties": {
                    "video": _ASSET,
                    "scale": {"type": "integer", "enum": [2, 4]},
                },
            },
            parameter_schema=_SEED_STEPS,
            output_schema=_VIDEO_OUT,
            capability_path=("video", "upscale"),
        ),
        Operation(
            id="video.interpolate",
            description="Frame interpolation.",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["video"],
                "properties": {
                    "video": _ASSET,
                    "target_fps": {"type": "number", "minimum": 16, "maximum": 60},
                    "multiplier": {"type": "integer", "enum": [2, 4]},
                },
            },
            parameter_schema={},
            output_schema=_VIDEO_OUT,
            capability_path=("video", "interpolation"),
        ),
        Operation(
            id="video.color_grade",
            description="Optional look / color-grade pass on a clip.",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["video"],
                "properties": {
                    "video": _ASSET,
                    "look": {"type": "string"},
                    "strength": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            parameter_schema=_SEED_ONLY,
            output_schema=_VIDEO_OUT,
            capability_path=("video", "color_grade"),
        ),
        Operation(
            id="video.lipsync",
            description="Lip-sync a start image (optional end/middle frames) to uploaded audio.",
            presets=VIDEO_PRESETS,
            input_schema={
                "type": "object",
                "required": ["audio", "start_image", "prompt"],
                "properties": {
                    "audio": _ASSET,
                    "start_image": _ASSET,
                    "end_image": _ASSET,
                    "middle_image": _ASSET,
                    "prompt": _PROMPT,
                    "duration_seconds": _DURATION_SHOT,
                    "aspect_ratio": _ASPECT,
                },
            },
            parameter_schema=_SEED_STEPS,
            output_schema=_VIDEO_OUT,
            implemented=True,
            required_models=["ltx-2.5-distilled", "ltx-iclora-lipdub"],
            capability_path=("video", "lipsync"),
        ),
        Operation(
            id="video.motion_transfer",
            description="Transfer motion from a reference video.",
            presets=VIDEO_PRESETS,
            input_schema={
                "type": "object",
                "required": ["reference_video", "prompt"],
                "properties": {
                    "reference_video": _ASSET,
                    "start_image": _ASSET,
                    "prompt": _PROMPT,
                    "audio_prompt": _PROMPT,
                    "duration_seconds": _DURATION_SHOT,
                    "aspect_ratio": _ASPECT,
                },
            },
            parameter_schema=_SEED_STEPS,
            output_schema=_VIDEO_OUT,
            implemented=True,
            required_models=["ltx-2.5-distilled", "ltx-iclora-motion-track"],
            capability_path=("video", "motion_transfer"),
        ),
        Operation(
            id="video.extract_frame",
            description="Extract a frame from video.",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["video"],
                "properties": {
                    "video": _ASSET,
                    "time_seconds": {"type": "number", "minimum": 0},
                    "frame_index": {"type": "integer", "minimum": 0},
                    "position": {"type": "string", "enum": ["first", "last", "middle"]},
                },
            },
            parameter_schema={},
            output_schema=_IMAGE_OUT,
            capability_path=("video", "extract_frame"),
        ),
        Operation(
            id="video.inspect",
            description="Inspect video metadata.",
            presets=GENERIC,
            input_schema={"type": "object", "required": ["video"], "properties": {"video": _ASSET}},
            parameter_schema={},
            output_schema=_INSPECT_OUT,
            capability_path=("video", "inspect"),
        ),
        Operation(
            id="video.mask",
            description="Video mask / tracking.",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["video"],
                "properties": {
                    "video": _ASSET,
                    "prompt": _PROMPT,
                    "start_image": _ASSET,
                },
            },
            parameter_schema=_SEED_ONLY,
            output_schema=_VIDEO_OUT,
            capability_path=("video", "mask"),
        ),
        Operation(
            id="audio.tts",
            description="Text to speech (narration or character dialogue).",
            presets=TTS_PRESETS,
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 8000},
                    "language": {"type": "string", "enum": ["hi", "en"]},
                    "voice": {"type": "string"},
                    "style": {"type": "string"},
                    "speaker_ref": _ASSET,
                },
            },
            parameter_schema={"properties": {"seed": _SEED, "speaking_rate": {"type": "number", "minimum": 0.5, "maximum": 2.0}}},
            output_schema=_AUDIO_OUT,
            implemented=True,
            required_models=["chatterbox-multilingual", "chatterbox-hi"],
            capability_path=("audio", "tts"),
        ),
        Operation(
            id="audio.sfx",
            description="Isolated sound effect.",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": _PROMPT,
                    "duration_seconds": _DURATION_STEM,
                },
            },
            parameter_schema=_SEED_ONLY,
            output_schema=_AUDIO_OUT,
            implemented=True,
            required_models=["ace-step-1.5"],
            capability_path=("audio", "sfx"),
        ),
        Operation(
            id="audio.ambience",
            description="Ambience stem.",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": _PROMPT,
                    "duration_seconds": _DURATION_STEM,
                    "loopable": {"type": "boolean"},
                },
            },
            parameter_schema=_SEED_ONLY,
            output_schema=_AUDIO_OUT,
            implemented=True,
            required_models=["ace-step-1.5"],
            capability_path=("audio", "ambience"),
        ),
        Operation(
            id="audio.foley",
            description="Performed object / footstep / cloth sounds timed to picture.",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": _PROMPT,
                    "video": _ASSET,
                    "duration_seconds": _DURATION_STEM,
                },
            },
            parameter_schema=_SEED_ONLY,
            output_schema=_AUDIO_OUT,
            implemented=True,
            required_models=["ace-step-1.5"],
            capability_path=("audio", "foley"),
        ),
        Operation(
            id="audio.music",
            description="Background / score generation.",
            presets=MUSIC_PRESETS,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": _PROMPT,
                    "duration_seconds": _DURATION_STEM,
                    "mood": {"type": "string"},
                    "no_vocals": {"type": "boolean"},
                },
            },
            parameter_schema=_SEED_ONLY,
            output_schema=_AUDIO_OUT,
            implemented=True,
            required_models=["ace-step-1.5"],
            capability_path=("audio", "music"),
        ),
        Operation(
            id="audio.mix",
            description="Mix audio stems on a timeline.",
            presets=MIX_PRESETS,
            input_schema={
                "type": "object",
                "required": ["stems"],
                "properties": {
                    "stems": {"type": "array", "minItems": 1, "items": _STEM},
                    "duration_seconds": _DURATION_STEM,
                    "target_lufs": {"type": "number"},
                },
            },
            parameter_schema={"properties": {"target_lufs": {"type": "number"}}},
            output_schema=_AUDIO_OUT,
            implemented=True,
            capability_path=("audio", "mix"),
        ),
        Operation(
            id="audio.normalize",
            description="Loudness / peak normalize before mux.",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["audio"],
                "properties": {
                    "audio": _ASSET,
                    "target_lufs": {"type": "number"},
                    "true_peak_db": {"type": "number"},
                },
            },
            parameter_schema={"properties": {"target_lufs": {"type": "number"}}},
            output_schema=_AUDIO_OUT,
            implemented=True,
            capability_path=("audio", "normalize"),
        ),
        Operation(
            id="audio.inspect",
            description="Inspect audio metadata.",
            presets=GENERIC,
            input_schema={"type": "object", "required": ["audio"], "properties": {"audio": _ASSET}},
            parameter_schema={},
            output_schema=_INSPECT_OUT,
            implemented=True,
            capability_path=("audio", "inspect"),
        ),
        Operation(
            id="media.concat",
            description="Concatenate clips into a longer reel or scene.",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["clips"],
                "properties": {
                    "clips": {"type": "array", "minItems": 2, "items": _CLIP},
                    "transition": {"type": "string", "enum": ["cut", "crossfade"]},
                    "transition_seconds": {"type": "number", "minimum": 0, "maximum": 2},
                },
            },
            parameter_schema={},
            output_schema=_VIDEO_OUT,
            implemented=True,
            capability_path=("media", "concat"),
        ),
        Operation(
            id="media.finalize",
            description="Mux picture + mixed audio and delivery-encode.",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["video"],
                "properties": {
                    "video": _ASSET,
                    "audio": _ASSET,
                    "container": {"type": "string", "enum": ["mp4", "webm"]},
                    "audio_offset_seconds": {"type": "number"},
                },
            },
            parameter_schema={},
            output_schema=_VIDEO_OUT,
            implemented=True,
            capability_path=("media", "finalize"),
        ),
        Operation(
            id="qc.image",
            description="Image quality checks.",
            presets=GENERIC,
            input_schema={"type": "object", "required": ["image"], "properties": {"image": _ASSET}},
            parameter_schema={},
            output_schema=_QC_OUT,
            capability_path=("qc", "image"),
        ),
        Operation(
            id="qc.video",
            description="Video quality checks.",
            presets=GENERIC,
            input_schema={"type": "object", "required": ["video"], "properties": {"video": _ASSET}},
            parameter_schema={},
            output_schema=_QC_OUT,
            capability_path=("qc", "video"),
        ),
        Operation(
            id="qc.audio",
            description="Audio quality checks.",
            presets=GENERIC,
            input_schema={"type": "object", "required": ["audio"], "properties": {"audio": _ASSET}},
            parameter_schema={},
            output_schema=_QC_OUT,
            capability_path=("qc", "audio"),
        ),
        Operation(
            id="qc.media",
            description="Combined media quality checks.",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["video"],
                "properties": {
                    "image": _ASSET,
                    "video": _ASSET,
                    "audio": _ASSET,
                },
            },
            parameter_schema={},
            output_schema=_QC_OUT,
            capability_path=("qc", "media"),
        ),
        Operation(
            id="text.chat",
            description="Synchronous text + vision chat via llama-server. Not a Comfy job.",
            presets=GENERIC,
            input_schema={
                "type": "object",
                "required": ["messages"],
                "properties": {
                    "model": {"type": "string"},
                    "messages": {"type": "array"},
                    "temperature": {"type": "number"},
                    "max_tokens": {"type": "integer"},
                    "stream": {"type": "boolean"},
                },
            },
            parameter_schema={},
            output_schema={
                "type": "object",
                "properties": {
                    "choices": {"type": "array"},
                    "model": {"type": "string"},
                },
            },
            implemented=True,
            capability_path=("text", "chat"),
        ),
    ]
