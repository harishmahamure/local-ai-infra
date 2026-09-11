"""Camera presets for F08. Each preset is a prompt fragment plus track geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import DomainError, ErrorCode


@dataclass(frozen=True)
class CameraPreset:
    id: str
    label: str
    prompt: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    mid_x: float = 0.5
    mid_y: float = 0.5
    bezier: bool = False
    num_tracks: int = 7
    track_spread: float = 0.04


CAMERA_PRESETS: dict[str, CameraPreset] = {
    "dolly": CameraPreset(
        id="dolly",
        label="Dolly",
        prompt="slow dolly move, camera travels smoothly along a rail, continuous shot",
        start_x=0.35,
        start_y=0.55,
        end_x=0.65,
        end_y=0.45,
        num_tracks=8,
    ),
    "orbit": CameraPreset(
        id="orbit",
        label="Orbit",
        prompt="orbital camera move around the subject, circular tracking, continuous shot",
        start_x=0.18,
        start_y=0.5,
        end_x=0.82,
        end_y=0.5,
        mid_x=0.5,
        mid_y=0.28,
        bezier=True,
        num_tracks=9,
    ),
    "crane": CameraPreset(
        id="crane",
        label="Crane",
        prompt="crane shot, camera rises and tilts, elevated reveal, continuous shot",
        start_x=0.5,
        start_y=0.82,
        end_x=0.5,
        end_y=0.22,
        num_tracks=6,
    ),
    "tracking": CameraPreset(
        id="tracking",
        label="Tracking",
        prompt="lateral tracking shot, camera follows the subject, locked horizon, continuous shot",
        start_x=0.15,
        start_y=0.52,
        end_x=0.85,
        end_y=0.48,
        num_tracks=8,
        track_spread=0.03,
    ),
    "handheld": CameraPreset(
        id="handheld",
        label="Handheld",
        prompt="handheld camera, subtle documentary shake, lived-in motion, continuous shot",
        start_x=0.42,
        start_y=0.48,
        end_x=0.58,
        end_y=0.54,
        mid_x=0.5,
        mid_y=0.4,
        bezier=True,
        num_tracks=11,
        track_spread=0.06,
    ),
    "push_in": CameraPreset(
        id="push_in",
        label="Push-in",
        prompt="slow push-in, camera advances toward the subject, tightening frame, continuous shot",
        start_x=0.28,
        start_y=0.28,
        end_x=0.5,
        end_y=0.5,
        mid_x=0.4,
        mid_y=0.4,
        bezier=True,
        num_tracks=10,
        track_spread=0.05,
    ),
}


def resolve_camera(value: Any) -> CameraPreset:
    key = str(value or "").strip()
    if not key:
        raise DomainError(ErrorCode.INVALID_PARAMETER, "camera preset is required")
    preset = CAMERA_PRESETS.get(key)
    if preset is None:
        raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unknown camera preset {key}")
    return preset


def camera_catalog() -> list[dict[str, str]]:
    return [{"id": item.id, "label": item.label} for item in CAMERA_PRESETS.values()]


def tracks_inputs(preset: CameraPreset, *, width: int, height: int, length: int) -> dict[str, Any]:
    return {
        "width": width,
        "height": height,
        "start_x": preset.start_x,
        "start_y": preset.start_y,
        "end_x": preset.end_x,
        "end_y": preset.end_y,
        "mid_x": preset.mid_x,
        "mid_y": preset.mid_y,
        "bezier": preset.bezier,
        "num_frames": length,
        "num_tracks": preset.num_tracks,
        "track_spread": preset.track_spread,
        "interpolation": "linear",
    }
