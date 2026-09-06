from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from ...application.ports import ExecutionOutput, ExecutionRequest, ExecutionResult
from ...domain.errors import DomainError, ErrorCode
from . import commands


class FfmpegExecutor:
    def __init__(self, runner=commands.run, prober=commands.probe) -> None:
        self._run = runner
        self._probe = prober

    def health(self) -> dict[str, Any]:
        try:
            commands._bin("ffmpeg")
            commands._bin("ffprobe")
            return {"ffmpeg": True}
        except commands.FfmpegError as exc:
            return {"ffmpeg": False, "error": str(exc)}

    def cancel(self, job_id: str) -> None:
        return None

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if request.cancel_check and request.cancel_check():
            raise DomainError(ErrorCode.CANCELLED, "Job cancelled")
        builder = request.workflow.builder
        plan = request.plan
        if request.on_phase:
            request.on_phase("ffmpeg.run", 0.4)
        try:
            if builder == "ffmpeg_inspect":
                return self._inspect(request)
            if builder == "ffmpeg_normalize":
                return self._normalize(request)
            if builder == "ffmpeg_mix":
                return self._mix(request)
            if builder == "ffmpeg_concat":
                return self._concat(request)
            if builder == "ffmpeg_finalize":
                return self._finalize(request)
        except commands.FfmpegError as exc:
            raise DomainError(ErrorCode.FFMPEG_ERROR, str(exc)) from exc
        raise DomainError(ErrorCode.WORKFLOW_INVALID, f"Unknown ffmpeg builder {builder}")

    def _inspect(self, request: ExecutionRequest) -> ExecutionResult:
        raw = request.input_files.get("audio") or request.input_files.get("video") or request.input_files.get("image")
        if not raw:
            raise DomainError(ErrorCode.INVALID_REQUEST, "An input asset is required")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.bin"
            src.write_bytes(raw)
            info = self._probe(str(src))
        payload = json.dumps(self._public_probe(info), indent=2).encode()
        duration = self._duration(info)
        return ExecutionResult(
            outputs=[ExecutionOutput(data=payload, mime_type="application/json", artifact_type="JSON", duration_seconds=duration)],
            model_ids=[],
            extra={"probe": info, "plan": request.plan},
        )

    def _normalize(self, request: ExecutionRequest) -> ExecutionResult:
        raw = request.input_files.get("audio")
        if not raw:
            raise DomainError(ErrorCode.INVALID_REQUEST, "audio.asset_id is required")
        target = float(request.plan.get("target_lufs") or -16)
        peak = float(request.plan.get("true_peak_db") or -1.5)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.wav"
            dest = Path(tmp) / "out.wav"
            src.write_bytes(raw)
            self._run(commands.loudnorm_argv(str(src), str(dest), target_lufs=target, true_peak=peak))
            out = dest.read_bytes()
        return ExecutionResult(
            outputs=[ExecutionOutput(data=out, mime_type="audio/wav", artifact_type="AUDIO")],
            model_ids=[],
            extra={"plan": request.plan},
        )

    def _mix(self, request: ExecutionRequest) -> ExecutionResult:
        stems_meta = list(request.plan.get("stems") or [])
        if not stems_meta:
            raise DomainError(ErrorCode.INVALID_REQUEST, "stems are required")
        paths: list[str] = []
        starts: list[float] = []
        gains: list[float] = []
        with tempfile.TemporaryDirectory() as tmp:
            for index, stem in enumerate(stems_meta):
                key = f"stems_{index}"
                raw = request.input_files.get(key)
                if not raw:
                    raise DomainError(ErrorCode.ASSET_NOT_FOUND, f"stem {index} asset is missing")
                path = Path(tmp) / f"stem_{index}.wav"
                path.write_bytes(raw)
                paths.append(str(path))
                starts.append(float(stem.get("start_seconds") or 0))
                gains.append(float(stem.get("gain_db") or 0))
            dest = Path(tmp) / "mix.wav"
            self._run(
                commands.mix_argv(
                    paths,
                    str(dest),
                    starts=starts,
                    gains_db=gains,
                    target_lufs=float(request.plan.get("target_lufs") or -16),
                    duration=request.plan.get("duration_seconds"),
                )
            )
            out = dest.read_bytes()
        return ExecutionResult(
            outputs=[ExecutionOutput(data=out, mime_type="audio/wav", artifact_type="AUDIO")],
            model_ids=[],
            extra={"plan": request.plan},
        )

    def _concat(self, request: ExecutionRequest) -> ExecutionResult:
        clips = list(request.plan.get("clips") or [])
        if len(clips) < 2:
            raise DomainError(ErrorCode.INVALID_REQUEST, "media.concat requires at least 2 clips")
        transition = str(request.plan.get("transition") or "cut")
        fade = float(request.plan.get("transition_seconds") or 0)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, _clip in enumerate(clips):
                raw = request.input_files.get(f"clips_{index}")
                if not raw:
                    raise DomainError(ErrorCode.ASSET_NOT_FOUND, f"clip {index} asset is missing")
                path = Path(tmp) / f"clip_{index}.mp4"
                path.write_bytes(raw)
                paths.append(str(path))
            dest = Path(tmp) / "out.mp4"
            if transition == "crossfade" and fade > 0:
                self._run(commands.concat_filter_argv(paths, str(dest), transition_seconds=fade))
            else:
                listing = Path(tmp) / "list.txt"
                listing.write_text("".join(f"file '{p}'\n" for p in paths))
                self._run(commands.concat_argv(str(listing), str(dest), transition="cut", transition_seconds=0, n=len(paths)))
            out = dest.read_bytes()
        return ExecutionResult(
            outputs=[ExecutionOutput(data=out, mime_type="video/mp4", artifact_type="VIDEO")],
            model_ids=[],
            extra={"plan": request.plan},
        )

    def _finalize(self, request: ExecutionRequest) -> ExecutionResult:
        video = request.input_files.get("video")
        if not video:
            raise DomainError(ErrorCode.INVALID_REQUEST, "video.asset_id is required")
        audio = request.input_files.get("audio")
        offset = float(request.plan.get("audio_offset_seconds") or 0)
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "in.mp4"
            vpath.write_bytes(video)
            apath = None
            if audio:
                apath = Path(tmp) / "in.wav"
                apath.write_bytes(audio)
            dest = Path(tmp) / "out.mp4"
            self._run(commands.finalize_argv(str(vpath), str(apath) if apath else None, str(dest), offset=offset))
            out = dest.read_bytes()
        return ExecutionResult(
            outputs=[ExecutionOutput(data=out, mime_type="video/mp4", artifact_type="VIDEO")],
            model_ids=[],
            extra={"plan": request.plan},
        )

    def _public_probe(self, info: dict[str, Any]) -> dict[str, Any]:
        fmt = info.get("format") or {}
        streams = info.get("streams") or []
        audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        return {
            "duration_seconds": self._duration(info),
            "sample_rate": int(audio["sample_rate"]) if audio.get("sample_rate") else None,
            "codec": audio.get("codec_name") or video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "format": fmt.get("format_name"),
            "size_bytes": int(fmt["size"]) if fmt.get("size") else None,
        }

    def _duration(self, info: dict[str, Any]) -> float | None:
        fmt = info.get("format") or {}
        try:
            return float(fmt.get("duration"))
        except (TypeError, ValueError):
            return None
