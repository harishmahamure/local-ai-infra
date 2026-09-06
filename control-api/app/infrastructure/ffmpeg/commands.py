from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class FfmpegError(RuntimeError):
    pass


def _bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FfmpegError(f"{name} is not installed")
    return path


def inspect_argv(path: str) -> list[str]:
    return [
        _bin("ffprobe"),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]


def loudnorm_argv(src: str, dest: str, *, target_lufs: float, true_peak: float) -> list[str]:
    return [
        _bin("ffmpeg"),
        "-y",
        "-i",
        src,
        "-af",
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11",
        "-ar",
        "48000",
        dest,
    ]


def mix_argv(
    stem_paths: list[str],
    dest: str,
    *,
    starts: list[float],
    gains_db: list[float],
    target_lufs: float,
    duration: float | None,
) -> list[str]:
    cmd = [_bin("ffmpeg"), "-y"]
    filters = []
    labels = []
    for index, path in enumerate(stem_paths):
        cmd.extend(["-i", path])
        delay_ms = max(0, int(starts[index] * 1000))
        gain = gains_db[index]
        chain = f"[{index}:a]adelay={delay_ms}|{delay_ms},volume={gain}dB[a{index}]"
        filters.append(chain)
        labels.append(f"[a{index}]")
    n = len(stem_paths)
    mix = "".join(labels) + f"amix=inputs={n}:normalize=0:duration=longest[m]"
    filters.append(mix)
    filters.append(f"[m]loudnorm=I={target_lufs}:TP=-1.5:LRA=11[out]")
    cmd.extend(["-filter_complex", ";".join(filters), "-map", "[out]", "-ar", "48000"])
    if duration:
        cmd.extend(["-t", f"{duration:.3f}"])
    cmd.append(dest)
    return cmd


def concat_argv(list_file: str, dest: str, *, transition: str, transition_seconds: float, n: int) -> list[str]:
    if transition == "crossfade" and n >= 2 and transition_seconds > 0:
        # Caller builds the filter graph separately; this argv is the cut path.
        pass
    return [
        _bin("ffmpeg"),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-c",
        "copy",
        dest,
    ]


def concat_filter_argv(paths: list[str], dest: str, *, transition_seconds: float) -> list[str]:
    cmd = [_bin("ffmpeg"), "-y"]
    for path in paths:
        cmd.extend(["-i", path])
    parts = []
    current = "[0:v][0:a]"
    # Sequential xfade + acrossfade; fallback to concat filter if a clip lacks audio.
    video_chain = "".join(f"[{i}:v]" for i in range(len(paths))) + f"concat=n={len(paths)}:v=1:a=0[v]"
    audio_chain = "".join(f"[{i}:a]" for i in range(len(paths))) + f"concat=n={len(paths)}:v=0:a=1[a]"
    if transition_seconds <= 0:
        parts = [video_chain, audio_chain]
    else:
        v = "[0:v]"
        a = "[0:a]"
        offset = 0.0
        for i in range(1, len(paths)):
            offset = max(0.1, offset + 1.0)
            nv = f"v{i}"
            na = f"a{i}"
            parts.append(f"{v}[{i}:v]xfade=transition=fade:duration={transition_seconds}:offset={offset}[{nv}]")
            parts.append(f"{a}[{i}:a]acrossfade=d={transition_seconds}[{na}]")
            v, a = f"[{nv}]", f"[{na}]"
        cmd.extend(["-filter_complex", ";".join(parts), "-map", v, "-map", a, dest])
        return cmd
    cmd.extend(["-filter_complex", ";".join(parts), "-map", "[v]", "-map", "[a]", dest])
    return cmd


def finalize_argv(video: str, audio: str | None, dest: str, *, offset: float) -> list[str]:
    cmd = [_bin("ffmpeg"), "-y", "-i", video]
    if audio:
        if offset:
            cmd.extend(["-itsoffset", f"{offset:.3f}"])
        cmd.extend(["-i", audio])
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest"])
    else:
        cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an"])
    cmd.append(dest)
    return cmd


def resample_wav_48k(raw: bytes) -> bytes:
    if shutil.which("ffmpeg") is None:
        return raw
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        dest = Path(tmp) / "out.wav"
        src.write_bytes(raw)
        cmd = [_bin("ffmpeg"), "-y", "-i", str(src), "-ar", "48000", "-ac", "1", str(dest)]
        run(cmd)
        return dest.read_bytes()


def run(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise FfmpegError(str(exc)) from exc
    if proc.returncode != 0:
        raise FfmpegError((proc.stderr or proc.stdout or "ffmpeg failed").strip())
    return proc.stdout


def probe(path: str) -> dict[str, Any]:
    raw = run(inspect_argv(path))
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise FfmpegError(f"ffprobe JSON invalid: {exc}") from exc
