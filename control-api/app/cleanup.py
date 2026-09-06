"""Remove generated images/videos from the GPU box. Never touches model weights."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

JOBS_ROOT = Path(config.LOGS) / "jobs"
JOB_STORES = (
    Path(config.LOGS) / "generate-jobs.json",
    Path(config.LOGS) / "ltx-video-jobs.json",
    Path(config.LOGS) / "upscale-jobs.json",
    Path(config.LOGS) / "character-master-jobs.json",
)
ACTIVE_STATUSES = frozenset(
    {
        "queued",
        "switching",
        "running",
        "load_models",
        "encode",
        "sample",
        "refine",
        "decode",
        "export",
    }
)
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"})
VIDEO_EXTS = frozenset({".mp4", ".webm", ".mov", ".mkv", ".avi"})
TARGETS = ("images", "videos", "comfy_outputs")


class CleanupError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _human_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _comfy_output_dirs() -> list[Path]:
    home = Path(os.environ.get("HOME", str(Path.home())))
    paths = [
        Path(os.environ.get("COMFYUI_OUTPUT", str(home / "ComfyUI" / "output"))),
        Path(os.environ.get("COMFY_LTX_OUTPUT", str(home / "ComfyUI-ltx" / "output"))),
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _active_job_ids() -> set[str]:
    active: set[str] = set()
    for store in JOB_STORES:
        if not store.is_file():
            continue
        try:
            data = json.loads(store.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for job_id, job in (data.get("jobs") or {}).items():
            status = str((job or {}).get("status") or "").lower()
            if status in ACTIVE_STATUSES:
                active.add(str(job_id))
    return active


def _file_kind(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "images"
    if ext in VIDEO_EXTS:
        return "videos"
    return None


def _iter_job_media(kind: str, skip_ids: set[str]) -> list[Path]:
    if not JOBS_ROOT.is_dir():
        return []
    found: list[Path] = []
    for job_dir in JOBS_ROOT.iterdir():
        if not job_dir.is_dir() or job_dir.name in skip_ids:
            continue
        folder = job_dir / kind
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and _file_kind(path) == kind:
                found.append(path)
    return found


def _iter_comfy_media() -> list[Path]:
    found: list[Path] = []
    for out_dir in _comfy_output_dirs():
        if not out_dir.is_dir():
            continue
        for path in out_dir.rglob("*"):
            if path.is_file() and _file_kind(path) in {"images", "videos"}:
                found.append(path)
    return found


def _collect(targets: list[str], skip_ids: set[str]) -> dict[str, list[Path]]:
    chosen = [t for t in TARGETS if t in targets]
    buckets: dict[str, list[Path]] = {t: [] for t in chosen}
    if "images" in buckets:
        buckets["images"] = _iter_job_media("images", skip_ids)
    if "videos" in buckets:
        buckets["videos"] = _iter_job_media("videos", skip_ids)
    if "comfy_outputs" in buckets:
        buckets["comfy_outputs"] = _iter_comfy_media()
    return buckets


def _summarize(buckets: dict[str, list[Path]]) -> dict[str, Any]:
    targets = []
    total_bytes = 0
    total_files = 0
    for name, files in buckets.items():
        nbytes = 0
        for path in files:
            try:
                nbytes += path.stat().st_size
            except OSError:
                continue
        total_bytes += nbytes
        total_files += len(files)
        targets.append(
            {
                "id": name,
                "files": len(files),
                "bytes": nbytes,
                "bytesHuman": _human_size(nbytes),
            }
        )
    return {
        "targets": targets,
        "files": total_files,
        "bytes": total_bytes,
        "bytesHuman": _human_size(total_bytes),
    }


def inventory() -> dict[str, Any]:
    skip = _active_job_ids()
    buckets = _collect(list(TARGETS), skip)
    summary = _summarize(buckets)
    return {
        "status": "idle",
        "note": "Removes generated job images/videos and optional ComfyUI output. Model weights are never deleted.",
        "skippedJobIds": sorted(skip),
        **summary,
    }


def run(targets: list[str] | None = None) -> dict[str, Any]:
    raw = list(TARGETS[:2]) if targets is None else targets
    chosen = [t for t in raw if t in TARGETS]
    if not chosen:
        raise CleanupError("INVALID_TARGET", "Choose images, videos, and/or comfy_outputs", 400)
    skip = _active_job_ids()
    buckets = _collect(chosen, skip)
    deleted_files = 0
    deleted_bytes = 0
    errors: list[str] = []
    for files in buckets.values():
        for path in files:
            try:
                size = path.stat().st_size
                path.unlink()
                deleted_files += 1
                deleted_bytes += size
            except OSError as exc:
                errors.append(f"{path.name}: {exc}")
    _prune_empty_dirs(chosen)
    return {
        "status": "completed" if not errors else "partial",
        "targets": chosen,
        "deletedFiles": deleted_files,
        "deletedBytes": deleted_bytes,
        "deletedBytesHuman": _human_size(deleted_bytes),
        "skippedJobIds": sorted(skip),
        "errors": errors[:20],
        "finishedAt": _now(),
        **_summarize(_collect(list(TARGETS), _active_job_ids())),
    }


def _prune_empty_dirs(targets: list[str]) -> None:
    if JOBS_ROOT.is_dir() and any(t in {"images", "videos"} for t in targets):
        for job_dir in JOBS_ROOT.iterdir():
            if not job_dir.is_dir():
                continue
            for name in ("images", "videos"):
                folder = job_dir / name
                if folder.is_dir() and not any(folder.iterdir()):
                    shutil.rmtree(folder, ignore_errors=True)
            if job_dir.is_dir() and not any(job_dir.iterdir()):
                shutil.rmtree(job_dir, ignore_errors=True)
    if "comfy_outputs" in targets:
        for out_dir in _comfy_output_dirs():
            if not out_dir.is_dir():
                continue
            for path in sorted(out_dir.rglob("*"), reverse=True):
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
