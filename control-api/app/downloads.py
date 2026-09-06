from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config, runtime


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


def _read_state() -> dict[str, Any]:
    if not config.DOWNLOAD_STATE.exists():
        return {
            "status": "idle",
            "current": None,
            "currentBundle": None,
            "currentFile": None,
            "currentBytes": None,
            "currentTotal": None,
            "currentPercent": None,
            "startedAt": None,
            "finishedAt": None,
            "error": None,
        }
    try:
        return json.loads(config.DOWNLOAD_STATE.read_text())
    except json.JSONDecodeError:
        return {"status": "idle", "current": None}


def _download_active() -> bool:
    proc = subprocess.run(
        ["systemctl", "--user", "is-active", "ai-download.service"],
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() in {"active", "activating"}


def _log_tail(lines: int = 80) -> list[str]:
    if not config.DOWNLOAD_LOG.exists():
        return []
    return config.DOWNLOAD_LOG.read_text(errors="replace").splitlines()[-lines:]


def _hf_incomplete_bytes(state: dict[str, Any]) -> int:
    """Newest Hugging Face *.incomplete blob for the in-flight repo (live size)."""
    current = str(state.get("current") or "")
    parts = [p for p in current.split("/") if p]
    if len(parts) < 2:
        return 0
    repo = f"{parts[0]}--{parts[1]}"
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    blobs = hf_home / "hub" / f"models--{repo}" / "blobs"
    if not blobs.is_dir():
        return 0
    newest = None
    newest_mtime = -1.0
    for path in blobs.glob("*.incomplete"):
        try:
            st = path.stat()
        except OSError:
            continue
        if st.st_mtime >= newest_mtime:
            newest_mtime = st.st_mtime
            newest = st.st_size
    return int(newest or 0)


def _build_progress(models: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    bundles = models.get("bundles", [])
    files: list[dict[str, Any]] = []
    required = 0
    ok = 0
    optional = 0
    optional_ok = 0
    bytes_on_disk = 0

    current = state.get("current") or ""
    current_file = state.get("currentFile") or ""
    failed_by_name = {
        str(item.get("name") or ""): str(item.get("error") or "download failed")
        for item in state.get("failedFiles") or []
        if item.get("name")
    }
    failed_by_bundle = {
        str(item.get("bundleId") or ""): str(item.get("error") or "download failed")
        for item in state.get("failedFiles") or []
        if item.get("bundleId")
    }
    current_bytes = int(state.get("currentBytes") or 0)
    current_total = int(state.get("currentTotal") or 0)
    if state.get("status") == "running" and current_bytes <= 0:
        current_bytes = _hf_incomplete_bytes(state)
    current_file_pct = state.get("currentPercent")
    if current_file_pct is None and current_total > 0:
        current_file_pct = round(100.0 * current_bytes / current_total, 1)
    in_flight = 0.0

    for bundle in bundles:
        for f in bundle.get("files", []):
            is_optional = bool(f.get("optional"))
            if is_optional:
                optional += 1
                if f.get("state") == "ok":
                    optional_ok += 1
            else:
                required += 1
                if f.get("state") == "ok":
                    ok += 1
            if f.get("state") == "ok":
                bytes_on_disk += int(f.get("bytes") or 0)

            name = f.get("name", "")
            remote_hint = f.get("remote") or name
            file_error = failed_by_name.get(name) or failed_by_bundle.get(str(bundle["id"]))
            is_current = bool(
                state.get("status") == "running"
                and (
                    (current_file and (name.endswith(current_file) or current_file == name))
                    or (current and (current.endswith(name) or name in current or remote_hint in current))
                )
            )
            if is_current and f.get("state") != "ok" and not is_optional and current_total > 0:
                in_flight = min(1.0, current_bytes / current_total)

            file_pct: float | None
            if f.get("state") == "ok":
                file_pct = 100.0
            elif is_current and current_file_pct is not None:
                file_pct = float(current_file_pct)
            else:
                file_pct = None

            files.append(
                {
                    "bundleId": bundle["id"],
                    "bundleStatus": bundle["status"],
                    "name": name,
                    "remote": remote_hint,
                    "optional": is_optional,
                    "state": (
                        "downloading"
                        if is_current and f.get("state") != "ok"
                        else "failed"
                        if f.get("state") != "ok" and file_error
                        else f.get("state")
                    ),
                    "error": file_error if f.get("state") != "ok" else None,
                    "bytes": f.get("bytes", 0),
                    "sizeHuman": _human_size(int(f.get("bytes") or 0)),
                    "percent": file_pct,
                    "isCurrent": is_current,
                }
            )

    pct = round(((ok + in_flight) / required) * 100, 1) if required else 0.0
    return {
        "bundlesTotal": len(bundles),
        "bundlesComplete": sum(1 for b in bundles if b["status"] == "complete"),
        "filesRequired": required,
        "filesComplete": ok,
        "filesOptional": optional,
        "filesOptionalComplete": optional_ok,
        "bytesOnDisk": bytes_on_disk,
        "bytesOnDiskHuman": _human_size(bytes_on_disk),
        "percentComplete": pct,
        "currentBytes": current_bytes or None,
        "currentTotal": current_total or None,
        "currentBytesHuman": _human_size(current_bytes) if current_bytes else None,
        "currentTotalHuman": _human_size(current_total) if current_total else None,
        "currentFilePercent": float(current_file_pct) if current_file_pct is not None else None,
        "files": files,
    }


def get_download_status() -> dict[str, Any]:
    state = _read_state()
    running = _download_active()
    if running and state.get("status") != "running":
        state["status"] = "running"
    if not running and state.get("status") == "running":
        state["status"] = "failed"
        state.setdefault("error", "download service stopped unexpectedly")
        state["finishedAt"] = state.get("finishedAt") or _now()

    try:
        models = runtime.get_models()
    except Exception:
        models = {"bundles": []}

    progress = _build_progress(models, state)
    return {
        **state,
        "running": running,
        "logTail": _log_tail(),
        "progress": progress,
    }


def start_download(ids: list[str] | None = None) -> dict[str, Any]:
    if _download_active():
        raise ConflictError("Download already running")

    config.CONFIG.mkdir(parents=True, exist_ok=True)
    config.LOGS.mkdir(parents=True, exist_ok=True)
    payload = {"ids": ids or [], "requestedAt": _now()}
    config.DOWNLOAD_REQUEST.write_text(json.dumps(payload, indent=2) + "\n")

    proc = subprocess.run(
        ["systemctl", "--user", "start", "--no-block", "ai-download.service"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "failed to start ai-download.service")

    state = _read_state()
    state.update(
        {
            "status": "running",
            "startedAt": _now(),
            "finishedAt": None,
            "error": None,
            "current": None,
            "currentBundle": None,
            "currentFile": None,
            "currentBytes": None,
            "currentTotal": None,
            "currentPercent": None,
        }
    )
    config.DOWNLOAD_STATE.write_text(json.dumps(state, indent=2) + "\n")
    return get_download_status()


class ConflictError(Exception):
    pass
