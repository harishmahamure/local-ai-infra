from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
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
    current_bundle = state.get("currentBundle") or ""

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
            is_current = bool(
                state.get("status") == "running"
                and (
                    (current_file and name.endswith(current_file) or current_file == name)
                    or (current and (current.endswith(name) or name in current or remote_hint in current))
                )
            )

            files.append(
                {
                    "bundleId": bundle["id"],
                    "bundleStatus": bundle["status"],
                    "name": name,
                    "remote": remote_hint,
                    "optional": is_optional,
                    "state": f.get("state"),
                    "bytes": f.get("bytes", 0),
                    "sizeHuman": _human_size(int(f.get("bytes") or 0)),
                    "isCurrent": is_current,
                }
            )

    pct = round((ok / required) * 100, 1) if required else 0.0
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
        }
    )
    config.DOWNLOAD_STATE.write_text(json.dumps(state, indent=2) + "\n")
    return get_download_status()


class ConflictError(Exception):
    pass
