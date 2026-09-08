from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from . import config

_CONTROL_TIMEOUT_SEC = 180
UNITS = ("llama-fast", "gemma")


def _run_control(*args: str, check: bool = True, timeout: int = _CONTROL_TIMEOUT_SEC) -> subprocess.CompletedProcess[str]:
    cmd = ["bash", str(config.CONTROL_SH), *args]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"control.sh timed out after {timeout}s: {' '.join(args)}") from exc


def _gpu_info() -> dict[str, Any]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            return {
                "name": parts[0],
                "memoryUsed": parts[1],
                "memoryTotal": parts[2],
                "utilization": parts[3],
            }
    except (subprocess.CalledProcessError, IndexError, FileNotFoundError):
        pass
    return {"name": None, "memoryUsed": None, "memoryTotal": None, "utilization": None}


def _gpu_processes() -> list[dict[str, str]]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        rows = []
        for line in out.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                rows.append({"pid": parts[0], "name": parts[1], "memory": parts[2]})
        return rows
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _service_states() -> dict[str, str]:
    states: dict[str, str] = {}
    for unit in UNITS:
        proc = subprocess.run(
            ["systemctl", "--user", "is-active", f"{unit}.service"],
            capture_output=True,
            text=True,
        )
        states[unit] = proc.stdout.strip() or "inactive"
    return states


def _active_profile(services: dict[str, str]) -> str:
    active = [u for u, s in services.items() if s == "active"]
    if not active:
        return "none"
    if len(active) > 1:
        return f"CONFLICT:{','.join(active)}"
    return active[0]


def _model_hint(profile: str) -> str:
    hints = {
        "gemma": "Gemma 4 E4B Q4 + mmproj (text + vision, 128K ctx)",
        "llama-fast": "Qwen3.6-35B-A3B RotorQuant Q4 + mmproj (text + vision, 262K ctx)",
        "none": "none — GPU idle",
    }
    if profile.startswith("CONFLICT"):
        return "MORE THAN ONE SERVICE — run stop then start one profile"
    return hints.get(profile, profile)


def _http_ok(url: str) -> bool:
    proc = subprocess.run(["curl", "-sf", url], capture_output=True)
    return proc.returncode == 0


def get_status() -> dict[str, Any]:
    services = _service_states()
    profile = _active_profile(services)
    llm_url = f"http://{config.LAN_IP}:{config.LLAMA_PORT}/v1"

    if profile == "none":
        load_state = "STOPPED"
        api_state = "down"
        active_url = None
    elif profile.startswith("CONFLICT"):
        load_state = "CONFLICT"
        api_state = "multiple services — exclusive GPU violated"
        active_url = None
    else:
        base = f"http://{config.LAN_IP}:{config.LLAMA_PORT}"
        if _http_ok(f"{base}/health") or _http_ok(f"{base}/v1/models"):
            load_state = "LOADED"
            api_state = "ready"
        else:
            load_state = "STARTING"
            api_state = "not responding"
        active_url = llm_url

    return {
        "exclusive": True,
        "loadState": load_state,
        "profile": profile,
        "model": _model_hint(profile),
        "apiState": api_state,
        "activeUrl": active_url,
        "endpoints": {
            "llm": llm_url,
            "control": f"http://{config.LAN_IP}:{config.CONTROL_PORT}",
        },
        "services": services,
        "gpu": _gpu_info(),
        "gpuProcesses": _gpu_processes(),
    }


def start_profile(profile_id: str) -> dict[str, Any]:
    if profile_id not in config.VALID_PROFILES:
        raise ValueError(f"Unknown profile: {profile_id}")
    proc = _run_control("start", profile_id, check=False)
    if proc.returncode != 0:
        parts = [p for p in (proc.stdout.strip(), proc.stderr.strip()) if p]
        detail = "\n".join(parts) if parts else "start failed"
        raise RuntimeError(detail)
    return get_status()


def wait_for_profile(
    profile_id: str,
    *,
    timeout_sec: float = 60.0,
    poll_sec: float = 2.0,
) -> dict[str, Any]:
    """Poll until the requested profile is LOADED or timeout/failure."""
    if profile_id not in config.VALID_PROFILES:
        raise ValueError(f"Unknown profile: {profile_id}")
    deadline = time.monotonic() + timeout_sec
    last: dict[str, Any] = {}

    while time.monotonic() < deadline:
        last = get_status()
        load_state = last.get("loadState")
        profile = last.get("profile", "none")

        if profile.startswith("CONFLICT"):
            raise RuntimeError(f"GPU profile conflict: {profile}")

        services = last.get("services") or {}
        if services.get(profile_id) == "failed":
            raise RuntimeError(f"{profile_id} service failed to start (check logs on GPU box)")

        if profile == profile_id and load_state == "LOADED":
            return last

        threading.Event().wait(poll_sec)

    raise RuntimeError(
        f"Profile {profile_id} did not reach LOADED within {int(timeout_sec)}s "
        f"(last state: profile={last.get('profile')}, loadState={last.get('loadState')})"
    )


def stop_profile() -> None:
    proc = _run_control("stop", check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "stop failed")


def get_models() -> dict[str, Any]:
    import sys

    candidates = [
        config.PYTHON,
        Path(sys.executable),
        Path("python3"),
    ]
    python_bin = next((str(p) for p in candidates if p and Path(p).exists()), sys.executable)
    proc = subprocess.run(
        [
            python_bin,
            str(config.ROOT / "scripts" / "model_status.py"),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_model_env(),
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "model_status failed").strip()
        return {"bundles": [], "error": detail}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"bundles": [], "error": f"Invalid model_status JSON: {exc}"}


def _model_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("LLAMACPP_MODELS", str(Path.home() / "ai-inference" / "models" / "llamacpp"))
    env.setdefault("INSTALLED_JSON", str(config.ROOT / "catalog" / "installed.json"))
    return env
