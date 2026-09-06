from __future__ import annotations

import threading
from typing import Any

from ..domain.errors import DomainError, ErrorCode
from ..domain.runtime import ResourceLease, ResourceRequirement

PROFILE_ALIASES = {
    "comfy": "comfy",
    "comfyui": "comfy",
    "comfy-ltx": "comfy-ltx",
    "comfyui-ltx": "comfy-ltx",
}


class ProfileResourceScheduler:
    """Single-GPU scheduler: concurrency 1, reuse profile when possible."""

    def __init__(self, runtime_module: Any, interrupt_fn=None) -> None:
        self._runtime = runtime_module
        self._interrupt_fn = interrupt_fn
        self._lock = threading.Lock()
        self._active_job: str | None = None
        self._loaded_profile: str | None = None
        self._estimated_vram: float = 0.0
        self._queue_depth = 0

    def set_queue_depth(self, depth: int) -> None:
        self._queue_depth = depth

    def acquire(self, job_id: str, requirement: ResourceRequirement) -> ResourceLease:
        profile = PROFILE_ALIASES.get(requirement.profile, requirement.profile)
        with self._lock:
            if self._active_job and self._active_job != job_id:
                raise DomainError(ErrorCode.INTERNAL_ERROR, "GPU worker invariant violated: concurrent acquire")
            self._active_job = job_id
        if requirement.gpu_required:
            self._ensure_profile(profile)
            if profile != "tts":
                self._check_vram(requirement.estimated_vram_gb)
        self._loaded_profile = profile
        self._estimated_vram = requirement.estimated_vram_gb
        return ResourceLease(job_id=job_id, profile=profile, estimated_vram_gb=requirement.estimated_vram_gb)

    def release(self, lease: ResourceLease) -> None:
        with self._lock:
            if self._active_job == lease.job_id:
                self._active_job = None

    def snapshot(self) -> dict[str, Any]:
        status = {}
        try:
            status = self._runtime.get_status()
        except Exception as exc:
            status = {"error": str(exc)}
        gpu = status.get("gpu") or {}
        return {
            "gpu_name": gpu.get("name"),
            "total_vram": gpu.get("memoryTotal"),
            "available_vram": gpu.get("memoryUsed"),
            "loaded_model": status.get("model"),
            "active_profile": status.get("profile"),
            "active_job": self._active_job,
            "queue_depth": self._queue_depth,
            "comfyui_health": status.get("loadState"),
            "worker_health": "alive",
        }

    def interrupt(self, job_id: str) -> None:
        if self._interrupt_fn:
            self._interrupt_fn(job_id)

    def _ensure_profile(self, profile: str) -> None:
        try:
            status = self._runtime.get_status()
        except Exception as exc:
            raise DomainError(ErrorCode.COMFYUI_UNAVAILABLE, str(exc)) from exc
        current = str(status.get("profile") or "none")
        if profile == "tts":
            try:
                self._runtime.start_profile("tts")
            except Exception as exc:
                raise DomainError(ErrorCode.COMFYUI_UNAVAILABLE, str(exc)) from exc
            return
        unit = "comfyui" if profile == "comfy" else "comfyui-ltx" if profile == "comfy-ltx" else profile
        if current == unit and status.get("loadState") == "LOADED":
            return
        try:
            self._runtime.start_profile(profile)
            if hasattr(self._runtime, "wait_for_profile"):
                self._runtime.wait_for_profile(profile, timeout_sec=90)
        except Exception as exc:
            raise DomainError(ErrorCode.COMFYUI_UNAVAILABLE, str(exc)) from exc

    def _check_vram(self, needed: float) -> None:
        if needed <= 0:
            return
        try:
            status = self._runtime.get_status()
            gpu = status.get("gpu") or {}
            total_s = str(gpu.get("memoryTotal") or "")
            used_s = str(gpu.get("memoryUsed") or "")
            total = _parse_mib(total_s)
            used = _parse_mib(used_s)
            if total is None:
                return
            free_gb = max(0.0, (total - (used or 0)) / 1024.0)
            if needed - 1.0 > free_gb and used and used > 1024:
                # After a profile switch Comfy already owns the VRAM; only fail if clearly over card size.
                card_gb = total / 1024.0
                if needed > card_gb + 0.5:
                    raise DomainError(ErrorCode.VRAM_INSUFFICIENT, f"Need {needed} GB VRAM, card has {card_gb:.1f} GB")
        except DomainError:
            raise
        except Exception:
            return


def _parse_mib(value: str) -> float | None:
    try:
        return float(value.replace("MiB", "").replace("GiB", "").strip()) * (1024.0 if "GiB" in value else 1.0)
    except (TypeError, ValueError):
        return None
