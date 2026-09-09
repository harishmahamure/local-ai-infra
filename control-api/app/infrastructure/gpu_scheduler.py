from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from ..domain.errors import DomainError, ErrorCode


@dataclass
class ResourceLease:
    job_id: str
    profile: str = "comfyui"


class GpuScheduler:
    """Single-GPU scheduler: concurrency 1, auto-switch to ComfyUI."""

    def __init__(self, runtime_module: Any) -> None:
        self._runtime = runtime_module
        self._lock = threading.Lock()
        self._active_job: str | None = None

    def is_busy(self) -> bool:
        return self._active_job is not None

    def active_job(self) -> str | None:
        return self._active_job

    def acquire(self, job_id: str) -> ResourceLease:
        with self._lock:
            if self._active_job and self._active_job != job_id:
                raise DomainError(
                    ErrorCode.INTERNAL_ERROR,
                    "GPU worker invariant violated: concurrent acquire",
                    {"active_job": self._active_job, "requested": job_id},
                )
            self._active_job = job_id
        try:
            self._ensure_comfyui()
        except Exception:
            with self._lock:
                if self._active_job == job_id:
                    self._active_job = None
            raise
        return ResourceLease(job_id=job_id, profile="comfyui")

    def release(self, lease: ResourceLease) -> None:
        with self._lock:
            if self._active_job == lease.job_id:
                self._active_job = None

    def snapshot(self) -> dict[str, Any]:
        return {"activeJobId": self._active_job, "busy": self._active_job is not None}

    def _ensure_comfyui(self) -> None:
        try:
            status = self._runtime.get_status()
        except Exception as exc:
            raise DomainError(ErrorCode.COMFYUI_UNAVAILABLE, str(exc)) from exc
        current = str(status.get("profile") or "none")
        if current == "comfyui" and status.get("loadState") == "LOADED":
            return
        try:
            if current not in {"none", "comfyui"} and not str(current).startswith("CONFLICT"):
                self._runtime.stop_profile()
            self._runtime.start_profile("comfyui")
            if hasattr(self._runtime, "wait_for_profile"):
                self._runtime.wait_for_profile("comfyui", timeout_sec=180)
        except DomainError:
            raise
        except Exception as exc:
            raise DomainError(ErrorCode.COMFYUI_UNAVAILABLE, str(exc)) from exc
