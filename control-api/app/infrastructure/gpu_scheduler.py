from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from ..domain.errors import DomainError, ErrorCode

COMFY_PROFILES = frozenset({"comfyui", "comfy-ltx"})


@dataclass
class ResourceLease:
    job_id: str
    profile: str = "comfyui"


class GpuScheduler:
    """Single-GPU scheduler: concurrency 1, auto-switch to the job's Comfy profile."""

    def __init__(self, runtime_module: Any) -> None:
        self._runtime = runtime_module
        self._lock = threading.Lock()
        self._active_job: str | None = None

    def is_busy(self) -> bool:
        return self._active_job is not None

    def active_job(self) -> str | None:
        return self._active_job

    def acquire(self, job_id: str, profile: str = "comfyui") -> ResourceLease:
        if profile not in COMFY_PROFILES:
            raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unknown GPU profile {profile}")
        with self._lock:
            if self._active_job and self._active_job != job_id:
                raise DomainError(
                    ErrorCode.INTERNAL_ERROR,
                    "GPU worker invariant violated: concurrent acquire",
                    {"active_job": self._active_job, "requested": job_id},
                )
            self._active_job = job_id
        try:
            self._ensure_profile(profile)
        except Exception:
            with self._lock:
                if self._active_job == job_id:
                    self._active_job = None
            raise
        return ResourceLease(job_id=job_id, profile=profile)

    def switch(self, lease: ResourceLease, profile: str) -> ResourceLease:
        if profile not in COMFY_PROFILES:
            raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unknown GPU profile {profile}")
        if profile == lease.profile:
            return lease
        with self._lock:
            if self._active_job and self._active_job != lease.job_id:
                raise DomainError(
                    ErrorCode.INTERNAL_ERROR,
                    "GPU worker invariant violated: concurrent switch",
                    {"active_job": self._active_job, "requested": lease.job_id},
                )
        self._ensure_profile(profile)
        return ResourceLease(job_id=lease.job_id, profile=profile)

    def release(self, lease: ResourceLease) -> None:
        with self._lock:
            if self._active_job == lease.job_id:
                self._active_job = None

    def snapshot(self) -> dict[str, Any]:
        return {"activeJobId": self._active_job, "busy": self._active_job is not None}

    def _ensure_profile(self, profile: str) -> None:
        try:
            status = self._runtime.get_status()
        except Exception as exc:
            raise DomainError(ErrorCode.COMFYUI_UNAVAILABLE, str(exc)) from exc
        current = str(status.get("profile") or "none")
        if current == profile and status.get("loadState") == "LOADED":
            return
        try:
            if current not in {"none", profile} and not str(current).startswith("CONFLICT"):
                self._runtime.stop_profile()
            self._runtime.start_profile(profile)
            if hasattr(self._runtime, "wait_for_profile"):
                timeout = 240 if profile == "comfy-ltx" else 180
                self._runtime.wait_for_profile(profile, timeout_sec=timeout)
        except DomainError:
            raise
        except Exception as exc:
            raise DomainError(ErrorCode.COMFYUI_UNAVAILABLE, str(exc)) from exc
