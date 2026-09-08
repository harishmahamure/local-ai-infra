from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RetryCategory(str, Enum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    OOM = "oom"


class ErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    MODEL_DOWNLOAD_FAILED = "MODEL_DOWNLOAD_FAILED"
    MODEL_INCOMPATIBLE = "MODEL_INCOMPATIBLE"
    LORA_INCOMPATIBLE = "LORA_INCOMPATIBLE"
    LICENSE_RESTRICTED = "LICENSE_RESTRICTED"
    WORKFLOW_NOT_AVAILABLE = "WORKFLOW_NOT_AVAILABLE"
    WORKFLOW_INVALID = "WORKFLOW_INVALID"
    NODE_NOT_INSTALLED = "NODE_NOT_INSTALLED"
    ASSET_NOT_FOUND = "ASSET_NOT_FOUND"
    ASSET_EXPIRED = "ASSET_EXPIRED"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    UNSUPPORTED_PRESET = "UNSUPPORTED_PRESET"
    UNSUPPORTED_PARAMETER = "UNSUPPORTED_PARAMETER"
    VRAM_INSUFFICIENT = "VRAM_INSUFFICIENT"
    COMFYUI_UNAVAILABLE = "COMFYUI_UNAVAILABLE"
    COMFYUI_ERROR = "COMFYUI_ERROR"
    FFMPEG_ERROR = "FFMPEG_ERROR"
    GENERATION_TIMEOUT = "GENERATION_TIMEOUT"
    QC_FAILED = "QC_FAILED"
    CANCELLED = "CANCELLED"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    GPU_OOM = "GPU_OOM"
    STAGE_TIMEOUT = "STAGE_TIMEOUT"
    LEASE_LOST = "LEASE_LOST"
    DEPENDENCY_FAILED = "DEPENDENCY_FAILED"
    WORKER_LOST = "WORKER_LOST"
    ARTIFACT_PERSIST_FAILED = "ARTIFACT_PERSIST_FAILED"

    @property
    def retry_category(self) -> RetryCategory:
        if self == ErrorCode.GPU_OOM:
            return RetryCategory.OOM
        if self in {
            ErrorCode.COMFYUI_UNAVAILABLE,
            ErrorCode.GENERATION_TIMEOUT,
            ErrorCode.MODEL_DOWNLOAD_FAILED,
            ErrorCode.INTERNAL_ERROR,
            ErrorCode.STAGE_TIMEOUT,
            ErrorCode.LEASE_LOST,
            ErrorCode.WORKER_LOST,
            ErrorCode.ARTIFACT_PERSIST_FAILED,
        }:
            return RetryCategory.RETRYABLE
        return RetryCategory.NON_RETRYABLE

    @property
    def retryable(self) -> bool:
        return self.retry_category == RetryCategory.RETRYABLE

    @property
    def http_status(self) -> int:
        mapping = {
            ErrorCode.INVALID_REQUEST: 400,
            ErrorCode.INVALID_PARAMETER: 400,
            ErrorCode.UNSUPPORTED_PARAMETER: 400,
            ErrorCode.UNSUPPORTED_OPERATION: 400,
            ErrorCode.UNSUPPORTED_PRESET: 400,
            ErrorCode.LORA_INCOMPATIBLE: 400,
            ErrorCode.MODEL_INCOMPATIBLE: 409,
            ErrorCode.LICENSE_RESTRICTED: 403,
            ErrorCode.MODEL_NOT_AVAILABLE: 409,
            ErrorCode.MODEL_DOWNLOAD_FAILED: 502,
            ErrorCode.WORKFLOW_NOT_AVAILABLE: 409,
            ErrorCode.WORKFLOW_INVALID: 409,
            ErrorCode.NODE_NOT_INSTALLED: 409,
            ErrorCode.ASSET_NOT_FOUND: 404,
            ErrorCode.ASSET_EXPIRED: 410,
            ErrorCode.VRAM_INSUFFICIENT: 409,
            ErrorCode.COMFYUI_UNAVAILABLE: 503,
            ErrorCode.COMFYUI_ERROR: 502,
            ErrorCode.FFMPEG_ERROR: 502,
            ErrorCode.GENERATION_TIMEOUT: 504,
            ErrorCode.QC_FAILED: 422,
            ErrorCode.CANCELLED: 409,
            ErrorCode.IDEMPOTENCY_CONFLICT: 409,
            ErrorCode.INTERNAL_ERROR: 500,
            ErrorCode.GPU_OOM: 507,
            ErrorCode.STAGE_TIMEOUT: 504,
            ErrorCode.LEASE_LOST: 503,
            ErrorCode.DEPENDENCY_FAILED: 409,
            ErrorCode.WORKER_LOST: 503,
            ErrorCode.ARTIFACT_PERSIST_FAILED: 500,
        }
        return mapping[self]


@dataclass
class DomainError(Exception):
    code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message
