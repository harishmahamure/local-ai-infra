from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RetryCategory(str, Enum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"


class ErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    MODEL_DOWNLOAD_FAILED = "MODEL_DOWNLOAD_FAILED"
    LICENSE_RESTRICTED = "LICENSE_RESTRICTED"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    GPU_BUSY = "GPU_BUSY"
    COMFYUI_UNAVAILABLE = "COMFYUI_UNAVAILABLE"
    GENERATION_FAILED = "GENERATION_FAILED"
    GENERATION_TIMEOUT = "GENERATION_TIMEOUT"
    ASSET_NOT_FOUND = "ASSET_NOT_FOUND"
    CANCELLED = "CANCELLED"

    @property
    def retry_category(self) -> RetryCategory:
        if self in {
            ErrorCode.MODEL_DOWNLOAD_FAILED,
            ErrorCode.INTERNAL_ERROR,
            ErrorCode.GPU_BUSY,
            ErrorCode.COMFYUI_UNAVAILABLE,
            ErrorCode.GENERATION_TIMEOUT,
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
            ErrorCode.UNSUPPORTED_OPERATION: 400,
            ErrorCode.LICENSE_RESTRICTED: 403,
            ErrorCode.MODEL_NOT_AVAILABLE: 409,
            ErrorCode.MODEL_DOWNLOAD_FAILED: 502,
            ErrorCode.INTERNAL_ERROR: 500,
            ErrorCode.GPU_BUSY: 409,
            ErrorCode.COMFYUI_UNAVAILABLE: 503,
            ErrorCode.GENERATION_FAILED: 500,
            ErrorCode.GENERATION_TIMEOUT: 504,
            ErrorCode.ASSET_NOT_FOUND: 404,
            ErrorCode.CANCELLED: 409,
        }
        return mapping[self]


@dataclass
class DomainError(Exception):
    code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message
