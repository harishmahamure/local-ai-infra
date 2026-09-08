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

    @property
    def retry_category(self) -> RetryCategory:
        if self in {ErrorCode.MODEL_DOWNLOAD_FAILED, ErrorCode.INTERNAL_ERROR}:
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
        }
        return mapping[self]


@dataclass
class DomainError(Exception):
    code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message
