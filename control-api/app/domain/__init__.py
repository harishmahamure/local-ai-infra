"""Framework-independent llama.cpp control domain."""

from .errors import DomainError, ErrorCode, RetryCategory
from .jobs import Job, JobStatus, can_transition
from .models import ModelFile, ModelRecord, ModelState
from .operations import OPERATIONS, VIDEO_OPERATIONS, OperationSpec
from .shots import FLOWS, SHOT_OPERATIONS

__all__ = [
    "DomainError",
    "ErrorCode",
    "FLOWS",
    "Job",
    "JobStatus",
    "ModelFile",
    "ModelRecord",
    "ModelState",
    "OPERATIONS",
    "SHOT_OPERATIONS",
    "VIDEO_OPERATIONS",
    "OperationSpec",
    "RetryCategory",
    "can_transition",
]
