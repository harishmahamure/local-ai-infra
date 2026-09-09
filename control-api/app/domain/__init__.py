"""Framework-independent llama.cpp control domain."""

from .errors import DomainError, ErrorCode, RetryCategory
from .jobs import Job, JobStatus, can_transition
from .models import ModelFile, ModelRecord, ModelState
from .operations import OPERATIONS, OperationSpec

__all__ = [
    "DomainError",
    "ErrorCode",
    "Job",
    "JobStatus",
    "ModelFile",
    "ModelRecord",
    "ModelState",
    "OPERATIONS",
    "OperationSpec",
    "RetryCategory",
    "can_transition",
]
