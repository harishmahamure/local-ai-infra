from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from .errors import ErrorCode, RetryCategory


class RetryDecision(str, Enum):
    RETRY = "RETRY"
    FAIL = "FAIL"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay_seconds: float = 5.0
    max_delay_seconds: float = 300.0
    multiplier: float = 2.0
    jitter: bool = True
    allow_oom_fallback: bool = False

    def delay_for(self, retry_count: int) -> float:
        delay = min(self.max_delay_seconds, self.base_delay_seconds * (self.multiplier**retry_count))
        if self.jitter:
            delay = delay * (0.5 + random.random())
        return delay


def classify_retry(code: ErrorCode, policy: RetryPolicy, retry_count: int) -> RetryDecision:
    if code.retry_category == RetryCategory.NON_RETRYABLE:
        return RetryDecision.FAIL
    if code == ErrorCode.GPU_OOM:
        if policy.allow_oom_fallback:
            return RetryDecision.FALLBACK
        return RetryDecision.FAIL
    if retry_count >= policy.max_retries:
        return RetryDecision.FAIL
    if code.retry_category == RetryCategory.RETRYABLE:
        return RetryDecision.RETRY
    return RetryDecision.FAIL
