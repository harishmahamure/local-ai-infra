from __future__ import annotations

from typing import Any

from ..application.ports import ExecutionRequest, ExecutionResult, WorkflowExecutor
from ..domain.errors import DomainError, ErrorCode


class RoutingExecutor:
    """Dispatch by workflow.executor (comfyui | tts | ffmpeg)."""

    def __init__(self, executors: dict[str, WorkflowExecutor]) -> None:
        self._executors = executors

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        key = str(request.workflow.executor or "")
        impl = self._executors.get(key)
        if impl is None:
            raise DomainError(ErrorCode.UNSUPPORTED_OPERATION, f"No executor for {key or request.workflow.id}")
        return impl.execute(request)

    def cancel(self, job_id: str) -> None:
        for impl in self._executors.values():
            try:
                impl.cancel(job_id)
            except Exception:
                continue

    def health(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, impl in self._executors.items():
            try:
                out[key] = impl.health()
            except Exception as exc:
                out[key] = {"error": str(exc)}
        return out
