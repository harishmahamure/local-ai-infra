from __future__ import annotations

import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from ...domain.errors import DomainError, ErrorCode


def error_payload(code: ErrorCode | str, message: str, request: Request, details: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(code, ErrorCode):
        retryable = code.retryable
        code_s = code.value
    else:
        retryable = False
        code_s = code
    return {
        "error": {
            "code": code_s,
            "message": message,
            "details": details or {},
            "retryable": retryable,
            "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
        }
    }


def domain_error_response(exc: DomainError, request: Request) -> JSONResponse:
    return JSONResponse(status_code=exc.code.http_status, content=error_payload(exc.code, exc.message, request, exc.details))
