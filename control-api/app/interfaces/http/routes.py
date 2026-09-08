from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...application.chat import ControlServices
from ...domain.errors import DomainError
from .errors import domain_error_response
from .schemas import DownloadRequest, TextChatRequest

router = APIRouter(tags=["llama.cpp"])


def _svc(request: Request) -> ControlServices:
    return request.app.state.engine


def _ok(payload: dict[str, Any], status: int = 200) -> JSONResponse:
    return JSONResponse(status_code=status, content=payload)


@router.get("/health", summary="Liveness probe", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe", tags=["Health"], description="200 if a llama.cpp profile is LOADED; 503 otherwise.")
def ready(request: Request):
    ok, details = _svc(request).ready()
    return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, **details})


@router.get("/v1/status", summary="GPU runtime status")
def status(request: Request):
    return _ok(_svc(request).status())


@router.get("/v1/models", summary="Catalog files on disk plus runtime state")
def list_models(request: Request):
    disk = _svc(request).disk_models()
    catalog = _svc(request).list_models()
    return _ok({**disk, **catalog})


@router.get("/v1/models/{model_id}", summary="Get one catalog model")
def get_model(model_id: str, request: Request):
    try:
        return _ok(_svc(request).get_model(model_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/models/{model_id}/load", summary="Load model onto the GPU")
def load_model(model_id: str, request: Request):
    try:
        return _ok(_svc(request).load_model(model_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/models/{model_id}/unload", summary="Unload model from the GPU")
def unload_model(model_id: str, request: Request):
    try:
        return _ok(_svc(request).unload_model(model_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/text/models", summary="List text/vision chat models")
def list_text_models(request: Request):
    return _ok(_svc(request).list_text_models())


@router.post(
    "/v1/text/chat",
    summary="Text + vision chat",
    description="Synchronous JSON or SSE when stream=true. Starts llama-fast or gemma as needed.",
)
def text_chat(body: TextChatRequest, request: Request):
    try:
        payload = body.model_dump()
        if body.stream:
            return StreamingResponse(_svc(request).chat_stream(payload), media_type="text/event-stream")
        return _ok(_svc(request).chat(payload))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/downloads", summary="GGUF download status")
def downloads_get(request: Request):
    return _ok(_svc(request).download_status())


@router.post("/v1/downloads", status_code=202, summary="Queue a GGUF download")
def downloads_post(body: DownloadRequest, request: Request):
    try:
        return _ok(_svc(request).start_download(body.ids), 202)
    except DomainError as exc:
        return domain_error_response(exc, request)
