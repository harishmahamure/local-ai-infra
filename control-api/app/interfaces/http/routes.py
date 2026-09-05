from __future__ import annotations

import json
import queue
from typing import Any

from fastapi import APIRouter, File, Header, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ...application.services import EngineServices
from ...domain.errors import DomainError
from .errors import domain_error_response
from .schemas import RetryJobRequest, SubmitJobRequest, TextChatRequest

router = APIRouter()


def _svc(request: Request) -> EngineServices:
    return request.app.state.engine


def _ok(payload: dict[str, Any], status: int = 200) -> JSONResponse:
    return JSONResponse(status_code=status, content=payload)


@router.post("/v1/jobs", status_code=202)
def submit_job(
    body: SubmitJobRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    try:
        job = _svc(request).submit_job(body.model_dump(), idempotency_key=idempotency_key, trace_id=getattr(request.state, "request_id", None))
        return _ok({"job_id": job.job_id, "status": "queued" if job.status.value == "QUEUED" else job.status.value, "operation": job.operation, "preset": job.preset}, 202)
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    try:
        return _ok(_svc(request).get_job(job_id).to_public_dict())
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/jobs/{job_id}/events")
def job_events(job_id: str, request: Request):
    services = _svc(request)
    try:
        services.get_job(job_id)
    except DomainError as exc:
        return domain_error_response(exc, request)

    def generate():
        seen = set()
        for event in services.events.history(job_id):
            key = event.get("timestamp", "") + event.get("event", "")
            seen.add(key)
            yield _sse(event)
            if event.get("event") in {"job.completed", "job.failed", "job.cancelled"}:
                return
        q = services.events.subscribe(job_id)
        while True:
            try:
                event = q.get(timeout=15)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            key = event.get("timestamp", "") + event.get("event", "")
            if key in seen:
                continue
            yield _sse(event)
            if event.get("event") in {"job.completed", "job.failed", "job.cancelled"}:
                return

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request):
    try:
        return _ok(_svc(request).cancel_job(job_id).to_public_dict())
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/jobs/{job_id}/retry")
def retry_job(job_id: str, body: RetryJobRequest, request: Request):
    try:
        job = _svc(request).retry_job(job_id, body.strategy, body.parameters, body.stage)
        return _ok({"job_id": job.job_id, "status": "queued", "operation": job.operation, "preset": job.preset}, 202)
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/operations")
def list_operations(request: Request):
    return _ok(_svc(request).list_operations())


@router.get("/v1/capabilities")
def list_capabilities(request: Request):
    return _ok(_svc(request).list_capabilities())


@router.get("/v1/presets")
def list_presets(request: Request):
    return _ok(_svc(request).list_presets())


@router.get("/v1/models")
def list_models(request: Request):
    return _ok(_svc(request).list_models())


@router.get("/v1/models/{model_id}")
def get_model(model_id: str, request: Request):
    try:
        return _ok(_svc(request).get_model(model_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/models/{model_id}/download", status_code=202)
def download_model(model_id: str, request: Request):
    try:
        return _ok(_svc(request).download_model(model_id), 202)
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/models/{model_id}/verify")
def verify_model(model_id: str, request: Request):
    try:
        return _ok(_svc(request).verify_model(model_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/models/{model_id}/load")
def load_model(model_id: str, request: Request):
    try:
        return _ok(_svc(request).load_model(model_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/models/{model_id}/unload")
def unload_model(model_id: str, request: Request):
    try:
        return _ok(_svc(request).unload_model(model_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/models/{model_id}/start")
def start_model(model_id: str, request: Request):
    try:
        return _ok(_svc(request).start_model(model_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/models/{model_id}/stop")
def stop_model(model_id: str, request: Request):
    try:
        return _ok(_svc(request).stop_model(model_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/text/models")
def list_text_models(request: Request):
    return _ok(_svc(request).list_text_models())


@router.post("/v1/text/chat")
def text_chat(body: TextChatRequest, request: Request):
    try:
        payload = body.model_dump()
        if body.stream:
            return StreamingResponse(_svc(request).chat_stream(payload), media_type="text/event-stream")
        return _ok(_svc(request).chat(payload))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/loras")
def list_loras(request: Request):
    return _ok(_svc(request).list_loras())


@router.get("/v1/workflows")
def list_workflows(request: Request):
    return _ok(_svc(request).list_workflows())


@router.get("/v1/workflows/{workflow_id}")
def get_workflow(workflow_id: str, request: Request):
    try:
        return _ok(_svc(request).get_workflow(workflow_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/workflows/{workflow_id}/validate")
def validate_workflow(workflow_id: str, request: Request):
    try:
        return _ok(_svc(request).validate_workflow(workflow_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/assets")
async def upload_asset(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    try:
        artifact = _svc(request).upload_asset(raw, file.content_type or "application/octet-stream", file.filename)
        return _ok(artifact.to_public_dict(), 201)
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/assets/{asset_id}")
def get_asset(asset_id: str, request: Request):
    try:
        return _ok(_svc(request).get_asset(asset_id).to_public_dict())
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.delete("/v1/assets/{asset_id}")
def delete_asset(asset_id: str, request: Request):
    try:
        _svc(request).delete_asset(asset_id)
        return Response(status_code=204)
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/runtime")
def runtime(request: Request):
    return _ok(_svc(request).runtime())


@router.get("/ready")
def ready(request: Request):
    ok, details = _svc(request).ready()
    return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, **details})


def _sse(event: dict[str, Any]) -> str:
    return f"event: {event.get('event', 'message')}\ndata: {json.dumps(event)}\n\n"
