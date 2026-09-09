from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Header, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ...application.chat import ControlServices
from ...domain.errors import DomainError, ErrorCode
from .errors import domain_error_response
from .schemas import DownloadRequest, ImageJobRequest, TextChatRequest

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


def _jobs(request: Request):
    jobs = _svc(request).jobs
    if jobs is None:
        raise DomainError(ErrorCode.INTERNAL_ERROR, "Image job engine is not enabled")
    return jobs


@router.get("/v1/image/operations", summary="List image operations", tags=["image"])
def list_image_operations(request: Request):
    try:
        return _ok(_jobs(request).list_operations())
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/image/jobs", status_code=202, summary="Queue an image job", tags=["image"])
def submit_image_job(
    body: ImageJobRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    try:
        payload = body.model_dump(exclude_none=True)
        return _ok(_jobs(request).submit(payload, idempotency_key=idempotency_key), 202)
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/jobs", summary="List jobs", tags=["image"])
def list_jobs(
    request: Request,
    status: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
):
    try:
        return _ok(_jobs(request).list_jobs(status=status, cursor=cursor, limit=limit))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.delete("/v1/jobs", summary="Delete jobs", tags=["image"])
def delete_jobs(
    request: Request,
    status: str | None = Query(default=None, description="If set, only delete jobs in this status"),
    delete_assets: bool = Query(default=True, description="Also delete generated images from disk"),
):
    try:
        return _ok(_jobs(request).delete_jobs(status=status, delete_assets=delete_assets))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/jobs/{job_id}", summary="Get job status", tags=["image"])
def get_job(job_id: str, request: Request):
    try:
        return _ok(_jobs(request).get(job_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/jobs/{job_id}/events", summary="Job status SSE stream", tags=["image"])
def job_events(job_id: str, request: Request):
    try:
        jobs = _jobs(request)

        def generate():
            for payload in jobs.events(job_id):
                yield f"event: job\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/jobs/{job_id}/cancel", summary="Cancel a job", tags=["image"])
def cancel_job(job_id: str, request: Request):
    try:
        return _ok(_jobs(request).cancel(job_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.delete("/v1/jobs/{job_id}", summary="Delete a job", tags=["image"])
def delete_job(
    job_id: str,
    request: Request,
    delete_assets: bool = Query(default=True, description="Also delete generated images from disk"),
):
    try:
        return _ok(_jobs(request).delete_job(job_id, delete_assets=delete_assets))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.delete("/v1/jobs/{job_id}/assets", summary="Delete all images for a job", tags=["image"])
def delete_job_assets(job_id: str, request: Request):
    try:
        return _ok(_jobs(request).delete_job_assets(job_id))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.post("/v1/assets", summary="Upload an asset", tags=["image"])
async def upload_asset(request: Request, file: UploadFile = File(...)):
    try:
        data = await file.read()
        record = _jobs(request).assets.write(data, mime_type=file.content_type, filename=file.filename)
        return _ok(_jobs(request).assets.public_dict(record), 201)
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.delete("/v1/assets", summary="Delete all assets from disk", tags=["image"])
def delete_all_assets(request: Request):
    try:
        return _ok(_jobs(request).delete_all_assets())
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/assets/{asset_id}", summary="Get asset metadata", tags=["image"])
def get_asset(asset_id: str, request: Request):
    try:
        record = _jobs(request).assets.get(asset_id)
        return _ok(_jobs(request).assets.public_dict(record))
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.get("/v1/assets/{asset_id}/content", summary="Download asset bytes", tags=["image"])
def get_asset_content(asset_id: str, request: Request):
    try:
        data, record = _jobs(request).assets.read(asset_id)
        return Response(content=data, media_type=record.get("mime_type") or "application/octet-stream")
    except DomainError as exc:
        return domain_error_response(exc, request)


@router.delete("/v1/assets/{asset_id}", summary="Delete an asset from disk", tags=["image"])
def delete_asset(asset_id: str, request: Request):
    try:
        return _ok(_jobs(request).delete_asset(asset_id))
    except DomainError as exc:
        return domain_error_response(exc, request)
