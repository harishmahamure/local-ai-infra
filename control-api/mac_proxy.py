"""Mac LAN proxy — forwards to GPU control API."""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

GPU_CONTROL_URL = os.environ.get("GPU_CONTROL_URL", "http://192.168.50.100:8090").rstrip("/")

# Docs live on the GPU control API. Disable them here so /docs, /redoc,
# and /openapi.json are forwarded instead of the proxy's two-route stub.
app = FastAPI(
    title="AI Control Proxy",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "proxy": "mac", "upstream": GPU_CONTROL_URL}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy(path: str, request: Request) -> Response:
    url = f"{GPU_CONTROL_URL}/{path}" if path else GPU_CONTROL_URL
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = {k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}}
    body = await request.body()

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            upstream = await client.request(
                request.method,
                url,
                headers=headers,
                content=body if body else None,
            )
    except httpx.ConnectError:
        return Response(
            content=f"GPU control API is unreachable at {GPU_CONTROL_URL}. Check ai-control.service on the GPU box.",
            status_code=502,
            media_type="text/plain",
        )
    except httpx.HTTPError as exc:
        return Response(
            content=f"GPU control API request failed: {exc}",
            status_code=502,
            media_type="text/plain",
        )

    skip = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in skip}
    return Response(content=upstream.content, status_code=upstream.status_code, headers=resp_headers)
