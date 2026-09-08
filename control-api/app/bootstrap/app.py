from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from ..application.chat import ControlServices, build_services
from ..interfaces.http.routes import router as v1_router


def create_app(engine: ControlServices | None = None) -> FastAPI:
    app = FastAPI(
        title="llama.cpp Control API",
        version="4.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        description=(
            "Minimal llama.cpp control plane: GPU profile load/unload, GGUF downloads, "
            "and `/v1/text/chat` (sync or SSE).\n\n"
            "Mac proxy: `http://127.0.0.1:8090` → GPU `http://192.168.50.100:8090`.\n"
            "Interactive docs: `/docs` (Swagger) and `/redoc`."
        ),
        servers=[
            {"url": "http://127.0.0.1:8090", "description": "Mac proxy (LAN)"},
            {"url": "http://192.168.50.100:8090", "description": "GPU box (direct)"},
        ],
        openapi_tags=[
            {"name": "Health", "description": "Liveness and readiness."},
            {"name": "llama.cpp", "description": "Text/vision chat, model load, GGUF downloads."},
        ],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        return await call_next(request)

    app.include_router(v1_router)
    app.state.engine = engine or build_services()
    return app
