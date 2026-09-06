from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..application.services import EngineServices
from ..interfaces.http.legacy import STATIC_DIR, legacy_router
from ..interfaces.http.routes import router as v1_router
from .container import build_engine


def create_app(engine: EngineServices | None = None) -> FastAPI:
    app = FastAPI(
        title="AI Media Inference Engine",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        description=(
            "Two surfaces on the same server:\n\n"
            "- **Engine `/v1`** — catalog operations, job queue, assets, text chat.\n"
            "- **Control `/api/v1`** — GPU profiles, catalog downloads, generate / upscale / LTX, cleanup.\n\n"
            "Mac proxy: `http://127.0.0.1:8090` → GPU `http://192.168.50.100:8090`.\n"
            "Interactive docs: `/docs` (Swagger) and `/redoc`."
        ),
        servers=[
            {"url": "http://127.0.0.1:8090", "description": "Mac proxy (LAN)"},
            {"url": "http://192.168.50.100:8090", "description": "GPU box (direct)"},
        ],
        openapi_tags=[
            {"name": "Health", "description": "Liveness and readiness."},
            {"name": "Engine (/v1)", "description": "Stable engine API: jobs, catalog, assets, text chat."},
            {"name": "Control UI (legacy /api/v1)", "description": "Control dashboard API used by the static UI."},
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
    app.include_router(legacy_router)
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.state.engine = engine or build_engine()
    return app
