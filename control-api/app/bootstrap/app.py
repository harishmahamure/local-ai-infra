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
