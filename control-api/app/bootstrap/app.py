from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..application.chat import ControlServices, build_services
from ..interfaces.http.routes import router as v1_router

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
RESERVED_PREFIXES = frozenset(
    {"v1", "static", "assets", "docs", "redoc", "health", "ready", "openapi.json", "openapi.yaml"}
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine: ControlServices = app.state.engine
    engine.start()
    yield
    engine.stop()


def create_app(engine: ControlServices | None = None) -> FastAPI:
    app = FastAPI(
        title="GPU Control API",
        version="5.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        description=(
            "Control plane: exclusive GPU profiles (llama.cpp chat + ComfyUI Qwen image), "
            "GGUF downloads, `/v1/text/chat`, and `/v1/image/jobs` (one GPU job at a time).\n\n"
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
            {"name": "image", "description": "Qwen image jobs, assets, and GPU queue."},
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

    def _index():
        index = STATIC_DIR / "index.html"
        if not index.is_file():
            return {"error": "dashboard not installed"}
        return FileResponse(index)

    @app.get("/", include_in_schema=False)
    def dashboard():
        return _index()

    app.include_router(v1_router)
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str):
        root = path.split("/", 1)[0]
        if root in RESERVED_PREFIXES:
            raise HTTPException(status_code=404, detail="Not found")
        existing = STATIC_DIR / path
        if existing.is_file():
            return FileResponse(existing)
        return _index()

    app.state.engine = engine or build_services()
    return app
