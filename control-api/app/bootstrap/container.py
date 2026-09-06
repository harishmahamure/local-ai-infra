from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import config, downloads, runtime
from ..application.services import EngineServices
from ..infrastructure.catalogs import load_catalogs
from ..infrastructure.clock import SystemClock, UlidGenerator
from ..infrastructure.comfyui.executor import ComfyUIExecutor
from ..infrastructure.ffmpeg.executor import FfmpegExecutor
from ..infrastructure.routing import RoutingExecutor
from ..infrastructure.tts.executor import TtsExecutor
from ..infrastructure.events import InProcessEventPublisher
from ..infrastructure.gpu import ProfileResourceScheduler
from ..infrastructure.inspect import BasicMediaInspector
from ..infrastructure.model_runtime import CatalogModelRuntime, SystemdModelDownloader
from ..infrastructure.persistence.memory import (
    MemoryArtifactRepository,
    MemoryIdempotencyRepository,
    MemoryJobRepository,
)
from ..infrastructure.persistence.sqlite import (
    SqliteArtifactRepository,
    SqliteEventStore,
    SqliteIdempotencyRepository,
    SqliteJobRepository,
    SqliteStore,
)
from ..infrastructure.storage import LocalArtifactStorage
from ..infrastructure.worker import JobWorker


def build_engine(
    *,
    memory: bool = False,
    executor: Any | None = None,
    model_runtime: Any | None = None,
    downloader: Any | None = None,
    scheduler: Any | None = None,
    start_worker: bool = True,
    catalog_dir: Path | None = None,
    artifact_root: Path | None = None,
    db_path: Path | None = None,
    commercial_mode: bool | None = None,
) -> EngineServices:
    catalog = load_catalogs(catalog_dir or config.CATALOG_DIR, commercial_mode=config.COMMERCIAL_MODE if commercial_mode is None else commercial_mode)
    clock = SystemClock()
    ids = UlidGenerator()
    storage = LocalArtifactStorage(artifact_root or config.ARTIFACT_ROOT)
    if memory:
        jobs = MemoryJobRepository()
        artifacts = MemoryArtifactRepository()
        idempotency = MemoryIdempotencyRepository()
        events = InProcessEventPublisher(clock=clock)
    else:
        store = SqliteStore(db_path or config.ENGINE_DB)
        jobs = SqliteJobRepository(store)
        artifacts = SqliteArtifactRepository(store)
        idempotency = SqliteIdempotencyRepository(store)
        events = InProcessEventPublisher(store=SqliteEventStore(store), clock=clock)

    exec_impl = executor or RoutingExecutor(
        {
            "comfyui": ComfyUIExecutor(),
            "tts": TtsExecutor(),
            "ffmpeg": FfmpegExecutor(),
        }
    )
    sched = scheduler or ProfileResourceScheduler(runtime, interrupt_fn=exec_impl.cancel)
    runtime_impl = model_runtime or CatalogModelRuntime(catalog, runtime, downloads)
    dl = downloader or SystemdModelDownloader(downloads)
    inspector = BasicMediaInspector()
    worker = JobWorker(
        jobs=jobs,
        artifacts=artifacts,
        storage=storage,
        events=events,
        scheduler=sched,
        executor=exec_impl,
        catalog=catalog,
        clock=clock,
        ids=ids,
        ttl_hours=config.ARTIFACT_TTL_HOURS,
        inspector=inspector,
    )
    if start_worker:
        worker.start()
        for job in jobs.list_active():
            if job.cancel_requested:
                continue
            worker.enqueue(job.job_id)
    return EngineServices(
        catalog=catalog,
        jobs=jobs,
        artifacts=artifacts,
        idempotency=idempotency,
        storage=storage,
        events=events,
        worker=worker,
        scheduler=sched,
        executor=exec_impl,
        clock=clock,
        ids=ids,
        model_runtime=runtime_impl,
        downloader=dl,
        inspector=inspector,
        ttl_hours=config.ARTIFACT_TTL_HOURS,
        max_upload_bytes=config.MAX_UPLOAD_BYTES,
    )
