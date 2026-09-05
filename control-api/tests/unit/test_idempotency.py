from __future__ import annotations

from pathlib import Path

from app.application.ports import ExecutionOutput, ExecutionRequest, ExecutionResult
from app.bootstrap.container import build_engine
from app.domain.errors import DomainError, ErrorCode
from app.domain.models import ModelState
from app.domain.runtime import ResourceLease, ResourceRequirement

REPO = Path(__file__).resolve().parents[3]
PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
    b"\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeExecutor:
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult(
            outputs=[ExecutionOutput(data=PNG, mime_type="image/png", artifact_type="IMAGE")],
            model_ids=["qwen-image-2512-fp8"],
            seed=1,
            duration_ms=1,
        )

    def cancel(self, job_id: str) -> None:
        return None

    def health(self) -> dict:
        return {"comfy": True}


class FakeRuntime:
    def __init__(self, catalog) -> None:
        self._catalog = catalog

    def disk_status(self) -> dict[str, str]:
        return {mid: "complete" for mid in self._catalog.models}

    def state(self, model_id: str) -> ModelState:
        return ModelState.AVAILABLE

    def load(self, model_id: str) -> dict:
        return {"ok": True}

    def unload(self, model_id: str) -> dict:
        return {"ok": True}


class FakeDownloader:
    def start(self, model_ids: list[str]) -> dict:
        return {"status": "queued", "ids": model_ids}

    def status(self) -> dict:
        return {"status": "idle"}


class FakeScheduler:
    def acquire(self, job_id: str, requirement: ResourceRequirement) -> ResourceLease:
        return ResourceLease(job_id=job_id, profile=requirement.profile, estimated_vram_gb=requirement.estimated_vram_gb)

    def release(self, lease: ResourceLease) -> None:
        return None

    def snapshot(self) -> dict:
        return {"gpu_name": "fake", "queue_depth": 0, "active_job": None, "worker_health": "alive"}

    def interrupt(self, job_id: str) -> None:
        return None

    def set_queue_depth(self, depth: int) -> None:
        return None


def make_engine(**kwargs):
    engine = build_engine(
        memory=True,
        executor=FakeExecutor(),
        scheduler=FakeScheduler(),
        downloader=FakeDownloader(),
        catalog_dir=REPO / "catalog",
        start_worker=kwargs.get("start_worker", False),
        commercial_mode=True,
    )
    engine.model_runtime = FakeRuntime(engine.catalog)
    return engine


def test_idempotency_returns_same_job() -> None:
    engine = make_engine()
    body = {"operation": "image.generate", "preset": "master", "inputs": {"prompt": "a person"}}
    a = engine.submit_job(body, idempotency_key="k1", trace_id="t1")
    b = engine.submit_job(body, idempotency_key="k1", trace_id="t1")
    assert a.job_id == b.job_id


def test_idempotency_conflict() -> None:
    engine = make_engine()
    engine.submit_job({"operation": "image.generate", "preset": "master", "inputs": {"prompt": "a"}}, idempotency_key="k1", trace_id="t")
    try:
        engine.submit_job({"operation": "image.generate", "preset": "master", "inputs": {"prompt": "b"}}, idempotency_key="k1", trace_id="t")
        raise AssertionError("expected conflict")
    except DomainError as exc:
        assert exc.code == ErrorCode.IDEMPOTENCY_CONFLICT
