from __future__ import annotations

from app.domain.jobs import Job, JobStatus
from app.infrastructure.persistence.sqlite import SqliteJobRepository, SqliteStore


def test_sqlite_job_roundtrip(tmp_path) -> None:
    store = SqliteStore(tmp_path / "engine.sqlite")
    repo = SqliteJobRepository(store)
    job = Job(
        job_id="job_test",
        operation="image.generate",
        preset="master",
        status=JobStatus.QUEUED,
        phase="queued",
        progress=0.0,
        inputs={"prompt": "x"},
        parameters={"seed": 1},
        client_context={"shot_id": "opaque"},
        created_at="2026-01-01T00:00:00+00:00",
        idempotency_key="abc",
    )
    repo.save(job)
    loaded = repo.get("job_test")
    assert loaded is not None
    assert loaded.operation == "image.generate"
    assert loaded.client_context["shot_id"] == "opaque"
    assert loaded.status == JobStatus.QUEUED
