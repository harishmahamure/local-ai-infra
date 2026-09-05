from __future__ import annotations

from app.domain.errors import ErrorCode
from app.domain.ids import new_asset_id, new_job_id
from app.domain.jobs import Job, JobStatus


def test_ids_have_prefixes() -> None:
    assert new_job_id().startswith("job_")
    assert new_asset_id().startswith("ast_")


def test_job_cancel_queued() -> None:
    job = Job(
        job_id="job_1",
        operation="image.generate",
        preset="master",
        status=JobStatus.QUEUED,
        phase="queued",
        progress=0,
        inputs={},
        parameters={},
        client_context={},
        created_at="2026-01-01T00:00:00+00:00",
    )
    job.request_cancel()
    assert job.status == JobStatus.CANCELLED
    assert job.error["code"] == ErrorCode.CANCELLED.value


def test_error_retryable() -> None:
    assert ErrorCode.COMFYUI_UNAVAILABLE.retryable is True
    assert ErrorCode.LICENSE_RESTRICTED.retryable is False
    assert ErrorCode.LICENSE_RESTRICTED.http_status == 403
