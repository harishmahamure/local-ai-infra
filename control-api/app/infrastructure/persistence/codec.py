from __future__ import annotations

import json
from typing import Any

from ...domain.artifacts import Artifact, ArtifactType, Lineage
from ...domain.jobs import Job, JobResult, JobStatus


def job_to_dict(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "operation": job.operation,
        "preset": job.preset,
        "status": job.status.value,
        "phase": job.phase,
        "progress": job.progress,
        "inputs": job.inputs,
        "parameters": job.parameters,
        "client_context": job.client_context,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error,
        "result": None
        if job.result is None
        else {"asset_ids": job.result.asset_ids, "manifest": job.result.manifest},
        "workflow_id": job.workflow_id,
        "workflow_version": job.workflow_version,
        "model_ids": job.model_ids,
        "retry_count": job.retry_count,
        "parent_job_id": job.parent_job_id,
        "idempotency_key": job.idempotency_key,
        "trace_id": job.trace_id,
        "cancel_requested": job.cancel_requested,
        "output": job.output,
    }


def job_from_dict(data: dict[str, Any]) -> Job:
    result = data.get("result")
    return Job(
        job_id=data["job_id"],
        operation=data["operation"],
        preset=data["preset"],
        status=JobStatus(data["status"]),
        phase=data.get("phase", ""),
        progress=float(data.get("progress") or 0),
        inputs=data.get("inputs") or {},
        parameters=data.get("parameters") or {},
        client_context=data.get("client_context") or {},
        created_at=data["created_at"],
        started_at=data.get("started_at"),
        finished_at=data.get("finished_at"),
        error=data.get("error"),
        result=None
        if not result
        else JobResult(asset_ids=list(result.get("asset_ids") or []), manifest=dict(result.get("manifest") or {})),
        workflow_id=data.get("workflow_id"),
        workflow_version=data.get("workflow_version"),
        model_ids=list(data.get("model_ids") or []),
        retry_count=int(data.get("retry_count") or 0),
        parent_job_id=data.get("parent_job_id"),
        idempotency_key=data.get("idempotency_key"),
        trace_id=data.get("trace_id") or "",
        cancel_requested=bool(data.get("cancel_requested")),
        output=data.get("output"),
    )


def artifact_to_dict(artifact: Artifact) -> dict[str, Any]:
    return {
        "asset_id": artifact.asset_id,
        "type": artifact.type.value,
        "mime_type": artifact.mime_type,
        "checksum": artifact.checksum,
        "size_bytes": artifact.size_bytes,
        "created_at": artifact.created_at,
        "expires_at": artifact.expires_at,
        "retention_policy": artifact.retention_policy,
        "width": artifact.width,
        "height": artifact.height,
        "duration_seconds": artifact.duration_seconds,
        "lineage": {
            "source_asset_ids": artifact.lineage.source_asset_ids,
            "parent_job_id": artifact.lineage.parent_job_id,
            "model_id": artifact.lineage.model_id,
            "workflow_id": artifact.lineage.workflow_id,
            "workflow_version": artifact.lineage.workflow_version,
            "seed": artifact.lineage.seed,
            "prompt": artifact.lineage.prompt,
            "control_inputs": artifact.lineage.control_inputs,
            "loras": artifact.lineage.loras,
            "generation_settings": artifact.lineage.generation_settings,
        },
        "manifest": artifact.manifest,
        "deleted": artifact.deleted,
    }


def artifact_from_dict(data: dict[str, Any]) -> Artifact:
    lin = data.get("lineage") or {}
    return Artifact(
        asset_id=data["asset_id"],
        type=ArtifactType(data["type"]),
        mime_type=data["mime_type"],
        checksum=data["checksum"],
        size_bytes=int(data["size_bytes"]),
        created_at=data["created_at"],
        expires_at=data["expires_at"],
        retention_policy=data.get("retention_policy", "ttl"),
        width=data.get("width"),
        height=data.get("height"),
        duration_seconds=data.get("duration_seconds"),
        lineage=Lineage(
            source_asset_ids=list(lin.get("source_asset_ids") or []),
            parent_job_id=lin.get("parent_job_id"),
            model_id=lin.get("model_id"),
            workflow_id=lin.get("workflow_id"),
            workflow_version=lin.get("workflow_version"),
            seed=lin.get("seed"),
            prompt=lin.get("prompt"),
            control_inputs=dict(lin.get("control_inputs") or {}),
            loras=list(lin.get("loras") or []),
            generation_settings=dict(lin.get("generation_settings") or {}),
        ),
        manifest=dict(data.get("manifest") or {}),
        deleted=bool(data.get("deleted")),
    )


def dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"))


def loads(raw: str) -> dict[str, Any]:
    return json.loads(raw)
