from __future__ import annotations

import json
from typing import Any

from ...domain.artifacts import Artifact, ArtifactType, Lineage
from ...domain.attempts import AttemptStatus, JobAttempt
from ...domain.batches import Batch, BatchStatus
from ...domain.dependencies import DependencyPolicy, DependencyState, JobDependency
from ...domain.jobs import Job, JobResult, JobStatus
from ...domain.nodes import ComputeNode, NodeMode
from ...domain.priority import Priority
from ...domain.queues import QueueClass
from ...domain.workers import WorkerRecord, WorkerStatus


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
        "priority": job.priority.value,
        "queue": job.queue.value,
        "queued_at": job.queued_at,
        "completed_at": job.completed_at,
        "max_retries": job.max_retries,
        "current_attempt": job.current_attempt,
        "batch_id": job.batch_id,
        "next_attempt_at": job.next_attempt_at,
        "lease_owner": job.lease_owner,
        "lease_expires_at": job.lease_expires_at,
        "compute_node": job.compute_node,
        "resolved_plan": job.resolved_plan,
        "error_summary": job.error_summary,
    }


def job_from_dict(data: dict[str, Any]) -> Job:
    result = data.get("result")
    priority = data.get("priority") or "NORMAL"
    queue = data.get("queue") or "gpu"
    try:
        pri = Priority(priority)
    except ValueError:
        pri = Priority.NORMAL
    try:
        q = QueueClass(queue)
    except ValueError:
        q = QueueClass.GPU
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
        priority=pri,
        queue=q,
        queued_at=data.get("queued_at"),
        completed_at=data.get("completed_at"),
        max_retries=int(data.get("max_retries") or 3),
        current_attempt=data.get("current_attempt"),
        batch_id=data.get("batch_id"),
        next_attempt_at=data.get("next_attempt_at"),
        lease_owner=data.get("lease_owner"),
        lease_expires_at=data.get("lease_expires_at"),
        compute_node=data.get("compute_node") or "gpu-box",
        resolved_plan=dict(data.get("resolved_plan") or {}),
        error_summary=data.get("error_summary"),
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


def attempt_to_dict(attempt: JobAttempt) -> dict[str, Any]:
    return {
        "attempt_id": attempt.attempt_id,
        "job_id": attempt.job_id,
        "attempt_number": attempt.attempt_number,
        "status": attempt.status.value,
        "compute_node": attempt.compute_node,
        "runtime": attempt.runtime,
        "model": attempt.model,
        "variant": attempt.variant,
        "workflow": attempt.workflow,
        "worker_id": attempt.worker_id,
        "queued_at": attempt.queued_at,
        "started_at": attempt.started_at,
        "ended_at": attempt.ended_at,
        "peak_vram": attempt.peak_vram,
        "execution_time": attempt.execution_time,
        "result": attempt.result,
        "error_code": attempt.error_code,
        "error_details": attempt.error_details,
        "lease_expires_at": attempt.lease_expires_at,
        "profile": attempt.profile,
        "queue": attempt.queue,
    }


def attempt_from_dict(data: dict[str, Any]) -> JobAttempt:
    return JobAttempt(
        attempt_id=data["attempt_id"],
        job_id=data["job_id"],
        attempt_number=int(data.get("attempt_number") or 1),
        status=AttemptStatus(data["status"]),
        compute_node=data.get("compute_node") or "gpu-box",
        runtime=data.get("runtime"),
        model=data.get("model"),
        variant=data.get("variant"),
        workflow=data.get("workflow"),
        worker_id=data.get("worker_id"),
        queued_at=data.get("queued_at"),
        started_at=data.get("started_at"),
        ended_at=data.get("ended_at"),
        peak_vram=data.get("peak_vram"),
        execution_time=data.get("execution_time"),
        result=data.get("result"),
        error_code=data.get("error_code"),
        error_details=dict(data.get("error_details") or {}),
        lease_expires_at=data.get("lease_expires_at"),
        profile=data.get("profile"),
        queue=data.get("queue") or "gpu",
    )


def batch_to_dict(batch: Batch) -> dict[str, Any]:
    return {
        "batch_id": batch.batch_id,
        "capability": batch.capability,
        "preset": batch.preset,
        "created_at": batch.created_at,
        "status": batch.status.value,
        "total": batch.total,
        "queued": batch.queued,
        "running": batch.running,
        "completed": batch.completed,
        "failed": batch.failed,
        "cancelled": batch.cancelled,
        "job_ids": batch.job_ids,
        "priority": batch.priority,
        "cancel_requested": batch.cancel_requested,
    }


def batch_from_dict(data: dict[str, Any]) -> Batch:
    return Batch(
        batch_id=data["batch_id"],
        capability=data.get("capability") or "",
        preset=data.get("preset") or "master",
        created_at=data["created_at"],
        status=BatchStatus(data.get("status") or BatchStatus.QUEUED.value),
        total=int(data.get("total") or 0),
        queued=int(data.get("queued") or 0),
        running=int(data.get("running") or 0),
        completed=int(data.get("completed") or 0),
        failed=int(data.get("failed") or 0),
        cancelled=int(data.get("cancelled") or 0),
        job_ids=list(data.get("job_ids") or []),
        priority=data.get("priority") or "NORMAL",
        cancel_requested=bool(data.get("cancel_requested")),
    )


def worker_to_dict(worker: WorkerRecord) -> dict[str, Any]:
    return {
        "worker_id": worker.worker_id,
        "role": worker.role,
        "status": worker.status.value,
        "current_job_id": worker.current_job_id,
        "current_attempt_id": worker.current_attempt_id,
        "runtime_status": worker.runtime_status,
        "gpu_status": worker.gpu_status,
        "last_heartbeat": worker.last_heartbeat,
        "started_at": worker.started_at,
        "hostname": worker.hostname,
        "queues": worker.queues,
    }


def worker_from_dict(data: dict[str, Any]) -> WorkerRecord:
    return WorkerRecord(
        worker_id=data["worker_id"],
        role=data.get("role") or "gpu",
        status=WorkerStatus(data.get("status") or WorkerStatus.STARTING.value),
        current_job_id=data.get("current_job_id"),
        current_attempt_id=data.get("current_attempt_id"),
        runtime_status=data.get("runtime_status") or "idle",
        gpu_status=data.get("gpu_status"),
        last_heartbeat=data.get("last_heartbeat"),
        started_at=data.get("started_at"),
        hostname=data.get("hostname") or "gpu-box",
        queues=list(data.get("queues") or []),
    )


def node_to_dict(node: ComputeNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "mode": node.mode.value,
        "hostname": node.hostname,
        "healthy": node.healthy,
        "updated_at": node.updated_at,
    }


def node_from_dict(data: dict[str, Any]) -> ComputeNode:
    return ComputeNode(
        node_id=data["node_id"],
        mode=NodeMode(data.get("mode") or NodeMode.NORMAL.value),
        hostname=data.get("hostname") or "gpu-box",
        healthy=bool(data.get("healthy", True)),
        updated_at=data.get("updated_at"),
    )


def dependency_from_row(job_id: str, depends_on: str, policy: str, bindings: str, state: str) -> JobDependency:
    try:
        artifact_bindings = json.loads(bindings) if bindings else {}
    except json.JSONDecodeError:
        artifact_bindings = {}
    return JobDependency(
        job_id=job_id,
        depends_on=depends_on,
        policy=DependencyPolicy(policy),
        artifact_bindings=artifact_bindings,
        state=DependencyState(state),
    )


def dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"))


def loads(raw: str) -> dict[str, Any]:
    return json.loads(raw)
