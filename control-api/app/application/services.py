from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ..domain.artifacts import Artifact, ArtifactType
from ..domain.errors import DomainError, ErrorCode
from ..domain.jobs import Job, JobStatus, RetryStrategy
from ..domain.models import ModelState
from .catalog import CatalogRegistry
from ..infrastructure.clock import expires_at, is_expired
from .resolvers import (
    ALLOWED_OVERRIDES,
    availability_reason,
    enforce_license,
    resolve_loras,
    resolve_operation,
    resolve_preset,
    resolve_workflow,
    validate_overrides,
)
from .ports import ArtifactRepository, ArtifactStorage, EventPublisher, IdempotencyRepository, JobRepository
from .text_chat import TEXT_CHAT_MODELS, build_llama_messages, llama_model_name

MAX_PROMPT = 8000


@dataclass
class EngineServices:
    catalog: CatalogRegistry
    jobs: JobRepository
    artifacts: ArtifactRepository
    idempotency: IdempotencyRepository
    storage: ArtifactStorage
    events: EventPublisher
    worker: Any
    scheduler: Any
    executor: Any
    clock: Any
    ids: Any
    model_runtime: Any
    downloader: Any
    inspector: Any
    ttl_hours: int
    max_upload_bytes: int
    llama_chat: Any = None

    def submit_job(self, body: dict[str, Any], *, idempotency_key: str | None, trace_id: str | None) -> Job:
        if (body.get("output") or {}).get("mode") == "signed_upload":
            raise DomainError(ErrorCode.UNSUPPORTED_PARAMETER, "signed_upload is not implemented in Phase 1")
        request_hash = _hash_request(body)
        if idempotency_key:
            existing = self.idempotency.get(idempotency_key)
            if existing:
                stored_hash, job_id = existing
                if stored_hash != request_hash:
                    raise DomainError(ErrorCode.IDEMPOTENCY_CONFLICT, "Idempotency-Key reused with a different body")
                job = self.jobs.get(job_id)
                if job:
                    return job
        operation = resolve_operation(self.catalog, str(body.get("operation") or ""))
        preset_id = str(body.get("preset") or "master")
        preset = resolve_preset(self.catalog, operation, preset_id)
        workflow = resolve_workflow(self.catalog, operation, preset)
        inputs = dict(body.get("inputs") or {})
        parameters = validate_overrides(dict(body.get("parameters") or {}), workflow)
        merged = preset.merged_parameters(parameters)
        job_params = {**merged, **parameters}
        _validate_inputs(operation.id, inputs, self.max_upload_bytes)
        model_ids = list(workflow.required_models)
        enforce_license(self.catalog, model_ids)
        loras = resolve_loras(self.catalog, preset, job_params, model_ids)
        job_params["_resolved_loras"] = loras
        reason = availability_reason(self.catalog, operation, self.model_runtime.disk_status())
        if reason == "not_implemented":
            raise DomainError(ErrorCode.UNSUPPORTED_OPERATION, f"{operation.id} is not implemented yet")
        if reason == "license_restricted":
            raise DomainError(ErrorCode.LICENSE_RESTRICTED, f"{operation.id} is license-restricted")
        if reason == "model_not_downloaded":
            raise DomainError(ErrorCode.MODEL_NOT_AVAILABLE, f"Required models for {operation.id} are not downloaded")

        now = self.clock.now_iso()
        job = Job(
            job_id=self.ids.job_id(),
            operation=operation.id,
            preset=preset.id,
            status=JobStatus.QUEUED,
            phase="queued",
            progress=0.0,
            inputs=inputs,
            parameters=job_params,
            client_context=dict(body.get("client_context") or {}),
            created_at=now,
            workflow_id=workflow.id,
            workflow_version=workflow.version,
            model_ids=model_ids,
            idempotency_key=idempotency_key,
            trace_id=trace_id or self.ids.trace_id(),
            output=body.get("output"),
        )
        self.jobs.save(job)
        if idempotency_key:
            self.idempotency.put(idempotency_key, request_hash, job.job_id)
        self.events.publish(job.job_id, "job.queued")
        self.worker.enqueue(job.job_id)
        return job

    def get_job(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if not job:
            raise DomainError(ErrorCode.INVALID_REQUEST, f"Job {job_id} not found")
        return job

    def cancel_job(self, job_id: str) -> Job:
        job = self.get_job(job_id)
        job.request_cancel()
        if not job.status.terminal:
            job.status = JobStatus.CANCELLED
            job.phase = "cancelled"
            job.finished_at = self.clock.now_iso()
            job.error = {"code": ErrorCode.CANCELLED.value, "message": "Job cancelled", "retryable": False}
        self.jobs.save(job)
        try:
            self.executor.cancel(job_id)
            self.scheduler.interrupt(job_id)
        except Exception:
            pass
        self.events.publish(job.job_id, "job.cancelled")
        return job

    def retry_job(self, job_id: str, strategy: str, parameters: dict[str, Any] | None = None, stage: str | None = None) -> Job:
        parent = self.get_job(job_id)
        try:
            kind = RetryStrategy(strategy)
        except ValueError as exc:
            raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unknown retry strategy: {strategy}") from exc
        body = {
            "operation": parent.operation,
            "preset": parent.preset,
            "inputs": parent.inputs,
            "parameters": {k: v for k, v in parent.parameters.items() if k in ALLOWED_OVERRIDES},
            "client_context": parent.client_context,
        }
        params = {k: v for k, v in body["parameters"].items() if k in ALLOWED_OVERRIDES}
        if kind == RetryStrategy.NEW_SEED:
            params.pop("seed", None)
        elif kind == RetryStrategy.OVERRIDE:
            params.update(parameters or {})
        elif kind == RetryStrategy.RERUN_STAGE and stage:
            params["rerun_stage"] = stage
        body["parameters"] = params
        child = self.submit_job(body, idempotency_key=None, trace_id=parent.trace_id)
        child.parent_job_id = parent.job_id
        child.retry_count = parent.retry_count + 1
        self.jobs.save(child)
        return child

    def upload_asset(self, data: bytes, mime_type: str, filename: str | None = None) -> Artifact:
        if not data:
            raise DomainError(ErrorCode.INVALID_REQUEST, "Empty upload")
        if len(data) > self.max_upload_bytes:
            raise DomainError(ErrorCode.INVALID_REQUEST, "Upload exceeds size limit")
        kind = _type_from_mime(mime_type, filename)
        now = self.clock.now_iso()
        asset_id = self.ids.asset_id()
        checksum = self.storage.write(asset_id, data)
        meta = self.inspector.inspect_image(data, mime_type) if kind == ArtifactType.IMAGE else {}
        artifact = Artifact(
            asset_id=asset_id,
            type=kind,
            mime_type=mime_type or "application/octet-stream",
            checksum=checksum,
            size_bytes=len(data),
            created_at=now,
            expires_at=expires_at(now, self.ttl_hours),
            width=meta.get("width"),
            height=meta.get("height"),
        )
        self.artifacts.save(artifact)
        return artifact

    def get_asset(self, asset_id: str) -> Artifact:
        artifact = self.artifacts.get(asset_id)
        if artifact is None or artifact.deleted:
            raise DomainError(ErrorCode.ASSET_NOT_FOUND, f"Asset {asset_id} not found")
        if is_expired(artifact.expires_at, self.clock.now_iso()):
            raise DomainError(ErrorCode.ASSET_EXPIRED, f"Asset {asset_id} has expired")
        return artifact

    def delete_asset(self, asset_id: str) -> None:
        artifact = self.get_asset(asset_id)
        artifact.deleted = True
        self.artifacts.save(artifact)
        self.storage.delete(asset_id)

    def list_operations(self) -> dict[str, Any]:
        disk = self.model_runtime.disk_status()
        items = []
        for op in self.catalog.operations.values():
            reason = availability_reason(self.catalog, op, disk)
            items.append(op.public_dict(available=reason is None, reason=reason))
        return {"operations": items}

    def list_capabilities(self) -> dict[str, Any]:
        disk = self.model_runtime.disk_status()
        caps: dict[str, Any] = {
            "image": {"generate": False, "edit": False, "upscale": False, "control": [], "layered": False},
            "video": {
                "text_to_video": False,
                "image_to_video": False,
                "audio_to_video": False,
                "first_last_frames": False,
                "continuation": False,
                "enhancement": False,
                "interpolation": False,
                "lipsync": False,
                "motion_transfer": False,
            },
            "audio": {"tts": [], "music": False, "sfx": False, "ambience": False, "mix": False},
            "text": {"chat": False, "vision": False},
        }
        for op in self.catalog.operations.values():
            if availability_reason(self.catalog, op, disk) is not None:
                continue
            if op.id == "image.generate":
                caps["image"]["generate"] = True
            elif op.id == "image.edit":
                caps["image"]["edit"] = True
            elif op.id == "image.controlled":
                caps["image"]["control"] = ["pose", "depth", "canny"]
            elif op.id == "image.layered":
                caps["image"]["layered"] = True
            elif op.id == "image.upscale":
                caps["image"]["upscale"] = True
            elif op.id == "video.generate":
                caps["video"]["text_to_video"] = True
            elif op.id == "video.image_to_video":
                caps["video"]["image_to_video"] = True
            elif op.id == "video.audio_to_video":
                caps["video"]["audio_to_video"] = True
            elif op.id == "video.first_last_frames":
                caps["video"]["first_last_frames"] = True
            elif op.id == "video.lipsync":
                caps["video"]["lipsync"] = True
            elif op.id == "video.motion_transfer":
                caps["video"]["motion_transfer"] = True
            elif op.id == "text.chat":
                caps["text"]["chat"] = True
                caps["text"]["vision"] = True
        return caps

    def list_presets(self) -> dict[str, Any]:
        return {"presets": [{"id": p.id, "label": p.label, "description": p.description, "operations": p.operations} for p in self.catalog.presets.values()]}

    def list_models(self) -> dict[str, Any]:
        return {"models": [self._public_model(m) for m in self.catalog.models.values()]}

    def get_model(self, model_id: str) -> dict[str, Any]:
        model = self.catalog.models.get(model_id)
        if not model:
            raise DomainError(ErrorCode.MODEL_NOT_AVAILABLE, f"Unknown model {model_id}")
        return self._public_model(model)

    def list_text_models(self) -> dict[str, Any]:
        return {
            "models": [
                self._public_model(model)
                for model in self.catalog.models.values()
                if "text.chat" in model.supported_operations
            ]
        }

    def start_model(self, model_id: str) -> dict[str, Any]:
        return self.load_model(model_id)

    def stop_model(self, model_id: str) -> dict[str, Any]:
        return self.unload_model(model_id)

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._llama_chat_payload(body)
        try:
            return self._llama_client().complete(payload)
        except DomainError:
            raise
        except Exception as exc:
            raise DomainError(ErrorCode.INTERNAL_ERROR, f"Text model request failed: {exc}") from exc

    def chat_stream(self, body: dict[str, Any]):
        payload = self._llama_chat_payload(body)
        for line in self._llama_client().stream(payload):
            if line.startswith("data:") or line.startswith("event:"):
                yield f"{line.rstrip()}\n\n" if not line.endswith("\n\n") else line
            else:
                yield f"data: {line.rstrip()}\n\n"

    def download_model(self, model_id: str) -> dict[str, Any]:
        if model_id not in self.catalog.models:
            raise DomainError(ErrorCode.MODEL_NOT_AVAILABLE, f"Unknown model {model_id}")
        return self.downloader.start([model_id])

    def verify_model(self, model_id: str) -> dict[str, Any]:
        state = self.model_runtime.state(model_id)
        return {"model_id": model_id, "state": state.value, "valid": state in {ModelState.AVAILABLE, ModelState.READY}}

    def load_model(self, model_id: str) -> dict[str, Any]:
        return self.model_runtime.load(model_id)

    def unload_model(self, model_id: str) -> dict[str, Any]:
        return self.model_runtime.unload(model_id)

    def list_loras(self) -> dict[str, Any]:
        return {"loras": [lora.to_public_dict() for lora in self.catalog.loras.values()]}

    def list_workflows(self) -> dict[str, Any]:
        return {
            "workflows": [
                {
                    "id": wf.id,
                    "version": wf.version,
                    "operation": wf.operation,
                    "description": wf.description,
                    "executor": wf.executor,
                    "supported_presets": wf.supported_presets,
                }
                for wf in self.catalog.workflows.values()
            ]
        }

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        wf = self.catalog.workflows.get(workflow_id)
        if not wf:
            raise DomainError(ErrorCode.WORKFLOW_NOT_AVAILABLE, f"Unknown workflow {workflow_id}")
        return {
            "id": wf.id,
            "version": wf.version,
            "operation": wf.operation,
            "description": wf.description,
            "executor": wf.executor,
            "required_models": wf.required_models,
            "required_nodes": wf.required_nodes,
            "supported_presets": wf.supported_presets,
            "bindings": wf.bindings,
        }

    def validate_workflow(self, workflow_id: str) -> dict[str, Any]:
        wf = self.catalog.workflows.get(workflow_id)
        if not wf:
            raise DomainError(ErrorCode.WORKFLOW_NOT_AVAILABLE, f"Unknown workflow {workflow_id}")
        disk = self.model_runtime.disk_status()
        missing_models = [mid for mid in wf.required_models if disk.get(mid) not in {"complete", "ok"}]
        license_warnings = []
        if self.catalog.commercial_mode:
            for mid in wf.required_models:
                model = self.catalog.models.get(mid)
                if model and (model.noncommercial_only or not model.commercial):
                    license_warnings.append(mid)
        compatibility_warnings = [
            lora.id for lora in self.catalog.loras.values() if lora.compatibility_test_required and lora.id in wf.optional_loras
        ]
        return {
            "valid": not missing_models and not license_warnings,
            "missing_models": missing_models,
            "missing_nodes": [],
            "missing_files": missing_models,
            "compatibility_warnings": compatibility_warnings,
            "license_warnings": license_warnings,
        }

    def runtime(self) -> dict[str, Any]:
        snap = self.scheduler.snapshot()
        snap["storage_writable"] = self.storage.writable()
        snap["worker_alive"] = bool(getattr(self.worker, "alive", False))
        return snap

    def ready(self) -> tuple[bool, dict[str, Any]]:
        details = {
            "worker": bool(getattr(self.worker, "alive", False)),
            "storage": self.storage.writable(),
            "registry": bool(self.catalog.operations),
        }
        return all(details.values()), details


    def _public_model(self, model) -> dict[str, Any]:
        payload = model.to_public_dict(
            state=self.model_runtime.state(model.id),
            disk_status=self.model_runtime.disk_status().get(model.id),
        )
        profile = None
        if hasattr(self.model_runtime, "profile_for"):
            try:
                profile = self.model_runtime.profile_for(model.id)
            except Exception:
                profile = None
        if profile is None and model.dest == "llamacpp":
            from ..infrastructure.model_runtime import profile_for_text_model

            profile = profile_for_text_model(model.id)
        if profile:
            payload["profile"] = profile
        live_ctx = _live_context_length(model.id, model.supported_context)
        if live_ctx:
            payload["context_length"] = live_ctx
        return payload

    def _llama_client(self):
        if self.llama_chat is None:
            from ..infrastructure.llama_chat import LlamaChatClient

            self.llama_chat = LlamaChatClient()
        return self.llama_chat

    def _llama_chat_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        model_id = str(body.get("model") or "").strip()
        model = self.catalog.models.get(model_id)
        if model is None or model_id not in TEXT_CHAT_MODELS:
            raise DomainError(ErrorCode.MODEL_NOT_AVAILABLE, f"Unknown text model {model_id}")
        if "text.chat" not in model.supported_operations:
            raise DomainError(ErrorCode.UNSUPPORTED_OPERATION, f"{model_id} does not support text.chat")
        enforce_license(self.catalog, [model_id])

        def resolve_asset(asset_id: str) -> tuple[bytes, str]:
            artifact = self.get_asset(asset_id)
            return self.storage.read(asset_id), artifact.mime_type or "image/png"

        messages = build_llama_messages(list(body.get("messages") or []), resolve_asset=resolve_asset)
        if self.model_runtime.state(model_id) != ModelState.READY:
            self.model_runtime.load(model_id)
        payload: dict[str, Any] = {
            "model": llama_model_name(model),
            "messages": messages,
        }
        if body.get("temperature") is not None:
            payload["temperature"] = body["temperature"]
        if body.get("max_tokens") is not None:
            payload["max_tokens"] = body["max_tokens"]
        return payload


def _live_context_length(model_id: str, fallback: int) -> int:
    from .. import config

    env_name = "gemma.env" if model_id.startswith("gemma") else "llama-fast.env" if model_id.startswith("qwen36") else ""
    if env_name:
        path = config.CONFIG / env_name
        if path.is_file():
            last = 0
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or not stripped.startswith("CTX="):
                    continue
                try:
                    last = int(stripped.split("=", 1)[1].strip())
                except ValueError:
                    continue
            if last:
                return last
    return fallback


def _hash_request(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _type_from_mime(mime: str, filename: str | None) -> ArtifactType:
    if mime.startswith("image/"):
        return ArtifactType.IMAGE
    if mime.startswith("video/"):
        return ArtifactType.VIDEO
    if mime.startswith("audio/"):
        return ArtifactType.AUDIO
    name = (filename or "").lower()
    if name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return ArtifactType.IMAGE
    if name.endswith((".mp4", ".webm", ".mov")):
        return ArtifactType.VIDEO
    if name.endswith((".wav", ".mp3", ".flac")):
        return ArtifactType.AUDIO
    return ArtifactType.TEMPORARY


def _validate_inputs(operation: str, inputs: dict[str, Any], _max_upload: int) -> None:
    prompt = inputs.get("prompt")
    if isinstance(prompt, str) and len(prompt) > MAX_PROMPT:
        raise DomainError(ErrorCode.INVALID_REQUEST, "Prompt exceeds maximum length")
    if operation == "image.generate" and not str(inputs.get("prompt") or "").strip():
        raise DomainError(ErrorCode.INVALID_REQUEST, "prompt is required")
    if operation == "image.edit":
        if not str(inputs.get("prompt") or "").strip():
            raise DomainError(ErrorCode.INVALID_REQUEST, "prompt is required")
        refs = inputs.get("reference_images")
        has_refs = isinstance(refs, list) and any(isinstance(item, dict) and item.get("asset_id") for item in refs)
        has_image = isinstance(inputs.get("image"), dict) and inputs["image"].get("asset_id")
        if not has_refs and not has_image:
            raise DomainError(ErrorCode.INVALID_REQUEST, "reference_images or image.asset_id is required")
        if isinstance(refs, list) and len(refs) > 3:
            raise DomainError(ErrorCode.INVALID_REQUEST, "image.edit accepts at most 3 reference images")
    if operation == "image.controlled":
        if not str(inputs.get("prompt") or "").strip():
            raise DomainError(ErrorCode.INVALID_REQUEST, "prompt is required")
        if not (isinstance(inputs.get("control_image"), dict) and inputs["control_image"].get("asset_id")):
            raise DomainError(ErrorCode.INVALID_REQUEST, "control_image.asset_id is required")
        control_type = str(inputs.get("control_type") or "pose")
        if control_type not in {"pose", "depth", "canny"}:
            raise DomainError(ErrorCode.INVALID_REQUEST, "control_type must be pose, depth, or canny")
    if operation == "image.layered":
        if not (isinstance(inputs.get("image"), dict) and inputs["image"].get("asset_id")):
            raise DomainError(ErrorCode.INVALID_REQUEST, "image.asset_id is required")
        layers = inputs.get("layers")
        if layers is not None and (int(layers) < 1 or int(layers) > 8):
            raise DomainError(ErrorCode.INVALID_PARAMETER, "layers must be 1-8")
    if operation == "image.upscale" and not (isinstance(inputs.get("image"), dict) and inputs["image"].get("asset_id")):
        raise DomainError(ErrorCode.INVALID_REQUEST, "image.asset_id is required")
    if operation == "video.generate" and not str(inputs.get("prompt") or "").strip():
        raise DomainError(ErrorCode.INVALID_REQUEST, "prompt is required")
    if operation == "video.image_to_video":
        if not str(inputs.get("prompt") or "").strip():
            raise DomainError(ErrorCode.INVALID_REQUEST, "prompt is required")
        if not (isinstance(inputs.get("start_image"), dict) and inputs["start_image"].get("asset_id")):
            raise DomainError(ErrorCode.INVALID_REQUEST, "start_image.asset_id is required")
    if operation == "video.audio_to_video":
        if not str(inputs.get("prompt") or "").strip():
            raise DomainError(ErrorCode.INVALID_REQUEST, "prompt is required")
        if not (isinstance(inputs.get("audio"), dict) and inputs["audio"].get("asset_id")):
            raise DomainError(ErrorCode.INVALID_REQUEST, "audio.asset_id is required")
    if operation == "video.first_last_frames":
        if not str(inputs.get("prompt") or "").strip():
            raise DomainError(ErrorCode.INVALID_REQUEST, "prompt is required")
        if not (isinstance(inputs.get("start_image"), dict) and inputs["start_image"].get("asset_id")):
            raise DomainError(ErrorCode.INVALID_REQUEST, "start_image.asset_id is required")
        if not (isinstance(inputs.get("end_image"), dict) and inputs["end_image"].get("asset_id")):
            raise DomainError(ErrorCode.INVALID_REQUEST, "end_image.asset_id is required")
    if operation == "video.lipsync":
        if not str(inputs.get("prompt") or "").strip():
            raise DomainError(ErrorCode.INVALID_REQUEST, "prompt is required")
        if not (isinstance(inputs.get("start_image"), dict) and inputs["start_image"].get("asset_id")):
            raise DomainError(ErrorCode.INVALID_REQUEST, "start_image.asset_id is required")
        if not (isinstance(inputs.get("audio"), dict) and inputs["audio"].get("asset_id")):
            raise DomainError(ErrorCode.INVALID_REQUEST, "audio.asset_id is required")
    if operation == "video.motion_transfer":
        if not str(inputs.get("prompt") or "").strip():
            raise DomainError(ErrorCode.INVALID_REQUEST, "prompt is required")
        if not (isinstance(inputs.get("reference_video"), dict) and inputs["reference_video"].get("asset_id")):
            raise DomainError(ErrorCode.INVALID_REQUEST, "reference_video.asset_id is required")
    count = inputs.get("candidate_count")
    if count is not None and (int(count) < 1 or int(count) > 8):
        raise DomainError(ErrorCode.INVALID_PARAMETER, "candidate_count must be 1-8")
    duration = inputs.get("duration_seconds")
    if duration is not None and (float(duration) < 1 or float(duration) > 30):
        raise DomainError(ErrorCode.INVALID_PARAMETER, "duration_seconds must be 1-30")
