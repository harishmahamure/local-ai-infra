from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from .. import config, downloads, runtime
from ..domain.errors import DomainError, ErrorCode
from ..domain.models import ModelState
from ..infrastructure.catalogs import CatalogRegistry, enforce_license
from ..infrastructure.llama_chat import LlamaChatClient
from ..infrastructure.model_runtime import CatalogModelRuntime, SystemdModelDownloader
from .jobs import JobService
from .text_chat import TEXT_CHAT_MODELS, build_llama_messages, llama_model_name


def _no_assets(asset_id: str) -> tuple[bytes, str]:
    raise DomainError(
        ErrorCode.INVALID_REQUEST,
        "image_asset is not supported; use image_url with a data: or https URL",
    )


def _live_context_length(model_id: str, fallback: int) -> int:
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


@dataclass
class ControlServices:
    catalog: CatalogRegistry
    model_runtime: CatalogModelRuntime
    downloader: SystemdModelDownloader
    llama_chat: LlamaChatClient | None = None
    jobs: JobService | None = None

    def start(self) -> None:
        if self.jobs:
            self.jobs.recover_and_start()

    def stop(self) -> None:
        if self.jobs:
            self.jobs.stop()

    def _llama_client(self) -> LlamaChatClient:
        if self.llama_chat is None:
            self.llama_chat = LlamaChatClient()
        return self.llama_chat

    def _public_model(self, model) -> dict[str, Any]:
        payload = model.to_public_dict(
            state=self.model_runtime.state(model.id),
            disk_status=self.model_runtime.disk_status().get(model.id),
        )
        payload["profile"] = self.model_runtime.profile_for(model.id)
        live_ctx = _live_context_length(model.id, model.supported_context)
        if live_ctx:
            payload["context_length"] = live_ctx
        return payload

    def status(self) -> dict[str, Any]:
        payload = runtime.get_status()
        if self.jobs:
            payload["queue"] = self.jobs.snapshot()
        return payload

    def disk_models(self) -> dict[str, Any]:
        return runtime.get_models()

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

    def load_model(self, model_id: str) -> dict[str, Any]:
        self._guard_gpu()
        try:
            return self.model_runtime.load(model_id)
        except DomainError:
            raise
        except Exception as exc:
            raise DomainError(ErrorCode.INTERNAL_ERROR, str(exc)) from exc

    def unload_model(self, model_id: str) -> dict[str, Any]:
        self._guard_gpu()
        return self.model_runtime.unload(model_id)

    def _guard_gpu(self) -> None:
        if self.jobs and self.jobs.scheduler.is_busy():
            raise DomainError(
                ErrorCode.GPU_BUSY,
                "An image job is using the GPU; wait for it to finish or cancel it",
                {"active_job": self.jobs.scheduler.active_job()},
            )

    def download_status(self) -> dict[str, Any]:
        return self.downloader.status()

    def start_download(self, ids: list[str] | None) -> dict[str, Any]:
        if ids:
            unknown = [mid for mid in ids if mid not in self.catalog.models]
            if unknown:
                raise DomainError(ErrorCode.MODEL_NOT_AVAILABLE, f"Unknown model {unknown[0]}")
        return self.downloader.start(ids or [])

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._llama_chat_payload(body)
        try:
            return self._llama_client().complete(payload)
        except DomainError:
            raise
        except Exception as exc:
            raise DomainError(ErrorCode.INTERNAL_ERROR, f"Text model request failed: {exc}") from exc

    def chat_stream(self, body: dict[str, Any]) -> Iterator[str]:
        payload = self._llama_chat_payload(body)
        for line in self._llama_client().stream(payload):
            if line.startswith("data:") or line.startswith("event:"):
                yield f"{line.rstrip()}\n\n" if not line.endswith("\n\n") else line
            else:
                yield f"data: {line.rstrip()}\n\n"

    def ready(self) -> tuple[bool, dict[str, Any]]:
        status = runtime.get_status()
        loaded = status.get("loadState") == "LOADED"
        details = {
            "loadState": status.get("loadState"),
            "profile": status.get("profile"),
            "catalog": bool(self.catalog.models),
        }
        return loaded, details

    def _llama_chat_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        model_id = str(body.get("model") or "").strip()
        model = self.catalog.models.get(model_id)
        if model is None or model_id not in TEXT_CHAT_MODELS:
            raise DomainError(ErrorCode.MODEL_NOT_AVAILABLE, f"Unknown text model {model_id}")
        if "text.chat" not in model.supported_operations:
            raise DomainError(ErrorCode.UNSUPPORTED_OPERATION, f"{model_id} does not support text.chat")
        enforce_license(self.catalog, [model_id])
        messages = build_llama_messages(list(body.get("messages") or []), resolve_asset=_no_assets)
        self._guard_gpu()
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


def build_services(
    *,
    catalog: CatalogRegistry | None = None,
    model_runtime: CatalogModelRuntime | None = None,
    downloader: SystemdModelDownloader | None = None,
    llama_chat: LlamaChatClient | None = None,
    commercial_mode: bool | None = None,
    jobs: JobService | None = None,
    enable_jobs: bool = True,
) -> ControlServices:
    from ..infrastructure.catalogs import load_catalogs

    cat = catalog or load_catalogs(
        config.CATALOG_DIR,
        commercial_mode=config.COMMERCIAL_MODE if commercial_mode is None else commercial_mode,
    )
    dl = downloader or SystemdModelDownloader(downloads)
    runtime_impl = model_runtime or CatalogModelRuntime(cat, runtime, downloads)
    job_svc = jobs
    if job_svc is None and enable_jobs:
        from datetime import datetime, timezone

        from ..infrastructure.assets import AssetStore
        from ..infrastructure.comfy_client import ComfyClient
        from ..infrastructure.gpu_scheduler import GpuScheduler
        from ..infrastructure.job_store import JobStore
        from ..infrastructure.job_worker import JobWorker

        store = JobStore(config.JOBS_DB)
        asset_store = AssetStore(config.ASSET_DIR, store)
        scheduler = GpuScheduler(runtime)
        worker = JobWorker(
            jobs=store,
            assets=asset_store,
            scheduler=scheduler,
            catalog=cat,
            client=ComfyClient(),
            now_iso=lambda: datetime.now(timezone.utc).isoformat(),
            disk_status=runtime_impl.disk_status,
        )
        job_svc = JobService(
            store=store,
            assets=asset_store,
            scheduler=scheduler,
            worker=worker,
            disk_status=runtime_impl.disk_status,
        )
    return ControlServices(
        catalog=cat,
        model_runtime=runtime_impl,
        downloader=dl,
        llama_chat=llama_chat,
        jobs=job_svc,
    )
