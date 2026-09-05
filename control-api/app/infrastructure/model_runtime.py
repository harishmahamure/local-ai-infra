from __future__ import annotations

from typing import Any

from ..domain.errors import DomainError, ErrorCode
from ..domain.models import ModelState
from ..application.catalog import CatalogRegistry

PROFILE_FOR_DEST = {
    "comfyui": "comfy",
    "llamacpp": "llama-fast",
}

LTX_MODELS = {
    "ltx-2.5-distilled",
    "ltx-2.5-studio",
    "ltx-2.5-prompt-enhancer",
    "ltx-camera-loras",
    "ltx-iclora-union",
    "ltx-iclora-detailer",
    "ltx-iclora-lipdub",
    "ltx-iclora-motion-track",
}


class CatalogModelRuntime:
    def __init__(self, catalog: CatalogRegistry, runtime_module: Any, downloader: Any) -> None:
        self._catalog = catalog
        self._runtime = runtime_module
        self._downloader = downloader

    def disk_status(self) -> dict[str, str]:
        report = self._runtime.get_models()
        out: dict[str, str] = {}
        for bundle in report.get("bundles") or []:
            mid = bundle.get("id")
            if mid:
                out[str(mid)] = str(bundle.get("status") or "missing")
        return out

    def state(self, model_id: str) -> ModelState:
        model = self._catalog.models.get(model_id)
        if model is None:
            return ModelState.INCOMPATIBLE
        if not model.enabled:
            return ModelState.DISABLED
        status = self.disk_status().get(model_id, "missing")
        if status == "complete":
            loaded = str((self._runtime.get_status() or {}).get("profile") or "")
            wanted = self._profile_for(model_id)
            unit = "comfyui" if wanted == "comfy" else "comfyui-ltx" if wanted == "comfy-ltx" else wanted
            if loaded == unit:
                return ModelState.READY
            return ModelState.AVAILABLE
        if status == "partial":
            return ModelState.NOT_DOWNLOADED
        return ModelState.NOT_DOWNLOADED

    def load(self, model_id: str) -> dict[str, Any]:
        if model_id not in self._catalog.models:
            raise DomainError(ErrorCode.MODEL_NOT_AVAILABLE, f"Unknown model {model_id}")
        if self.state(model_id) == ModelState.NOT_DOWNLOADED:
            raise DomainError(ErrorCode.MODEL_NOT_AVAILABLE, f"{model_id} is not downloaded")
        profile = self._profile_for(model_id)
        return self._runtime.start_profile(profile)

    def unload(self, model_id: str) -> dict[str, Any]:
        self._runtime.stop_profile()
        return self._runtime.get_status()

    def _profile_for(self, model_id: str) -> str:
        if model_id in LTX_MODELS:
            return "comfy-ltx"
        model = self._catalog.models[model_id]
        if model.dest == "llamacpp":
            return profile_for_text_model(model_id)
        if model_id.startswith("gemma"):
            return "gemma"
        return "comfy"

    def profile_for(self, model_id: str) -> str:
        return self._profile_for(model_id)


def profile_for_text_model(model_id: str) -> str:
    if model_id.startswith("gemma"):
        return "gemma"
    return "llama-fast"


class SystemdModelDownloader:
    def __init__(self, downloads_module: Any) -> None:
        self._downloads = downloads_module

    def start(self, model_ids: list[str]) -> dict[str, Any]:
        try:
            return self._downloads.start_download(model_ids or None)
        except self._downloads.ConflictError as exc:
            raise DomainError(ErrorCode.MODEL_DOWNLOAD_FAILED, str(exc)) from exc

    def status(self) -> dict[str, Any]:
        return self._downloads.get_download_status()
