from __future__ import annotations

from pathlib import Path

from app.application.chat import build_services
from app.application.text_chat import build_llama_messages, is_filesystem_url, llama_model_name
from app.domain.errors import DomainError, ErrorCode
from app.domain.models import ModelFile, ModelRecord, ModelState
from app.infrastructure.catalogs import load_catalogs
from app.infrastructure.model_runtime import CatalogModelRuntime, profile_for_text_model

REPO = Path(__file__).resolve().parents[3]
PNG = b"\x89PNG\r\n\x1a\n"


class FakeDownloader:
    class ConflictError(Exception):
        pass

    def start_download(self, ids):
        return {"status": "running", "ids": ids or []}

    def get_download_status(self):
        return {"status": "idle"}


class ProfileModule:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped = 0
        self.profile = "none"

    def get_models(self) -> dict:
        return {
            "bundles": [
                {"id": "gemma-4-e4b", "status": "complete"},
                {"id": "qwen36-35b-a3b-rq", "status": "complete"},
            ]
        }

    def get_status(self) -> dict:
        return {"profile": self.profile}

    def start_profile(self, profile: str) -> dict:
        self.started.append(profile)
        self.profile = "gemma" if profile == "gemma" else "llama-fast"
        return {"profile": self.profile}

    def stop_profile(self) -> None:
        self.stopped += 1
        self.profile = "none"


class TrackingRuntime(CatalogModelRuntime):
    def __init__(self, catalog) -> None:
        super().__init__(catalog, ProfileModule(), FakeDownloader())
        self.loads: list[str] = []
        self.unloads: list[str] = []
        self.ready: set[str] = set()

    def load(self, model_id: str) -> dict:
        self.loads.append(model_id)
        self.ready.add(model_id)
        return {"profile": profile_for_text_model(model_id)}

    def unload(self, model_id: str) -> dict:
        self.unloads.append(model_id)
        self.ready.discard(model_id)
        return {"ok": True}

    def state(self, model_id: str) -> ModelState:
        return ModelState.READY if model_id in self.ready else ModelState.AVAILABLE

    def profile_for(self, model_id: str) -> str:
        return profile_for_text_model(model_id)


class FakeLlama:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def complete(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    def stream(self, payload: dict):
        self.payloads.append(payload)
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'


def _engine(*, runtime=None, llama=None):
    catalog = load_catalogs(REPO / "catalog")
    module = ProfileModule()
    model_runtime = runtime or CatalogModelRuntime(catalog, module, FakeDownloader())
    return build_services(
        catalog=catalog,
        model_runtime=model_runtime,
        downloader=FakeDownloader(),
        llama_chat=llama or FakeLlama(),
        enable_jobs=False,
    ), module


def test_build_llama_messages_image_url() -> None:
    messages = build_llama_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this frame."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}},
                ],
            }
        ],
        resolve_asset=lambda _asset_id: (PNG, "image/png"),
    )
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "Describe this frame."}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_rejects_filesystem_image_url() -> None:
    assert is_filesystem_url("/tmp/frame.png") is True
    try:
        build_llama_messages(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "/tmp/frame.png"}}]}],
            resolve_asset=lambda _asset_id: (PNG, "image/png"),
        )
        raise AssertionError("expected filesystem rejection")
    except DomainError as exc:
        assert "Filesystem" in exc.message


def test_llama_model_name() -> None:
    model = ModelRecord(
        id="qwen36-35b-a3b-rq",
        license="Apache-2.0",
        commercial=True,
        dest="llamacpp",
        files=[
            ModelFile("r", "Qwen3.6-35B-A3B-Q4_K_M.gguf", "Qwen3.6-35B-A3B-Q4_K_M.gguf"),
            ModelFile("r", "mmproj.gguf", "Qwen3.6-35B-A3B-mmproj-F16.gguf"),
        ],
    )
    assert llama_model_name(model) == "Qwen3.6-35B-A3B-Q4_K_M"


def test_start_stop_aliases_use_matching_profiles() -> None:
    catalog = load_catalogs(REPO / "catalog")
    module = ProfileModule()
    runtime = CatalogModelRuntime(catalog, module, FakeDownloader())
    runtime.load("gemma-4-e4b")
    assert module.started == ["gemma"]
    runtime.load("qwen36-35b-a3b-rq")
    assert module.started[-1] == "llama-fast"
    runtime.unload("qwen36-35b-a3b-rq")
    assert module.stopped == 1


def test_chat_proxy_includes_image_url() -> None:
    runtime = TrackingRuntime(load_catalogs(REPO / "catalog"))
    llama = FakeLlama()
    engine, _ = _engine(runtime=runtime, llama=llama)
    result = engine.chat(
        {
            "model": "gemma-4-e4b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this frame."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}},
                    ],
                }
            ],
        }
    )
    assert result["choices"][0]["message"]["content"] == "ok"
    assert runtime.loads == ["gemma-4-e4b"]
    payload = llama.payloads[0]
    assert payload["model"] == "gemma-4-E4B-it-Q4_K_M"
    url = payload["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_qwen_chat_starts_llama_fast() -> None:
    runtime = TrackingRuntime(load_catalogs(REPO / "catalog"))
    llama = FakeLlama()
    engine, _ = _engine(runtime=runtime, llama=llama)
    engine.chat({"model": "qwen36-35b-a3b-rq", "messages": [{"role": "user", "content": "hi"}]})
    assert runtime.loads == ["qwen36-35b-a3b-rq"]
    assert llama.payloads[0]["model"] == "Qwen3.6-35B-A3B-Q4_K_M"


def test_image_asset_is_rejected() -> None:
    engine, _ = _engine()
    try:
        engine.chat(
            {
                "model": "gemma-4-e4b",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_asset", "asset_id": "ast_01KTEST"}],
                    }
                ],
            }
        )
        raise AssertionError("expected image_asset rejection")
    except DomainError as exc:
        assert exc.code == ErrorCode.INVALID_REQUEST
