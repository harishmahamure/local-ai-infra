from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.bootstrap.app import create_app
from app.bootstrap.container import build_engine
from tests.unit.test_idempotency import PNG, FakeDownloader, FakeExecutor, FakeRuntime, FakeScheduler
from tests.unit.test_text_chat import FakeLlama, TrackingRuntime

REPO = Path(__file__).resolve().parents[3]


def _engine():
    engine = build_engine(
        memory=True,
        executor=FakeExecutor(),
        scheduler=FakeScheduler(),
        downloader=FakeDownloader(),
        catalog_dir=REPO / "catalog",
        start_worker=True,
        commercial_mode=True,
    )
    engine.model_runtime = FakeRuntime(engine.catalog)
    return engine


def _client():
    return TestClient(create_app(_engine()))


def test_operations_and_capabilities() -> None:
    client = _client()
    ops = client.get("/v1/operations").json()["operations"]
    ids = {item["id"] for item in ops}
    assert "image.generate" in ids
    assert "audio.tts" in ids
    audio = next(item for item in ops if item["id"] == "audio.tts")
    assert audio["available"] is False
    caps = client.get("/v1/capabilities").json()
    assert caps["image"]["generate"] is True
    assert caps["audio"]["music"] is False
    text = next(item for item in ops if item["id"] == "text.chat")
    assert text["available"] is True
    assert caps["text"]["chat"] is True
    assert caps["text"]["vision"] is True


def test_submit_get_complete() -> None:
    client = _client()
    res = client.post(
        "/v1/jobs",
        json={"operation": "image.generate", "preset": "character_master", "inputs": {"prompt": "a historical inspector"}},
        headers={"Idempotency-Key": "itest-1"},
    )
    assert res.status_code == 202
    job_id = res.json()["job_id"]
    assert job_id.startswith("job_")
    deadline = time.time() + 5
    body = {}
    while time.time() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body.get("status") == "COMPLETED":
            break
        time.sleep(0.05)
    assert body["status"] == "COMPLETED"
    assert body["result"]["asset_ids"]
    asset_id = body["result"]["asset_ids"][0]
    asset = client.get(f"/v1/assets/{asset_id}").json()
    assert asset["asset_id"] == asset_id
    assert "expires_at" in asset


def test_upload_and_upscale() -> None:
    client = _client()
    up = client.post("/v1/assets", files={"file": ("t.png", PNG, "image/png")})
    assert up.status_code == 201
    asset_id = up.json()["asset_id"]
    res = client.post(
        "/v1/jobs",
        json={"operation": "image.upscale", "preset": "conservative_upscale", "inputs": {"image": {"asset_id": asset_id}}},
    )
    assert res.status_code == 202


def test_signed_upload_rejected() -> None:
    client = _client()
    res = client.post(
        "/v1/jobs",
        json={
            "operation": "image.generate",
            "preset": "master",
            "inputs": {"prompt": "x"},
            "output": {"mode": "signed_upload", "destination": {"url": "https://example.com"}},
        },
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "UNSUPPORTED_PARAMETER"


def test_unimplemented_operation() -> None:
    client = _client()
    res = client.post("/v1/jobs", json={"operation": "audio.tts", "preset": "master", "inputs": {"text": "hi"}})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "UNSUPPORTED_OPERATION"


def test_cancel_and_retry() -> None:
    engine = build_engine(
        memory=True,
        executor=FakeExecutor(),
        scheduler=FakeScheduler(),
        downloader=FakeDownloader(),
        catalog_dir=REPO / "catalog",
        start_worker=False,
        commercial_mode=True,
    )
    engine.model_runtime = FakeRuntime(engine.catalog)
    client = TestClient(create_app(engine))
    res = client.post("/v1/jobs", json={"operation": "image.generate", "preset": "master", "inputs": {"prompt": "x"}})
    job_id = res.json()["job_id"]
    cancelled = client.post(f"/v1/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "CANCELLED"
    retry = client.post(f"/v1/jobs/{job_id}/retry", json={"strategy": "new_seed"})
    assert retry.status_code == 202
    assert retry.json()["job_id"] != job_id


def test_health_and_ready() -> None:
    client = _client()
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True


def test_legacy_status_still_present() -> None:
    client = _client()
    paths = client.app.openapi()["paths"]
    assert "/api/v1/generate" in paths
    assert "/v1/jobs" in paths


def test_text_start_stop_and_vision_chat() -> None:
    engine = _engine()
    runtime = TrackingRuntime(engine.catalog)
    engine.model_runtime = runtime
    llama = FakeLlama()
    engine.llama_chat = llama
    client = TestClient(create_app(engine))

    start = client.post("/v1/models/gemma-4-e4b/start")
    assert start.status_code == 200
    assert runtime.loads == ["gemma-4-e4b"]
    stop = client.post("/v1/models/gemma-4-e4b/stop")
    assert stop.status_code == 200
    assert runtime.unloads == ["gemma-4-e4b"]

    listed = client.get("/v1/text/models").json()["models"]
    ids = {item["id"] for item in listed}
    assert ids == {"gemma-4-e4b", "qwen36-35b-a3b-rq"}
    gemma = next(item for item in listed if item["id"] == "gemma-4-e4b")
    assert gemma["vision"] is True
    assert gemma["profile"] == "gemma"
    assert gemma["context_length"] == 131072

    up = client.post("/v1/assets", files={"file": ("t.png", PNG, "image/png")})
    asset_id = up.json()["asset_id"]
    chat = client.post(
        "/v1/text/chat",
        json={
            "model": "qwen36-35b-a3b-rq",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this frame."},
                        {"type": "image_asset", "asset_id": asset_id},
                    ],
                }
            ],
        },
    )
    assert chat.status_code == 200
    payload = llama.payloads[0]
    assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert runtime.loads[-1] == "qwen36-35b-a3b-rq"
