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
    generate = next(item for item in ops if item["id"] == "image.generate")
    assert "face_lock" in generate["presets"]
    assert "audio.tts" in ids
    assert "audio.music" in ids
    assert "audio.foley" in ids
    assert "video.upscale" in ids
    assert "media.finalize" in ids
    audio = next(item for item in ops if item["id"] == "audio.tts")
    assert audio["available"] is True
    assert audio["implemented"] is True
    assert "text" in (audio["input_schema"].get("required") or [])
    assert audio["output_schema"]
    music = next(item for item in ops if item["id"] == "audio.music")
    assert music["implemented"] is True
    assert music["available"] is True
    assert "prompt" in (music["input_schema"].get("required") or [])
    lipsync = next(item for item in ops if item["id"] == "video.lipsync")
    assert lipsync["implemented"] is True
    caps = client.get("/v1/capabilities").json()
    assert caps["image"]["generate"] is True
    assert caps["audio"]["music"] is True
    assert caps["audio"]["foley"] is True
    assert caps["audio"]["tts"]
    assert caps["video"]["upscale"] is False
    assert caps["media"]["concat"] is True
    text = next(item for item in ops if item["id"] == "text.chat")
    assert text["available"] is True
    assert text["implemented"] is True
    assert caps["text"]["chat"] is True
    assert caps["text"]["vision"] is True


def test_get_operation_contract() -> None:
    client = _client()
    lipsync = client.get("/v1/operations/video.lipsync").json()
    assert lipsync["id"] == "video.lipsync"
    assert lipsync["implemented"] is True
    assert lipsync["available"] is True
    assert "ltx-iclora-lipdub" in lipsync["required_models"]
    assert "ltx25-lipsync" in lipsync["workflow_ids"]
    assert "audio" in (lipsync["input_schema"].get("required") or [])
    music = client.get("/v1/operations/audio.music").json()
    assert music["implemented"] is True
    assert music["available"] is True
    assert music.get("reason") in {None, ""}
    assert "ace-step-1.5" in music["required_models"]
    assert "prompt" in (music["input_schema"].get("required") or [])
    assert music["output_schema"]
    assert "audio-music" in music["workflow_ids"]
    missing = client.get("/v1/operations/movie.direct")
    assert missing.status_code == 400
    assert missing.json()["error"]["code"] == "UNSUPPORTED_OPERATION"


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


def test_image_edit_control_layered_jobs() -> None:
    client = _client()
    up = client.post("/v1/assets", files={"file": ("t.png", PNG, "image/png")})
    asset_id = up.json()["asset_id"]
    edit = client.post(
        "/v1/jobs",
        json={
            "operation": "image.edit",
            "preset": "master",
            "inputs": {"prompt": "change the sign", "reference_images": [{"asset_id": asset_id}]},
        },
    )
    assert edit.status_code == 202, edit.text
    control = client.post(
        "/v1/jobs",
        json={
            "operation": "image.controlled",
            "preset": "balanced",
            "inputs": {"prompt": "match this pose", "control_image": {"asset_id": asset_id}, "control_type": "pose"},
        },
    )
    assert control.status_code == 202, control.text
    layered = client.post(
        "/v1/jobs",
        json={
            "operation": "image.layered",
            "preset": "master",
            "inputs": {"image": {"asset_id": asset_id}, "layers": 3},
        },
    )
    assert layered.status_code == 202, layered.text
    missing = client.post(
        "/v1/jobs",
        json={"operation": "image.edit", "preset": "master", "inputs": {"prompt": "no image"}},
    )
    assert missing.status_code == 400


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
    res = client.post("/v1/jobs", json={"operation": "video.continue", "preset": "master", "inputs": {"source_video": {"asset_id": "ast_x"}, "prompt": "next shot"}})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "UNSUPPORTED_OPERATION"


def test_phase_a_jobs_accepted() -> None:
    client = _client()
    tts = client.post("/v1/jobs", json={"operation": "audio.tts", "preset": "narrator_hindi", "inputs": {"text": "नमस्ते"}})
    assert tts.status_code == 202, tts.text
    music = client.post(
        "/v1/jobs",
        json={"operation": "audio.music", "preset": "cinematic_master", "inputs": {"prompt": "sparse drone", "duration_seconds": 24}},
    )
    assert music.status_code == 202, music.text
    missing_tts = client.post("/v1/jobs", json={"operation": "audio.tts", "preset": "master", "inputs": {}})
    assert missing_tts.status_code == 400
    missing_music = client.post("/v1/jobs", json={"operation": "audio.music", "preset": "master", "inputs": {}})
    assert missing_music.status_code == 400

    up = client.post("/v1/assets", files={"file": ("t.png", PNG, "image/png")})
    asset_id = up.json()["asset_id"]
    mix = client.post(
        "/v1/jobs",
        json={
            "operation": "audio.mix",
            "preset": "cinematic",
            "inputs": {"stems": [{"asset_id": asset_id, "role": "music", "gain_db": -3}]},
        },
    )
    assert mix.status_code == 202, mix.text
    concat = client.post(
        "/v1/jobs",
        json={
            "operation": "media.concat",
            "preset": "master",
            "inputs": {"clips": [{"asset_id": asset_id}, {"asset_id": asset_id}], "transition": "cut"},
        },
    )
    assert concat.status_code == 202, concat.text
    finalize = client.post(
        "/v1/jobs",
        json={
            "operation": "media.finalize",
            "preset": "master",
            "inputs": {"video": {"asset_id": asset_id}, "audio": {"asset_id": asset_id}},
        },
    )
    assert finalize.status_code == 202, finalize.text
    missing_mix = client.post("/v1/jobs", json={"operation": "audio.mix", "preset": "master", "inputs": {"stems": []}})
    assert missing_mix.status_code == 400
    missing_concat = client.post(
        "/v1/jobs",
        json={"operation": "media.concat", "preset": "master", "inputs": {"clips": [{"asset_id": asset_id}]}},
    )
    assert missing_concat.status_code == 400
    missing_final = client.post("/v1/jobs", json={"operation": "media.finalize", "preset": "master", "inputs": {}})
    assert missing_final.status_code == 400


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


def test_legacy_generate_flow_validates() -> None:
    client = _client()
    missing = client.post("/api/v1/generate", json={"flow": "merge", "prompt": "combine these"})
    assert missing.status_code == 422
    ok = client.post("/api/v1/generate", json={"flow": "t2i", "prompt": "a quiet harbor"})
    assert ok.status_code == 202, ok.text
    assert ok.json().get("jobId")
    img2img_missing = client.post("/api/v1/generate", json={"flow": "img2img", "prompt": "same person", "denoise": 0.65})
    assert img2img_missing.status_code == 422
    png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    img2img_ok = client.post(
        "/api/v1/generate",
        json={"flow": "img2img", "prompt": "same person, new lighting", "image": png, "denoise": 0.65},
    )
    assert img2img_ok.status_code == 202, img2img_ok.text
    character_missing = client.post(
        "/api/v1/generate",
        json={"flow": "character", "prompt": "same man in steel armor, dusk courtyard"},
    )
    assert character_missing.status_code == 422
    character_ok = client.post(
        "/api/v1/generate",
        json={
            "flow": "character",
            "prompt": "same man in steel armor, dusk courtyard",
            "image": png,
        },
    )
    assert character_ok.status_code == 202, character_ok.text
    no_prompt = client.post("/api/v1/generate", json={"flow": "character", "image": png, "prompt": ""})
    assert no_prompt.status_code == 422


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
