from __future__ import annotations

from fastapi.testclient import TestClient

from app.application.chat import build_services
from app.bootstrap.app import create_app


def _client() -> TestClient:
    return TestClient(create_app(build_services(enable_jobs=False)))


def test_ui_paths_serve_react_shell() -> None:
    client = _client()
    for path in (
        "/",
        "/runtime",
        "/images",
        "/images/generate_prop",
        "/jobs",
        "/jobs/job_example",
        "/models",
        "/downloads",
    ):
        res = client.get(path)
        assert res.status_code == 200, path
        assert "text/html" in res.headers["content-type"]
        assert b'id="root"' in res.content
        assert b"/assets/" in res.content
        assert b"app.js" not in res.content


def test_unknown_ui_path_serves_spa() -> None:
    client = _client()
    res = client.get("/not-a-page")
    assert res.status_code == 200
    assert b'id="root"' in res.content


def test_api_docs_and_assets_untouched() -> None:
    client = _client()
    assert client.get("/health").json()["status"] == "ok"
    spec = client.get("/openapi.json").json()
    assert "/v1/status" in spec["paths"]
    assert "/v1/text/chat" in spec["paths"]
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "text/html" in docs.headers["content-type"]
    index = client.get("/").text
    js_name = next(part for part in index.split('"') if part.startswith("/assets/") and part.endswith(".js"))
    js = client.get(js_name)
    assert js.status_code == 200
    assert "text/javascript" in js.headers.get("content-type", "") or js.content
