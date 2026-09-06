from __future__ import annotations

from app.downloads import _build_progress, _hf_incomplete_bytes


def _models(*files: dict) -> dict:
    by_bundle: dict[str, list] = {}
    for f in files:
        by_bundle.setdefault(f["bundleId"], []).append(f)
    bundles = [
        {
            "id": bid,
            "status": "complete" if all(x.get("state") == "ok" for x in flist if not x.get("optional")) else "partial",
            "files": flist,
        }
        for bid, flist in by_bundle.items()
    ]
    return {"bundles": bundles}


def test_percent_complete_counts_finished_required_files() -> None:
    models = _models(
        {"bundleId": "a", "name": "one.gguf", "state": "ok", "bytes": 100, "optional": False},
        {"bundleId": "a", "name": "two.gguf", "state": "missing", "bytes": 0, "optional": False},
    )
    progress = _build_progress(models, {"status": "idle"})
    assert progress["filesComplete"] == 1
    assert progress["filesRequired"] == 2
    assert progress["percentComplete"] == 50.0


def test_running_file_adds_in_flight_percent() -> None:
    models = _models(
        {"bundleId": "a", "name": "one.gguf", "state": "ok", "bytes": 100, "optional": False},
        {"bundleId": "a", "name": "two.gguf", "state": "missing", "bytes": 0, "optional": False},
    )
    progress = _build_progress(
        models,
        {
            "status": "running",
            "currentFile": "two.gguf",
            "currentBytes": 50,
            "currentTotal": 200,
            "currentPercent": 25.0,
        },
    )
    assert progress["percentComplete"] == 62.5
    assert progress["currentFilePercent"] == 25.0
    current = next(f for f in progress["files"] if f["name"] == "two.gguf")
    assert current["state"] == "downloading"
    assert current["isCurrent"] is True
    assert current["percent"] == 25.0


def test_failed_file_uses_state_error() -> None:
    models = _models(
        {"bundleId": "ltx-iclora-lipdub", "name": "loras/lipdub.safetensors", "state": "missing", "bytes": 0, "optional": False},
    )
    progress = _build_progress(
        models,
        {
            "status": "failed",
            "failedFiles": [
                {
                    "bundleId": "ltx-iclora-lipdub",
                    "name": "loras/lipdub.safetensors",
                    "error": "Gated HuggingFace repo",
                }
            ],
        },
    )
    row = progress["files"][0]
    assert row["state"] == "failed"
    assert "Gated" in row["error"]


def test_incomplete_blob_fallback(tmp_path, monkeypatch) -> None:
    blobs = tmp_path / "hub" / "models--Lightricks--LTX-2.5" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "old.incomplete").write_bytes(b"x" * 10)
    newer = blobs / "new.incomplete"
    newer.write_bytes(b"y" * 42)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert (
        _hf_incomplete_bytes({"current": "Lightricks/LTX-2.5/loras/file.safetensors"})
        == 42
    )
