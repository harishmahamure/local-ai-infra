from __future__ import annotations

import json

from app import cleanup


def test_inventory_and_run(tmp_path, monkeypatch) -> None:
    jobs = tmp_path / "jobs"
    images = jobs / "job-old" / "images"
    videos = jobs / "job-old" / "videos"
    active = jobs / "job-live" / "videos"
    images.mkdir(parents=True)
    videos.mkdir(parents=True)
    active.mkdir(parents=True)
    (images / "still.png").write_bytes(b"png" * 10)
    (videos / "clip.mp4").write_bytes(b"mp4" * 20)
    (active / "keep.mp4").write_bytes(b"live" * 8)
    (jobs / "job-old" / "notes.txt").write_text("leave me")

    store = tmp_path / "ltx-video-jobs.json"
    store.write_text(json.dumps({"jobs": {"job-live": {"status": "running"}}}))
    comfy_out = tmp_path / "comfy-output"
    comfy_out.mkdir()
    (comfy_out / "comfy.png").write_bytes(b"c" * 5)

    monkeypatch.setattr(cleanup, "JOBS_ROOT", jobs)
    monkeypatch.setattr(cleanup, "JOB_STORES", (store,))
    monkeypatch.setattr(cleanup, "_comfy_output_dirs", lambda: [comfy_out])

    inv = cleanup.inventory()
    assert inv["files"] == 3
    assert inv["skippedJobIds"] == ["job-live"]
    ids = {t["id"]: t["files"] for t in inv["targets"]}
    assert ids["images"] == 1
    assert ids["videos"] == 1
    assert ids["comfy_outputs"] == 1

    result = cleanup.run(["images", "videos"])
    assert result["deletedFiles"] == 2
    assert not (images / "still.png").exists()
    assert not (videos / "clip.mp4").exists()
    assert (active / "keep.mp4").exists()
    assert (comfy_out / "comfy.png").exists()

    comfy = cleanup.run(["comfy_outputs"])
    assert comfy["deletedFiles"] == 1
    assert not (comfy_out / "comfy.png").exists()


def test_run_rejects_empty_targets() -> None:
    try:
        cleanup.run([])
    except cleanup.CleanupError as exc:
        assert exc.status == 400
    else:
        raise AssertionError("expected CleanupError")
