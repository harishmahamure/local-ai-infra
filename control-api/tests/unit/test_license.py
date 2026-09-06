from __future__ import annotations

from pathlib import Path

from app.domain.errors import DomainError, ErrorCode
from app.infrastructure.catalogs import load_catalogs
from tests.unit.test_idempotency import FakeDownloader, FakeExecutor, FakeRuntime, FakeScheduler
from app.bootstrap.container import build_engine

REPO = Path(__file__).resolve().parents[3]


def test_commercial_mode_blocks_nc_model() -> None:
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
    engine.catalog.models["qwen-image-2512-fp8"].noncommercial_only = True
    try:
        engine.submit_job({"operation": "image.generate", "preset": "master", "inputs": {"prompt": "x"}}, idempotency_key=None, trace_id="t")
        raise AssertionError("expected license error")
    except DomainError as exc:
        assert exc.code == ErrorCode.LICENSE_RESTRICTED


def test_catalog_default_commercial() -> None:
    catalog = load_catalogs(REPO / "catalog")
    assert catalog.models["qwen-image-2512-fp8"].commercial is True
    assert catalog.models["qwen-image-2512-fp8"].noncommercial_only is False
    assert catalog.models["chatterbox-multilingual"].commercial is True
    assert catalog.models["chatterbox-hi"].commercial is True
    assert catalog.models["ace-step-1.5"].commercial is True
