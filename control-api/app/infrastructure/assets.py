from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..domain.errors import DomainError, ErrorCode
from .job_store import JobStore

MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def png_size(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def sniff_mime(data: bytes, fallback: str = "application/octet-stream") -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    if data[:4] == b"\x1aE\xdf\xa3":
        return "video/webm"
    return fallback


class AssetStore:
    def __init__(self, root: Path, jobs: JobStore) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs = jobs

    def write(
        self,
        data: bytes,
        *,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        if not data:
            raise DomainError(ErrorCode.INVALID_REQUEST, "Empty file")
        mime = sniff_mime(data, mime_type or "application/octet-stream")
        digest = sha256_bytes(data)
        ext = MIME_EXT.get(mime, ".bin")
        rel = Path(digest[:2]) / f"{digest}{ext}"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        width, height = png_size(data) if mime == "image/png" else (None, None)
        record = {
            "asset_id": f"ast_{uuid.uuid4().hex}",
            "sha256": digest,
            "mime_type": mime,
            "size_bytes": len(data),
            "width": width,
            "height": height,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "path": str(path),
            "filename": filename,
        }
        self._jobs.save_asset(record)
        return record

    def get(self, asset_id: str) -> dict[str, Any]:
        record = self._jobs.get_asset(asset_id)
        if not record:
            raise DomainError(ErrorCode.ASSET_NOT_FOUND, f"Asset {asset_id} not found")
        return record

    def read(self, asset_id: str) -> tuple[bytes, dict[str, Any]]:
        record = self.get(asset_id)
        path = Path(record["path"])
        if not path.is_file():
            raise DomainError(ErrorCode.ASSET_NOT_FOUND, f"Asset file missing for {asset_id}")
        return path.read_bytes(), record

    def public_dict(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "asset_id": record["asset_id"],
            "mime_type": record["mime_type"],
            "size_bytes": record["size_bytes"],
            "width": record.get("width"),
            "height": record.get("height"),
            "sha256": record["sha256"],
            "created_at": record["created_at"],
            "filename": record.get("filename"),
        }

    def delete(self, asset_id: str) -> dict[str, Any]:
        record = self.get(asset_id)
        path = Path(str(record["path"])).resolve()
        root = self.root.resolve()
        if path != root and root not in path.parents:
            raise DomainError(ErrorCode.INTERNAL_ERROR, f"Asset path is outside the assets root for {asset_id}")
        remaining = self._jobs.count_assets_with_path(str(record["path"]))
        file_removed = False
        if remaining <= 1 and path.is_file():
            path.unlink()
            file_removed = True
        deleted = self._jobs.delete_asset_record(asset_id)
        if deleted is None:
            raise DomainError(ErrorCode.ASSET_NOT_FOUND, f"Asset {asset_id} not found")
        self._jobs.unref_asset(asset_id)
        return {**self.public_dict(record), "deleted": True, "file_removed": file_removed}
