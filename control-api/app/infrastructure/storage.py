from __future__ import annotations

import hashlib
import os
from pathlib import Path


class LocalArtifactStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, asset_id: str) -> Path:
        if "/" in asset_id or ".." in asset_id or not asset_id.startswith("ast_"):
            raise ValueError("invalid asset_id")
        path = self.root / asset_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write(self, asset_id: str, data: bytes) -> str:
        directory = self._dir(asset_id)
        tmp = directory / "data.tmp"
        final = directory / "data"
        tmp.write_bytes(data)
        os.replace(tmp, final)
        return hashlib.sha256(data).hexdigest()

    def read(self, asset_id: str) -> bytes:
        path = self.root / asset_id / "data"
        if not path.is_file():
            raise FileNotFoundError(asset_id)
        return path.read_bytes()

    def delete(self, asset_id: str) -> None:
        path = self.root / asset_id / "data"
        if path.is_file():
            path.unlink()

    def writable(self) -> bool:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            probe = self.root / ".writable"
            probe.write_text("ok")
            probe.unlink()
            return True
        except OSError:
            return False


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
