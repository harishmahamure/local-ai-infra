"""Persist character-master assets and reproducibility metadata."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .. import config

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

MASTER_FILENAME = "CHARACTER_MASTER.png"
METADATA_FILENAME = "metadata.json"


def assets_root(root: Path | None = None) -> Path:
    base = root if root is not None else Path(config.LOGS)
    return base / "assets" / "characters"


def character_dir(character_id: str, *, root: Path | None = None) -> Path:
    if not is_safe_character_id(character_id):
        raise ValueError(f"Invalid characterId: {character_id}")
    path = assets_root(root) / character_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "candidates").mkdir(parents=True, exist_ok=True)
    return path


def is_safe_character_id(character_id: str) -> bool:
    return bool(character_id) and bool(_SAFE_ID.match(character_id))


def candidate_filename(index: int, seed: int | None) -> str:
    seed_part = str(seed) if seed is not None else "random"
    return f"{index:02d}_{seed_part}.png"


def persist_candidate(
    character_id: str,
    index: int,
    seed: int | None,
    image_bytes: bytes,
    *,
    root: Path | None = None,
) -> Path:
    dest = character_dir(character_id, root=root) / "candidates" / candidate_filename(index, seed)
    dest.write_bytes(image_bytes)
    return dest


def write_metadata(
    character_id: str,
    metadata: dict[str, Any],
    *,
    root: Path | None = None,
) -> Path:
    dest = character_dir(character_id, root=root) / METADATA_FILENAME
    dest.write_text(json.dumps(metadata, indent=2) + "\n")
    return dest


def load_metadata(character_id: str, *, root: Path | None = None) -> dict[str, Any] | None:
    path = character_dir(character_id, root=root) / METADATA_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def promote_candidate(
    character_id: str,
    candidate_path: Path,
    metadata: dict[str, Any],
    *,
    selected_index: int,
    root: Path | None = None,
) -> Path:
    if not candidate_path.is_file():
        raise FileNotFoundError(str(candidate_path))
    dest = character_dir(character_id, root=root) / MASTER_FILENAME
    shutil.copy2(candidate_path, dest)
    updated = dict(metadata)
    updated["selected_candidate_index"] = selected_index
    write_metadata(character_id, updated, root=root)
    return dest


def master_path(character_id: str, *, root: Path | None = None) -> Path | None:
    path = character_dir(character_id, root=root) / MASTER_FILENAME
    return path if path.is_file() else None


def load_character(character_id: str, *, root: Path | None = None) -> dict[str, Any] | None:
    if not is_safe_character_id(character_id):
        return None
    directory = assets_root(root) / character_id
    if not directory.is_dir():
        return None
    candidates_dir = directory / "candidates"
    candidates = []
    if candidates_dir.is_dir():
        for path in sorted(candidates_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                candidates.append(path.name)
    return {
        "characterId": character_id,
        "directory": str(directory),
        "masterFilename": MASTER_FILENAME if (directory / MASTER_FILENAME).is_file() else None,
        "candidates": candidates,
        "metadata": load_metadata(character_id, root=root),
    }
