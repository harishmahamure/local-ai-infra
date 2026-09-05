from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Callable

from ..domain.errors import DomainError, ErrorCode
from ..domain.models import ModelRecord

TEXT_CHAT_MODELS = {"gemma-4-e4b", "qwen36-35b-a3b-rq"}


def llama_model_name(model: ModelRecord) -> str:
    for item in model.files:
        name = Path(item.local).name
        if "mmproj" not in name.lower():
            return Path(name).stem
    return model.id


def is_filesystem_url(url: str) -> bool:
    value = url.strip()
    if value.startswith(("data:", "http://", "https://")):
        return False
    return True


def build_llama_messages(
    messages: list[Any],
    *,
    resolve_asset: Callable[[str], tuple[bytes, str]],
) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise DomainError(ErrorCode.INVALID_REQUEST, "messages is required")
    out: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise DomainError(ErrorCode.INVALID_REQUEST, "Each message must be an object")
        role = str(item.get("role") or "").strip()
        if role not in {"system", "user", "assistant"}:
            raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unsupported role: {role}")
        content = _content_parts(item.get("content"), resolve_asset=resolve_asset)
        out.append({"role": role, "content": content})
    return out


def _content_parts(content: Any, *, resolve_asset: Callable[[str], tuple[bytes, str]]) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise DomainError(ErrorCode.INVALID_REQUEST, "message.content must be a string or array")
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise DomainError(ErrorCode.INVALID_REQUEST, "content part must be an object")
        kind = str(part.get("type") or "")
        if kind == "text":
            parts.append({"type": "text", "text": str(part.get("text") or "")})
            continue
        if kind == "image_asset":
            asset_id = str(part.get("asset_id") or "").strip()
            if not asset_id:
                raise DomainError(ErrorCode.INVALID_REQUEST, "image_asset.asset_id is required")
            data, mime = resolve_asset(asset_id)
            encoded = base64.b64encode(data).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            continue
        if kind == "image_url":
            raw = part.get("image_url")
            url = raw.get("url") if isinstance(raw, dict) else None
            if not url:
                raise DomainError(ErrorCode.INVALID_REQUEST, "image_url.url is required")
            if is_filesystem_url(str(url)):
                raise DomainError(ErrorCode.INVALID_REQUEST, "Filesystem paths are not allowed for images")
            parts.append({"type": "image_url", "image_url": {"url": str(url)}})
            continue
        raise DomainError(ErrorCode.INVALID_PARAMETER, f"Unsupported content type: {kind}")
    return parts
