"""Thin httpx client for ComfyUI REST API."""

from __future__ import annotations

import io
import uuid
from typing import Any
from urllib.parse import urlencode

import httpx

from .. import config


class ComfyClient:
    def __init__(self, port: int | None = None, *, lan_ip: str | None = None) -> None:
        self.lan_ip = lan_ip or config.LAN_IP
        self.port = int(port if port is not None else config.COMFY_PORT)
        self.base = f"http://{self.lan_ip}:{self.port}"
        self.timeout = httpx.Timeout(600.0, connect=15.0)
        self._object_info_cache: dict[str, Any] | None = None

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base, timeout=self.timeout)

    def get_object_info(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._object_info_cache is not None and not refresh:
            return self._object_info_cache
        with self._client() as client:
            response = client.get("/object_info")
            response.raise_for_status()
            self._object_info_cache = response.json()
        return self._object_info_cache

    def has_node(self, class_type: str) -> bool:
        try:
            return class_type in self.get_object_info()
        except httpx.HTTPError:
            return False

    def missing_nodes(self, names: list[str]) -> list[str]:
        return [name for name in names if not self.has_node(name)]

    def is_ready(self) -> bool:
        try:
            with self._client() as client:
                response = client.get("/system_stats")
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    def queue_prompt(self, prompt: dict[str, Any], client_id: str | None = None) -> dict[str, Any]:
        cid = client_id or str(uuid.uuid4())
        payload = {"prompt": prompt, "client_id": cid}
        with self._client() as client:
            response = client.post("/prompt", json=payload)
            if response.status_code >= 400:
                detail = response.text.strip()
                try:
                    body = response.json()
                    node_errors = body.get("node_errors") or {}
                    if node_errors:
                        detail = f"{body.get('error') or body} node_errors={node_errors}"
                    else:
                        detail = str(body.get("error") or body.get("message") or body)
                except Exception:
                    pass
                raise httpx.HTTPStatusError(
                    f"ComfyUI rejected workflow ({response.status_code}): {detail}",
                    request=response.request,
                    response=response,
                )
            data = response.json()
            data["client_id"] = cid
            return data

    def get_history(self, prompt_id: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.get(f"/history/{prompt_id}")
            response.raise_for_status()
            return response.json()

    def fetch_media(self, filename: str, subfolder: str = "", folder_type: str = "output") -> tuple[bytes, str]:
        params = urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        with self._client() as client:
            response = client.get(f"/view?{params}")
            response.raise_for_status()
            raw = response.content
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
        mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "gif": "image/gif",
        }.get(ext, "application/octet-stream")
        return raw, mime

    def upload_image(self, image_bytes: bytes, filename: str) -> dict[str, Any]:
        with self._client() as client:
            files = {"image": (filename, io.BytesIO(image_bytes), "application/octet-stream")}
            data = {"overwrite": "true"}
            response = client.post("/upload/image", files=files, data=data)
            response.raise_for_status()
            return response.json()

    def interrupt(self) -> None:
        with self._client() as client:
            response = client.post("/interrupt")
            response.raise_for_status()
