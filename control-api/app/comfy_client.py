"""Thin httpx client for ComfyUI REST API."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import select
import socket
import uuid
from typing import Any
from urllib.parse import urlencode

import httpx

from . import config


class ComfyClient:
    """ComfyUI REST client bound to a host:port."""

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
        with self._client() as c:
            r = c.get("/object_info")
            r.raise_for_status()
            self._object_info_cache = r.json()
        return self._object_info_cache

    def has_node(self, class_type: str) -> bool:
        try:
            return class_type in self.get_object_info()
        except httpx.HTTPError:
            return False

    def is_ready(self) -> bool:
        try:
            with self._client() as c:
                r = c.get("/system_stats")
                return r.status_code == 200
        except httpx.HTTPError:
            return False

    def queue_prompt(self, prompt: dict[str, Any], client_id: str | None = None) -> dict[str, Any]:
        cid = client_id or str(uuid.uuid4())
        payload = {"prompt": prompt, "client_id": cid}
        with self._client() as c:
            r = c.post("/prompt", json=payload)
            if r.status_code >= 400:
                detail = r.text.strip()
                try:
                    body = r.json()
                    node_errors = body.get("node_errors") or {}
                    if node_errors:
                        detail = f"{body.get('error') or body} node_errors={node_errors}"
                    else:
                        detail = str(body.get("error") or body.get("message") or body)
                except Exception:
                    pass
                raise httpx.HTTPStatusError(
                    f"ComfyUI rejected workflow ({r.status_code}): {detail}",
                    request=r.request,
                    response=r,
                )
            data = r.json()
            data["client_id"] = cid
            return data

    def get_history(self, prompt_id: str) -> dict[str, Any]:
        with self._client() as c:
            r = c.get(f"/history/{prompt_id}")
            r.raise_for_status()
            return r.json()

    def view_url(self, filename: str, subfolder: str = "", folder_type: str = "output") -> str:
        params = urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        return f"{self.base}/view?{params}"

    def fetch_media(self, filename: str, subfolder: str = "", folder_type: str = "output") -> tuple[bytes, str]:
        params = urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        with self._client() as c:
            r = c.get(f"/view?{params}")
            r.raise_for_status()
            raw = r.content
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
        mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "gif": "image/gif",
            "mp4": "video/mp4",
            "webm": "video/webm",
            "mov": "video/quicktime",
        }.get(ext, "application/octet-stream")
        return raw, mime

    def fetch_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> tuple[bytes, str]:
        return self.fetch_media(filename, subfolder=subfolder, folder_type=folder_type)

    def upload_image(self, image_bytes: bytes, filename: str) -> dict[str, Any]:
        with self._client() as c:
            files = {"image": (filename, io.BytesIO(image_bytes), "application/octet-stream")}
            data = {"overwrite": "true"}
            r = c.post("/upload/image", files=files, data=data)
            r.raise_for_status()
            return r.json()

    def upload_video(self, video_bytes: bytes, filename: str) -> dict[str, Any]:
        with self._client() as c:
            files = {"image": (filename, io.BytesIO(video_bytes), "video/mp4")}
            data = {"overwrite": "true"}
            r = c.post("/upload/image", files=files, data=data)
            r.raise_for_status()
            return r.json()

    def interrupt(self) -> None:
        with self._client() as c:
            r = c.post("/interrupt")
            r.raise_for_status()

    def open_progress_ws(self, client_id: str) -> "ComfyProgressWS | None":
        try:
            ws = ComfyProgressWS(self.lan_ip, self.port, client_id)
            ws.connect()
            return ws
        except OSError:
            return None


class ComfyProgressWS:
    """Minimal RFC 6455 client for ComfyUI `/ws` progress messages (text JSON only)."""

    def __init__(self, host: str, port: int, client_id: str) -> None:
        self.host = host
        self.port = int(port)
        self.client_id = client_id
        self.sock: socket.socket | None = None
        self._buf = b""
        self._text_parts: list[str] = []

    def connect(self, timeout: float = 10.0) -> None:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        sock = socket.create_connection((self.host, self.port), timeout=timeout)
        path = f"/ws?clientId={self.client_id}"
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        sock.sendall(req.encode("ascii"))
        sock.settimeout(timeout)
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = sock.recv(4096)
            if not chunk:
                sock.close()
                raise OSError("ComfyUI websocket handshake closed")
            raw += chunk
        header, _, rest = raw.partition(b"\r\n\r\n")
        status = header.split(b"\r\n", 1)[0]
        if b"101" not in status:
            sock.close()
            raise OSError(f"ComfyUI websocket handshake failed: {status.decode('ascii', 'ignore')}")
        expected = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
        if expected not in header:
            sock.close()
            raise OSError("ComfyUI websocket accept mismatch")
        sock.settimeout(None)
        sock.setblocking(False)
        self.sock = sock
        self._buf = rest

    def poll_messages(self, timeout: float = 1.0) -> list[dict[str, Any]]:
        if not self.sock:
            return []
        ready, _, _ = select.select([self.sock], [], [], timeout)
        if ready:
            try:
                chunk = self.sock.recv(65536)
            except BlockingIOError:
                chunk = b""
            if not chunk:
                self.close()
                return []
            self._buf += chunk
        messages: list[dict[str, Any]] = []
        while True:
            frame = self._take_frame()
            if frame is None:
                break
            opcode, payload, fin = frame
            if opcode == 0x8:
                self.close()
                break
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode == 0x2:
                continue
            if opcode in (0x0, 0x1):
                try:
                    text = payload.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if opcode == 0x1:
                    self._text_parts = [text]
                else:
                    self._text_parts.append(text)
                if not fin:
                    continue
                joined = "".join(self._text_parts)
                self._text_parts = []
                try:
                    parsed = json.loads(joined)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    messages.append(parsed)
        return messages

    def close(self) -> None:
        sock = self.sock
        self.sock = None
        if not sock:
            return
        try:
            self._send_frame(0x8, b"")
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def _take_frame(self) -> tuple[int, bytes, bool] | None:
        buf = self._buf
        if len(buf) < 2:
            return None
        b0, b1 = buf[0], buf[1]
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F
        offset = 2
        if length == 126:
            if len(buf) < 4:
                return None
            length = int.from_bytes(buf[2:4], "big")
            offset = 4
        elif length == 127:
            if len(buf) < 10:
                return None
            length = int.from_bytes(buf[2:10], "big")
            offset = 10
        if masked:
            if len(buf) < offset + 4:
                return None
            mask = buf[offset : offset + 4]
            offset += 4
        else:
            mask = None
        if len(buf) < offset + length:
            return None
        payload = buf[offset : offset + length]
        self._buf = buf[offset + length :]
        if mask:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return opcode, payload, fin

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        if not self.sock:
            return
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        header = bytearray()
        header.append(0x80 | opcode)
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header.extend(n.to_bytes(2, "big"))
        else:
            header.append(0x80 | 127)
            header.extend(n.to_bytes(8, "big"))
        header.extend(mask)
        self.sock.setblocking(True)
        try:
            self.sock.sendall(header + masked)
        finally:
            self.sock.setblocking(False)


# Primary Qwen-Image ComfyUI (:8188) — module-level helpers delegate here for backward compatibility.
default = ComfyClient(config.COMFY_PORT)
ltx = ComfyClient(config.COMFY_LTX_PORT)

BASE = default.base


def get_object_info(*, refresh: bool = False) -> dict[str, Any]:
    return default.get_object_info(refresh=refresh)


def has_node(class_type: str) -> bool:
    return default.has_node(class_type)


def is_ready() -> bool:
    return default.is_ready()


def ltx_is_ready() -> bool:
    return ltx.is_ready()


def ltx_has_node(class_type: str) -> bool:
    return ltx.has_node(class_type)


def queue_prompt(prompt: dict[str, Any], client_id: str | None = None) -> dict[str, Any]:
    return default.queue_prompt(prompt, client_id=client_id)


def get_history(prompt_id: str) -> dict[str, Any]:
    return default.get_history(prompt_id)


def view_url(filename: str, subfolder: str = "", folder_type: str = "output") -> str:
    return default.view_url(filename, subfolder=subfolder, folder_type=folder_type)


def fetch_image(filename: str, subfolder: str = "", folder_type: str = "output") -> tuple[bytes, str]:
    return default.fetch_image(filename, subfolder=subfolder, folder_type=folder_type)


def fetch_media(filename: str, subfolder: str = "", folder_type: str = "output") -> tuple[bytes, str]:
    return default.fetch_media(filename, subfolder=subfolder, folder_type=folder_type)


def upload_image(image_bytes: bytes, filename: str) -> dict[str, Any]:
    return default.upload_image(image_bytes, filename)


def interrupt() -> None:
    default.interrupt()


def ltx_interrupt() -> None:
    ltx.interrupt()


def decode_image_field(image_field: str) -> tuple[bytes, str]:
    """Accept raw base64 or data URL; return (bytes, suggested filename)."""
    if image_field.startswith("data:"):
        header, _, b64 = image_field.partition(",")
        ext = "png"
        if "jpeg" in header or "jpg" in header:
            ext = "jpg"
        elif "webp" in header:
            ext = "webp"
        return base64.b64decode(b64), f"upload.{ext}"
    try:
        raw = base64.b64decode(image_field, validate=True)
        return raw, "upload.png"
    except Exception:
        return b"", image_field


def decode_video_field(video_field: str) -> tuple[bytes, str]:
    """Accept raw base64 or data URL; return (bytes, suggested filename)."""
    if video_field.startswith("data:"):
        header, _, b64 = video_field.partition(",")
        ext = "mp4"
        if "webm" in header:
            ext = "webm"
        elif "quicktime" in header or "mov" in header:
            ext = "mov"
        return base64.b64decode(b64), f"upload.{ext}"
    try:
        raw = base64.b64decode(video_field, validate=True)
        return raw, "upload.mp4"
    except Exception:
        return b"", video_field


def decode_audio_field(audio_field: str) -> tuple[bytes, str]:
    """Accept raw base64 or data URL; return (bytes, suggested filename)."""
    if audio_field.startswith("data:"):
        header, _, b64 = audio_field.partition(",")
        ext = "wav"
        if "mpeg" in header or "mp3" in header:
            ext = "mp3"
        elif "flac" in header:
            ext = "flac"
        elif "mp4" in header or "m4a" in header or "aac" in header:
            ext = "m4a"
        return base64.b64decode(b64), f"upload.{ext}"
    try:
        raw = base64.b64decode(audio_field, validate=True)
        return raw, "upload.wav"
    except Exception:
        return b"", audio_field
