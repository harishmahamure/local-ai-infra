from __future__ import annotations

import struct
from typing import Any


class BasicMediaInspector:
    def inspect_image(self, data: bytes, mime_type: str) -> dict[str, Any]:
        width = height = None
        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
            width, height = struct.unpack(">II", data[16:24])
        elif data[:2] == b"\xff\xd8":
            width, height = _jpeg_size(data)
        return {"width": width, "height": height, "mime_type": mime_type, "size_bytes": len(data)}


def _jpeg_size(data: bytes) -> tuple[int | None, int | None]:
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            return None, None
        marker = data[i + 1]
        if marker in {0xC0, 0xC1, 0xC2}:
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return width, height
        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
        i += 2 + length
    return None, None
