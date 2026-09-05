from __future__ import annotations

from typing import Any, Iterator

import httpx

from .. import config

TIMEOUT = httpx.Timeout(300.0, connect=15.0)


class LlamaChatClient:
    def __init__(self, base_url: str | None = None, timeout: httpx.Timeout | None = None) -> None:
        self.base_url = base_url or f"http://{config.LAN_IP}:{config.LLAMA_PORT}/v1"
        self.timeout = timeout or TIMEOUT

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            response = client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("llama-server returned a non-object chat response")
        return data

    def stream(self, payload: dict[str, Any]) -> Iterator[str]:
        body = {**payload, "stream": True}
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            with client.stream("POST", "/chat/completions", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        yield line if line.endswith("\n") else f"{line}\n"
