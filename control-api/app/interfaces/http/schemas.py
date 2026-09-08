from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TextChatMessage(BaseModel):
    role: str
    content: Any


class TextChatRequest(BaseModel):
    model: str = Field(..., description="gemma-4-e4b or qwen36-35b-a3b-rq")
    messages: list[TextChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = Field(False, description="If true, response is text/event-stream")


class DownloadRequest(BaseModel):
    ids: list[str] = Field(default_factory=list, description="Catalog ids to fetch. Empty = all llamacpp bundles.")
