from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SubmitJobRequest(BaseModel):
    operation: str
    preset: str = "master"
    inputs: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    client_context: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] | None = None


class RetryJobRequest(BaseModel):
    strategy: Literal["same", "new_seed", "override", "rerun_stage"] = "same"
    parameters: dict[str, Any] | None = None
    stage: str | None = None


class JobAccepted(BaseModel):
    job_id: str
    status: str
    operation: str
    preset: str


class TextChatMessage(BaseModel):
    role: str
    content: Any


class TextChatRequest(BaseModel):
    model: str
    messages: list[TextChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
