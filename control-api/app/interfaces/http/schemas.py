from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SubmitJobRequest(BaseModel):
    operation: str = Field(..., description="Catalog operation id, e.g. image.generate, video.generate, video.lipsync")
    preset: str = Field("master", description="draft | balanced | master | character_master | …")
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Operation inputs. Use {asset_id} objects for files, not filesystem paths.",
    )
    parameters: dict[str, Any] = Field(default_factory=dict, description="seed, steps, cfg, loras, duration_seconds, …")
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
    model: str = Field(..., description="gemma-4-e4b or qwen36-35b-a3b-rq")
    messages: list[TextChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = Field(False, description="If true, response is text/event-stream")
