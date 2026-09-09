from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
    ids: list[str] = Field(default_factory=list, description="Catalog ids to fetch. Empty = all catalog bundles.")


class AssetRef(BaseModel):
    asset_id: str


class ImageJobRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    operation: str
    prompt: str = ""
    negative_prompt: str = ""
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    steps: int | None = None
    cfg: float | None = None
    denoise: float | None = None
    fast: bool = False
    style_preset: str | None = Field(None, description="cinematic_naturalism (default), painterly_concept, graphic_still, photoreal_cinematic, documentary, or off")
    sampler_name: str | None = None
    scheduler: str | None = None
    shift: float | None = None
    lora_strength: float | None = None
    control_strength: float | None = None
    scale: int | None = Field(None, description="2 or 4 for upscale_asset")
    left: int | None = None
    right: int | None = None
    top: int | None = None
    bottom: int | None = None
    feathering: int | None = None
    framing: str | None = None
    lens: str | None = None
    camera_height: str | None = None
    image: AssetRef | None = None
    mask: AssetRef | None = None
    character: AssetRef | None = None
    attire: AssetRef | None = None
    location: AssetRef | None = None
    references: list[AssetRef] = Field(default_factory=list)
