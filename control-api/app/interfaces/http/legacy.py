from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ... import config, downloads, runtime, generate as generate_module, ltx_video, upscale as upscale_module
from ...workflows import assets as workflow_assets
from ...workflows import character_master

STATIC_DIR = Path(__file__).resolve().parents[3] / "static"

legacy_router = APIRouter()


class ErrorBody(BaseModel):
    code: str
    message: str
    requestId: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class ProfileRequest(BaseModel):
    id: str = Field(..., pattern="^(llama-fast|comfy|comfy-ltx|gemma)$")


class DownloadRequest(BaseModel):
    ids: list[str] = Field(default_factory=list)


class PromptItem(BaseModel):
    prompt: str = ""
    image: str | None = None
    plan: dict[str, Any] | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    fastGenMode: bool | None = None
    steps: int | None = Field(None, description='Omit or null for auto; integer to override')
    upscale: bool | None = Field(None, description='Omit or null for auto; true/false to override')


class GenerateRequest(BaseModel):
    prompt: str = ""
    image: str | None = None
    plan: dict[str, Any] | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    count: int = Field(1, ge=1, le=10)
    prompts: list[PromptItem] | None = None
    fastGenMode: bool | None = Field(None, description='Omit or null for auto; true forces Lightning 4-step')
    steps: int | None = Field(None, description='Omit or null for auto; integer to override')
    upscale: bool | None = Field(None, description='Omit or null for auto; true/false to override')


class UpscaleRequest(BaseModel):
    image: str | None = None
    sourceJobId: str | None = None
    sourceFilename: str | None = None
    scale: Literal[2, 4] = Field(4, description="RealESRGAN scale factor: 2 or 4")


class CharacterMasterRequest(BaseModel):
    characterDescription: str
    style: str | None = None
    quality: Literal["draft", "master"] = "master"
    aspectRatio: Literal["9:16", "16:9", "1:1", "4:5"] = "9:16"
    width: int | None = Field(None, ge=256, le=4096)
    height: int | None = Field(None, ge=256, le=4096)
    seed: int | None = None
    candidateCount: int | None = Field(None, ge=1, le=8)
    image: str | None = None
    denoise: float | None = Field(None, ge=0.0, le=1.0)
    realismLora: float | bool | None = Field(False, description="Optional realism LoRA strength 0-1; default off")
    upscale: bool = False
    characterId: str | None = None


class CharacterMasterSelectRequest(BaseModel):
    candidateIndex: int = Field(..., ge=0, le=7)


class LtxVideoRequest(BaseModel):
    mode: str = Field("t2v", pattern="^(t2v|i2v|a2v|flf2v|lipsync|motion_transfer)$")
    prompt: str = ""
    audioPrompt: str | None = None
    image: str | None = None
    endImage: str | None = None
    middleImage: str | None = None
    audio: str | None = None
    sourceJobId: str | None = None
    sourceFilename: str | None = None
    duration: float | None = Field(None, ge=1.0, le=10.0)
    preset: str | None = None
    orientation: str | None = Field(None, pattern="^(portrait|landscape|square)$")
    speed: str | None = Field(None, pattern="^(fast|standard|quality)$")
    plan: dict[str, Any] | None = None
    seed: int | None = None
    width: int | None = Field(None, ge=256, le=2048)
    height: int | None = Field(None, ge=256, le=2048)
    length: int | None = Field(None, ge=17, le=241)
    fps: float | None = Field(None, ge=16.0, le=30.0)
    steps: int | None = Field(None, ge=1, le=60)
    videoCfg: float | None = Field(None, ge=1.0, le=15.0)
    audioCfg: float | None = Field(None, ge=1.0, le=15.0)
    negativePrompt: str | None = None
    samplerName: str | None = None
    maxShift: float | None = None
    baseShift: float | None = None
    terminal: float | None = None
    stretch: bool | None = None
    strength: float | None = Field(None, ge=0.0, le=1.0)
    refine: bool | None = None
    refineSteps: int | None = Field(None, ge=1, le=40)
    refineDenoise: float | None = Field(None, ge=0.05, le=1.0)
    tiledDecode: bool | None = None
    promptEnhance: bool | None = None
    cameraMotion: str | None = Field(
        None,
        pattern="^(auto|none|dolly_in|dolly_out|dolly_left|dolly_right|jib_up|jib_down|static)$",
    )
    referenceVideo: str | None = None
    icLora: str | None = Field(None, pattern="^(auto|none|union|detailer)$")
    controlType: str | None = Field(None, pattern="^(auto|depth|canny|pose)$")
    detailer: bool | None = None
    loraStrength: float | None = Field(None, ge=0.0, le=2.0)
    icLoraStrength: float | None = Field(None, ge=0.0, le=1.0)


def _workflow_capabilities() -> dict[str, Any]:
    return {"characterMaster": character_master.get_capabilities()}


def _error(status: int, code: str, message: str, request: Request) -> JSONResponse:
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "requestId": rid}},
    )


@legacy_router.get("/api/v1/status")
def api_status(request: Request) -> dict[str, Any]:
    return runtime.get_status()


@legacy_router.get("/api/v1/models")
def api_models() -> dict[str, Any]:
    try:
        return runtime.get_models()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@legacy_router.get("/api/v1/capabilities")
def api_capabilities() -> dict[str, Any]:
    caps: dict[str, Any] = {}
    errors: list[str] = []

    try:
        caps = generate_module.get_capabilities()
    except Exception as exc:
        errors.append(f"generate: {exc}")

    for key, getter in (
        ("ltxVideo", ltx_video.get_capabilities),
        ("upscale", upscale_module.get_capabilities),
        ("workflows", _workflow_capabilities),
    ):
        try:
            caps[key] = getter()
        except Exception as exc:
            caps[key] = {"ready": False, "error": str(exc)}
            errors.append(f"{key}: {exc}")

    if errors and not caps:
        raise HTTPException(status_code=500, detail="; ".join(errors))
    if errors:
        caps["partialErrors"] = errors
    return caps


@legacy_router.post("/api/v1/generate", status_code=202)
def api_generate_post(body: GenerateRequest, request: Request):
    try:
        prompts_payload = None
        if body.prompts:
            prompts_payload = [p.model_dump() for p in body.prompts]
        return generate_module.submit(
            prompt=body.prompt,
            image=body.image,
            plan=body.plan,
            seed=body.seed,
            width=body.width,
            height=body.height,
            count=body.count,
            prompts=prompts_payload,
            fast_gen_mode=body.fastGenMode,
            steps=body.steps,
            upscale=body.upscale,
        )
    except generate_module.GenerateError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "GENERATE_FAILED", str(exc), request)


@legacy_router.get("/api/v1/generate/{job_id}/images/{filename}")
def api_generate_image(job_id: str, filename: str):
    path = generate_module.get_job_image_path(job_id, filename)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@legacy_router.post("/api/v1/workflows/character-master", status_code=202)
def api_character_master_post(body: CharacterMasterRequest, request: Request):
    try:
        return character_master.submit(body.model_dump())
    except character_master.WorkflowError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "CHARACTER_MASTER_FAILED", str(exc), request)


@legacy_router.get("/api/v1/workflows/character-master/{job_id}/images/{filename}")
def api_character_master_image(job_id: str, filename: str):
    path = character_master.get_job_image_path(job_id, filename)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@legacy_router.get("/api/v1/workflows/character-master/{job_id}")
def api_character_master_get(job_id: str, request: Request):
    try:
        return character_master.get_job(job_id)
    except character_master.WorkflowError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "CHARACTER_MASTER_GET_FAILED", str(exc), request)


@legacy_router.post("/api/v1/workflows/character-master/{job_id}/select")
def api_character_master_select(job_id: str, body: CharacterMasterSelectRequest, request: Request):
    try:
        return character_master.select_candidate(job_id, body.candidateIndex)
    except character_master.WorkflowError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "CHARACTER_MASTER_SELECT_FAILED", str(exc), request)


@legacy_router.get("/api/v1/assets/characters/{character_id}")
def api_character_asset(character_id: str):
    data = workflow_assets.load_character(character_id)
    if not data:
        raise HTTPException(status_code=404, detail="Character asset not found")
    return data


@legacy_router.get("/api/v1/assets/characters/{character_id}/files/{filename}")
def api_character_asset_file(character_id: str, filename: str):
    path = character_master.get_character_file(character_id, filename)
    if not path:
        raise HTTPException(status_code=404, detail="Character file not found")
    return FileResponse(path)


@legacy_router.get("/api/v1/generate/{job_id}")
def api_generate_get(job_id: str, request: Request):
    try:
        return generate_module.get_job(job_id)
    except generate_module.GenerateError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "GENERATE_GET_FAILED", str(exc), request)


@legacy_router.post("/api/v1/upscale", status_code=202)
def api_upscale_post(body: UpscaleRequest, request: Request):
    try:
        return upscale_module.submit(
            image=body.image,
            source_job_id=body.sourceJobId,
            source_filename=body.sourceFilename,
            scale=body.scale,
        )
    except upscale_module.UpscaleError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "UPSCALE_FAILED", str(exc), request)


@legacy_router.get("/api/v1/upscale/{job_id}/images/{filename}")
def api_upscale_image(job_id: str, filename: str):
    path = upscale_module.get_job_image_path(job_id, filename)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@legacy_router.get("/api/v1/upscale/{job_id}")
def api_upscale_get(job_id: str, request: Request):
    try:
        return upscale_module.get_job(job_id)
    except upscale_module.UpscaleError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "UPSCALE_GET_FAILED", str(exc), request)


@legacy_router.post("/api/v1/ltx-video", status_code=202)
def api_ltx_video_post(body: LtxVideoRequest, request: Request):
    try:
        return ltx_video.submit(
            mode=body.mode,
            prompt=body.prompt,
            audio_prompt=body.audioPrompt,
            image=body.image,
            end_image=body.endImage,
            middle_image=body.middleImage,
            audio=body.audio,
            source_job_id=body.sourceJobId,
            source_filename=body.sourceFilename,
            duration=body.duration,
            preset=body.preset,
            orientation=body.orientation,
            speed=body.speed,
            plan=body.plan,
            seed=body.seed,
            width=body.width,
            height=body.height,
            length=body.length,
            fps=body.fps,
            steps=body.steps,
            video_cfg=body.videoCfg,
            audio_cfg=body.audioCfg,
            negative_prompt=body.negativePrompt,
            sampler_name=body.samplerName,
            max_shift=body.maxShift,
            base_shift=body.baseShift,
            terminal=body.terminal,
            stretch=body.stretch,
            strength=body.strength,
            refine=body.refine,
            refine_steps=body.refineSteps,
            refine_denoise=body.refineDenoise,
            tiled_decode=body.tiledDecode,
            prompt_enhance=body.promptEnhance,
            camera_motion=body.cameraMotion,
            reference_video=body.referenceVideo,
            ic_lora=body.icLora,
            control_type=body.controlType,
            detailer=body.detailer,
            lora_strength=body.loraStrength,
            ic_lora_strength=body.icLoraStrength,
        )
    except ltx_video.LtxVideoError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "LTX_VIDEO_FAILED", str(exc), request)


@legacy_router.get("/api/v1/ltx-video/{job_id}/videos/{filename}")
def api_ltx_video_file(job_id: str, filename: str):
    path = ltx_video.get_job_video_path(job_id, filename)
    if not path:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path)


@legacy_router.get("/api/v1/ltx-video/{job_id}")
def api_ltx_video_get(job_id: str, request: Request):
    try:
        return ltx_video.get_job(job_id)
    except ltx_video.LtxVideoError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "LTX_VIDEO_GET_FAILED", str(exc), request)


@legacy_router.get("/api/v1/downloads")
def api_downloads_get(request: Request) -> dict[str, Any]:
    return downloads.get_download_status()


@legacy_router.post("/api/v1/downloads", status_code=202)
def api_downloads_post(body: DownloadRequest, request: Request):
    try:
        return downloads.start_download(body.ids or None)
    except downloads.ConflictError as exc:
        return _error(409, "DOWNLOAD_RUNNING", str(exc), request)
    except Exception as exc:
        return _error(500, "DOWNLOAD_START_FAILED", str(exc), request)


@legacy_router.put("/api/v1/profile")
def api_profile_put(body: ProfileRequest, request: Request):
    try:
        return runtime.start_profile(body.id)
    except ValueError as exc:
        return _error(400, "INVALID_PROFILE", str(exc), request)
    except Exception as exc:
        return _error(500, "PROFILE_START_FAILED", str(exc), request)


@legacy_router.delete("/api/v1/profile", status_code=204, response_class=Response)
def api_profile_delete():
    try:
        runtime.stop_profile()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@legacy_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@legacy_router.get("/")
def ui_index():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "AI control API — see /docs"}


@legacy_router.get("/api/v1/urls")
def api_urls() -> dict[str, str]:
    s = runtime.get_status()
    return s["endpoints"]
