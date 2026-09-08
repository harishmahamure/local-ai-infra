from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ... import cleanup, config, downloads, runtime, generate as generate_module, ltx_video, upscale as upscale_module
from ...workflows import assets as workflow_assets
from ...workflows import character_master

STATIC_DIR = Path(__file__).resolve().parents[3] / "static"

legacy_router = APIRouter(tags=["Control UI (legacy /api/v1)"])


class ErrorBody(BaseModel):
    code: str
    message: str
    requestId: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class ProfileRequest(BaseModel):
    id: str = Field(..., pattern="^(llama-fast|comfy|comfy-ltx|gemma)$", description="Exclusive GPU profile")


class DownloadRequest(BaseModel):
    ids: list[str] = Field(default_factory=list, description="Catalog bundle ids; empty means all remaining")


class CleanupRequest(BaseModel):
    targets: list[str] = Field(
        default_factory=lambda: ["images", "videos"],
        description="images, videos, and/or comfy_outputs",
    )


class PromptItem(BaseModel):
    prompt: str = ""
    image: str | None = None
    images: list[str] | None = None
    plan: dict[str, Any] | None = None
    flow: str | None = None
    controlType: str | None = None
    controlStrength: float | None = Field(None, ge=0.0, le=2.0)
    layers: int | None = Field(None, ge=1, le=8)
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    fastGenMode: bool | None = None
    steps: int | None = Field(None, description='Omit or null for auto; integer to override')
    upscale: bool | None = Field(None, description='Omit or null for auto; true/false to override')
    denoise: float | None = Field(None, ge=0.0, le=1.0)


class GenerateRequest(BaseModel):
    prompt: str = Field("", description="Natural-language image request (shorthand when prompts is omitted)")
    image: str | None = Field(None, description="Optional base64 or data URL reference image")
    images: list[str] | None = Field(None, description="Extra references for merge (2–3) or extra uploads")
    plan: dict[str, Any] | None = Field(None, description="Explicit plan; skips the Gemma planner")
    flow: str | None = Field(None, pattern="^(t2i|img2img|character|text_edit|merge|semantic_edit|layered|control|lightning)$")
    controlType: str | None = Field(None, pattern="^(pose|depth|canny)$")
    controlStrength: float | None = Field(None, ge=0.0, le=2.0)
    layers: int | None = Field(None, ge=1, le=8)
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    count: int = Field(1, ge=1, le=10, description="Same-prompt batch size (shorthand mode)")
    prompts: list[PromptItem] | None = Field(None, description="Multi-prompt batch (preferred). Takes precedence over prompt.")
    fastGenMode: bool | None = Field(None, description="Omit or null for auto; true forces Lightning 4-step")
    steps: int | None = Field(None, description="Omit or null for auto; integer to override")
    upscale: bool | None = Field(None, description="Omit or null for auto; true/false to override")
    denoise: float | None = Field(None, ge=0.0, le=1.0, description="Img2img/character denoise (0–1); img2img default 0.65, character default 0.40")


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
    audioPrompt: str | None = Field(None, description='Sound design; use "silent, no sound" for silence')
    image: str | None = Field(None, description="Base64 or data URL start frame (required for i2v)")
    endImage: str | None = Field(None, description="Last-frame image for flf2v")
    middleImage: str | None = Field(None, description="Optional mid-clip keyframe")
    audio: str | None = Field(None, description="Base64 or data URL audio for lipsync / a2v")
    sourceJobId: str | None = None
    sourceFilename: str | None = None
    duration: float | None = Field(None, ge=1.0, le=30.0, description="Clip length in seconds (1–30)")
    preset: str | None = Field(None, description="ltx_reel | ltx_landscape | ltx_square | ltx_quality | ltx_studio | ltx_fast")
    orientation: str | None = Field(None, pattern="^(portrait|landscape|square)$")
    speed: str | None = Field(None, pattern="^(fast|standard|quality)$", description="fast=12 steps; standard=20; quality=28+refine")
    plan: dict[str, Any] | None = Field(None, description="Explicit plan; skips preset merge")
    seed: int | None = None
    width: int | None = Field(None, ge=256, le=2048)
    height: int | None = Field(None, ge=256, le=2048)
    length: int | None = Field(None, ge=17, le=897, description="Frame count snapped to 8k+1. 30s @ 24fps = 721")
    fps: float | None = Field(None, ge=16.0, le=30.0)
    steps: int | None = Field(None, ge=1, le=60)
    videoCfg: float | None = Field(None, ge=1.0, le=15.0, description="Distilled LTX uses 1.0")
    audioCfg: float | None = Field(None, ge=1.0, le=15.0, description="Distilled LTX uses 1.0")
    negativePrompt: str | None = None
    samplerName: str | None = None
    maxShift: float | None = None
    baseShift: float | None = None
    terminal: float | None = None
    stretch: bool | None = None
    strength: float | None = Field(None, ge=0.0, le=1.0, description="I2V image conditioning (0–1)")
    refine: bool | None = Field(None, description="2-stage latent upscale refine")
    refineSteps: int | None = Field(None, ge=1, le=40)
    refineDenoise: float | None = Field(None, ge=0.05, le=1.0)
    tiledDecode: bool | None = Field(None, description="Tiled VAE decode (lower VRAM)")
    promptEnhance: bool | None = Field(None, description="Expand prompt with LTX Studio TextGenerateLTX2Prompt")
    cameraMotion: str | None = Field(
        None,
        pattern="^(auto|none|dolly_in|dolly_out|dolly_left|dolly_right|jib_up|jib_down|static)$",
        description="Prompt-only camera language. LTX-2 19B camera LoRAs are not loaded on 2.5.",
    )
    referenceVideo: str | None = Field(None, description="Base64 or data URL reference video for IC-LoRA")
    icLora: str | None = Field(None, pattern="^(auto|none|union)$", description="LTX-2.3 Union Control IC-LoRA; skipped without referenceVideo")
    controlType: str | None = Field(None, pattern="^(auto|depth|canny|pose)$")
    loraStrength: float | None = Field(None, ge=0.0, le=2.0, description="Unused on LTX-2.5 (camera is prompt-only)")
    icLoraStrength: float | None = Field(None, ge=0.0, le=1.0)


def _workflow_capabilities() -> dict[str, Any]:
    return {"characterMaster": character_master.get_capabilities()}


def _error(status: int, code: str, message: str, request: Request) -> JSONResponse:
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "requestId": rid}},
    )


@legacy_router.get("/api/v1/status", summary="GPU runtime status", description="loadState, active profile, VRAM, process list.")
def api_status(request: Request) -> dict[str, Any]:
    return runtime.get_status()


@legacy_router.get("/api/v1/models", summary="Catalog files on disk", description="Each bundle: complete | partial | missing plus per-file state.")
def api_models() -> dict[str, Any]:
    try:
        return runtime.get_models()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@legacy_router.get("/api/v1/capabilities", summary="Control UI capabilities", description="Image presets, LTX video, upscale, character-master readiness.")
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


@legacy_router.post(
    "/api/v1/generate",
    status_code=202,
    summary="Queue image generation",
    description="Multi-prompt batch (`prompts[]`) or shorthand (`prompt` + `count`). Gemma plans once, then Comfy runs items. Partial failures continue.",
)
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
            flow=body.flow,
            extra_images=body.images,
            control_type=body.controlType,
            control_strength=body.controlStrength,
            layers=body.layers,
            denoise=body.denoise,
        )
    except generate_module.GenerateError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "GENERATE_FAILED", str(exc), request)


@legacy_router.get("/api/v1/generate/{job_id}/images/{filename}", summary="Download generate output image")
def api_generate_image(job_id: str, filename: str):
    path = generate_module.get_job_image_path(job_id, filename)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@legacy_router.post("/api/v1/workflows/character-master", status_code=202, summary="Queue Character Master candidates")
def api_character_master_post(body: CharacterMasterRequest, request: Request):
    try:
        return character_master.submit(body.model_dump())
    except character_master.WorkflowError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "CHARACTER_MASTER_FAILED", str(exc), request)


@legacy_router.get("/api/v1/workflows/character-master/{job_id}/images/{filename}", summary="Download Character Master image")
def api_character_master_image(job_id: str, filename: str):
    path = character_master.get_job_image_path(job_id, filename)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@legacy_router.get("/api/v1/workflows/character-master/{job_id}", summary="Poll Character Master job")
def api_character_master_get(job_id: str, request: Request):
    try:
        return character_master.get_job(job_id)
    except character_master.WorkflowError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "CHARACTER_MASTER_GET_FAILED", str(exc), request)


@legacy_router.post("/api/v1/workflows/character-master/{job_id}/select", summary="Promote a Character Master candidate")
def api_character_master_select(job_id: str, body: CharacterMasterSelectRequest, request: Request):
    try:
        return character_master.select_candidate(job_id, body.candidateIndex)
    except character_master.WorkflowError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "CHARACTER_MASTER_SELECT_FAILED", str(exc), request)


@legacy_router.get("/api/v1/assets/characters/{character_id}", summary="Load persisted character asset")
def api_character_asset(character_id: str):
    data = workflow_assets.load_character(character_id)
    if not data:
        raise HTTPException(status_code=404, detail="Character asset not found")
    return data


@legacy_router.get("/api/v1/assets/characters/{character_id}/files/{filename}", summary="Download character file")
def api_character_asset_file(character_id: str, filename: str):
    path = character_master.get_character_file(character_id, filename)
    if not path:
        raise HTTPException(status_code=404, detail="Character file not found")
    return FileResponse(path)


@legacy_router.get("/api/v1/generate/{job_id}", summary="Poll generate job")
def api_generate_get(job_id: str, request: Request):
    try:
        return generate_module.get_job(job_id)
    except generate_module.GenerateError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "GENERATE_GET_FAILED", str(exc), request)


@legacy_router.post("/api/v1/upscale", status_code=202, summary="Queue RealESRGAN upscale")
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


@legacy_router.get("/api/v1/upscale/{job_id}/images/{filename}", summary="Download upscale output image")
def api_upscale_image(job_id: str, filename: str):
    path = upscale_module.get_job_image_path(job_id, filename)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@legacy_router.get("/api/v1/upscale/{job_id}", summary="Poll upscale job")
def api_upscale_get(job_id: str, request: Request):
    try:
        return upscale_module.get_job(job_id)
    except upscale_module.UpscaleError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "UPSCALE_GET_FAILED", str(exc), request)


@legacy_router.post(
    "/api/v1/ltx-video",
    status_code=202,
    summary="Queue LTX-2.5 video",
    description="t2v / i2v / a2v / flf2v / lipsync / motion_transfer on Comfy LTX :8189. Duration 1–30s. Camera is prompt-only. audioPrompt recommended.",
)
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
            lora_strength=body.loraStrength,
            ic_lora_strength=body.icLoraStrength,
        )
    except ltx_video.LtxVideoError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "LTX_VIDEO_FAILED", str(exc), request)


@legacy_router.get("/api/v1/ltx-video/{job_id}/videos/{filename}", summary="Download LTX MP4")
def api_ltx_video_file(job_id: str, filename: str):
    path = ltx_video.get_job_video_path(job_id, filename)
    if not path:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path)


@legacy_router.get("/api/v1/ltx-video/{job_id}", summary="Poll LTX video job")
def api_ltx_video_get(job_id: str, request: Request):
    try:
        return ltx_video.get_job(job_id)
    except ltx_video.LtxVideoError as exc:
        return _error(exc.status, exc.code, exc.message, request)
    except Exception as exc:
        return _error(500, "LTX_VIDEO_GET_FAILED", str(exc), request)


@legacy_router.get("/api/v1/downloads", summary="Catalog download status")
def api_downloads_get(request: Request) -> dict[str, Any]:
    return downloads.get_download_status()


@legacy_router.post("/api/v1/downloads", status_code=202, summary="Start catalog download")
def api_downloads_post(body: DownloadRequest, request: Request):
    try:
        return downloads.start_download(body.ids or None)
    except downloads.ConflictError as exc:
        return _error(409, "DOWNLOAD_RUNNING", str(exc), request)
    except Exception as exc:
        return _error(500, "DOWNLOAD_START_FAILED", str(exc), request)


@legacy_router.get("/api/v1/cleanup", summary="List generated media on GPU")
def api_cleanup_get() -> dict[str, Any]:
    return cleanup.inventory()


@legacy_router.post("/api/v1/cleanup", summary="Delete generated images/videos")
def api_cleanup_post(body: CleanupRequest, request: Request):
    try:
        return cleanup.run(body.targets)
    except cleanup.CleanupError as exc:
        return _error(exc.status, exc.code, str(exc), request)
    except Exception as exc:
        return _error(500, "CLEANUP_FAILED", str(exc), request)


@legacy_router.put("/api/v1/profile", summary="Start exclusive GPU profile")
def api_profile_put(body: ProfileRequest, request: Request):
    try:
        return runtime.start_profile(body.id)
    except ValueError as exc:
        return _error(400, "INVALID_PROFILE", str(exc), request)
    except Exception as exc:
        return _error(500, "PROFILE_START_FAILED", str(exc), request)


@legacy_router.delete("/api/v1/profile", status_code=204, response_class=Response, summary="Stop all GPU profiles")
def api_profile_delete():
    try:
        runtime.stop_profile()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@legacy_router.get("/health", summary="Liveness probe", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@legacy_router.get("/", summary="Control UI", include_in_schema=False)
def ui_index():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "AI control API — see /docs"}


@legacy_router.get("/api/v1/urls", summary="LAN service URLs")
def api_urls() -> dict[str, str]:
    s = runtime.get_status()
    return s["endpoints"]
