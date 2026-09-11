from __future__ import annotations

from app.application.image_ops import plan_operation, step_to_plan
from app.application.video_ops import plan_video_operation
from app.domain.errors import DomainError
from app.domain.operations import (
    DEVOTION_WALLPAPER_NEGATIVE,
    DEVOTION_WALLPAPER_SUFFIX,
    LIVE_WALLPAPER_NEGATIVE,
    LIVE_WALLPAPER_SUFFIX,
    LTX_VIDEO_NEGATIVE,
    LTX_VIDEO_SUFFIX,
)
from app.infrastructure.comfy.ltx_graph import MODELS as LTX_MODELS
from app.infrastructure.comfy.ltx_graph import build_i2v, build_t2v
from app.infrastructure.comfy.graphs import MODELS
from app.infrastructure.gpu_scheduler import GpuScheduler


def _class_types(graph: dict) -> set[str]:
    return {node["class_type"] for node in graph.values()}


_LATENT_INPUTS = {
    "latent",
    "latent_image",
    "video_latent",
    "audio_latent",
    "av_latent",
    "samples",
}
_CONDITIONING_SLOTS = {
    "CLIPTextEncode": {0},
    "LTXVConditioning": {0, 1},
    "LTXVImgToVideo": {0, 1},
}


def _assert_latent_inputs_are_latents(graph: dict) -> None:
    for node in graph.values():
        for name, ref in node["inputs"].items():
            if name not in _LATENT_INPUTS or not isinstance(ref, list) or len(ref) != 2:
                continue
            src, slot = str(ref[0]), int(ref[1])
            src_type = graph[src]["class_type"]
            assert slot not in _CONDITIONING_SLOTS.get(src_type, set()), (
                f"{node['class_type']}.{name} linked to {src_type}[{slot}] (CONDITIONING)"
            )


def test_devotion_wallpaper_desktop_default_upscale() -> None:
    steps, _, bundles = plan_operation("generate_devotion_wallpaper", {"prompt": "temple at dawn"})
    assert len(steps) == 2
    assert steps[0].mode == "txt2img"
    assert steps[0].width == 1664
    assert steps[0].height == 928
    assert steps[0].prompt.startswith("temple at dawn")
    assert DEVOTION_WALLPAPER_SUFFIX in steps[0].prompt
    assert DEVOTION_WALLPAPER_NEGATIVE in steps[0].negative_prompt
    assert steps[0].profile == "comfyui"
    assert steps[1].mode == "upscale"
    assert steps[1].upscale_scale == 2
    assert steps[1].image_keys == ["__previous__"]
    assert bundles == ["qwen-image-2512-fp8", "upscalers-esrgan"]


def test_devotion_wallpaper_mobile_and_skip_upscale() -> None:
    steps, _, bundles = plan_operation(
        "generate_devotion_wallpaper",
        {"prompt": "lotus lamp", "target": "mobile", "upscale": False},
    )
    assert len(steps) == 1
    assert steps[0].width == 928
    assert steps[0].height == 1664
    assert bundles == ["qwen-image-2512-fp8"]


def test_devotion_wallpaper_explicit_size_wins() -> None:
    steps, _, _ = plan_operation(
        "generate_devotion_wallpaper",
        {"prompt": "river shrine", "target": "mobile", "width": 1104, "height": 1472, "upscale": False},
    )
    assert steps[0].width == 1104
    assert steps[0].height == 1472


def test_devotion_wallpaper_user_negative_and_fast() -> None:
    custom, _, _ = plan_operation(
        "generate_devotion_wallpaper",
        {"prompt": "temple", "negative_prompt": "neon signs", "upscale": False},
    )
    assert custom[0].negative_prompt.startswith("neon signs")
    assert DEVOTION_WALLPAPER_NEGATIVE not in custom[0].negative_prompt
    fast, _, _ = plan_operation(
        "generate_devotion_wallpaper",
        {"prompt": "temple", "fast": True, "upscale": False},
    )
    assert fast[0].steps == 4
    assert fast[0].cfg == 1.0
    assert fast[0].loras == [{"name": MODELS["qwen_lora_lightning"], "strength": 1.0}]
    assert fast[0].negative_prompt == ""


def test_devotion_rejects_bad_target() -> None:
    try:
        plan_operation("generate_devotion_wallpaper", {"prompt": "x", "target": "watch"})
        raise AssertionError("expected invalid target")
    except DomainError as exc:
        assert exc.code.value == "INVALID_PARAMETER"


def test_live_wallpaper_requires_image() -> None:
    try:
        plan_video_operation("generate_live_wallpaper", {"prompt": "soft incense smoke"})
        raise AssertionError("expected image required")
    except DomainError as exc:
        assert exc.code.value == "INVALID_REQUEST"


def test_live_wallpaper_mobile_quality_defaults() -> None:
    steps, refs, bundles = plan_video_operation(
        "generate_live_wallpaper",
        {"prompt": "lamp flame", "image": {"asset_id": "ast_still"}},
    )
    assert refs == {"image": "ast_still"}
    assert len(steps) == 1
    step = steps[0]
    assert step.mode == "i2v"
    assert step.profile == "comfy-ltx"
    assert step.width == 704
    assert step.height == 1216
    assert step.fps == 24
    assert step.duration == 4.0
    assert step.length == 97
    assert step.refine is True
    assert step.steps == 8
    assert LIVE_WALLPAPER_SUFFIX in step.prompt
    assert step.negative_prompt == LIVE_WALLPAPER_NEGATIVE
    assert bundles == ["ltx-2.5-distilled", "ltx-2.5-studio"]
    plan = step_to_plan(step, uploaded={"image": "in.png"}, previous_name=None)
    assert plan["mode"] == "i2v"
    assert plan["refine"] is True
    graph = build_i2v(plan)
    types = _class_types(graph)
    assert {
        "UNETLoader",
        "LTXVPreprocess",
        "LTXVImgToVideoInplace",
        "LTXVConcatAVLatent",
        "LTXVDualCFGGuider",
        "SamplerCustomAdvanced",
        "CreateVideo",
        "SaveVideo",
    } <= types
    unets = [node for node in graph.values() if node["class_type"] == "UNETLoader"]
    assert unets[0]["inputs"]["unet_name"] == LTX_MODELS["distilled"]
    assert graph[next(k for k, n in graph.items() if n["class_type"] == "CLIPLoader")]["inputs"]["clip_name"] == LTX_MODELS["clip"]
    vaes = [node for node in graph.values() if node["class_type"] == "VAELoader"]
    assert {node["inputs"]["vae_name"] for node in vaes} == {LTX_MODELS["vae"], LTX_MODELS["audio_vae"]}
    assert "LTXVAudioVAELoader" not in types
    assert len([node for node in graph.values() if node["class_type"] == "SamplerCustomAdvanced"]) == 2
    _assert_latent_inputs_are_latents(graph)


def test_live_wallpaper_video_target_no_refine() -> None:
    steps, _, bundles = plan_video_operation(
        "generate_live_wallpaper",
        {
            "prompt": "prayer flags",
            "image": {"asset_id": "ast_still"},
            "target": "video",
            "refine": False,
        },
    )
    assert steps[0].width == 1216
    assert steps[0].height == 704
    assert steps[0].refine is False
    assert bundles == ["ltx-2.5-distilled"]
    plan = step_to_plan(steps[0], uploaded={"image": "in.png"}, previous_name=None)
    graph = build_i2v(plan)
    unets = [node for node in graph.values() if node["class_type"] == "UNETLoader"]
    assert unets[0]["inputs"]["unet_name"] == LTX_MODELS["distilled"]
    vaes = [node for node in graph.values() if node["class_type"] == "VAELoader"]
    assert {node["inputs"]["vae_name"] for node in vaes} == {LTX_MODELS["vae"], LTX_MODELS["audio_vae"]}
    assert "LTXVAudioVAELoader" not in _class_types(graph)
    assert len([node for node in graph.values() if node["class_type"] == "SamplerCustomAdvanced"]) == 1
    _assert_latent_inputs_are_latents(graph)


def test_ltx_video_t2v_no_image() -> None:
    try:
        plan_video_operation("generate_live_wallpaper", {"prompt": "soft incense smoke"})
        raise AssertionError("expected image required")
    except DomainError as exc:
        assert exc.code.value == "INVALID_REQUEST"
    steps, refs, bundles = plan_video_operation(
        "generate_video",
        {"prompt": "oil lamp flame in a dark shrine"},
    )
    assert refs == {}
    assert steps[0].mode == "t2v"
    assert steps[0].profile == "comfy-ltx"
    assert steps[0].width == 1216
    assert steps[0].height == 704
    assert steps[0].image_keys == []
    assert steps[0].refine is True
    assert LTX_VIDEO_SUFFIX in steps[0].prompt
    assert steps[0].negative_prompt == LTX_VIDEO_NEGATIVE
    assert bundles == ["ltx-2.5-distilled", "ltx-2.5-studio"]
    plan = step_to_plan(steps[0], uploaded={}, previous_name=None)
    graph = build_t2v(plan)
    types = _class_types(graph)
    assert "LoadImage" not in types
    assert "LTXVImgToVideoInplace" not in types
    assert {"EmptyLTXVLatentVideo", "LTXVConcatAVLatent", "SamplerCustomAdvanced", "SaveVideo"} <= types
    vaes = [node for node in graph.values() if node["class_type"] == "VAELoader"]
    assert {node["inputs"]["vae_name"] for node in vaes} == {LTX_MODELS["vae"], LTX_MODELS["audio_vae"]}
    assert len([node for node in graph.values() if node["class_type"] == "SamplerCustomAdvanced"]) == 2
    _assert_latent_inputs_are_latents(graph)


def test_ltx_video_mobile_no_refine() -> None:
    steps, _, bundles = plan_video_operation(
        "generate_video",
        {"prompt": "temple bells", "target": "mobile", "refine": False},
    )
    assert steps[0].width == 704
    assert steps[0].height == 1216
    assert steps[0].refine is False
    assert bundles == ["ltx-2.5-distilled"]
    graph = build_t2v(step_to_plan(steps[0], uploaded={}, previous_name=None))
    assert "LoadImage" not in _class_types(graph)
    assert len([node for node in graph.values() if node["class_type"] == "SamplerCustomAdvanced"]) == 1
    _assert_latent_inputs_are_latents(graph)


def test_scheduler_switches_profile() -> None:
    started: list[str] = []

    class FakeRuntime:
        def __init__(self) -> None:
            self.profile = "none"
            self.load = "STOPPED"

        def get_status(self):
            return {"profile": self.profile, "loadState": self.load}

        def start_profile(self, profile):
            started.append(profile)
            self.profile = profile
            self.load = "LOADED"
            return {"profile": profile}

        def stop_profile(self):
            self.profile = "none"
            self.load = "STOPPED"

        def wait_for_profile(self, profile, timeout_sec=180):
            self.profile = profile
            self.load = "LOADED"
            return self.get_status()

    runtime = FakeRuntime()
    scheduler = GpuScheduler(runtime)
    still = scheduler.acquire("job_still", profile="comfyui")
    assert still.profile == "comfyui"
    scheduler.release(still)
    live = scheduler.acquire("job_live", profile="comfy-ltx")
    assert live.profile == "comfy-ltx"
    scheduler.release(live)
    assert started == ["comfyui", "comfy-ltx"]


def test_job_service_lists_wallpaper_ops(tmp_path) -> None:
    from app.application.jobs import JobService
    from app.infrastructure.job_store import JobStore

    class Dummy:
        def start(self):
            return None

        def stop(self):
            return None

        def interrupt_active(self):
            return None

    class Sched:
        def snapshot(self):
            return {}

        def is_busy(self):
            return False

        def active_job(self):
            return None

    svc = JobService(store=JobStore(tmp_path / "jobs.db"), assets=None, scheduler=Sched(), worker=Dummy())
    images = svc.list_operations()
    assert any(op["id"] == "generate_devotion_wallpaper" for op in images["operations"])
    videos = svc.list_video_operations()
    live = next(op for op in videos["operations"] if op["id"] == "generate_live_wallpaper")
    t2v = next(op for op in videos["operations"] if op["id"] == "generate_video")
    assert live["profile"] == "comfy-ltx"
    assert t2v["requiresImage"] is False
    assert videos["capabilities"]["targets"] == ["mobile", "video"]
    created = svc.submit(
        {"operation": "generate_live_wallpaper", "prompt": "flame", "image": {"asset_id": "ast_still"}, "refine": False}
    )
    assert created["operation"] == "generate_live_wallpaper"
    t2v_job = svc.submit({"operation": "generate_video", "prompt": "temple courtyard at dusk", "refine": False})
    assert t2v_job["operation"] == "generate_video"

