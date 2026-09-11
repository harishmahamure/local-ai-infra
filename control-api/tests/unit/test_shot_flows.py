from __future__ import annotations

from app.application.image_ops import plan_job, step_to_plan
from app.application.shot_ops import align_length, plan_shot_flow, snap_frame_idx, validate_guides
from app.domain.errors import DomainError
from app.domain.quality import UNET_INT8, UNET_NVFP4, estimate_budget, select_unet
from app.domain.shots import FLOWS, Guide
from app.infrastructure.comfy.ltx_shot_graph import build as build_shot
from app.infrastructure.comfy.post_graph import build as build_post
from app.infrastructure.gpu_scheduler import GpuScheduler


def _class_types(graph: dict) -> set[str]:
    return {node["class_type"] for node in graph.values()}


def test_select_unet_lora_stays_on_int8() -> None:
    assert select_unet(use_lora=True, prefer_nvfp4=True) == UNET_INT8
    assert select_unet(use_lora=False, prefer_nvfp4=True) == UNET_NVFP4
    assert select_unet(use_lora=False, prefer_nvfp4=False) == UNET_INT8


def test_token_budget_windows_instead_of_rejecting() -> None:
    small = estimate_budget(704, 1216, 97, 1)
    assert small.windowed is False
    huge = estimate_budget(1920, 1088, 897, 12, video_guides=3, envelope=1000)
    assert huge.windowed is True
    assert huge.warning


def test_frame_index_snap_and_validate() -> None:
    assert snap_frame_idx(-1, 97) == 96
    assert snap_frame_idx(9, 97, video_guide=True) == 8
    guides = validate_guides([Guide("image", 0), Guide("end_image", -1)], 97)
    assert [item.frame_idx for item in guides] == [0, 96]
    try:
        validate_guides([Guide("a", 8), Guide("b", 8)], 97)
        raise AssertionError("expected duplicate frame error")
    except DomainError as exc:
        assert "duplicate" in exc.message
    try:
        validate_guides([Guide("a", 24), Guide("b", 8)], 97)
        raise AssertionError("expected order error")
    except DomainError as exc:
        assert "non-decreasing" in exc.message


def test_length_aligns_to_8n1() -> None:
    assert align_length(4.0, 24) == 97


def _ref(aid: str) -> dict:
    return {"asset_id": aid}


def test_f01_single_image_plans_shot() -> None:
    steps, refs, bundles = plan_shot_flow(
        "shot_single_image",
        {"prompt": "a lantern sways", "image": _ref("ast_start"), "prefer_nvfp4": False},
    )
    assert refs["image"] == "ast_start"
    assert len(steps) == 1
    assert steps[0].mode == "shot"
    assert steps[0].profile == "comfy-ltx"
    assert steps[0].guides[0]["frame_idx"] == 0
    assert steps[0].unet_name == UNET_INT8
    assert "ltx-2.5-distilled" in bundles


def test_f02_start_end_guides() -> None:
    steps, _, _ = plan_shot_flow(
        "shot_start_end",
        {
            "prompt": "walk to the door",
            "image": _ref("ast_a"),
            "end_image": _ref("ast_b"),
            "prefer_nvfp4": False,
        },
    )
    idxs = [guide["frame_idx"] for guide in steps[0].guides]
    assert idxs == [0, 96]


def test_f03_compose_then_shot() -> None:
    steps, _, bundles = plan_shot_flow(
        "shot_multi_reference",
        {
            "prompt": "they stand in the temple",
            "character": _ref("ast_c"),
            "location": _ref("ast_l"),
            "prefer_nvfp4": False,
        },
    )
    assert [step.mode for step in steps] == ["edit", "shot"]
    assert steps[0].profile == "comfyui"
    assert steps[1].guides[0]["key"] == "__previous__"
    assert "qwen-image-edit-2511-fp8" in bundles


def test_f04_keyframes_ordered() -> None:
    steps, refs, _ = plan_shot_flow(
        "shot_multi_keyframe",
        {
            "prompt": "ritual beat",
            "image": _ref("ast_0"),
            "keyframes": [
                {"asset_id": "ast_1", "frame_idx": 24},
                {"asset_id": "ast_2", "frame_idx": 64, "strength": 0.8},
            ],
            "prefer_nvfp4": False,
        },
    )
    assert refs["keyframe_1"] == "ast_2"
    assert [guide["frame_idx"] for guide in steps[0].guides] == [0, 24, 64]


def test_f05_continuation_uses_video_guide() -> None:
    steps, _, _ = plan_shot_flow(
        "shot_continuation",
        {"prompt": "continue the walk", "video": _ref("ast_clip"), "prefer_nvfp4": False},
    )
    assert steps[0].guides[0]["kind"] == "video"
    assert steps[0].video_keys == ["video"]


def test_f06_control_uses_int8_and_union_lora() -> None:
    steps, _, bundles = plan_shot_flow(
        "shot_motion_reference",
        {
            "prompt": "same action",
            "image": _ref("ast_i"),
            "video": _ref("ast_v"),
            "control": "canny",
        },
    )
    assert steps[0].unet_name == UNET_INT8
    assert steps[0].control["lora"].startswith("ltx-2.3-22b-ic-lora-union")
    assert "ltx-2.5-control" in bundles


def test_f08_camera_requires_preset() -> None:
    try:
        plan_shot_flow("shot_camera", {"prompt": "move", "image": _ref("ast_i")})
        raise AssertionError("expected camera error")
    except DomainError as exc:
        assert "camera" in exc.message
    steps, _, _ = plan_shot_flow(
        "shot_camera",
        {"prompt": "move", "image": _ref("ast_i"), "camera": "orbit"},
    )
    assert steps[0].camera["id"] == "orbit"
    assert steps[0].control["kind"] == "motion_track"


def test_f09_action_suffix() -> None:
    steps, _, _ = plan_shot_flow(
        "shot_action",
        {"prompt": "the gate falls", "image": _ref("ast_i"), "action": "destruction", "prefer_nvfp4": False},
    )
    assert "destruction" in steps[0].prompt


def test_f11_extend_and_f12_stitch() -> None:
    extend, _, _ = plan_shot_flow(
        "shot_extend",
        {"prompt": "keep going", "video": _ref("ast_v"), "duration": 8, "prefer_nvfp4": False},
    )
    assert extend[0].guides[0]["kind"] == "video"
    stitch, refs, _ = plan_shot_flow(
        "shot_stitch",
        {"clips": [_ref("ast_1"), _ref("ast_2")], "upscale": True},
    )
    assert stitch[0].mode == "post"
    assert refs["clip_0"] == "ast_1"
    assert stitch[0].post["upscale"] is True


def test_plan_job_dispatches_shots() -> None:
    steps, _, _ = plan_job("shot_single_image", {"prompt": "x", "image": _ref("ast_i"), "prefer_nvfp4": False})
    assert steps[0].mode == "shot"


def test_shot_graph_has_guides_and_crop() -> None:
    steps, refs, _ = plan_shot_flow(
        "shot_start_end",
        {"prompt": "cross the room", "image": _ref("ast_a"), "end_image": _ref("ast_b"), "prefer_nvfp4": False, "refine": True},
    )
    plan = step_to_plan(steps[0], uploaded={"image": "a.png", "end_image": "b.png"}, previous_name=None)
    graph = build_shot(plan)
    types = _class_types(graph)
    assert "LTXVAddGuide" in types
    assert "LTXVCropGuides" in types
    assert "CLIPLoader" in types
    clip = next(node for node in graph.values() if node["class_type"] == "CLIPLoader")
    assert clip["inputs"]["device"] == "cpu"
    assert any(node["class_type"] == "LTXVLatentUpsampler" for node in graph.values())


def test_control_graph_loads_iclora() -> None:
    steps, _, _ = plan_shot_flow(
        "shot_motion_reference",
        {"prompt": "copy motion", "image": _ref("ast_i"), "video": _ref("ast_v"), "control": "canny"},
    )
    plan = step_to_plan(steps[0], uploaded={"image": "a.png", "video": "m.mp4"}, previous_name=None)
    graph = build_shot(plan)
    types = _class_types(graph)
    assert "LoraLoaderModelOnly" in types
    assert "GetICLoRAParameters" in types
    assert "Canny" in types
    assert "LoadVideo" in types


def test_camera_graph_uses_tracks() -> None:
    steps, _, _ = plan_shot_flow(
        "shot_camera",
        {"prompt": "orbit", "image": _ref("ast_i"), "camera": "orbit"},
    )
    plan = step_to_plan(steps[0], uploaded={"image": "a.png"}, previous_name=None)
    graph = build_shot(plan)
    types = _class_types(graph)
    assert "GenerateTracks" in types
    assert "WanMoveVisualizeTracks" in types
    assert "LTXVContextWindows" not in types or steps[0].windowed


def test_post_graph_joins_clips() -> None:
    graph = build_post({"video_names": ["a.mp4", "b.mp4"], "fps": 24, "post": {"upscale": True}})
    types = _class_types(graph)
    assert "LoadVideo" in types
    assert "ImageBatch" in types
    assert "ImageUpscaleWithModel" in types
    assert "SaveVideo" in types


def test_scheduler_switch_keeps_lease() -> None:
    started: list[str] = []

    class FakeRuntime:
        def __init__(self):
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

    scheduler = GpuScheduler(FakeRuntime())
    lease = scheduler.acquire("job_1", profile="comfyui")
    switched = scheduler.switch(lease, "comfy-ltx")
    assert switched.job_id == "job_1"
    assert switched.profile == "comfy-ltx"
    assert scheduler.active_job() == "job_1"
    scheduler.release(switched)
    assert started == ["comfyui", "comfy-ltx"]


def test_job_service_regenerate_rehydrates_parent(tmp_path) -> None:
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

    svc = JobService(store=JobStore(tmp_path / "jobs.db"), assets=None, scheduler=Sched(), worker=Dummy())
    flows = svc.list_shot_flows()
    assert any(item["id"] == "shot_single_image" for item in flows["operations"])
    assert "orbit" in {item["id"] for item in flows["capabilities"]["cameras"]}
    parent = svc.submit(
        {"operation": "shot_single_image", "prompt": "lantern", "image": {"asset_id": "ast_i"}, "prefer_nvfp4": False}
    )
    child = svc.submit({"operation": "shot_regenerate", "parent_job_id": parent["job_id"], "seed": 99})
    assert child["operation"] == "shot_single_image"
    assert child["inputs"]["parent_job_id"] == parent["job_id"]
    assert child["inputs"]["prompt"] == "lantern"
    assert child["seed"] == 99


def test_all_flows_registered() -> None:
    assert len(FLOWS) == 12
    for flow_id in (
        "shot_single_image",
        "shot_start_end",
        "shot_multi_reference",
        "shot_multi_keyframe",
        "shot_continuation",
        "shot_motion_reference",
        "shot_character_interaction",
        "shot_camera",
        "shot_action",
        "shot_regenerate",
        "shot_extend",
        "shot_stitch",
    ):
        assert flow_id in FLOWS
