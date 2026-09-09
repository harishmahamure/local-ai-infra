from __future__ import annotations

import threading

from app.application.image_ops import plan_operation, step_to_plan
from app.application.jobs import JobService
from app.domain.errors import DomainError
from app.domain.jobs import Job, JobStatus, can_transition
from app.domain.operations import DEFAULT_NEGATIVE, DEFAULT_SHIFT, STYLE_PRESETS
from app.infrastructure.comfy.graphs import (
    MODELS,
    build_edit,
    build_inpaint,
    build_outpaint,
    build_txt2img,
    build_upscale,
)
from app.infrastructure.gpu_scheduler import GpuScheduler
from app.infrastructure.job_store import JobStore


def _class_types(graph: dict) -> set[str]:
    return {node["class_type"] for node in graph.values()}


def _nodes(graph: dict, class_type: str) -> list[dict]:
    return [node for node in graph.values() if node["class_type"] == class_type]


def test_txt2img_graph_nodes() -> None:
    graph = build_txt2img(
        {"prompt": "a knight", "width": 1024, "height": 1024, "steps": 50, "cfg": 4.0, "seed": 1}
    )
    types = _class_types(graph)
    assert {"UNETLoader", "CLIPLoader", "VAELoader", "EmptySD3LatentImage", "KSampler", "VAEDecode", "SaveImage", "ModelSamplingAuraFlow"} <= types
    assert "VAEEncode" not in types
    aura = _nodes(graph, "ModelSamplingAuraFlow")[0]
    assert aura["inputs"]["shift"] == 3.1
    sampler = _nodes(graph, "KSampler")[0]
    assert sampler["inputs"]["sampler_name"] == "euler"
    assert sampler["inputs"]["scheduler"] == "simple"


def test_txt2img_lightning_zeroes_negative() -> None:
    graph = build_txt2img(
        {
            "prompt": "a knight",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "cfg": 1.0,
            "seed": 1,
            "loras": [{"name": MODELS["qwen_lora_lightning"], "strength": 1.0}],
        }
    )
    assert "ConditioningZeroOut" in _class_types(graph)
    assert len(_nodes(graph, "CLIPTextEncode")) == 1


def test_edit_graph_uses_qwen_edit_plus() -> None:
    graph = build_edit(
        {
            "prompt": "change outfit",
            "image_name": "a.png",
            "image_names": ["a.png"],
            "width": 1024,
            "height": 1024,
            "seed": 2,
        }
    )
    types = _class_types(graph)
    assert "TextEncodeQwenImageEditPlus" in types
    assert "CFGNorm" in types
    assert "ModelSamplingAuraFlow" in types
    assert "EmptySD3LatentImage" in types
    assert "FluxKontextMultiReferenceLatentMethod" not in types
    encode = _nodes(graph, "TextEncodeQwenImageEditPlus")
    assert encode[0]["inputs"]["image1"]
    negatives = [node for node in encode if node["inputs"]["prompt"] == ""]
    assert len(negatives) == 1
    assert _nodes(graph, "ModelSamplingAuraFlow")[0]["inputs"]["shift"] == 3.1


def test_edit_graph_multi_reference_uses_kontext() -> None:
    graph = build_edit(
        {
            "prompt": "compose",
            "image_names": ["a.png", "b.png"],
            "width": 1664,
            "height": 928,
            "seed": 2,
        }
    )
    assert "FluxKontextMultiReferenceLatentMethod" in _class_types(graph)
    encode = next(n for n in graph.values() if n["class_type"] == "TextEncodeQwenImageEditPlus" and n["inputs"]["prompt"])
    assert "image1" in encode["inputs"]
    assert "image2" in encode["inputs"]


def test_edit_applies_loras() -> None:
    graph = build_edit(
        {
            "prompt": "change outfit",
            "image_name": "a.png",
            "width": 1024,
            "height": 1024,
            "cfg": 1.0,
            "steps": 4,
            "seed": 2,
            "loras": [{"name": MODELS["qwen_edit_lora_lightning"], "strength": 1.0}],
        }
    )
    loras = _nodes(graph, "LoraLoaderModelOnly")
    assert loras
    assert loras[0]["inputs"]["lora_name"] == MODELS["qwen_edit_lora_lightning"]
    assert "ConditioningZeroOut" in _class_types(graph)


def test_inpaint_and_outpaint_graphs() -> None:
    inpaint = build_inpaint({"prompt": "fix", "image_name": "a.png", "mask_name": "m.png", "seed": 1})
    assert "QwenImageDiffsynthControlnet" in _class_types(inpaint)
    assert "ImageToMask" in _class_types(inpaint)
    assert "ImageScaleToTotalPixels" in _class_types(inpaint)
    assert "ModelSamplingAuraFlow" in _class_types(inpaint)
    outpaint = build_outpaint({"prompt": "expand", "image_name": "a.png", "pad": {"left": 64, "right": 64}, "seed": 1})
    assert "ImagePadForOutpaint" in _class_types(outpaint)
    assert "ImageScaleToTotalPixels" in _class_types(outpaint)
    cn = next(n for n in outpaint.values() if n["class_type"] == "QwenImageDiffsynthControlnet")
    assert cn["inputs"]["mask"]


def test_inpaint_quality_skips_lightning() -> None:
    graph = build_inpaint({"prompt": "fix", "image_name": "a.png", "mask_name": "m.png", "seed": 1, "steps": 50, "cfg": 4.0})
    assert "LoraLoaderModelOnly" not in _class_types(graph)
    assert "CLIPTextEncode" in _class_types(graph)


def test_upscale_graph() -> None:
    graph = build_upscale({"image_name": "a.png", "upscale_scale": 2})
    assert {"UpscaleModelLoader", "ImageUpscaleWithModel", "SaveImage"} <= _class_types(graph)
    loader = next(n for n in graph.values() if n["class_type"] == "UpscaleModelLoader")
    assert "x2" in loader["inputs"]["model_name"]


def test_character_without_reference_is_txt2img() -> None:
    steps, _refs, bundles = plan_operation("generate_character", {"prompt": "elf ranger"})
    assert len(steps) == 1
    assert steps[0].mode == "txt2img"
    assert steps[0].steps == 50
    assert steps[0].cfg == 4.0
    assert steps[0].shift == DEFAULT_SHIFT
    assert DEFAULT_NEGATIVE in steps[0].negative_prompt
    assert steps[0].prompt == f"elf ranger. {STYLE_PRESETS['cinematic_naturalism'].suffix}"
    assert STYLE_PRESETS["cinematic_naturalism"].negative in steps[0].negative_prompt
    assert "qwen-image-2512-fp8" in bundles


def test_character_with_reference_uses_edit() -> None:
    steps, refs, bundles = plan_operation(
        "generate_character",
        {"prompt": "elf ranger in a cloak", "image": {"asset_id": "ast_face"}},
    )
    assert len(steps) == 1
    assert steps[0].mode == "edit"
    assert steps[0].steps == 40
    assert steps[0].cfg == 3.0
    assert steps[0].image_keys == ["image"]
    assert "image 1" in steps[0].prompt
    assert "elf ranger in a cloak" in steps[0].prompt
    assert "full-body" not in steps[0].prompt
    assert STYLE_PRESETS["cinematic_naturalism"].suffix in steps[0].prompt
    assert refs["image"] == "ast_face"
    assert bundles == ["qwen-image-edit-2511-fp8"]


def test_turnaround_plans_four_edit_views() -> None:
    steps, refs, bundles = plan_operation(
        "generate_character_turnaround",
        {"prompt": "elf ranger", "character": {"asset_id": "ast_abc"}},
    )
    assert len(steps) == 4
    assert all(s.mode == "edit" for s in steps)
    assert all(s.image_keys == ["character"] for s in steps)
    assert all(s.steps == 40 for s in steps)
    assert "qwen-image-edit-2511-fp8" in bundles
    assert refs["character"] == "ast_abc"


def test_turnaround_from_prompt_uses_base_key() -> None:
    steps, _refs, bundles = plan_operation("generate_character_turnaround", {"prompt": "elf ranger"})
    assert len(steps) == 5
    assert steps[0].mode == "txt2img"
    assert all(s.mode == "edit" for s in steps[1:])
    assert all(s.image_keys == ["__base__"] for s in steps[1:])
    assert "qwen-image-2512-fp8" in bundles
    plan = step_to_plan(steps[1], uploaded={}, previous_name="prev.png", base_name="base.png")
    assert plan["image_name"] == "base.png"


def test_keyframe_uses_three_references() -> None:
    steps, refs, _ = plan_operation(
        "generate_keyframe",
        {
            "prompt": "dusk duel",
            "character": {"asset_id": "ast_c"},
            "attire": {"asset_id": "ast_a"},
            "location": {"asset_id": "ast_l"},
        },
    )
    assert len(steps) == 1
    assert steps[0].mode == "edit"
    assert steps[0].image_keys == ["character", "attire", "location"]
    assert steps[0].steps == 40
    assert set(refs) == {"character", "attire", "location"}
    assert steps[0].prompt.startswith("dusk duel")
    assert STYLE_PRESETS["cinematic_naturalism"].suffix in steps[0].prompt


def test_shot_and_prop_put_user_prompt_first() -> None:
    shot, _, _ = plan_operation(
        "generate_shot_reference",
        {"prompt": "the hero draws", "image": {"asset_id": "ast_c"}, "framing": "wide", "lens": "24"},
    )
    assert shot[0].prompt.startswith("the hero draws")
    assert STYLE_PRESETS["cinematic_naturalism"].suffix in shot[0].prompt
    prop, _, _ = plan_operation("generate_prop", {"prompt": "brass lantern"})
    assert prop[0].prompt.startswith("brass lantern")
    assert STYLE_PRESETS["cinematic_naturalism"].suffix not in prop[0].prompt


def test_style_presets_and_opt_out() -> None:
    cinematic = STYLE_PRESETS["cinematic_naturalism"]
    painterly = STYLE_PRESETS["painterly_concept"]
    graphic = STYLE_PRESETS["graphic_still"]
    for operation, body in (
        ("generate_character", {"prompt": "wanderer"}),
        ("generate_character_turnaround", {"prompt": "wanderer", "character": {"asset_id": "ast_c"}}),
        (
            "generate_keyframe",
            {
                "prompt": "dusk duel",
                "character": {"asset_id": "ast_c"},
                "attire": {"asset_id": "ast_a"},
                "location": {"asset_id": "ast_l"},
            },
        ),
        ("generate_shot_reference", {"prompt": "the hero draws", "image": {"asset_id": "ast_c"}}),
    ):
        steps, _, _ = plan_operation(operation, body)
        assert steps[0].prompt.endswith(cinematic.suffix)
        assert cinematic.negative in steps[0].negative_prompt
        assert steps[0].steps in {40, 50}
        assert steps[0].cfg in {3.0, 4.0}

    painted, _, _ = plan_operation(
        "generate_character",
        {"prompt": "wanderer", "style_preset": "painterly_concept"},
    )
    assert painted[0].prompt == f"wanderer. {painterly.suffix}"
    assert painterly.negative in painted[0].negative_prompt

    still, _, _ = plan_operation(
        "generate_keyframe",
        {
            "prompt": "dusk duel",
            "character": {"asset_id": "ast_c"},
            "attire": {"asset_id": "ast_a"},
            "location": {"asset_id": "ast_l"},
            "style_preset": "graphic_still",
        },
    )
    assert still[0].prompt == f"dusk duel. {graphic.suffix}"
    assert graphic.negative in still[0].negative_prompt

    custom_neg, _, _ = plan_operation(
        "generate_character",
        {"prompt": "wanderer", "negative_prompt": "muddy colors"},
    )
    assert custom_neg[0].negative_prompt.startswith("muddy colors")
    assert cinematic.negative in custom_neg[0].negative_prompt

    off, _, _ = plan_operation(
        "generate_character",
        {"prompt": "wanderer", "style_preset": "off"},
    )
    assert off[0].prompt == "wanderer"
    assert cinematic.suffix not in off[0].prompt
    assert cinematic.negative not in off[0].negative_prompt

    try:
        plan_operation("generate_character", {"prompt": "wanderer", "style_preset": "neon_vapor"})
        raise AssertionError("expected unknown style preset")
    except DomainError as exc:
        assert exc.code.value == "INVALID_PARAMETER"

    location, _, _ = plan_operation("generate_location", {"prompt": "rainy street"})
    assert location[0].prompt == "rainy street"
    assert cinematic.suffix not in location[0].prompt
    assert cinematic.negative not in location[0].negative_prompt

    face, _, _ = plan_operation(
        "generate_character",
        {"prompt": "portrait master", "image": {"asset_id": "ast_face"}, "style_preset": "off"},
    )
    assert face[0].prompt.startswith("Generate the character of the person in image 1.")
    assert "full-body" not in face[0].prompt


def test_fast_edit_uses_edit_lightning_lora() -> None:
    steps, _, _ = plan_operation(
        "generate_attire",
        {"prompt": "red coat", "character": {"asset_id": "ast_c"}, "fast": True},
    )
    assert steps[0].mode == "edit"
    assert steps[0].cfg == 1.0
    assert steps[0].steps == 4
    assert steps[0].loras == [{"name": MODELS["qwen_edit_lora_lightning"], "strength": 1.0}]


def test_fast_txt2img_uses_base_lightning_lora() -> None:
    steps, _, _ = plan_operation("generate_prop", {"prompt": "lantern", "fast": True})
    assert steps[0].cfg == 1.0
    assert steps[0].negative_prompt == ""
    assert steps[0].loras == [{"name": MODELS["qwen_lora_lightning"], "strength": 1.0}]


def test_advanced_overrides_and_validation() -> None:
    steps, _, _ = plan_operation(
        "generate_location",
        {
            "prompt": "rainy street",
            "steps": 28,
            "cfg": 3.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "shift": 2.8,
        },
    )
    assert steps[0].steps == 28
    assert steps[0].cfg == 3.5
    assert steps[0].sampler_name == "dpmpp_2m"
    assert steps[0].scheduler == "karras"
    assert steps[0].shift == 2.8
    plan = step_to_plan(steps[0], uploaded={}, previous_name=None)
    assert plan["sampler_name"] == "dpmpp_2m"
    assert plan["scheduler"] == "karras"
    assert plan["shift"] == 2.8
    try:
        plan_operation("generate_location", {"prompt": "x", "sampler_name": "not-a-sampler"})
        raise AssertionError("expected invalid sampler")
    except DomainError as exc:
        assert exc.code.value == "INVALID_PARAMETER"


def test_inpaint_quality_uses_spec_steps() -> None:
    steps, _, _ = plan_operation(
        "inpaint_asset",
        {"prompt": "fix sleeve", "image": {"asset_id": "ast_i"}, "mask": {"asset_id": "ast_m"}},
    )
    assert steps[0].steps == 50
    assert steps[0].cfg == 4.0
    fast, _, _ = plan_operation(
        "inpaint_asset",
        {"prompt": "fix sleeve", "image": {"asset_id": "ast_i"}, "mask": {"asset_id": "ast_m"}, "fast": True},
    )
    assert fast[0].steps == 4
    assert fast[0].cfg == 1.0


def test_job_state_machine_rejects_illegal_jumps() -> None:
    job = Job(
        job_id="job_1",
        operation="generate_prop",
        status=JobStatus.QUEUED,
        phase="queued",
        progress=0,
        inputs={},
        parameters={},
        created_at="2026-01-01T00:00:00Z",
    )
    assert can_transition(JobStatus.QUEUED, JobStatus.RUNNING)
    job.transition(JobStatus.RUNNING, phase="rendering")
    try:
        job.transition(JobStatus.QUEUED)
        raise AssertionError("expected illegal transition")
    except DomainError:
        pass
    job.transition(JobStatus.SUCCEEDED, phase="done", progress=1)
    assert job.status.terminal


def test_scheduler_is_single_slot() -> None:
    class FakeRuntime:
        def get_status(self):
            return {"profile": "comfyui", "loadState": "LOADED"}

        def start_profile(self, profile):
            return {"profile": profile}

        def stop_profile(self):
            return None

    scheduler = GpuScheduler(FakeRuntime())
    first = scheduler.acquire("job_a")
    errors: list[str] = []

    def second():
        try:
            scheduler.acquire("job_b")
        except DomainError as exc:
            errors.append(exc.code.value)

    thread = threading.Thread(target=second)
    thread.start()
    thread.join()
    assert errors == ["INTERNAL_ERROR"]
    scheduler.release(first)
    lease = scheduler.acquire("job_b")
    scheduler.release(lease)


def test_idempotency_returns_same_job(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.db")

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

    svc = JobService(store=store, assets=None, scheduler=Sched(), worker=Dummy())
    first = svc.submit({"operation": "generate_prop", "prompt": "a lantern"}, idempotency_key="k1")
    second = svc.submit({"operation": "generate_prop", "prompt": "a lantern"}, idempotency_key="k1")
    assert first["job_id"] == second["job_id"]
    assert store.queued_count() == 1
    listed = svc.list_operations()
    assert "euler" in listed["capabilities"]["samplers"]
    assert "simple" in listed["capabilities"]["schedulers"]
    character = next(op for op in listed["operations"] if op["id"] == "generate_character")
    assert character["defaultSteps"] == 50
    keyframe = next(op for op in listed["operations"] if op["id"] == "generate_keyframe")
    assert keyframe["defaultSteps"] == 40
    assert keyframe["defaultCfg"] == 3.0
    assert character["defaultShift"] == DEFAULT_SHIFT
    assert {item["id"] for item in listed["capabilities"]["stylePresets"]} == set(STYLE_PRESETS)
