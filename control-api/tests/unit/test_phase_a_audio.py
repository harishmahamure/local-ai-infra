from __future__ import annotations

from pathlib import Path

import pytest

from app.ace_step_graph import CLIP, UNET, VAE, build as build_ace
from app.application.ports import ExecutionRequest
from app.domain.errors import DomainError, ErrorCode
from app.domain.jobs import Job, JobStatus
from app.domain.presets import Preset
from app.domain.workflows import WorkflowDefinition
from app.infrastructure.comfyui.plans import build_plan
from app.infrastructure.ffmpeg import commands
from app.infrastructure.ffmpeg.executor import FfmpegExecutor
from app.infrastructure.routing import RoutingExecutor
from app.infrastructure.tts.executor import TtsExecutor


def _job(operation: str, inputs: dict, parameters: dict | None = None) -> Job:
    return Job(
        job_id="job_test",
        operation=operation,
        preset="master",
        status=JobStatus.QUEUED,
        phase="queued",
        progress=0.0,
        inputs=inputs,
        parameters=parameters or {},
        client_context={},
        created_at="2026-01-01T00:00:00Z",
    )


def _workflow(builder: str, executor: str = "comfyui") -> WorkflowDefinition:
    return WorkflowDefinition(
        id="wf",
        version="v1",
        operation="op",
        description="",
        executor=executor,
        builder=builder,
    )


def test_tts_plan_uses_hindi_preset_language() -> None:
    job = _job("audio.tts", {"text": "नमस्ते"}, {"language": "hi", "speaking_rate": 1.1, "seed": 7})
    preset = Preset(id="narrator_hindi", label="", description="", operations=["audio.tts"], parameters={"language": "hi"})
    plan = build_plan(job, _workflow("tts_generate", "tts"), preset, [])
    assert plan["text"] == "नमस्ते"
    assert plan["language"] == "hi"
    assert plan["speaking_rate"] == 1.1
    assert plan["seed"] == 7


def test_ace_music_plan_appends_no_vocals() -> None:
    job = _job("audio.music", {"prompt": "sparse drone", "duration_seconds": 8}, {"seed": 2})
    preset = Preset(
        id="cinematic_master",
        label="",
        description="",
        operations=["audio.music"],
        parameters={"no_vocals": True},
    )
    plan = build_plan(job, _workflow("ace_step_music"), preset, [])
    assert plan["kind"] == "music"
    assert "no vocals" in plan["prompt"]
    assert plan["duration_seconds"] == 8.0
    graph = build_ace(plan)
    assert graph["1"]["inputs"]["unet_name"] == UNET
    assert graph["2"]["inputs"]["clip_name"] == CLIP
    assert graph["3"]["inputs"]["vae_name"] == VAE
    assert graph["4"]["inputs"]["text"] == plan["prompt"]
    assert graph["5"]["class_type"] == "EmptyLatentAudio"
    assert graph["8"]["class_type"] == "SaveAudio"


def test_foley_plan_ignores_video() -> None:
    job = _job("audio.foley", {"prompt": "footsteps on gravel", "video": {"asset_id": "ast_1"}})
    preset = Preset(id="master", label="", description="", operations=["audio.foley"], parameters={})
    plan = build_plan(job, _workflow("ace_step_foley"), preset, [])
    assert plan["video_ignored"] is True
    assert plan["kind"] == "foley"


def test_ffmpeg_mix_plan() -> None:
    stems = [{"asset_id": "a", "role": "music", "start_seconds": 1, "gain_db": -3}]
    job = _job("audio.mix", {"stems": stems, "target_lufs": -18})
    preset = Preset(id="cinematic", label="", description="", operations=["audio.mix"], parameters={"target_lufs": -16})
    plan = build_plan(job, _workflow("ffmpeg_mix", "ffmpeg"), preset, [])
    assert plan["builder"] == "ffmpeg_mix"
    assert plan["target_lufs"] == -18
    assert plan["stems"] == stems


def test_ffmpeg_argv_no_encode(monkeypatch) -> None:
    monkeypatch.setattr(commands, "_bin", lambda name: name)
    mix = commands.mix_argv(["a.wav", "b.wav"], "out.wav", starts=[0, 1.5], gains_db=[0, -6], target_lufs=-16, duration=4)
    assert mix[0] == "ffmpeg"
    assert "adelay=1500|1500" in mix[mix.index("-filter_complex") + 1]
    assert "loudnorm=I=-16" in mix[mix.index("-filter_complex") + 1]
    assert "-t" in mix
    norm = commands.loudnorm_argv("in.wav", "out.wav", target_lufs=-14, true_peak=-1.0)
    assert "loudnorm=I=-14:TP=-1.0:LRA=11" in norm
    cut = commands.concat_argv("list.txt", "out.mp4", transition="cut", transition_seconds=0, n=2)
    assert "-f" in cut and "concat" in cut
    fade = commands.concat_filter_argv(["a.mp4", "b.mp4"], "out.mp4", transition_seconds=0.5)
    assert "xfade" in fade[fade.index("-filter_complex") + 1]
    final = commands.finalize_argv("v.mp4", "a.wav", "out.mp4", offset=0.25)
    assert "-itsoffset" in final
    assert "libx264" in final
    assert "aac" in final
    inspect = commands.inspect_argv("in.wav")
    assert inspect[0] == "ffprobe"


def test_tts_executor_mocked(monkeypatch, tmp_path: Path) -> None:
    seen: dict = {}

    def generate_fn(**kwargs):
        seen.update(kwargs)
        return b"RIFF-fake-wav"

    monkeypatch.setattr("app.infrastructure.tts.executor.resample_wav_48k", lambda raw: raw)
    executor = TtsExecutor(generate_fn=generate_fn, model_root=tmp_path)
    result = executor.execute(
        ExecutionRequest(
            job=_job("audio.tts", {"text": "hello"}),
            workflow=_workflow("tts_generate", "tts"),
            plan={"text": "hello", "language": "hi", "speaking_rate": 1.0, "seed": 3},
        )
    )
    assert result.outputs[0].mime_type == "audio/wav"
    assert result.outputs[0].artifact_type == "AUDIO"
    assert "chatterbox-hi" in result.model_ids
    assert seen["language"] == "hi"
    assert seen["text"] == "hello"


def test_tts_executor_cancel() -> None:
    executor = TtsExecutor(generate_fn=lambda **_: b"x")
    executor.cancel("job_test")
    with pytest.raises(DomainError) as exc:
        executor.execute(
            ExecutionRequest(
                job=_job("audio.tts", {"text": "x"}),
                workflow=_workflow("tts_generate", "tts"),
                plan={"text": "x", "language": "en"},
            )
        )
    assert exc.value.code == ErrorCode.CANCELLED


def test_ffmpeg_executor_uses_injected_runner(tmp_path: Path) -> None:
    def runner(cmd: list[str]) -> str:
        Path(cmd[-1]).write_bytes(b"encoded")
        return ""

    executor = FfmpegExecutor(runner=runner, prober=lambda path: {"format": {"duration": "2.5"}, "streams": []})
    job = _job("audio.normalize", {"audio": {"asset_id": "a"}})
    result = executor.execute(
        ExecutionRequest(
            job=job,
            workflow=_workflow("ffmpeg_normalize", "ffmpeg"),
            plan={"target_lufs": -16, "true_peak_db": -1.5},
            input_files={"audio": b"wav"},
        )
    )
    assert result.outputs[0].data == b"encoded"
    inspect = executor.execute(
        ExecutionRequest(
            job=_job("audio.inspect", {"audio": {"asset_id": "a"}}),
            workflow=_workflow("ffmpeg_inspect", "ffmpeg"),
            plan={},
            input_files={"audio": b"wav"},
        )
    )
    assert inspect.outputs[0].artifact_type == "JSON"
    assert inspect.outputs[0].duration_seconds == 2.5


def test_routing_executor_dispatches() -> None:
    hits: list[str] = []

    class _Inner:
        def execute(self, request):
            hits.append(request.workflow.executor)
            return request

        def cancel(self, job_id: str) -> None:
            return None

        def health(self) -> dict:
            return {"ok": True}

    router = RoutingExecutor({"tts": _Inner(), "ffmpeg": _Inner(), "comfyui": _Inner()})
    router.execute(ExecutionRequest(job=_job("audio.tts", {}), workflow=_workflow("tts_generate", "tts"), plan={}))
    router.execute(ExecutionRequest(job=_job("audio.mix", {}), workflow=_workflow("ffmpeg_mix", "ffmpeg"), plan={}))
    assert hits == ["tts", "ffmpeg"]
    with pytest.raises(DomainError) as exc:
        router.execute(ExecutionRequest(job=_job("x", {}), workflow=_workflow("", "none"), plan={}))
    assert exc.value.code == ErrorCode.UNSUPPORTED_OPERATION
