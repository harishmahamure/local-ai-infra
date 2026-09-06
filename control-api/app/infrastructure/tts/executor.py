from __future__ import annotations

import os
import struct
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, Callable

from ...application.ports import ExecutionOutput, ExecutionRequest, ExecutionResult
from ...domain.errors import DomainError, ErrorCode
from ..ffmpeg.commands import resample_wav_48k

GenerateFn = Callable[..., bytes]


def _ckpt_dir() -> Path:
    raw = os.environ.get("CHATTERBOX_MODELS") or str(Path.home() / "ai-inference" / "models" / "chatterbox")
    return Path(os.path.expanduser(raw))


def _pcm16_wav(samples: list[int], sample_rate: int) -> bytes:
    buf = tempfile.SpooledTemporaryFile()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(struct.pack("<h", max(-32768, min(32767, s))) for s in samples))
    buf.seek(0)
    return buf.read()


def chatterbox_generate(
    *,
    text: str,
    language: str,
    speaker_wav: bytes | None,
    speaking_rate: float,
    seed: int | None,
    ckpt_dir: Path,
) -> bytes:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as exc:
        raise DomainError(
            ErrorCode.WORKFLOW_NOT_AVAILABLE,
            "chatterbox is not installed in the control venv. pip install chatterbox-tts on the GPU box.",
        ) from exc

    if not ckpt_dir.is_dir():
        raise DomainError(ErrorCode.MODEL_NOT_AVAILABLE, f"Chatterbox checkpoint dir missing: {ckpt_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_local(ckpt_dir=str(ckpt_dir), device=device)
    prompt_path = None
    tmp = None
    if speaker_wav:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(speaker_wav)
        tmp.close()
        prompt_path = tmp.name
    try:
        kwargs: dict[str, Any] = {"language_id": language or "en"}
        if prompt_path:
            kwargs["audio_prompt_path"] = prompt_path
        if speaking_rate and abs(speaking_rate - 1.0) > 0.01:
            kwargs["exaggeration"] = max(0.25, min(2.0, speaking_rate))
        if seed is not None:
            torch.manual_seed(int(seed))
        wav = model.generate(text, **kwargs)
        audio = wav.detach().cpu().flatten()
        sample_rate = int(getattr(model, "sr", None) or getattr(model, "sample_rate", 24000))
        samples = [int(float(x) * 32767) for x in audio.tolist()]
        return _pcm16_wav(samples, sample_rate)
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)


class TtsExecutor:
    def __init__(self, generate_fn: GenerateFn | None = None, model_root: Path | None = None) -> None:
        self._generate = generate_fn or chatterbox_generate
        self._model_root = model_root
        self._lock = threading.Lock()
        self._cancelled: set[str] = set()

    def health(self) -> dict[str, Any]:
        root = self._model_root or _ckpt_dir()
        return {"tts": True, "ckpt_dir": str(root), "present": root.is_dir()}

    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        job = request.job
        if job.job_id in self._cancelled or (request.cancel_check and request.cancel_check()):
            raise DomainError(ErrorCode.CANCELLED, "Job cancelled")
        plan = request.plan
        text = str(plan.get("text") or "").strip()
        if not text:
            raise DomainError(ErrorCode.INVALID_REQUEST, "text is required")
        language = str(plan.get("language") or "en")
        if language not in {"hi", "en"}:
            raise DomainError(ErrorCode.INVALID_PARAMETER, "language must be hi or en")
        if request.on_phase:
            request.on_phase("tts.generate", 0.3)
        raw = self._generate(
            text=text,
            language=language,
            speaker_wav=request.input_files.get("speaker_ref"),
            speaking_rate=float(plan.get("speaking_rate") or 1.0),
            seed=plan.get("seed"),
            ckpt_dir=self._model_root or _ckpt_dir(),
        )
        if job.job_id in self._cancelled or (request.cancel_check and request.cancel_check()):
            raise DomainError(ErrorCode.CANCELLED, "Job cancelled")
        wav = resample_wav_48k(raw)
        if request.on_phase:
            request.on_phase("tts.done", 0.85)
        models = ["chatterbox-multilingual"]
        if language == "hi":
            models.append("chatterbox-hi")
        return ExecutionResult(
            outputs=[ExecutionOutput(data=wav, mime_type="audio/wav", artifact_type="AUDIO")],
            model_ids=models,
            seed=plan.get("seed"),
            extra={"language": language, "plan": plan},
        )
