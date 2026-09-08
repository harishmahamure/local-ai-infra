from __future__ import annotations

from enum import Enum

from .runtime import ResourceRequirement
from .workflows import WorkflowDefinition


class QueueClass(str, Enum):
    GPU = "gpu"
    CPU_MEDIA = "cpu-media"
    AUDIO = "audio"
    IO = "io"


QUEUE_ALIASES = {
    "gpu": QueueClass.GPU,
    "gpu-high": QueueClass.GPU,
    "gpu-normal": QueueClass.GPU,
    "gpu-low": QueueClass.GPU,
    "cpu": QueueClass.CPU_MEDIA,
    "cpu-media": QueueClass.CPU_MEDIA,
    "audio": QueueClass.AUDIO,
    "tts": QueueClass.AUDIO,
    "io": QueueClass.IO,
}


def parse_queue(value: str | None) -> QueueClass:
    if value is None or value == "":
        return QueueClass.GPU
    key = str(value).strip().lower()
    if key in QUEUE_ALIASES:
        return QUEUE_ALIASES[key]
    raise ValueError(f"Unknown queue: {value}")


def queue_for_workflow(workflow: WorkflowDefinition) -> QueueClass:
    return queue_for_resource(workflow.resource, workflow.executor)


def queue_for_resource(resource: ResourceRequirement, executor: str = "") -> QueueClass:
    if executor == "tts" or resource.profile == "tts":
        return QueueClass.AUDIO
    if executor == "ffmpeg" or resource.profile in {"none", ""} or not resource.gpu_required:
        return QueueClass.CPU_MEDIA
    if executor == "io":
        return QueueClass.IO
    return QueueClass.GPU


def worker_role_for_queue(queue: QueueClass) -> str:
    if queue == QueueClass.GPU:
        return "gpu"
    if queue == QueueClass.AUDIO:
        return "audio"
    if queue == QueueClass.IO:
        return "io"
    return "cpu-media"


def queues_for_role(role: str) -> list[QueueClass]:
    if role == "gpu":
        return [QueueClass.GPU]
    if role == "audio":
        return [QueueClass.AUDIO]
    if role == "io":
        return [QueueClass.IO]
    if role == "cpu-media":
        return [QueueClass.CPU_MEDIA]
    if role == "all":
        return list(QueueClass)
    return [QueueClass.GPU]
