from __future__ import annotations

from copy import deepcopy

from ...domain.artifacts import Artifact
from ...domain.jobs import Job
from .codec import artifact_from_dict, artifact_to_dict, job_from_dict, job_to_dict


class MemoryJobRepository:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}

    def save(self, job: Job) -> None:
        self._jobs[job.job_id] = job_to_dict(job)

    def get(self, job_id: str) -> Job | None:
        data = self._jobs.get(job_id)
        return job_from_dict(deepcopy(data)) if data else None

    def list_active(self) -> list[Job]:
        out: list[Job] = []
        for data in self._jobs.values():
            job = job_from_dict(deepcopy(data))
            if not job.status.terminal:
                out.append(job)
        return out


class MemoryArtifactRepository:
    def __init__(self) -> None:
        self._items: dict[str, dict] = {}

    def save(self, artifact: Artifact) -> None:
        self._items[artifact.asset_id] = artifact_to_dict(artifact)

    def get(self, asset_id: str) -> Artifact | None:
        data = self._items.get(asset_id)
        return artifact_from_dict(deepcopy(data)) if data else None

    def delete(self, asset_id: str) -> None:
        item = self._items.get(asset_id)
        if item:
            item["deleted"] = True


class MemoryIdempotencyRepository:
    def __init__(self) -> None:
        self._items: dict[str, tuple[str, str]] = {}

    def get(self, key: str) -> tuple[str, str] | None:
        return self._items.get(key)

    def put(self, key: str, request_hash: str, job_id: str) -> None:
        self._items[key] = (request_hash, job_id)


class MemoryRuntimeStateRepository:
    def __init__(self) -> None:
        self._items: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._items.get(key)

    def put(self, key: str, value: str) -> None:
        self._items[key] = value
