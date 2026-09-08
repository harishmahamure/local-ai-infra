from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DependencyPolicy(str, Enum):
    REQUIRE_SUCCESS = "REQUIRE_SUCCESS"
    ALLOW_FAILED = "ALLOW_FAILED"
    OPTIONAL = "OPTIONAL"


class DependencyState(str, Enum):
    BLOCKED = "BLOCKED"
    READY = "READY"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class JobDependency:
    job_id: str
    depends_on: str
    policy: DependencyPolicy = DependencyPolicy.REQUIRE_SUCCESS
    artifact_bindings: dict[str, str] | None = None
    state: DependencyState = DependencyState.BLOCKED

    def to_public_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "depends_on": self.depends_on,
            "policy": self.policy.value,
            "artifact_bindings": dict(self.artifact_bindings or {}),
            "state": self.state.value,
        }
