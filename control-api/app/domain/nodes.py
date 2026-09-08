from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class NodeMode(str, Enum):
    NORMAL = "NORMAL"
    DRAINING = "DRAINING"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class ComputeNode:
    node_id: str
    mode: NodeMode = NodeMode.NORMAL
    hostname: str = "gpu-box"
    healthy: bool = True
    updated_at: str | None = None

    def accepts_new_work(self) -> bool:
        return self.mode == NodeMode.NORMAL and self.healthy

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "mode": self.mode.value,
            "hostname": self.hostname,
            "healthy": self.healthy,
            "updated_at": self.updated_at,
        }
