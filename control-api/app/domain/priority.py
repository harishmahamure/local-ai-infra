from __future__ import annotations

from enum import Enum


class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"
    BACKGROUND = "BACKGROUND"

    @property
    def rank(self) -> int:
        return {
            Priority.CRITICAL: 5,
            Priority.HIGH: 4,
            Priority.NORMAL: 3,
            Priority.LOW: 2,
            Priority.BACKGROUND: 1,
        }[self]

    def is_high_band(self) -> bool:
        return self in {Priority.CRITICAL, Priority.HIGH}


DEFAULT_PRIORITY_WEIGHTS: dict[Priority, int] = {
    Priority.CRITICAL: 1000,
    Priority.HIGH: 100,
    Priority.NORMAL: 10,
    Priority.LOW: 2,
    Priority.BACKGROUND: 1,
}


def parse_priority(value: str | None) -> Priority:
    if value is None or value == "":
        return Priority.NORMAL
    key = str(value).strip().upper()
    aliases = {"CRIT": "CRITICAL", "BG": "BACKGROUND"}
    key = aliases.get(key, key)
    try:
        return Priority(key)
    except ValueError as exc:
        raise ValueError(f"Unknown priority: {value}") from exc
