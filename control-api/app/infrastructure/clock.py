from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..domain.ids import new_asset_id, new_job_id, new_trace_id


class SystemClock:
    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def now_ts(self) -> float:
        return datetime.now(timezone.utc).timestamp()


class UlidGenerator:
    def job_id(self) -> str:
        return new_job_id()

    def asset_id(self) -> str:
        return new_asset_id()

    def trace_id(self) -> str:
        return new_trace_id()


def expires_at(now_iso: str, hours: int) -> str:
    now = datetime.fromisoformat(now_iso)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now + timedelta(hours=hours)).isoformat()


def is_expired(expires: str, now_iso: str) -> bool:
    try:
        exp = datetime.fromisoformat(expires)
        now = datetime.fromisoformat(now_iso)
    except ValueError:
        return False
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now >= exp
