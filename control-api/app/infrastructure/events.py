from __future__ import annotations

import json
import queue
import threading
from typing import Any

from .clock import SystemClock
from .persistence.sqlite import SqliteEventStore


class InProcessEventPublisher:
    def __init__(self, store: SqliteEventStore | None = None, clock: SystemClock | None = None) -> None:
        self._store = store
        self._clock = clock or SystemClock()
        self._lock = threading.Lock()
        self._history: dict[str, list[dict[str, Any]]] = {}
        self._subs: dict[str, list[queue.Queue]] = {}

    def publish(self, job_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {
            "event": event_type,
            "job_id": job_id,
            "timestamp": self._clock.now_iso(),
            **(payload or {}),
        }
        raw = json.dumps(event)
        if self._store:
            self._store.append(job_id, event_type, raw, event["timestamp"])
        with self._lock:
            self._history.setdefault(job_id, []).append(event)
            for q in self._subs.get(job_id, []):
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass
        return event

    def history(self, job_id: str) -> list[dict[str, Any]]:
        if self._store:
            stored = self._store.list_for(job_id)
            if stored:
                return stored
        with self._lock:
            return list(self._history.get(job_id, []))

    def subscribe(self, job_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._subs.setdefault(job_id, []).append(q)
        return q
