from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from ...domain.artifacts import Artifact
from ...domain.jobs import Job
from .codec import artifact_from_dict, artifact_to_dict, dumps, job_from_dict, job_to_dict, loads

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
  asset_id TEXT PRIMARY KEY,
  payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS idempotency (
  key TEXT PRIMARY KEY,
  request_hash TEXT NOT NULL,
  job_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS runtime_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  payload TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


class SqliteStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn


class SqliteJobRepository:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def save(self, job: Job) -> None:
        payload = dumps(job_to_dict(job))
        with self._store._lock, self._store._connect() as conn:
            conn.execute(
                "INSERT INTO jobs(job_id, payload) VALUES(?, ?) ON CONFLICT(job_id) DO UPDATE SET payload=excluded.payload",
                (job.job_id, payload),
            )
            conn.commit()

    def get(self, job_id: str) -> Job | None:
        with self._store._lock, self._store._connect() as conn:
            row = conn.execute("SELECT payload FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return job_from_dict(loads(row["payload"])) if row else None

    def list_active(self) -> list[Job]:
        with self._store._lock, self._store._connect() as conn:
            rows = conn.execute("SELECT payload FROM jobs").fetchall()
        jobs: list[Job] = []
        for row in rows:
            job = job_from_dict(loads(row["payload"]))
            if not job.status.terminal:
                jobs.append(job)
        return jobs


class SqliteArtifactRepository:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def save(self, artifact: Artifact) -> None:
        payload = dumps(artifact_to_dict(artifact))
        with self._store._lock, self._store._connect() as conn:
            conn.execute(
                "INSERT INTO artifacts(asset_id, payload) VALUES(?, ?) ON CONFLICT(asset_id) DO UPDATE SET payload=excluded.payload",
                (artifact.asset_id, payload),
            )
            conn.commit()

    def get(self, asset_id: str) -> Artifact | None:
        with self._store._lock, self._store._connect() as conn:
            row = conn.execute("SELECT payload FROM artifacts WHERE asset_id=?", (asset_id,)).fetchone()
        return artifact_from_dict(loads(row["payload"])) if row else None

    def delete(self, asset_id: str) -> None:
        artifact = self.get(asset_id)
        if artifact:
            artifact.deleted = True
            self.save(artifact)


class SqliteIdempotencyRepository:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def get(self, key: str) -> tuple[str, str] | None:
        with self._store._lock, self._store._connect() as conn:
            row = conn.execute("SELECT request_hash, job_id FROM idempotency WHERE key=?", (key,)).fetchone()
        return (row["request_hash"], row["job_id"]) if row else None

    def put(self, key: str, request_hash: str, job_id: str) -> None:
        with self._store._lock, self._store._connect() as conn:
            conn.execute(
                "INSERT INTO idempotency(key, request_hash, job_id) VALUES(?, ?, ?) ON CONFLICT(key) DO NOTHING",
                (key, request_hash, job_id),
            )
            conn.commit()


class SqliteRuntimeStateRepository:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def get(self, key: str) -> str | None:
        with self._store._lock, self._store._connect() as conn:
            row = conn.execute("SELECT value FROM runtime_state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def put(self, key: str, value: str) -> None:
        with self._store._lock, self._store._connect() as conn:
            conn.execute(
                "INSERT INTO runtime_state(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            conn.commit()


class SqliteEventStore:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def append(self, job_id: str, event_type: str, payload: str, created_at: str) -> None:
        with self._store._lock, self._store._connect() as conn:
            conn.execute(
                "INSERT INTO job_events(job_id, event_type, payload, created_at) VALUES(?,?,?,?)",
                (job_id, event_type, payload, created_at),
            )
            conn.commit()

    def list_for(self, job_id: str) -> list[dict]:
        with self._store._lock, self._store._connect() as conn:
            rows = conn.execute(
                "SELECT event_type, payload, created_at FROM job_events WHERE job_id=? ORDER BY id",
                (job_id,),
            ).fetchall()
        return [loads(row["payload"]) for row in rows]
