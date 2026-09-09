from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..domain.jobs import Job, JobStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    phase TEXT NOT NULL,
    progress REAL NOT NULL,
    inputs TEXT NOT NULL,
    parameters TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    asset_ids TEXT NOT NULL,
    idempotency_key TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    seed INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency ON jobs(idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    created_at TEXT NOT NULL,
    path TEXT NOT NULL,
    filename TEXT
);
CREATE INDEX IF NOT EXISTS idx_assets_sha ON assets(sha256);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(SCHEMA)

    def recover_stale(self) -> int:
        """Mark RUNNING jobs as FAILED after a process restart."""
        with self._lock, self._connect() as conn:
            cur = conn.execute("SELECT job_id FROM jobs WHERE status = ?", (JobStatus.RUNNING.value,))
            ids = [row["job_id"] for row in cur.fetchall()]
            if not ids:
                return 0
            error = json.dumps(
                {
                    "code": "INTERRUPTED_BY_RESTART",
                    "message": "Job was running when the control API restarted",
                    "retryable": True,
                }
            )
            now = _now()
            conn.executemany(
                "UPDATE jobs SET status = ?, phase = ?, finished_at = ?, error = ? WHERE job_id = ?",
                [(JobStatus.FAILED.value, "interrupted", now, error, job_id) for job_id in ids],
            )
            return len(ids)

    def save(self, job: Job) -> None:
        payload = (
            job.job_id,
            job.operation,
            job.status.value,
            job.phase,
            job.progress,
            json.dumps(job.inputs),
            json.dumps(job.parameters),
            job.created_at,
            job.started_at,
            job.finished_at,
            json.dumps(job.error) if job.error else None,
            json.dumps(job.asset_ids),
            job.idempotency_key,
            1 if job.cancel_requested else 0,
            job.seed,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, operation, status, phase, progress, inputs, parameters,
                    created_at, started_at, finished_at, error, asset_ids,
                    idempotency_key, cancel_requested, seed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    operation=excluded.operation,
                    status=excluded.status,
                    phase=excluded.phase,
                    progress=excluded.progress,
                    inputs=excluded.inputs,
                    parameters=excluded.parameters,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    error=excluded.error,
                    asset_ids=excluded.asset_ids,
                    cancel_requested=excluded.cancel_requested,
                    seed=excluded.seed
                """,
                payload,
            )

    def get(self, job_id: str) -> Job | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def get_by_idempotency(self, key: str) -> Job | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE idempotency_key = ?", (key,)).fetchone()
        return _row_to_job(row) if row else None

    def claim_next(self) -> Job | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1",
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if not row:
                return None
            job = _row_to_job(row)
            if job.cancel_requested:
                now = _now()
                conn.execute(
                    "UPDATE jobs SET status = ?, phase = ?, finished_at = ? WHERE job_id = ?",
                    (JobStatus.CANCELLED.value, "cancelled", now, job.job_id),
                )
                job.status = JobStatus.CANCELLED
                job.phase = "cancelled"
                job.finished_at = now
                return job
            now = _now()
            conn.execute(
                "UPDATE jobs SET status = ?, phase = ?, started_at = ? WHERE job_id = ? AND status = ?",
                (JobStatus.RUNNING.value, "waiting_for_gpu", now, job.job_id, JobStatus.QUEUED.value),
            )
            job.status = JobStatus.RUNNING
            job.phase = "waiting_for_gpu"
            job.started_at = now
            return job

    def queued_count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status = ?",
                (JobStatus.QUEUED.value,),
            ).fetchone()
        return int(row["n"] if row else 0)

    def queue_position(self, job_id: str) -> int | None:
        job = self.get(job_id)
        if job is None or job.status != JobStatus.QUEUED:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status = ? AND created_at <= ?",
                (JobStatus.QUEUED.value, job.created_at),
            ).fetchone()
        return int(row["n"] if row else 1)

    def list_jobs(
        self,
        *,
        status: str | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> tuple[list[Job], str | None]:
        limit = max(1, min(100, int(limit)))
        sql = "SELECT * FROM jobs"
        args: list[Any] = []
        clauses: list[str] = []
        if status:
            clauses.append("status = ?")
            args.append(status)
        if cursor:
            clauses.append("created_at < ?")
            args.append(cursor)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit + 1)
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        jobs = [_row_to_job(row) for row in rows[:limit]]
        next_cursor = rows[limit]["created_at"] if len(rows) > limit else None
        return jobs, next_cursor

    def list_all_jobs(self, *, status: str | None = None) -> list[Job]:
        sql = "SELECT * FROM jobs"
        args: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            args.append(status)
        sql += " ORDER BY created_at DESC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [_row_to_job(row) for row in rows]

    def list_all_assets(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM assets ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def save_asset(self, record: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO assets (asset_id, sha256, mime_type, size_bytes, width, height, created_at, path, filename)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    sha256=excluded.sha256,
                    mime_type=excluded.mime_type,
                    size_bytes=excluded.size_bytes,
                    width=excluded.width,
                    height=excluded.height,
                    path=excluded.path,
                    filename=excluded.filename
                """,
                (
                    record["asset_id"],
                    record["sha256"],
                    record["mime_type"],
                    record["size_bytes"],
                    record.get("width"),
                    record.get("height"),
                    record["created_at"],
                    record["path"],
                    record.get("filename"),
                ),
            )

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,)).fetchone()
        if not row:
            return None
        return dict(row)

    def delete_job(self, job_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            return cur.rowcount > 0

    def count_assets_with_path(self, path: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM assets WHERE path = ?", (path,)).fetchone()
        return int(row["n"] if row else 0)

    def delete_asset_record(self, asset_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,)).fetchone()
            if not row:
                return None
            record = dict(row)
            conn.execute("DELETE FROM assets WHERE asset_id = ?", (asset_id,))
        return record

    def unref_asset(self, asset_id: str) -> None:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs WHERE instr(asset_ids, ?) > 0", (asset_id,)).fetchall()
        for row in rows:
            job = _row_to_job(row)
            if asset_id not in job.asset_ids:
                continue
            job.asset_ids = [item for item in job.asset_ids if item != asset_id]
            self.save(job)


def _row_to_job(row: sqlite3.Row) -> Job:
    error_raw = row["error"]
    return Job(
        job_id=row["job_id"],
        operation=row["operation"],
        status=JobStatus(row["status"]),
        phase=row["phase"],
        progress=float(row["progress"] or 0),
        inputs=json.loads(row["inputs"] or "{}"),
        parameters=json.loads(row["parameters"] or "{}"),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        error=json.loads(error_raw) if error_raw else None,
        asset_ids=json.loads(row["asset_ids"] or "[]"),
        idempotency_key=row["idempotency_key"],
        cancel_requested=bool(row["cancel_requested"]),
        seed=row["seed"],
    )
