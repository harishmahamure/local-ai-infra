from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 2

_MIGRATIONS: dict[int, str] = {
    1: """
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
""",
    2: """
CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY
);
ALTER TABLE jobs ADD COLUMN status TEXT;
ALTER TABLE jobs ADD COLUMN priority TEXT;
ALTER TABLE jobs ADD COLUMN queue TEXT;
ALTER TABLE jobs ADD COLUMN operation TEXT;
ALTER TABLE jobs ADD COLUMN created_at TEXT;
ALTER TABLE jobs ADD COLUMN queued_at TEXT;
ALTER TABLE jobs ADD COLUMN next_attempt_at TEXT;
ALTER TABLE jobs ADD COLUMN batch_id TEXT;
ALTER TABLE jobs ADD COLUMN lease_owner TEXT;
ALTER TABLE jobs ADD COLUMN lease_expires_at TEXT;
ALTER TABLE jobs ADD COLUMN current_attempt TEXT;
ALTER TABLE jobs ADD COLUMN compute_node TEXT;
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_queue_status ON jobs(queue, status);
CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs(batch_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id, id);
CREATE TABLE IF NOT EXISTS job_attempts (
  attempt_id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  attempt_number INTEGER NOT NULL,
  status TEXT NOT NULL,
  payload TEXT NOT NULL,
  worker_id TEXT,
  queued_at TEXT,
  started_at TEXT,
  ended_at TEXT,
  lease_expires_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_attempts_job ON job_attempts(job_id, attempt_number);
CREATE INDEX IF NOT EXISTS idx_attempts_status ON job_attempts(status);
CREATE TABLE IF NOT EXISTS job_dependencies (
  job_id TEXT NOT NULL,
  depends_on TEXT NOT NULL,
  policy TEXT NOT NULL,
  artifact_bindings TEXT NOT NULL DEFAULT '{}',
  state TEXT NOT NULL,
  PRIMARY KEY (job_id, depends_on)
);
CREATE INDEX IF NOT EXISTS idx_deps_upstream ON job_dependencies(depends_on);
CREATE TABLE IF NOT EXISTS batches (
  batch_id TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS workers (
  worker_id TEXT PRIMARY KEY,
  role TEXT NOT NULL,
  status TEXT NOT NULL,
  payload TEXT NOT NULL,
  last_heartbeat TEXT
);
CREATE TABLE IF NOT EXISTS queue_state (
  queue TEXT PRIMARY KEY,
  paused INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT
);
CREATE TABLE IF NOT EXISTS node_state (
  node_id TEXT PRIMARY KEY,
  mode TEXT NOT NULL,
  payload TEXT NOT NULL,
  updated_at TEXT
);
""",
}


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'").fetchone()
    if row:
        ver = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()
        return int(ver[0] or 0)
    jobs = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'").fetchone()
    return 1 if jobs else 0


def apply_migrations(conn: sqlite3.Connection) -> int:
    conn.execute("PRAGMA foreign_keys=ON")
    version = current_version(conn)
    for target in range(version + 1, SCHEMA_VERSION + 1):
        sql = _MIGRATIONS[target]
        for statement in _split_sql(sql):
            try:
                conn.execute(statement)
            except sqlite3.OperationalError as exc:
                if "duplicate column" in str(exc).lower():
                    continue
                raise
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
        conn.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (?)", (target,))
        conn.commit()
    _backfill_job_columns(conn)
    return SCHEMA_VERSION


def _split_sql(sql: str) -> list[str]:
    return [part.strip() for part in sql.split(";") if part.strip()]


def _backfill_job_columns(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT job_id, payload, status FROM jobs").fetchall()
    import json

    for row in rows:
        if row["status"]:
            continue
        try:
            data = json.loads(row["payload"])
        except Exception:
            continue
        conn.execute(
            """UPDATE jobs SET status=?, priority=?, queue=?, operation=?, created_at=?, queued_at=?,
               next_attempt_at=?, batch_id=?, lease_owner=?, lease_expires_at=?, current_attempt=?, compute_node=?
               WHERE job_id=?""",
            (
                data.get("status"),
                data.get("priority") or "NORMAL",
                data.get("queue") or "gpu",
                data.get("operation"),
                data.get("created_at"),
                data.get("queued_at") or data.get("created_at"),
                data.get("next_attempt_at"),
                data.get("batch_id"),
                data.get("lease_owner"),
                data.get("lease_expires_at"),
                data.get("current_attempt"),
                data.get("compute_node") or "gpu-box",
                row["job_id"],
            ),
        )
    conn.commit()
