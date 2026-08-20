from __future__ import annotations

import json
import os
import queue
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from .config import (
    HISTORY_LIMIT,
    LEASE_SECONDS,
    MAX_ATTEMPTS,
    MODEL_SPECS,
    QUEUE_SIZE,
    STATE_DB,
    WORKER_TTL_SECONDS,
)


class ClosingConnection(sqlite3.Connection):
    """Commit/rollback like sqlite3.Connection and always release the file handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class HubState:
    """Process-safe, durable Hub state backed by SQLite."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or STATE_DB).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.clients: set[Any] = set()  # WebSockets are necessarily process-local.
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30,
            isolation_level=None,
            factory=ClosingConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_ids (
                    id INTEGER PRIMARY KEY AUTOINCREMENT
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('queued', 'inflight')),
                    payload TEXT NOT NULL,
                    worker_id TEXT,
                    claimed_at REAL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS tasks_model_status_id
                    ON tasks(model, status, id);
                CREATE INDEX IF NOT EXISTS tasks_inflight_claimed
                    ON tasks(status, claimed_at);
                CREATE TABLE IF NOT EXISTS failed (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    failed_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS history (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS history_model_event
                    ON history(model, event_id);
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    registered_at REAL NOT NULL,
                    last_seen REAL NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES ('hub_instance_id', ?)",
                (uuid.uuid4().hex,),
            )

    @property
    def instance_id(self) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'hub_instance_id'"
            ).fetchone()
        return str(row["value"])

    @property
    def storage_label(self) -> str:
        return "sqlite"

    def next_id(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute("INSERT INTO task_ids DEFAULT VALUES")
            return int(cursor.lastrowid)

    @staticmethod
    def _dump(item: dict[str, Any]) -> str:
        return json.dumps(item, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _load(value: str) -> dict[str, Any]:
        return json.loads(value)

    def enqueue(self, task: dict[str, Any], *, block: bool = False) -> None:
        del block  # Compatibility with the previous in-memory API.
        self.enqueue_many([task])

    def enqueue_many(self, tasks: Iterable[dict[str, Any]]) -> None:
        items = list(tasks)
        if not items:
            return
        counts: dict[str, int] = {}
        for task in items:
            model = str(task["model"])
            if model not in MODEL_SPECS:
                raise KeyError(model)
            counts[model] = counts.get(model, 0) + 1

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for model, added in counts.items():
                    queued = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM tasks WHERE model = ? AND status = 'queued'",
                            (model,),
                        ).fetchone()[0]
                    )
                    if QUEUE_SIZE and queued + added > QUEUE_SIZE:
                        raise queue.Full
                connection.executemany(
                    """
                    INSERT INTO tasks(id, model, status, payload, created_at)
                    VALUES (?, ?, 'queued', ?, ?)
                    """,
                    [
                        (
                            int(task["id"]),
                            str(task["model"]),
                            self._dump(task),
                            float(task.get("created_at", time.time())),
                        )
                        for task in items
                    ],
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def queue_size(self, model: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM tasks WHERE model = ? AND status = 'queued'",
                    (model,),
                ).fetchone()[0]
            )

    def has_queue_capacity(self, model: str, count: int) -> bool:
        return not QUEUE_SIZE or self.queue_size(model) + count <= QUEUE_SIZE

    def _prune_locked(self, connection: sqlite3.Connection, table: str) -> None:
        connection.execute(
            f"DELETE FROM {table} WHERE event_id NOT IN "
            f"(SELECT event_id FROM {table} ORDER BY event_id DESC LIMIT ?)",
            (HISTORY_LIMIT,),
        )

    def _requeue_expired_locked(self, connection: sqlite3.Connection, now: float) -> int:
        rows = connection.execute(
            """
            SELECT id, payload, worker_id FROM tasks
            WHERE status = 'inflight' AND claimed_at <= ?
            """,
            (now - LEASE_SECONDS,),
        ).fetchall()
        requeued = 0
        for row in rows:
            task = self._load(row["payload"])
            error = f"lease expired from worker {row['worker_id']}"
            task["last_error"] = error
            if int(task.get("attempt", 0)) >= MAX_ATTEMPTS:
                failed = {**task, "error": error, "failed_at": now}
                connection.execute(
                    "INSERT INTO failed(task_id, payload, failed_at) VALUES (?, ?, ?)",
                    (int(row["id"]), self._dump(failed), now),
                )
                connection.execute("DELETE FROM tasks WHERE id = ?", (int(row["id"]),))
            else:
                connection.execute(
                    """
                    UPDATE tasks
                    SET status = 'queued', payload = ?, worker_id = NULL, claimed_at = NULL
                    WHERE id = ?
                    """,
                    (self._dump(task), int(row["id"])),
                )
                requeued += 1
        self._prune_locked(connection, "failed")
        return requeued

    def requeue_expired(self) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                count = self._requeue_expired_locked(connection, time.time())
                connection.commit()
                return count
            except Exception:
                connection.rollback()
                raise

    def _claim_once(self, model: str, worker_id: str) -> dict[str, Any] | None:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._requeue_expired_locked(connection, now)
                row = connection.execute(
                    """
                    SELECT id, payload FROM tasks
                    WHERE model = ? AND status = 'queued'
                    ORDER BY id LIMIT 1
                    """,
                    (model,),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                task = self._load(row["payload"])
                task["attempt"] = int(task.get("attempt", 0)) + 1
                cursor = connection.execute(
                    """
                    UPDATE tasks
                    SET status = 'inflight', payload = ?, worker_id = ?, claimed_at = ?
                    WHERE id = ? AND status = 'queued'
                    """,
                    (self._dump(task), worker_id, now, int(row["id"])),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    return None
                connection.commit()
                return task
            except Exception:
                connection.rollback()
                raise

    def claim(self, model: str, worker_id: str, timeout: float = 25) -> dict[str, Any] | None:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            task = self._claim_once(model, worker_id)
            if task is not None:
                return task
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.2, remaining))

    def _lease_row(self, task_id: int) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT payload, worker_id FROM tasks WHERE id = ? AND status = 'inflight'",
                (task_id,),
            ).fetchone()

    def lease_owner(self, task_id: int) -> str | None:
        row = self._lease_row(task_id)
        return str(row["worker_id"]) if row else None

    def lease_task(self, task_id: int) -> dict[str, Any] | None:
        row = self._lease_row(task_id)
        return self._load(row["payload"]) if row else None

    def complete(self, task_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

    def renew_lease(self, task_id: int, worker_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks SET claimed_at = ?
                WHERE id = ? AND status = 'inflight' AND worker_id = ?
                """,
                (time.time(), task_id, worker_id),
            )
        return cursor.rowcount == 1

    def fail(self, task_id: int, error: str, requeue: bool = True) -> tuple[bool, dict[str, Any] | None]:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT payload FROM tasks WHERE id = ? AND status = 'inflight'",
                    (task_id,),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return False, None
                task = self._load(row["payload"])
                task["last_error"] = error
                should_requeue = requeue and int(task.get("attempt", 0)) < MAX_ATTEMPTS
                if should_requeue:
                    connection.execute(
                        """
                        UPDATE tasks
                        SET status = 'queued', payload = ?, worker_id = NULL, claimed_at = NULL
                        WHERE id = ?
                        """,
                        (self._dump(task), task_id),
                    )
                else:
                    failed = {**task, "error": error, "failed_at": now}
                    connection.execute(
                        "INSERT INTO failed(task_id, payload, failed_at) VALUES (?, ?, ?)",
                        (task_id, self._dump(failed), now),
                    )
                    connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                    self._prune_locked(connection, "failed")
                connection.commit()
                return should_requeue, task
            except Exception:
                connection.rollback()
                raise

    def register_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        worker_id = str(payload["worker_id"])
        with self._connect() as connection:
            old = connection.execute(
                "SELECT registered_at FROM workers WHERE worker_id = ?", (worker_id,)
            ).fetchone()
            registered_at = float(old["registered_at"]) if old else float(payload.get("registered_at", now))
            item = {**payload, "registered_at": registered_at, "last_seen": now}
            connection.execute(
                """
                INSERT INTO workers(worker_id, payload, registered_at, last_seen)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    payload = excluded.payload,
                    registered_at = excluded.registered_at,
                    last_seen = excluded.last_seen
                """,
                (worker_id, self._dump(item), registered_at, now),
            )
        return item

    def heartbeat(self, worker_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        now = time.time()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload, registered_at FROM workers WHERE worker_id = ?", (worker_id,)
            ).fetchone()
            if row is None:
                return None
            item = self._load(row["payload"])
            item.update(updates)
            item["registered_at"] = float(row["registered_at"])
            item["last_seen"] = now
            connection.execute(
                "UPDATE workers SET payload = ?, last_seen = ? WHERE worker_id = ?",
                (self._dump(item), now, worker_id),
            )
        return item

    def record_history(self, item: dict[str, Any]) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO history(task_id, model, payload, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (int(item["id"]), str(item["model"]), self._dump(item), float(item.get("time", time.time()))),
                )
                result = {**item, "event_id": int(cursor.lastrowid)}
                connection.execute(
                    "UPDATE history SET payload = ? WHERE event_id = ?",
                    (self._dump(result), int(cursor.lastrowid)),
                )
                self._prune_locked(connection, "history")
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    def finish(self, task_id: int, item: dict[str, Any]) -> dict[str, Any]:
        """Atomically remove an inflight task and publish its history item."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                exists = connection.execute(
                    "SELECT 1 FROM tasks WHERE id = ? AND status = 'inflight'", (task_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(task_id)
                cursor = connection.execute(
                    """
                    INSERT INTO history(task_id, model, payload, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (task_id, str(item["model"]), self._dump(item), float(item.get("time", time.time()))),
                )
                result = {**item, "event_id": int(cursor.lastrowid)}
                connection.execute(
                    "UPDATE history SET payload = ? WHERE event_id = ?",
                    (self._dump(result), int(cursor.lastrowid)),
                )
                connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                self._prune_locked(connection, "history")
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    def history_items(self, model: str | None = None, after: int = 0, limit: int = 300) -> list[dict[str, Any]]:
        clauses = ["event_id > ?"]
        params: list[Any] = [after]
        if model:
            clauses.append("model = ?")
            params.append(model)
        params.append(max(1, min(limit, HISTORY_LIMIT)))
        with self._connect() as connection:
            if after:
                rows = connection.execute(
                    f"SELECT payload FROM history WHERE {' AND '.join(clauses)} "
                    "ORDER BY event_id ASC LIMIT ?",
                    params,
                ).fetchall()
                return [self._load(row["payload"]) for row in rows]
            rows = connection.execute(
                f"SELECT payload FROM history WHERE {' AND '.join(clauses)} "
                "ORDER BY event_id DESC LIMIT ?",
                params,
            ).fetchall()
            return [self._load(row["payload"]) for row in reversed(rows)]

    def delete_history_items(self, event_ids: Iterable[int]) -> list[dict[str, Any]]:
        """Delete history rows and return their payloads for associated file cleanup."""
        ids = sorted({int(event_id) for event_id in event_ids})
        if not ids:
            return []

        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    f"SELECT event_id, payload FROM history WHERE event_id IN ({placeholders})",
                    ids,
                ).fetchall()
                connection.execute(
                    f"DELETE FROM history WHERE event_id IN ({placeholders})",
                    ids,
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return [self._load(row["payload"]) for row in rows]

    def failed_items(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM failed ORDER BY event_id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._load(row["payload"]) for row in reversed(rows)]

    def snapshot(self) -> dict[str, Any]:
        self.requeue_expired()
        now = time.time()
        with self._connect() as connection:
            queued_by_model = {model: 0 for model in MODEL_SPECS}
            inflight_by_model = {model: 0 for model in MODEL_SPECS}
            for row in connection.execute(
                "SELECT model, status, COUNT(*) AS count FROM tasks GROUP BY model, status"
            ):
                target = queued_by_model if row["status"] == "queued" else inflight_by_model
                target[str(row["model"])] = int(row["count"])
            workers = []
            for row in connection.execute("SELECT payload, last_seen FROM workers ORDER BY last_seen DESC"):
                item = self._load(row["payload"])
                item["online"] = now - float(row["last_seen"]) <= WORKER_TTL_SECONDS
                workers.append(item)
            history_rows = connection.execute(
                "SELECT payload FROM history ORDER BY event_id DESC LIMIT ?", (HISTORY_LIMIT,)
            ).fetchall()
            history = [self._load(row["payload"]) for row in history_rows]
            failed_count = int(connection.execute("SELECT COUNT(*) FROM failed").fetchone()[0])
        return {
            "hub_instance_id": self.instance_id,
            "process_id": os.getpid(),
            "storage": self.storage_label,
            "queued": sum(queued_by_model.values()),
            "queued_by_model": queued_by_model,
            "inflight": sum(inflight_by_model.values()),
            "inflight_by_model": inflight_by_model,
            "results": len(history),
            "images": sum(1 for item in history if item.get("kind", "image") == "image"),
            "artifacts": sum(1 for item in history if item.get("kind") == "artifact"),
            "failed": failed_count,
            "workers": workers,
        }


state = HubState()
