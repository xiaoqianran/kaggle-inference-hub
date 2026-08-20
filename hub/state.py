from __future__ import annotations

import itertools
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .config import HISTORY_LIMIT, LEASE_SECONDS, MAX_ATTEMPTS, MODEL_SPECS, QUEUE_SIZE, WORKER_TTL_SECONDS


@dataclass
class Lease:
    task: dict[str, Any]
    worker_id: str
    claimed_at: float


class HubState:
    def __init__(self) -> None:
        self.queues = {model: queue.Queue(maxsize=QUEUE_SIZE) for model in MODEL_SPECS}
        self.ids = itertools.count(1)
        self.inflight: dict[int, Lease] = {}
        self.failed: deque[dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)
        self.history: deque[dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)
        self.workers: dict[str, dict[str, Any]] = {}
        self.clients: set[Any] = set()
        self.lock = threading.RLock()

    def next_id(self) -> int:
        return next(self.ids)

    def enqueue(self, task: dict[str, Any], *, block: bool = False) -> None:
        q = self.queues[task["model"]]
        if block:
            q.put(task)
        else:
            q.put_nowait(task)

    def requeue_expired(self) -> int:
        now = time.time()
        expired: list[Lease] = []
        with self.lock:
            for task_id, lease in list(self.inflight.items()):
                if now - lease.claimed_at >= LEASE_SECONDS:
                    expired.append(self.inflight.pop(task_id))
        count = 0
        for lease in expired:
            task = lease.task
            task["last_error"] = f"lease expired from worker {lease.worker_id}"
            if int(task.get("attempt", 0)) >= MAX_ATTEMPTS:
                with self.lock:
                    self.failed.append({**task, "error": task["last_error"], "failed_at": now})
                continue
            try:
                self.enqueue(task)
                count += 1
            except queue.Full:
                with self.lock:
                    self.failed.append({**task, "error": "queue full while requeueing expired lease", "failed_at": now})
        return count

    def claim(self, model: str, worker_id: str, timeout: float = 25) -> dict[str, Any] | None:
        self.requeue_expired()
        try:
            task = self.queues[model].get(True, timeout)
        except queue.Empty:
            return None
        task["attempt"] = int(task.get("attempt", 0)) + 1
        with self.lock:
            self.inflight[task["id"]] = Lease(task=task, worker_id=worker_id, claimed_at=time.time())
        return task

    def lease_owner(self, task_id: int) -> str | None:
        with self.lock:
            lease = self.inflight.get(task_id)
            return lease.worker_id if lease else None

    def lease_task(self, task_id: int) -> dict[str, Any] | None:
        with self.lock:
            lease = self.inflight.get(task_id)
            return dict(lease.task) if lease else None

    def complete(self, task_id: int) -> None:
        with self.lock:
            self.inflight.pop(task_id, None)

    def fail(self, task_id: int, error: str, requeue: bool = True) -> tuple[bool, dict[str, Any] | None]:
        with self.lock:
            lease = self.inflight.pop(task_id, None)
        if lease is None:
            return False, None
        task = lease.task
        task["last_error"] = error
        if requeue and int(task.get("attempt", 0)) < MAX_ATTEMPTS:
            self.enqueue(task, block=False)
            return True, task
        with self.lock:
            self.failed.append({**task, "error": error, "failed_at": time.time()})
        return False, task

    def register_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        item = {**payload, "registered_at": payload.get("registered_at", now), "last_seen": now}
        with self.lock:
            old = self.workers.get(payload["worker_id"], {})
            item["registered_at"] = old.get("registered_at", item["registered_at"])
            self.workers[payload["worker_id"]] = item
        return item

    def heartbeat(self, worker_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        with self.lock:
            item = self.workers.get(worker_id)
            if item is None:
                return None
            item.update(updates)
            item["last_seen"] = time.time()
            return dict(item)

    def snapshot(self) -> dict[str, Any]:
        self.requeue_expired()
        now = time.time()
        with self.lock:
            workers = []
            for item in self.workers.values():
                x = dict(item)
                x["online"] = now - float(x.get("last_seen", 0)) <= WORKER_TTL_SECONDS
                workers.append(x)
            inflight_by_model = {m: 0 for m in MODEL_SPECS}
            for lease in self.inflight.values():
                inflight_by_model[lease.task["model"]] += 1
            queued_by_model = {m: q.qsize() for m, q in self.queues.items()}
            images = sum(1 for item in self.history if item.get("kind", "image") == "image")
            artifacts = sum(1 for item in self.history if item.get("kind") == "artifact")
            return {
                "queued": sum(queued_by_model.values()),
                "queued_by_model": queued_by_model,
                "inflight": len(self.inflight),
                "inflight_by_model": inflight_by_model,
                "results": len(self.history),
                "images": images,
                "artifacts": artifacts,
                "failed": len(self.failed),
                "workers": workers,
            }


state = HubState()
