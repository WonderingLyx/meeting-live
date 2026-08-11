"""Small in-process download progress registry."""
from __future__ import annotations

import threading
import time
from typing import Any

_lock = threading.Lock()
_states: dict[str, dict[str, Any]] = {}


def start_progress(key: str, *, label: str | None = None, total_bytes: int | None = None) -> None:
    now = time.time()
    with _lock:
        _states[key] = {
            "key": key,
            "label": label,
            "status": "downloading",
            "downloaded_bytes": 0,
            "total_bytes": total_bytes,
            "percent": 0.0 if total_bytes else None,
            "started_at": now,
            "updated_at": now,
        }


def update_progress(
    key: str,
    *,
    downloaded_bytes: int | None = None,
    total_bytes: int | None = None,
    status: str | None = None,
    message: str | None = None,
) -> None:
    now = time.time()
    with _lock:
        state = _states.setdefault(
            key,
            {
                "key": key,
                "status": "downloading",
                "downloaded_bytes": 0,
                "total_bytes": total_bytes,
                "percent": None,
                "started_at": now,
            },
        )
        if total_bytes:
            state["total_bytes"] = int(total_bytes)
        if downloaded_bytes is not None:
            state["downloaded_bytes"] = max(
                int(state.get("downloaded_bytes") or 0),
                int(downloaded_bytes),
            )
        if status:
            state["status"] = status
        if message:
            state["message"] = message
        total = state.get("total_bytes")
        downloaded = state.get("downloaded_bytes") or 0
        state["percent"] = (
            min(100.0, round(downloaded / total * 100, 1))
            if isinstance(total, int) and total > 0
            else None
        )
        state["updated_at"] = now


def finish_progress(key: str) -> None:
    with _lock:
        state = _states.get(key)
        if not state:
            return
        total = state.get("total_bytes")
        if isinstance(total, int) and total > 0:
            state["downloaded_bytes"] = total
            state["percent"] = 100.0
        state["status"] = "ready"
        state["updated_at"] = time.time()


def fail_progress(key: str, error: str) -> None:
    with _lock:
        state = _states.setdefault(key, {"key": key})
        state["status"] = "error"
        state["error"] = error
        state["updated_at"] = time.time()


def get_progress(key: str) -> dict[str, Any] | None:
    with _lock:
        state = _states.get(key)
        return dict(state) if state else None
