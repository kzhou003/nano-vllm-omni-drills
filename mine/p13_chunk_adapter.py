"""P13 · ChunkAdapter: the async-chunk data plane of one stage (sender + receiver halves).

Sender (scheduler thread calls save_async; a background SAVE thread does the work)
- save_async(req, snapshot, is_final) returns immediately; it must snapshot (deep-copy)
  its argument because the caller keeps mutating the live request.
- The save thread calls build_payload(snapshot) OUTSIDE any lock; None means "nothing to
  send this step" (skip, unless is_final). It puts {"data": ..., "finished": is_final} at
  key f"{req}_{stage_id}_{chunk_id}"; chunk_id starts at 0 and advances only after a
  successful put, so ids stay contiguous.
- cancel_sender(req): later non-final chunks are dropped, but the final chunk is still
  sent (with data=None) because it is the downstream end-of-stream marker.
Receiver (background RECV thread polls the connector)
- load_async(req): start polling f"{req}_{src_stage_id}_{next_id}"; the get happens
  OUTSIDE your locks; back off (~1 ms) when nothing arrived, sleep on a Condition when idle.
- poll_ready() (scheduler thread): {req: [chunk dicts in order]} committed since last call.
- cleanup_receiver(req): from now on nothing for req appears in poll_ready -- including a
  chunk the recv thread is reading at this very moment (re-check a per-registration token
  under the lock before committing).
- start()/stop() manage both threads; stop() must return promptly.
"""
import copy
import threading
from collections import deque
from collections.abc import Callable
from typing import Any


# ---------------------------------------------------------------- given types
class InMemoryConnector:
    """Transport only: put/get by key, get consumes. Thread-safe. Hooks let tests add latency."""

    def __init__(self, on_put: Callable[[str], None] | None = None, on_get: Callable[[str], None] | None = None):
        self._store: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.on_put, self.on_get = on_put, on_get
        self.puts: list[str] = []

    def put(self, key: str, obj: Any) -> None:
        if self.on_put:
            self.on_put(key)
        with self._lock:
            self._store[key] = obj
            self.puts.append(key)

    def get(self, key: str) -> Any | None:
        with self._lock:
            present = key in self._store
        if not present:
            return None
        if self.on_get:
            self.on_get(key)
        with self._lock:
            return self._store.pop(key, None)


# ------------------------------------------------------------ implement below
class ChunkAdapter:
    def __init__(
        self,
        connector: InMemoryConnector,
        stage_id: int,
        src_stage_id: int | None,
        build_payload: Callable[[Any], Any | None],
    ) -> None:
        self.connector = connector
        self.stage_id = stage_id
        self.src_stage_id = src_stage_id
        self.build_payload = build_payload
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def save_async(self, req_id: str, snapshot: Any, is_final: bool) -> None:
        raise NotImplementedError

    def cancel_sender(self, req_id: str) -> None:
        raise NotImplementedError

    def load_async(self, req_id: str) -> None:
        raise NotImplementedError

    def poll_ready(self) -> dict[str, list]:
        raise NotImplementedError

    def cleanup_receiver(self, req_id: str) -> None:
        raise NotImplementedError
