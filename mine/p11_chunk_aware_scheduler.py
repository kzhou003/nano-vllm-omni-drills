"""P11 · Park / restore around an upstream scheduler that knows nothing about chunks.

A request with needs_chunk=True receives its input from an upstream stage. Whenever its
next chunk is not ready (source.is_ready(id) is False) it must be invisible to
BaseScheduler.schedule():
- schedule(): (a) restore parked requests that became ready, (b) park every not-ready
  needs_chunk request from base.waiting / base.running (status "WAITING_FOR_CHUNK",
  remember origin queue and previous status, call source.load_async(id) once per parking),
  (c) call base.schedule() and (d) in a `finally`, restore any parked request that is ready
  -- this must also happen when base.schedule() raises.
- Restoring puts a request back into its ORIGIN queue with its previous status, in
  arrival order among the requests restored together.
- abort(id) works at any time: parked or queued, the request disappears and is never
  scheduled or restored again.
"""
from collections import deque
from dataclasses import dataclass
from typing import Protocol


# ---------------------------------------------------------------- given types
@dataclass
class Req:
    req_id: str
    arrival: int
    needs_chunk: bool = False
    status: str = "WAITING"


class BaseScheduler:
    """Stand-in for the upstream vLLM scheduler: knows nothing about chunks."""

    def __init__(self) -> None:
        self.waiting: deque[Req] = deque()
        self.running: list[Req] = []

    def schedule(self) -> list[str]:
        for r in self.running + list(self.waiting):
            if r.status in ("WAITING_FOR_CHUNK", "ABORTED"):
                raise AssertionError(f"upstream saw {r.req_id} in state {r.status}")
        out = [r.req_id for r in self.running] + [r.req_id for r in self.waiting]
        for r in self.waiting:
            r.status = "RUNNING"
        self.running += list(self.waiting)
        self.waiting.clear()
        return out


class ChunkSource(Protocol):
    def load_async(self, req_id: str) -> None: ...

    def is_ready(self, req_id: str) -> bool: ...


# ------------------------------------------------------------ implement below
class ChunkAwareScheduler:
    def __init__(self, base: BaseScheduler, source: ChunkSource) -> None:
        self.base = base
        self.source = source
        self.parked: dict[str, tuple[Req, str, str]] = {}  # id -> (req, origin, previous status)

    def add(self, req: Req) -> None:
        self.base.waiting.append(req)

    def abort(self, req_id: str) -> None:
        raise NotImplementedError

    def schedule(self) -> list[str]:
        raise NotImplementedError
