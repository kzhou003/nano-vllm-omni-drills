"""P04 · StagePool (sticky replica binding) + two-phase abort + dead-replica isolation.

StagePool
- pick(rid): sticky -- a request keeps its replica; new requests round-robin over LIVE
  replicas; raise NoLiveReplica if none.
- evict(replica): mark dead, drop and return the request ids bound to it.
- abort(ids): ONE abort call per replica with the ids bound to it; ids without a binding
  are ignored; bindings are released only after that replica's abort succeeded
  (if it raises, propagate and keep the bindings so the abort can be retried).

abort_request(pools, processor, rid)  -- two phase
  1. partial = processor.collect_partial(rid)        (read only)
  2. await pool.abort([rid]) for every pool          (any failure propagates; no commit)
  3. processor.commit_abort(rid)
  return OrchOutput(rid, partial, finished=True, error="abort")

handle_dead_replica(pool, replica) -> one terminal error OrchOutput per affected request;
requests bound to other replicas are untouched.
"""
from dataclasses import dataclass
from typing import Any, Protocol


# ---------------------------------------------------------------- given types
class NoLiveReplica(Exception):
    pass


class ReplicaClient(Protocol):
    async def abort(self, request_ids: list[str]) -> None: ...


class PartialProcessor(Protocol):
    def collect_partial(self, request_id: str) -> Any: ...

    def commit_abort(self, request_id: str) -> None: ...


@dataclass
class OrchOutput:
    request_id: str
    data: Any
    finished: bool
    error: str | None = None


# ------------------------------------------------------------ implement below
class StagePool:
    def __init__(self, stage_id: int, replicas: list[ReplicaClient]) -> None:
        self.stage_id = stage_id
        self.replicas = replicas
        raise NotImplementedError

    def live_replicas(self) -> list[int]:
        raise NotImplementedError

    def pick(self, request_id: str) -> int:
        raise NotImplementedError

    def bound(self, request_id: str) -> int | None:
        raise NotImplementedError

    def release(self, request_id: str) -> None:
        raise NotImplementedError

    def evict(self, replica_id: int) -> list[str]:
        raise NotImplementedError

    async def abort(self, request_ids: list[str]) -> None:
        raise NotImplementedError


async def abort_request(pools: list[StagePool], processor: PartialProcessor, request_id: str) -> OrchOutput:
    raise NotImplementedError


def handle_dead_replica(pool: StagePool, replica_id: int) -> list[OrchOutput]:
    raise NotImplementedError
