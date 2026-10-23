"""P02 · Orchestrator v1: linear multi-stage routing with a single-writer dispatcher.

Requirements
- start(): one reader task per stage client (awaits client.get_output()) feeding ONE
  ready queue; one dispatcher task processes it serially; one request-handler task
  consumes add_request() messages.
- add: register ReqState in self.request_states BEFORE awaiting stages[0].submit().
- When stage s finishes a request and s < max(final_stage_ids): build the next stage's
  inputs with forward_fn(s+1, prompt, all data stage s produced) and submit to s+1.
- Outputs of stages in final_stage_ids go to the client (next_output()). Exactly one
  message per request has finished=True: the last one, sent once every final stage finished.
- After the terminal message the request state is removed. Outputs for unknown request
  ids are dropped and counted in self.dropped_outputs.
"""
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------- given types
@dataclass
class StageOutput:
    request_id: str
    data: Any
    finished: bool


@dataclass
class OrchOutput:
    request_id: str
    stage_id: int | None
    data: Any
    finished: bool
    error: str | None = None


class StageClient(Protocol):
    async def submit(self, request_id: str, inputs: Any) -> None: ...

    async def get_output(self) -> list[StageOutput]: ...

    async def abort(self, request_ids: list[str]) -> None: ...


@dataclass
class ReqState:
    request_id: str
    prompt: Any
    submitted: set[int] = field(default_factory=set)
    finished: set[int] = field(default_factory=set)
    stage_data: dict[int, list] = field(default_factory=dict)
    pending_final: OrchOutput | None = None  # used by P03


# ------------------------------------------------------------ implement below
class Orchestrator:
    def __init__(
        self,
        stages: list[StageClient],
        final_stage_ids: set[int],
        forward_fn: Callable[[int, Any, list], Any],
    ) -> None:
        self.stages = stages
        self.final_stage_ids = frozenset(final_stage_ids)
        self.last_stage = max(final_stage_ids)
        self.forward_fn = forward_fn
        self.request_states: dict[str, ReqState] = {}
        self.dropped_outputs = 0
        self._req_q: asyncio.Queue = asyncio.Queue()
        self._ready_q: asyncio.Queue = asyncio.Queue()
        self._out_q: asyncio.Queue = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        raise NotImplementedError

    async def shutdown(self) -> None:
        raise NotImplementedError

    async def add_request(self, request_id: str, prompt: Any) -> None:
        raise NotImplementedError

    async def next_output(self, timeout: float | None = None) -> OrchOutput:
        raise NotImplementedError

    # Suggested internals (P03 overrides the last two):
    #   async def _reader(self, sid, client)
    #   async def _request_handler(self)
    #   async def _dispatcher(self)
    #   async def _handle_add(self, rid, prompt)
    #   async def _handle_output(self, sid, o: StageOutput)
    #   def _cleanup(self, rid)
