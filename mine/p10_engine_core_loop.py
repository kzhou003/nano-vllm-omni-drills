"""P10 · The EngineCore busy loop (what runs inside every stage process).

IO threads (not part of this drill) push Msg objects into self.input_queue and ship
whatever you pass to emit(). Implement run_busy_loop() (it runs on its own thread):
- While the engine has no work, BLOCK on input_queue (no polling / spinning).
- Before every step, drain all queued messages in FIFO order:
    ADD     payload (req_id, data)          -> engine.add
    ABORT   payload [req_ids]               -> engine.abort
    UTILITY payload (call_id, name, args)   -> emit(("utility", call_id, result, None)),
            or emit(("utility", call_id, None, repr(exc))) if it raised (engine keeps running)
    WAKEUP  no-op (only unblocks the queue)
- If the engine has work: for out in engine.step(): emit(("output", out)).
- request_shutdown() may be called from any thread / a signal handler: the loop must exit
  promptly even when blocked, then emit(("shutdown",)).
- An exception from engine.step() is fatal: emit(("fatal", repr(exc))) and return.
"""
import queue
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol


# ---------------------------------------------------------------- given types
@dataclass
class Msg:
    kind: Literal["ADD", "ABORT", "UTILITY", "WAKEUP"]
    payload: Any = None


class Engine(Protocol):
    def add(self, req_id: str, data: Any) -> None: ...

    def abort(self, req_ids: list[str]) -> None: ...

    def has_work(self) -> bool: ...

    def step(self) -> list[Any]: ...

    def utility(self, name: str, args: tuple) -> Any: ...


# ------------------------------------------------------------ implement below
class EngineCoreLoop:
    def __init__(self, engine: Engine, emit: Callable[[tuple], None]) -> None:
        self.engine = engine
        self.emit = emit
        self.input_queue: queue.Queue[Msg] = queue.Queue()

    def request_shutdown(self) -> None:
        raise NotImplementedError

    def run_busy_loop(self) -> None:
        raise NotImplementedError
