"""P05 · RpcRouter: correlate results arriving on one shared channel with blocked callers.

- register(key) -> waiter (a queue.Queue(maxsize=1)); duplicate live key -> ValueError.
- deliver(key, result) -> True if a waiter got it, else False and self.late_results += 1.
- call(key, send, timeout): register FIRST, then send(), then wait; always unregister in
  finally; raise TimeoutError on timeout; if the result is an exception, raise it.
- fail_all(err): wake every current waiter with err and latch it: later register()/call()
  raise err immediately.
All methods are called from arbitrary threads.
"""
import queue
from collections.abc import Callable, Hashable
from typing import Any


class RpcRouter:
    def __init__(self) -> None:
        self.late_results = 0
        raise NotImplementedError

    def register(self, key: Hashable) -> queue.Queue:
        raise NotImplementedError

    def unregister(self, key: Hashable, waiter: queue.Queue) -> None:
        raise NotImplementedError

    def deliver(self, key: Hashable, result: Any) -> bool:
        raise NotImplementedError

    def fail_all(self, error: BaseException) -> None:
        raise NotImplementedError

    def call(self, key: Hashable, send: Callable[[], None], timeout: float) -> Any:
        raise NotImplementedError
