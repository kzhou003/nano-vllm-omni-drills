"""P01 · OutputSink: many producer threads -> one asyncio consumer.

Requirements
- put(item): callable from ANY thread, never blocks, FIFO per producer.
- At most ONE loop.call_soon_threadsafe(...) per "wake cycle": while a wakeup is
  already in flight, further puts must not schedule another. Count every scheduled
  wakeup in self.wakeups_scheduled.
- get_batch(max_items, timeout): runs on the bound loop; returns 1..max_items items,
  or [] on timeout. It must never sleep while items are available (no lost wakeups).
- Test hook: call self.after_empty_drain() right after a drain that found nothing and
  BEFORE you re-arm the wakeup. Tests use it to inject a put() from another thread into
  exactly the window where naive implementations lose the wakeup.
"""
import asyncio


class OutputSink:
    def __init__(self) -> None:
        self.wakeups_scheduled = 0
        self.after_empty_drain = lambda: None
        raise NotImplementedError

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Remember the consumer loop (default: the running loop) and create its wakeup Event."""
        raise NotImplementedError

    def put(self, item) -> None:
        raise NotImplementedError

    async def get_batch(self, max_items: int = 256, timeout: float | None = None) -> list:
        raise NotImplementedError
