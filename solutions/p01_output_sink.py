"""P01 reference solution · OutputSink：多个生产者线程 -> 一个 asyncio 消费者，合并唤醒且不丢唤醒。

核心难点：
  1. 跨线程唤醒 event loop 只能用 loop.call_soon_threadsafe（asyncio.Event 不是线程安全的）；
  2. 一个“唤醒周期”内最多调度一次 call_soon_threadsafe，同时消费者绝不能在有数据时睡下去。
不变量：“队列非空 ⇒ 要么消费者醒着，要么有一次唤醒在途”。提示页：hints/p01_overview.html（A-11）。
"""
import asyncio
import threading
from collections import deque


# 状态分工：
#   _items           数据本身；deque 的 append / popleft 在 CPython 下是原子的，不加锁
#   _wake_in_flight  “唤醒在途”标志（门上的小旗）：True 表示消费者之后一定会再 drain，put 无需再唤醒
#   _lock            只保护 _wake_in_flight + wakeups_scheduled 这一对状态的“检查并设置”
class OutputSink:
    def __init__(self) -> None:
        self.wakeups_scheduled = 0             # 唤醒周期数 = call_soon_threadsafe 的调用次数
        self.after_empty_drain = lambda: None  # 测试钩子：空 drain 之后、重新武装之前调用
        self._items: deque = deque()
        self._lock = threading.Lock()
        self._wake_in_flight = False           # 初始 False：第一次 put 一定会调度唤醒
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Remember the consumer loop (default: the running loop) and create its wakeup Event."""
        # 在消费者所在的 loop 上调用；Event 必须属于这个 loop，之后只允许 loop 线程直接操作它
        self._loop = loop or asyncio.get_running_loop()
        self._event = asyncio.Event()

    # 任意生产者线程调用；永不阻塞（锁只包住几条赋值）。
    def put(self, item) -> None:
        # 易错：必须先 append 再看标志。这样任何人看到“标志已被置起”时，数据一定已在队列里：
        #  - 看到 True：消费者还没重新武装，它之后的 drain（含武装后那次）一定能看到 item
        #  - 看到 False：由本次 put 负责调度唤醒
        # 反过来（先看标志再 append）：看到 False 调度唤醒 -> 消费者醒来 drain 为空 -> 武装、睡下
        # -> 这时才 append 且不再唤醒，数据卡住（lost wakeup）
        self._items.append(item)
        with self._lock:
            if self._wake_in_flight:
                return                         # 唤醒已在途：合并，不再调度
            self._wake_in_flight = True
            self.wakeups_scheduled += 1
        # 锁外调用，缩短持锁时间。易错：不能在生产者线程里直接 self._event.set()——
        # asyncio.Event 不是线程安全的，也不会唤醒阻塞在 selector 上的 loop；
        # call_soon_threadsafe 会写 self-pipe 叫醒 loop，再在 loop 线程里执行 event.set
        self._loop.call_soon_threadsafe(self._event.set)

    # 只在消费者（loop 线程）里调用；单消费者，最多取 max_items 条，剩下的留给下次
    def _drain(self, max_items: int) -> list:
        out = []
        while len(out) < max_items:
            try:
                out.append(self._items.popleft())
            except IndexError:
                break
        return out

    # 只在绑定的 loop 上运行。返回 1..max_items 条；只有超时才返回 []。
    async def get_batch(self, max_items: int = 256, timeout: float | None = None) -> list:
        # 入口算一次绝对 deadline：循环可能转好几圈（伪唤醒），总等待时长不会被重置
        deadline = None if timeout is None else self._loop.time() + timeout
        while True:
            # ① 快路径：有数据直接返回，不武装、不清标志。此时标志保持 True，
            #    消费者忙着处理时生产者不会发多余唤醒；下次 get_batch 一定先 drain，数据不会丢
            out = self._drain(max_items)
            if out:
                return out
            # 测试在这里注入一次 put：它会看到标志仍是 True（上一轮留下的）而不唤醒。调用时不能持锁（否则死锁）
            self.after_empty_drain()
            # ② 武装：清 Event（丢掉上一周期遗留的 set，否则下面的 wait 立即返回、空转一圈），
            #    并把标志置 False——从这一刻起的第一次 put 会调度唤醒。
            #    clear 与置 False 之间没有 await，loop 上的 event.set 回调插不进来，顺序安全
            with self._lock:  # arm: the next put must schedule a wakeup
                self._wake_in_flight = False
                self._event.clear()
            # ③ 易错：这次二次 drain 不能省。它关住了“①空 → 生产者 append 且看到 True 不唤醒 → ②武装”
            #    这个窗口：那次 put 的 append 早于武装，这里一定能拿到
            out = self._drain(max_items)  # catch puts that raced with arming
            if out:
                # 返回时标志是 False（已武装），下一次 put 会多调度一次唤醒——只是多一次 wakeup，正确性不受影响
                return out
            remaining = None if deadline is None else deadline - self._loop.time()
            if remaining is not None and remaining <= 0:
                return []
            try:
                # ④ 睡下：await 期间生产者的 put 会通过 call_soon_threadsafe(event.set) 把我们唤醒
                await asyncio.wait_for(self._event.wait(), remaining)
            except asyncio.TimeoutError:
                # 超时那一刻可能恰好有数据进来（set 回调还没跑），最后再 drain 一次而不是直接返回 []
                return self._drain(max_items)
            # 被唤醒后回到循环顶部 drain；若是伪唤醒（拿不到数据），会重新武装再睡
