"""P05 reference solution · RpcRouter：在一条共享结果通道上把结果关联回正在阻塞等待的调用方。

所有方法都可能从任意线程调用。状态（全部由 _lock 保护）：
  _waiters  key -> 只容纳一个结果的 queue.Queue（每个在途调用一个“取餐器”）
  _fatal    fail_all 锁存的致命错误；一旦设置，之后的 register / call 立即失败
不变量：同一 key 最多一个活着的 waiter；每个 waiter 最多收到一个结果；登记一定先于发送。
提示页：hints/p05_overview.html（A-13）。
"""
import queue
import threading
from collections.abc import Callable, Hashable
from typing import Any


class RpcRouter:
    def __init__(self) -> None:
        self.late_results = 0
        self._lock = threading.Lock()
        self._waiters: dict[Hashable, queue.Queue] = {}
        self._fatal: BaseException | None = None

    # 调用方线程：登记一个等待者
    def register(self, key: Hashable) -> queue.Queue:
        with self._lock:
            # 易错：_fatal 检查必须和插入 _waiters 在同一把锁内。否则 fail_all 可能恰好在
            # “检查通过”与“插入”之间清空 _waiters，新插入的 waiter 永远没人唤醒，只能等到超时
            if self._fatal is not None:
                raise self._fatal
            # 同一个 key 两个在途调用会互相抢结果，直接拒绝
            if key in self._waiters:
                raise ValueError(f"duplicate rpc key {key!r}")
            w: queue.Queue = queue.Queue(maxsize=1)   # 每次调用恰好一个结果
            self._waiters[key] = w
            return w

    # 调用方线程（call 的 finally）：只移除“自己的” waiter
    def unregister(self, key: Hashable, waiter: queue.Queue) -> None:
        with self._lock:
            # 易错：按身份（is）比较再删。这个 key 可能已被 deliver / fail_all 弹出，并被一个新调用重新注册；
            # 直接 pop(key) 会误删别人的 waiter，对方的结果被当成 late、白等到超时
            if self._waiters.get(key) is waiter:
                del self._waiters[key]

    # 结果通道线程：把结果交给对应 waiter；有人接就返回 True，否则计为迟到结果
    def deliver(self, key: Hashable, result: Any) -> bool:
        with self._lock:
            # pop 而不是 get：一个 waiter 最多被交付一次；deliver 与 fail_all 在同一把锁下互斥，只有一个能拿到它
            w = self._waiters.pop(key, None)
            if w is None:
                # 调用方已超时 / 已注销，或 key 从未注册：计数后丢弃（在锁内自增，多线程下不丢计数）
                self.late_results += 1
                return False
        # 锁外 put（持锁做可能阻塞的事会堵住其他线程）：waiter 已从表里摘掉、只有我们持有它，
        # maxsize=1 且只 put 一次，正常不会 Full。边界：若调用方恰好刚超时，结果落进没人读的信箱被丢弃——可接受
        try:
            w.put_nowait(result)
        except queue.Full:
            return False
        return True

    # 致命错误（如 orchestrator 线程挂了）：立刻唤醒所有在途调用，并锁存错误让后续调用快速失败
    def fail_all(self, error: BaseException) -> None:
        with self._lock:
            # 锁存与清空在同一把锁内完成：之后的 register 一定能看到 _fatal
            self._fatal = error
            waiters = list(self._waiters.values())
            self._waiters.clear()
        # 锁外逐个唤醒：放入的是异常对象本身，call 端识别后 raise
        for w in waiters:
            try:
                w.put_nowait(error)
            except queue.Full:
                pass   # 防御性：已被摘出字典、不会再被 deliver，正常不会满

    # 调用方线程：注册 -> 发送 -> 阻塞等结果。四种结局：返回结果 / raise 结果里的异常 /
    # TimeoutError / raise 锁存的致命错误；无论哪种，返回时表里都不再有它的 waiter
    def call(self, key: Hashable, send: Callable[[], None], timeout: float) -> Any:
        # 易错：必须先 register 再 send。回包在另一个线程上到达，可能比 send() 返回还快（甚至在 send 内部）；
        # 若先 send，回包到达时找不到 waiter -> 计为 late 被丢弃 -> 本调用白等到超时。
        # register 放在 try 外：它失败时（重复 key / 已锁存）没有东西需要注销，send 也不会被调用
        w = self.register(key)
        try:
            send()
            try:
                result = w.get(timeout=timeout)   # 阻塞等待，绝不持锁
            except queue.Empty:
                # from None：隐藏内部的 queue.Empty，调用方只看到内置 TimeoutError
                raise TimeoutError(f"rpc {key!r} timed out after {timeout}s") from None
        finally:
            # 无论成功 / 超时 / send 抛异常都要注销：否则 _waiters 泄漏、同 key 再次调用报 duplicate；
            # 超时后注销也保证之后迟到的结果被正确计为 late_results
            self.unregister(key, w)
        # 结果本身是异常对象（远端执行失败，或 fail_all 广播的致命错误）时 raise 而不是 return
        if isinstance(result, BaseException):
            raise result
        return result
