"""P10 reference solution · EngineCore busy loop（每个 stage 进程里跑的主循环）。

IO 线程把 Msg 放进 input_queue；busy loop：空闲时阻塞等消息 -> 按 FIFO 抽干所有已排队消息 -> 有活就 step
一次并 emit 输出。引擎只由 busy loop 一个线程碰（单写者）；跨线程共享的只有 input_queue 和 shutdown 标志。
输出协议：("output", x) / ("utility", call_id, result, err) / ("fatal", repr) / ("shutdown",)。
提示页：hints/p10_overview.html（A-14）。
"""
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass
class Msg:
    # ADD: (req_id, data) · ABORT: [req_ids] · UTILITY: (call_id, name, args) · WAKEUP: 无（只用来叫醒 get()）
    kind: Literal["ADD", "ABORT", "UTILITY", "WAKEUP"]
    payload: Any = None


class Engine(Protocol):
    def add(self, req_id: str, data: Any) -> None: ...
    def abort(self, req_ids: list[str]) -> None: ...
    def has_work(self) -> bool: ...
    def step(self) -> list[Any]: ...
    def utility(self, name: str, args: tuple) -> Any: ...


class EngineCoreLoop:
    def __init__(self, engine: Engine, emit: Callable[[tuple], None]) -> None:
        self.engine = engine
        self.emit = emit
        # 线程安全的 FIFO：ADD / ABORT / UTILITY / WAKEUP 共用一条，所以“先 ADD 后 ABORT”的顺序天然保持
        self.input_queue: queue.Queue[Msg] = queue.Queue()
        # 关闭标志：threading.Event 可跨线程安全地 set / 查询
        self._shutdown = threading.Event()

    # 可从任意线程或 signal handler 调用：只设标志 + 放 WAKEUP，不 emit、不 join、不阻塞。
    def request_shutdown(self) -> None:
        self._shutdown.set()
        # 易错：只设标志不够——空闲时主循环阻塞在 input_queue.get() 上看不到标志，必须塞一条消息把它叫醒
        self.input_queue.put(Msg("WAKEUP"))

    # 把一条输入消息作用到 engine 上。阻塞拿到的第一条和抽干的其余消息都走这里
    def _handle(self, msg: Msg) -> None:
        if msg.kind == "ADD":
            req_id, data = msg.payload
            self.engine.add(req_id, data)
        elif msg.kind == "ABORT":
            self.engine.abort(list(msg.payload))   # payload 可能是任意可迭代对象，统一转 list
        elif msg.kind == "UTILITY":
            call_id, name, args = msg.payload
            # utility（RPC 式）的异常只影响这一次调用：按 call_id 回传错误，引擎继续运行。
            # args 作为一个 tuple 整体传入，不是 *args
            try:
                result = self.engine.utility(name, args)
            except Exception as e:  # noqa: BLE001 - utility errors are per-call
                self.emit(("utility", call_id, None, repr(e)))
            else:
                self.emit(("utility", call_id, result, None))
        # WAKEUP：什么都不做，它只负责把阻塞的 get() 叫醒

    def run_busy_loop(self) -> None:
        while not self._shutdown.is_set():
            if not self.engine.has_work():
                # 空闲时阻塞等待（不带 timeout）：零 CPU。易错：get(timeout=...) 或 sleep 轮询都是自旋。
                # 拿到的这条也是正常消息，必须处理（丢掉的话第一条 ADD 就没了）
                self._handle(self.input_queue.get())  # block: zero CPU while idle
            # 非阻塞地把队列里已有的消息按 FIFO 全部处理掉，保证 ADD / ABORT 在下一次 step 之前生效。
            # 易错：每步只处理一条消息时，ABORT 会晚于 step 生效，被 abort 的请求仍会产出输出；
            #       也不要先分类（先所有 ADD 再所有 ABORT），那会打乱 ADD→ABORT→ADD 的语义
            while True:
                try:
                    self._handle(self.input_queue.get_nowait())
                except queue.Empty:
                    break
            # 处理消息期间收到了关闭请求：不再 step，直接退出
            if self._shutdown.is_set():
                break
            # 再问一次 has_work：刚处理的可能只是 WAKEUP / UTILITY / ABORT，引擎仍然没活
            if self.engine.has_work():
                try:
                    outs = self.engine.step()
                except Exception as e:  # noqa: BLE001 - a failed step is fatal
                    # step 抛异常（如 CUDA 非法访存、NCCL 超时）是致命的：报告 fatal 后立即返回，
                    # 由外层进程监控把它报告为 stage 死亡（P04）；不再发 shutdown
                    self.emit(("fatal", repr(e)))
                    return
                for out in outs:
                    self.emit(("output", out))
        # 正常关闭路径的最后一条消息（注意是单元素元组）
        self.emit(("shutdown",))
