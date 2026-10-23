"""P11 reference solution · WAITING_FOR_CHUNK：在不懂 chunk 的上游 scheduler 外面做停靠（park）与归还（restore）。

下一个 chunk 还没到的 needs_chunk 请求在上游眼里必须“不存在”：每次 schedule 前把它们从 base 的队列里
摘出来暂存（park），chunk 就绪后按原队列、原状态放回（restore）；上游调度结束（含抛异常）后在 finally 里
再 restore 一次。全部在同一个调度线程里执行（单写者，不用锁）；唯一的“并发”是 is_ready() 的答案会被
别的线程（P13 的 recv 线程）推进。提示页：hints/p11_overview.html（A-15）。
"""
from collections import deque
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Req:
    req_id: str
    arrival: int                 # 到达顺序；同一批归还时按它排序
    needs_chunk: bool = False    # 输入由上游 stage 一块一块送来
    status: str = "WAITING"


class BaseScheduler:
    """Stand-in for the upstream vLLM scheduler: knows nothing about chunks."""

    def __init__(self) -> None:
        self.waiting: deque[Req] = deque()
        self.running: list[Req] = []

    # 返回 running 的 id + waiting 的 id，然后把 waiting 全部挪进 running 并设为 RUNNING。
    # 注意它从不改 running 中请求的 status——恢复 status 的责任完全在 ChunkAwareScheduler
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
    def load_async(self, req_id: str) -> None: ...   # 请求：把下一块送来（非阻塞）
    def is_ready(self, req_id: str) -> bool: ...     # 回答：下一块到了吗（可能 True -> False -> True 反复变化）


# 不变量：① 上游 schedule 执行期间队列里没有 WAITING_FOR_CHUNK / ABORTED 的请求；
#         ② 每个请求任一时刻只在 waiting、running、parked 之一；③ 每次停靠恰好一次 load_async；
#         ④ 归还 = 原队列 + 原状态，同一批按 arrival 排序。
class ChunkAwareScheduler:
    def __init__(self, base: BaseScheduler, source: ChunkSource) -> None:
        self.base = base
        self.source = source
        # 暂存区（等料夹）：id -> (req, 来源队列 "waiting"/"running", 停靠前的 status)
        self.parked: dict[str, tuple[Req, str, str]] = {}

    def add(self, req: Req) -> None:
        self.base.waiting.append(req)

    # abort 任何时候都可以调用：请求可能藏在 parked、waiting、running 三处之一，找到就彻底移除。
    # 易错：只设 ABORTED 不移除，上游看到会断言失败；只删 parked 而它其实在队列里，上游照样调度它
    def abort(self, req_id: str) -> None:
        entry = self.parked.pop(req_id, None)
        if entry is not None:
            # 从停靠表删掉即可阻止归还（归还只扫描 parked）；之后 is_ready 就算返回 True 也没有单子可挂回去。
            # 停靠中的请求不会同时在队列里（不变量 ②），所以直接返回
            entry[0].status = "ABORTED"
            return
        # 遍历拷贝（list(...)），避免边遍历边删除
        for r in list(self.base.waiting):
            if r.req_id == req_id:
                self.base.waiting.remove(r)
                r.status = "ABORTED"
        for r in list(self.base.running):
            if r.req_id == req_id:
                self.base.running.remove(r)
                r.status = "ABORTED"

    # restore：把 chunk 已就绪的暂存请求放回原队列末尾、恢复原 status。
    def _restore_ready(self) -> None:
        # 先收集再删除：不能在遍历 dict 的同时 del
        ready = [e for rid, e in self.parked.items() if self.source.is_ready(rid)]
        # 同一批归还按 Req.arrival 排序（到达顺序），而不是按停靠 / dict 插入顺序
        for req, origin, prev in sorted(ready, key=lambda e: e[0].arrival):
            del self.parked[req.req_id]
            # 易错：恢复成停靠前记录的状态。写死 "WAITING" 时，从 running 停靠的请求会带着错误状态待在 running
            #       （上游只改 waiting 里请求的 status，永远不会帮你纠正）
            req.status = prev
            if origin == "waiting":
                self.base.waiting.append(req)
            else:
                self.base.running.append(req)

    # park：扫描 waiting 与 running，把 needs_chunk 且未就绪的请求移进 parked。
    # “停靠”必须是移出队列，只改 status 不够（上游遍历队列时会看到 WAITING_FOR_CHUNK 并断言失败）
    def _park_not_ready(self) -> None:
        for origin, q in (("waiting", self.base.waiting), ("running", self.base.running)):
            # 遍历快照：list 边遍历边 remove 会跳元素，deque 会直接抛 RuntimeError
            for r in list(q):
                if r.needs_chunk and not self.source.is_ready(r.req_id):
                    q.remove(r)
                    # 易错：必须在改 status 之前保存原 status，否则 previous status 永远是 WAITING_FOR_CHUNK
                    self.parked[r.req_id] = (r, origin, r.status)
                    r.status = "WAITING_FOR_CHUNK"
                    # 每次停靠恰好一次 load_async：停靠后请求已不在队列里，下一轮这里根本看不到它，
                    # 所以连续多轮未就绪也不会重复请求；被归还、吃完一块再次停靠时才会再调一次
                    self.source.load_async(r.req_id)

    # 顺序：(a) restore 已就绪 -> (b) park 未就绪 -> (c) 上游调度 -> (d) finally 中再 restore。
    def schedule(self) -> list[str]:
        # (a) 先 restore：两次 schedule 之间刚就绪的请求本轮就能被调度（漏了它会晚一轮）
        self._restore_ready()
        # (b) 再 park：上游只看得到有数据的请求
        self._park_not_ready()
        try:
            return self.base.schedule()
        finally:
            # (d) 易错：上游抛异常时也必须 restore（否则就绪的请求一直卡在 parked），且异常要照常往上抛——
            #     finally 里不能写 return（会吞掉异常）。这里归还的请求不在本轮返回值里，下一轮被调度
            self._restore_ready()
