"""P02 reference solution · Orchestrator v1：线性多 stage 路由（单写者 dispatcher）。

Task 结构（全部跑在同一个 event loop 上）：
  N 个 _reader（每个 stage 一个） --(sid, StageOutput)--> _ready_q --> 1 个 _dispatcher
  add_request() --> _req_q --> 1 个 _request_handler
  _dispatcher --> _out_q --> next_output()（用户侧）
单写者：所有 StageOutput 汇入一个 _ready_q，由唯一的 dispatcher 串行处理；asyncio 协作式调度，
只要“读-改-写”中间没有 await 就不需要锁。提示页：hints/p02_overview.html、p02_flow.html（A-9）。
"""
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


# StageOutput：stage -> orchestrator（每个 stage 都会产生，包括非 final stage）。注意没有 stage_id 字段。
@dataclass
class StageOutput:
    request_id: str
    data: Any
    finished: bool   # 只表示“这个 stage 对这个请求完成了”，不是整个请求完成


# OrchOutput：orchestrator -> 用户（只来自 final_stage_ids 里的 stage）
@dataclass
class OrchOutput:
    request_id: str
    stage_id: int | None
    data: Any
    finished: bool   # 请求级终态：每个请求恰好一条 True，且是最后一条
    error: str | None = None


class StageClient(Protocol):
    async def submit(self, request_id: str, inputs: Any) -> None: ...
    async def get_output(self) -> list[StageOutput]: ...
    async def abort(self, request_ids: list[str]) -> None: ...


# 每个在途请求一份路由状态；只由 orchestrator 内部的 task 读写（同一个 event loop，无需加锁）
@dataclass
class ReqState:
    request_id: str
    prompt: Any
    submitted: set[int] = field(default_factory=set)   # 已提交过的 stage
    finished: set[int] = field(default_factory=set)    # 已报告 finished=True 的 stage
    # 每个 stage 累积的全部输出（按到达顺序）；stage s 完成时整份交给 forward_fn 构造 s+1 的输入
    stage_data: dict[int, list] = field(default_factory=dict)
    pending_final: OrchOutput | None = None            # P02 不用；P03 用来暂扣终态


class Orchestrator:
    def __init__(
        self,
        stages: list[StageClient],
        final_stage_ids: set[int],
        forward_fn: Callable[[int, Any, list], Any],
    ) -> None:
        self.stages = stages
        self.final_stage_ids = frozenset(final_stage_ids)
        # 最后一个需要运行的 stage：比它更靠后的 stage 对用户没有意义，永远不会被提交
        self.last_stage = max(final_stage_ids)
        self.forward_fn = forward_fn
        self.request_states: dict[str, ReqState] = {}
        self.dropped_outputs = 0
        # 三个无界队列：put 永不阻塞，reader / dispatcher 不会因为下游慢而互相卡死
        self._req_q: asyncio.Queue = asyncio.Queue()     # add_request -> request_handler：(rid, prompt)
        self._ready_q: asyncio.Queue = asyncio.Queue()   # readers -> dispatcher（fan-in）：(sid, StageOutput)
        self._out_q: asyncio.Queue = asyncio.Queue()     # dispatcher -> next_output：OrchOutput
        self._tasks: list[asyncio.Task] = []

    # ---------------------------------------------------------------- public API
    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        # 每个独立阻塞源（stage.get_output）配一个专门守着它的 reader：
        # 某个 stage 迟迟不出输出也不会挡住其他 stage（让 dispatcher 轮流等会死锁）
        for sid, client in enumerate(self.stages):
            self._tasks.append(loop.create_task(self._reader(sid, client)))
        # 只有一个 dispatcher——这就是“单写者”。task 句柄都保存下来：防止被 GC，shutdown 也要用
        self._tasks.append(loop.create_task(self._dispatcher()))
        self._tasks.append(loop.create_task(self._request_handler()))

    async def shutdown(self) -> None:
        for t in self._tasks:
            t.cancel()   # 在各自的 await 处抛 CancelledError
        # cancel 只是请求取消；gather 等它们真正退出。return_exceptions=True 吞掉 CancelledError，
        # 否则第一个 CancelledError 就会让 shutdown 自己抛出、其余 task 没被等待（“Task was destroyed but it is pending”）
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    # 公开 API 要薄：只投递消息，真正的注册 / 提交在 request handler task 里做
    async def add_request(self, request_id: str, prompt: Any) -> None:
        await self._req_q.put((request_id, prompt))

    async def next_output(self, timeout: float | None = None) -> OrchOutput:
        return await asyncio.wait_for(self._out_q.get(), timeout)   # timeout=None 表示一直等

    # ---------------------------------------------------------------- internals
    # reader：只搬运，不碰 request_states（避免多写者）。
    # 带上 sid 一起入队：StageOutput 不知道自己来自哪个 stage，只有 reader 知道。
    # 一个 stage 经“一个 reader → 一个 FIFO 队列 → 一个 dispatcher”，同一 stage 内的顺序天然保持
    async def _reader(self, sid: int, client: StageClient) -> None:
        while True:
            for o in await client.get_output():   # 一次返回一批（可能混着多个请求）
                await self._ready_q.put((sid, o))

    async def _request_handler(self) -> None:
        while True:
            rid, prompt = await self._req_q.get()
            await self._handle_add(rid, prompt)

    # 唯一的决策者：按到达顺序逐条处理，同一请求的输出不会并发处理
    async def _dispatcher(self) -> None:
        while True:
            sid, o = await self._ready_q.get()
            await self._handle_output(sid, o)

    # 只在 request handler task 里运行（P03 覆盖它）
    async def _handle_add(self, rid: str, prompt: Any) -> None:
        st = ReqState(rid, prompt)
        st.submitted.add(0)
        # 易错：必须在第一个 await 之前注册。await submit 会让出控制权；很快的 stage 可能在 submit
        # 返回前就吐出输出，reader -> _ready_q -> dispatcher 若查不到 ReqState，输出会被当成未知请求丢弃。
        # dict 赋值是同步的，放在 await 前就没有任何窗口
        self.request_states[rid] = st  # register BEFORE the first await
        await self.stages[0].submit(rid, prompt)

    # 只在 dispatcher task 里运行（P03 覆盖它）。
    # 处理一条 StageOutput：记录 ->（若 final stage）发给用户 ->（若终态）清理并 return ->（若该 stage 完成）转发
    async def _handle_output(self, sid: int, o: StageOutput) -> None:
        st = self.request_states.get(o.request_id)
        if st is None:
            # 请求已 cleanup（终态已发出）或从未注册：丢弃并计数，绝不抛异常（会让 dispatcher 退出、所有请求卡住）。
            # 易错：绝不能在这里新建状态（如 setdefault）——那会“复活”已结束的请求，状态泄漏、孤儿输出发给用户
            self.dropped_outputs += 1
            return
        # 先累积数据：stage 完成时 forward_fn 需要该 stage 产生的全部数据（含这条 finished 的最后一块）
        st.stage_data.setdefault(sid, []).append(o.data)
        terminal = False
        if o.finished:
            st.finished.add(sid)
            # 易错：OrchOutput.finished 是请求级的，条件是“所有 final stage 都完成”（final ⊆ finished），
            # 而不是 o.finished。例：final={0,2}，S0 的 "there"(finished=True) 到达时 finished={0}，
            # 发给用户的是 finished=False；只有 S2 的最后一块才是 True
            terminal = self.final_stage_ids <= st.finished
        if sid in self.final_stage_ids:
            # 非 final stage（如 S1 talker）的输出不发给用户。put_nowait：无界队列不会满，也就没有让出点
            self._out_q.put_nowait(OrchOutput(o.request_id, sid, o.data, terminal))
        if terminal:
            # 终态已发：立刻移除状态，之后同一请求的任何输出都走上面的 dropped 分支 -> 恰好一条 finished=True。
            # 并且不再往下转发（请求已结束，再 submit 会产生孤儿工作）
            self._cleanup(o.request_id)
            return
        # “发给用户”和“转发”是两个独立的 if（不是 if/elif）：S0 的最后一块既要给用户、又要触发转发。
        # 只有 stage 完成、且后面还有需要运行的 stage 时才转发；每来一块就转发会让 S1 被 submit 多次
        if o.finished and sid < self.last_stage:
            nxt = sid + 1
            # 先改状态再 await：await 期间别的 task 看到的状态已经自洽
            st.submitted.add(nxt)
            # 把 stage sid 的“全部”数据交给 forward_fn（不是只有最后一条 o.data）
            inputs = self.forward_fn(nxt, st.prompt, st.stage_data[sid])
            # 这个 await 期间 dispatcher 挂起，其他输出暂存在 _ready_q 里，不会交错处理
            await self.stages[nxt].submit(o.request_id, inputs)

    # 移除请求状态；pop(..., None) 使重复调用无害
    def _cleanup(self, rid: str) -> None:
        self.request_states.pop(rid, None)
