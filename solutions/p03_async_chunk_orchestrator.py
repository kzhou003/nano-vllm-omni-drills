"""P03 reference solution · Orchestrator v2：async_chunk 预热 + 终态守门（继承 P02）。

async_chunk 模式下 stage 之间经 connector 直接传数据，orchestrator 不再转发。它只做两件事：
  1. add 时一次性 prewarm 所有下游 stage（让它们提前在 scheduler 里等第一个 chunk）；
  2. 守住终态：finished=True 只能在“所有已提交的 stage 都完成”之后发出（控制面上完成报告可能乱序）。
reader / dispatcher / 队列 / _cleanup 全部复用 P02，只覆写 _handle_add / _handle_output。
提示页：hints/p03_overview.html（A-10）。
"""
from collections.abc import Callable
from typing import Any

from .p02_orchestrator import Orchestrator, OrchOutput, ReqState, StageClient, StageOutput

__all__ = ["AsyncChunkOrchestrator", "OrchOutput", "StageOutput", "ReqState"]


class AsyncChunkOrchestrator(Orchestrator):
    def __init__(
        self,
        stages: list[StageClient],
        final_stage_ids: set[int],
        prewarm_fn: Callable[[int, Any], Any],
    ) -> None:
        # forward_fn 用不到（不再由 orchestrator 转发），传一个占位函数满足父类签名
        super().__init__(stages, final_stage_ids, forward_fn=lambda *_: None)
        self.prewarm_fn = prewarm_fn

    # 只在 request handler task 里运行
    async def _handle_add(self, rid: str, prompt: Any) -> None:
        st = ReqState(rid, prompt)
        # 易错：一次性把 0..last_stage 全部标成 submitted，而且要在任何 await 之前。
        # submitted 是 all_done 的比较基准；如果像 P02 那样只标 {0} 或边 submit 边 add，
        # dispatcher 可能在某个 await 期间看到不完整的 submitted，误判“所有 stage 都完成了”，
        # 过早放出终态（之后剩余 stage 的完成报告只能被 drop，资源也被提前回收）
        st.submitted.update(range(self.last_stage + 1))
        # 与 P02 相同：在任何 await 之前注册
        self.request_states[rid] = st
        await self.stages[0].submit(rid, prompt)   # stage 0 拿真实 prompt
        # 下游 stage 收到的是占位输入（prewarm），真实数据之后通过 connector 到达。每个 stage 只 submit 一次
        for sid in range(1, self.last_stage + 1):
            await self.stages[sid].submit(rid, self.prewarm_fn(sid, prompt))

    # 只在 dispatcher task 里运行（单写者）。不累积 stage_data，也不转发——数据走 connector。
    # 易错：直接沿用 P02 的 _handle_output 会在 S0 完成时再 submit 一次 S1（输入是 None）
    async def _handle_output(self, sid: int, o: StageOutput) -> None:
        st = self.request_states.get(o.request_id)
        if st is None:
            # 已 cleanup 的请求：丢弃计数，绝不复活状态（复活的 ReqState submitted 为空，门形同虚设）
            self.dropped_outputs += 1
            return
        if o.finished:
            st.finished.add(sid)
        # 两个不同的“完成”：
        #   all_final = 所有 final stage 完成          -> 决定 OrchOutput.finished 的值
        #   all_done  = 所有已提交 stage（含非 final，如 S1 talker）完成 -> 决定终态能否立即放行
        all_final = self.final_stage_ids <= st.finished
        all_done = st.submitted <= st.finished
        if sid in self.final_stage_ids:
            out = OrchOutput(o.request_id, sid, o.data, False)
            # “让 all_final 第一次成立的那条输出”就是终态消息（它的 stage_id 不一定是 last_stage）
            if o.finished and all_final and st.pending_final is None:
                out.finished = True
                if not all_done:
                    # 例：final={0,2}，S2 最后一块先到，但非 final 的 S1 还没报告完成。
                    # 若此时发出并 cleanup，S1 的完成报告会被 drop，而 S1 可能仍在用 SHM / 绑定。
                    # 所以扣下终态（不发！），等剩下的 stage 完成后原样放行
                    st.pending_final = out  # hold the terminal until every stage is done
                    return
                # 所有已提交 stage 也都完成了：终态直接发出并清理
                self._out_q.put_nowait(out)
                self._cleanup(o.request_id)
                return
            # 普通（非终态）的 final stage 输出照常发给用户
            self._out_q.put_nowait(out)
        # 非 final stage 的完成事件也可能是“最后一块拼图”：all_done 成立且终态正被暂扣时，现在放行。
        # 易错：漏了这一步，用户永远收不到终态、request_states 泄漏。
        # 放行的仍是那条最后的 final stage 输出 -> “终态永远是最后一条”
        if st.pending_final is not None and all_done:
            self._out_q.put_nowait(st.pending_final)
            self._cleanup(o.request_id)
