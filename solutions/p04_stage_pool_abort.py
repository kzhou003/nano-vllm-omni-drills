"""P04 reference solution · StagePool（粘性绑定）+ 两阶段 abort + 死 replica 隔离。

所有方法都在 orchestrator 的 event loop 上调用：同步方法（pick / evict / release）之间不会交错，
唯一的让出点是 abort 里的 await。提示页：hints/p04_overview.html（A-12）。
"""
from dataclasses import dataclass
from typing import Any, Protocol


class NoLiveReplica(Exception):
    pass


class ReplicaClient(Protocol):
    async def abort(self, request_ids: list[str]) -> None: ...   # 一次调用 = 一次 IPC


class PartialProcessor(Protocol):
    def collect_partial(self, request_id: str) -> Any: ...   # 只读：已生成的部分结果
    def commit_abort(self, request_id: str) -> None: ...     # 不可逆：丢弃 detokenizer 等前端状态


@dataclass
class OrchOutput:   # 注意：本题的 OrchOutput 没有 stage_id 字段
    request_id: str
    data: Any
    finished: bool
    error: str | None = None


# 一个逻辑 stage 的多副本池。状态：
#   _dead   已死亡的 replica id（replica id = self.replicas 的下标）
#   _bind   request_id -> replica_id 的粘性绑定（请求的 KV cache / 流式中间状态都在那个 replica 上）
#   _rr     round-robin 游标
# 不变量：绑定存在 ⇔ 那个 replica 上（可能）还有这个请求的状态；pick 永不返回死 replica。
class StagePool:
    def __init__(self, stage_id: int, replicas: list[ReplicaClient]) -> None:
        self.stage_id = stage_id
        self.replicas = replicas
        self._dead: set[int] = set()
        self._bind: dict[str, int] = {}
        self._rr = 0

    def live_replicas(self) -> list[int]:
        return [i for i in range(len(self.replicas)) if i not in self._dead]   # 按下标升序

    # 为请求选择 replica：已绑定 -> 粘住原来的；否则在存活 replica 上 round-robin 并绑定。
    def pick(self, request_id: str) -> int:
        r = self._bind.get(request_id)
        if r is not None:
            # 粘性命中：原样返回、不推进游标。evict 会删掉死 replica 上的绑定，所以这里拿到的一定是活的
            return r
        live = self.live_replicas()
        if not live:
            raise NoLiveReplica(f"stage {self.stage_id} has no live replica")
        # 对全部 replica 取模、跳过死的（live 非空保证循环会结束）。
        # 也可以对 live 列表取模：分布更均匀（这种写法会让死 replica 的下一个邻居多拿一份），两种都满足 spec
        n = len(self.replicas)
        while True:
            r = self._rr % n
            self._rr += 1
            if r not in self._dead:
                break
        self._bind[request_id] = r
        return r

    # 只读查询：没有绑定就返回 None，不创建绑定
    def bound(self, request_id: str) -> int | None:
        return self._bind.get(request_id)

    # 请求正常结束时释放绑定（不存在也不报错）；不释放会让绑定表只增不减
    def release(self, request_id: str) -> None:
        self._bind.pop(request_id, None)

    # replica 挂了：标记死亡（之后 pick 不会再选它），摘掉并返回绑在它上面的请求，由调用方发错误终态。
    # 易错：只标记不删绑定 -> 之后对这些请求 pick / abort 会去找一个死进程
    def evict(self, replica_id: int) -> list[str]:
        self._dead.add(replica_id)
        # 先收集再删除：不能边遍历 dict 边删（RuntimeError: dict changed size during iteration）
        affected = [rid for rid, r in self._bind.items() if r == replica_id]
        for rid in affected:
            del self._bind[rid]
        return affected

    # 按 replica 分组：每个有 id 的 replica 只发一次 abort（批量 IPC），而不是每个请求一次。
    async def abort(self, request_ids: list[str]) -> None:
        groups: dict[int, list[str]] = {}
        for rid in request_ids:
            r = self._bind.get(rid)
            # 没有绑定的 id（从未派发到本 stage / 已释放 / 已 evict）直接忽略；没有 id 的 replica 不调用
            if r is not None:
                groups.setdefault(r, []).append(rid)
        # 顺序逐组 await（不用 gather：一组失败时其他组的状态会难以推理）
        for r, ids in groups.items():
            # 易错：必须等 replica 确认 abort 成功后才释放绑定。若抛异常，异常直接向上传播，
            # 这组和尚未处理的组的绑定都保留，调用方可以重试；先释放再 await 的话，失败后就再也不知道它们在哪
            await self.replicas[r].abort(ids)  # raises -> bindings kept for retry
            for rid in ids:
                # pop(…, None)：await 期间 evict 等操作可能已经删掉了这个绑定
                self._bind.pop(rid, None)


# 两阶段 abort（类似数据库 prepare/commit：不可逆的一步放在最后）：
#   ① 只读收集 partial -> ② 所有 stage 的引擎确认停止 -> ③ 才 commit。
async def abort_request(pools: list[StagePool], processor: PartialProcessor, request_id: str) -> OrchOutput:
    # 阶段 1：只读。即使后面 abort 失败，processor 状态仍完好，重试时能拿到同样的 partial；
    # 也必须在 commit 之前拿，commit 之后部分文本就没了
    partial = processor.collect_partial(request_id)
    for pool in pools:
        # 任一 pool 失败都直接抛出（不要 try/except 吞掉，也不要用 finally 保证 commit），跳过 commit。
        # 没绑定该请求的池分组为空、不产生任何调用，不需要在外面判断
        await pool.abort([request_id])
    # 易错：commit 必须放在所有 abort 成功之后。先 commit 再 abort 且 abort 失败：
    # 引擎还在为它产出 token，前端却已不认识这个请求（孤儿输出），重试时 partial 也丢了
    processor.commit_abort(request_id)
    # 终态：finished=True + error="abort"，带上已生成的部分结果
    return OrchOutput(request_id, partial, finished=True, error="abort")


# 死 replica 隔离：只给绑在该 replica 上的请求各发一条错误终态，其他 replica 上的请求照常。
# evict 已删除它们的绑定，所以重复调用不会再返回它们 -> 每个受影响请求恰好一条终态
def handle_dead_replica(pool: StagePool, replica_id: int) -> list[OrchOutput]:
    msg = f"stage {pool.stage_id} replica {replica_id} died"
    return [OrchOutput(rid, None, finished=True, error=msg) for rid in pool.evict(replica_id)]
