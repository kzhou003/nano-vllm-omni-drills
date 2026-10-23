"""P07 reference solution · Scheduler：continuous batching + chunked prefill + recompute 抢占。

每步在 token budget（max_num_batched_tokens）内：先给 running 里的序列分配（decode 或继续 chunked prefill），
再严格 FCFS 地从 waiting 接纳新序列；KV block 不够时抢占最晚加入 running 的序列（释放 KV、之后重算）。
抢占只丢缓存不丢进度：最终输出与从不抢占时逐 token 相同。提示页：hints/p07_overview.html（A-16）。

不变量（可以直接写成 assert）：
  I1 blocks_used == sum(s.num_blocks for s in running) <= num_blocks；全部结束后为 0
  I2 每步 sum(n) <= max_num_batched_tokens，且每个 n >= 1
  I3 0 <= num_computed <= num_tokens；只有相等时才采样
  I4 抢占只重置 num_computed / num_blocks / status，output 保留
  I5 被抢占者进 waiting 队首；waiting 内部始终是到达顺序
  I6 每个序列同时只在 waiting / running / finished 之一
"""
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


# Seq：调度器视角下的一个请求。只记账 token 数与 block 数，不持有真实 KV。
@dataclass
class Seq:
    seq_id: str
    prompt_len: int
    max_tokens: int
    output: list[int] = field(default_factory=list)   # 已生成的 token（抢占时保留）
    # 已写入 KV cache 的 token 数；num_tokens - num_computed = 还需计算的 token 数：
    # chunked prefill 未完成时 > 1，decode 阶段恒为 1（刚采样的那个 token 还没算 KV）
    num_computed: int = 0
    num_blocks: int = 0      # 当前占用的 KV block 数（只增不减，直到结束或被抢占时归零）
    status: str = "WAITING"  # WAITING | RUNNING | FINISHED

    # 总 token 数 = prompt + 已生成；被抢占后 recompute 也要覆盖这些全部 token
    @property
    def num_tokens(self) -> int:
        return self.prompt_len + len(self.output)


class Scheduler:
    def __init__(self, num_blocks: int, block_size: int, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Seq] = deque()   # FCFS；被抢占的序列放回队头
        self.running: list[Seq] = []         # 按接纳顺序，越靠后优先级越低（抢占的 victim 从尾部取）
        self.finished: list[Seq] = []
        self.blocks_used = 0
        self.num_preemptions = 0
        # 上一次 schedule() 的结果（seq_id -> 本步计算的 token 数）。update() 没有参数，只能靠它知道调度了谁
        self._scheduled: dict[str, int] = {}

    def add(self, seq: Seq) -> None:
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # ---------------------------------------------------------------- helpers
    # 本步要让 seq 持有 num_computed + n 个 token，还需要再申请几个 block（≤ 0 时为 0）。
    # 易错：按 num_tokens（整个 prompt）而不是 num_computed + n 算——chunked prefill 时提前占满 block、无谓抢占
    def _blocks_needed(self, seq: Seq, n: int) -> int:
        need = -(-(seq.num_computed + n) // self.block_size)   # ceil
        return max(0, need - seq.num_blocks)

    # 原子地增长：放得下就同时更新 blocks_used 和 seq.num_blocks，否则什么都不改并返回 False
    def _try_grow(self, seq: Seq, n: int) -> bool:
        extra = self._blocks_needed(seq, n)
        if self.blocks_used + extra > self.num_blocks:
            return False
        seq.num_blocks += extra
        self.blocks_used += extra
        return True

    # 归还 seq 占用的全部 block
    def _free(self, seq: Seq) -> None:
        self.blocks_used -= seq.num_blocks
        seq.num_blocks = 0

    # recompute 式抢占：丢弃全部 KV，放回 waiting 队头，之后重新 prefill（prompt + 已生成的 output）
    def _preempt(self, victim: Seq) -> None:
        self.running.remove(victim)
        self._free(victim)
        # 易错：必须把 num_computed 清零——KV 已随 block 释放，不清零的话重新进入时只调度 1 个 token，
        # 模型读到的是别人的 KV（还不会报错）。output 保留（清空 output 会重复生成、输出变样）
        victim.num_computed = 0
        victim.status = "WAITING"
        # appendleft：被抢占的序列比 waiting 里所有序列都更早到达，放队首保持 FCFS（放队尾会饿死它）
        self.waiting.appendleft(victim)
        self.num_preemptions += 1

    # ---------------------------------------------------------------- step
    # 返回本步调度：seq_id -> 本步要计算的 token 数。
    def schedule(self) -> dict[str, int]:
        budget = self.max_num_batched_tokens   # token budget 是局部变量，每步重新满额
        out: dict[str, int] = {}
        # ---- 阶段 1：running（decode，或继续 chunked prefill）----
        # 易错：用下标 while 而不是 for——抢占会从 running 里删元素，for 会跳元素或访问到已被抢占的序列
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            # decode 与 prefill 同一个公式：剩余未算 token，最多 budget 个（chunked prefill）
            n = min(seq.num_tokens - seq.num_computed, budget)
            if n <= 0:
                # budget 已用完（或没有剩余 token）：本步不调度它——这不是抢占，只是这步不跑
                i += 1
                continue
            scheduled = True
            # block 不够：反复抢占 running 尾部（最晚加入、优先级最低）的序列，直到放得下或抢到 seq 自己
            while not self._try_grow(seq, n):
                victim = self.running[-1]
                # victim 位于 running 尾部（下标 >= i），本步尚未被调度，所以不会残留在 out 里
                self._preempt(victim)
                if victim is seq:
                    scheduled = False   # 抢到了自己：seq 已不在 running 中，放弃本 seq
                    break
            if not scheduled:
                # 易错：seq 被抢占后 running 缩短，下标 i 现在指向下一个序列，不能 i += 1
                continue  # seq was removed from running; index i now points at the next one
            out[seq.seq_id] = n
            budget -= n
            i += 1
        # ---- 阶段 2：waiting，严格 FCFS；同时受 budget 和 max_num_seqs（只在这里作为准入条件）约束 ----
        while self.waiting and len(self.running) < self.max_num_seqs and budget > 0:
            seq = self.waiting[0]   # 只看队首（被抢占过的序列这里会算出 prompt + 已生成 的 token 数）
            n = min(seq.num_tokens - seq.num_computed, budget)
            # 易错：队首放不下就 break，不能 continue 去调度后面更小的请求（违反 FCFS，大请求永远饿死）；
            # 接纳新序列时也不会为它去抢占 running 里的序列
            if not self._try_grow(seq, n):
                break
            self.waiting.popleft()
            seq.status = "RUNNING"
            self.running.append(seq)   # 末尾 = 最低优先级 = 抢占首选
            out[seq.seq_id] = n
            budget -= n
        self._scheduled = out
        return out

    # update：模型 forward 结束后推进状态。只有算完全部 token（prefill 最后一块或 decode）的序列才采样新 token。
    def update(self, sample_fn: Callable[[str], int], eos_token: int) -> list[Seq]:
        done = []
        # 遍历 running 的拷贝：结束的序列会在循环里被移出 running
        for seq in list(self.running):
            n = self._scheduled.get(seq.seq_id)
            if n is None:
                continue   # 本步没被调度（budget 用完）
            seq.num_computed += n
            if seq.num_computed < seq.num_tokens:
                # 易错：chunked prefill 中间块的 logits 没有意义，不能采样（否则多出 token、提前达到 max_tokens）
                continue  # chunked prefill still in progress
            tok = sample_fn(seq.seq_id)
            # 先 append 再判结束：eos 本身也是输出的一部分。append 后 num_tokens 比 num_computed 多 1，
            # 下一步正好 decode 这 1 个 token
            seq.output.append(tok)
            if tok == eos_token or len(seq.output) >= seq.max_tokens:
                seq.status = "FINISHED"
                self.running.remove(seq)
                self._free(seq)          # 结束即释放 block
                self.finished.append(seq)
                done.append(seq)
        self._scheduled = {}   # 清空，防止重复 update 同一批调度
        return done
