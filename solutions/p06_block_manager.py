"""P06 reference solution · BlockManager：paged KV-cache 分配器（引用计数 + 链式哈希前缀缓存 + LRU 驱逐）。

- 每个 seq 有一张 block table（逻辑块下标 -> 物理 block id）；slot = table[pos // bs] * bs + pos % bs。
- 物理 block 带 refcount；开启 prefix cache 时多个 seq 可以共享同一前缀 block。
- refcount 归零但仍带 hash 的 block 进入 evictable LRU 池，仍可被命中复用；空间不够时才按 LRU 驱逐。
- allocate / append_slot 失败（OutOfBlocks）时不改变任何状态：先检查、再修改。
提示页：hints/p06_overview.html（A-17）。
"""
from collections import OrderedDict, deque
from dataclasses import dataclass, field


# 分配失败时抛出；调用方（scheduler）据此决定抢占或等待
class OutOfBlocks(Exception):
    pass


# 每个序列的四份记录
@dataclass
class _SeqState:
    table: list[int] = field(default_factory=list)    # block table：逻辑块 -> 物理块
    tokens: list[int] = field(default_factory=list)   # 全部 token（prompt + 追加的），算后续满块的 hash 要用
    hashes: list[int] = field(default_factory=list)  # chain hash of every FULL block（长度 = len(tokens) // bs）
    cached_blocks: int = 0                            # allocate 时前缀缓存命中的 block 数


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = False) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        # 空闲表：无 hash、ref==0 的 block，FIFO（popleft 取、append 还）
        self._free: deque[int] = deque(range(num_blocks))
        # 不变量：_evictable 里的 block 满足 ref==0 且带 hash；OrderedDict 头部最老（最先被驱逐）
        self._evictable: OrderedDict[int, None] = OrderedDict()  # LRU: oldest first
        self._ref = [0] * num_blocks       # ref[b] = 包含 b 的 block table 个数
        # 两个方向的索引必须互指：_hash_to_block[h] == b ⟹ _block_hash[b] == h
        self._block_hash: dict[int, int] = {}       # 物理 block -> 登记的前缀 hash（不在其中 = 不可被命中）
        self._hash_to_block: dict[int, int] = {}    # 前缀 hash -> 物理 block（命中查找）
        self._seqs: dict[str, _SeqState] = {}

    # 可用 block 数 = 真空闲 + 可驱逐（evictable 的 block 需要时可以被回收）
    @property
    def num_free_blocks(self) -> int:
        return len(self._free) + len(self._evictable)

    # 返回拷贝，防止调用方修改内部 block table
    def block_table(self, seq_id: str) -> list[int]:
        return list(self._seqs[seq_id].table)

    # 永远是 block_size 的整数倍，且 < len(prompt)（至少留 1 个 token 要算）
    def num_cached_tokens(self, seq_id: str) -> int:
        return self._seqs[seq_id].cached_blocks * self.block_size

    # 逻辑位置 pos -> KV cache 展平后的 slot 编号（P09 的 write_kv 就用它）。
    # 例：table=[0, 1]，bs=4，pos=5 -> 1*4 + 1 = 5
    def slot(self, seq_id: str, pos: int) -> int:
        table = self._seqs[seq_id].table
        return table[pos // self.block_size] * self.block_size + pos % self.block_size

    # 粗检查：ceil(num_tokens / bs) <= 可用块数，不考虑前缀命中（偏保守）。
    # 易错：allocate 内部不能用它当原子性检查——那里必须精确扣除“命中的可驱逐块”
    def can_allocate(self, num_tokens: int) -> bool:
        return -(-num_tokens // self.block_size) <= self.num_free_blocks   # -(-a // b) = ceil(a / b)

    # 链式哈希：每个满 block 的 hash 依赖前一个 block 的 hash，因此同一个 hash 代表
    # “从序列开头到该 block 为止的整段前缀”。易错：只 hash 本 block 的 tokens 会让不同前缀的相同片段误命中。
    # 起点固定用 None 当 h_{-1}；allocate 和 append_slot 必须用同一套规则，否则 decode 填满的块永远不会被命中
    @staticmethod
    def _chain(prev: int | None, tokens: list[int]) -> int:
        return hash((prev, tuple(tokens)))

    # 取一个新的物理 block：优先空闲表，没有再从 evictable 池驱逐最老的一个。
    # 调用方保证此时一定有可用块（容量已提前检查），所以这里不会 raise。
    def _take_block(self) -> int:
        if self._free:
            b = self._free.popleft()
        else:
            # popitem(last=False) 弹出 LRU 头部，即最久未使用的缓存 block
            b, _ = self._evictable.popitem(last=False)
            # 驱逐后该 block 内容将被覆盖：必须撕掉它的 hash 登记（两处），否则旧前缀会“命中”被覆盖的块，KV 错乱
            h = self._block_hash.pop(b)
            # 易错：只有当索引仍指向这个 block 时才删；同一 hash 可能登记在别的 block 上（内容相同的副本）
            if self._hash_to_block.get(h) == b:
                del self._hash_to_block[h]
        self._ref[b] = 1
        return b

    # 把一个刚写满的 block 登记到前缀缓存（未开启时什么也不做）。
    # 同一前缀已有 block 登记时保留旧的、新块不登记（一个 hash 只指向一个物理块）
    def _register(self, b: int, h: int) -> None:
        if self.enable_prefix_cache and h not in self._hash_to_block:
            self._hash_to_block[h] = b
            self._block_hash[b] = h

    # ---------------------------------------------------------------------------
    # allocate：为新 seq 的 prompt 分配 block table。先算、再查、最后改：
    #   1) 只对满 block 计算链式 hash；2) 从头连续匹配最长前缀；3) 精确检查容量（全有或全无）；
    #   4) 一次性执行：复用命中 block（ref+1，必要时移出可驱逐池）+ 为其余部分取新 block 并登记满块。
    # ---------------------------------------------------------------------------
    def allocate(self, seq_id: str, token_ids: list[int]) -> list[int]:
        bs = self.block_size
        n_blocks = -(-len(token_ids) // bs)   # 总共需要的 block 数（含最后一个未满 block）
        hashes: list[int] = []
        prev = None
        # 只对满 block 计算 hash。易错：给半满的最后一块也算 hash，之后更长的 prompt 会命中只装了部分 token 的块
        for i in range(len(token_ids) // bs):
            prev = self._chain(prev, token_ids[i * bs:(i + 1) * bs])
            hashes.append(prev)
        hits: list[int] = []
        if self.enable_prefix_cache:
            # 易错：最多匹配 (len-1)//bs 个 block（不是 len//bs）——即使 prompt 完全在缓存里，
            # 也要留至少 1 个 token 给模型算，才能拿到最后位置的 logits 采出下一个 token。例：len=8, bs=4 只能命中 1 块
            for h in hashes[: (len(token_ids) - 1) // bs]:
                b = self._hash_to_block.get(h)
                if b is None:
                    break   # 前缀匹配必须连续：第一次 miss 就停
                hits.append(b)
        need = n_blocks - len(hits)
        # 命中的块里 ref==0 的正躺在可驱逐池里——它们原本被算在 num_free_blocks 里，复用后就不能再当新块用了。
        # 易错（原子性陷阱）：只检查 need <= num_free_blocks 会“通过检查”，然后在取新块途中才失败，
        # 而此时命中块的 ref 已经加过了，状态被破坏
        reusable_from_pool = sum(1 for b in hits if self._ref[b] == 0)
        if need > self.num_free_blocks - reusable_from_pool:
            raise OutOfBlocks(f"{seq_id}: need {need} blocks, {self.num_free_blocks - reusable_from_pool} free")
        # ---- 从这里开始才修改状态，且之后不会再失败 ----
        table = []
        for b in hits:
            if self._ref[b] == 0:
                # 从可驱逐池“复活”：必须移出池子，否则之后会被当成候补驱逐掉（两个序列共享一块却内容不同）
                del self._evictable[b]
            self._ref[b] += 1
            table.append(b)
        for i in range(len(hits), n_blocks):
            b = self._take_block()
            table.append(b)
            if i < len(hashes):   # 满块才登记 hash，供后续请求复用
                self._register(b, hashes[i])
        self._seqs[seq_id] = _SeqState(table, list(token_ids), hashes, len(hits))
        return list(table)

    # append_slot：decode 时为新 token 追加一个 slot，返回该 token 的 slot；写满一个 block 时为它登记 hash。
    def append_slot(self, seq_id: str, token_id: int) -> int:
        st = self._seqs[seq_id]
        bs = self.block_size
        pos = len(st.tokens)   # 新 token 的位置 = 当前 token 数
        if pos % bs == 0:
            # 新 token 落在一个还不存在的逻辑块上，需要一个新 block（等价于 pos // bs == len(table)）。
            # 原子性：先检查、再取块，且都在 tokens.append 之前——失败时 seq 状态完全不变。
            # 易错：先 append token 再取块，OutOfBlocks 后 tokens 多了一个，之后的 slot 全错
            if self.num_free_blocks == 0:
                raise OutOfBlocks(f"{seq_id}: no block for position {pos}")
            st.table.append(self._take_block())
        st.tokens.append(token_id)
        if (pos + 1) % bs == 0:
            # 这个 block 刚好写满：用链尾当父 hash 计算新 hash、入链并登记，
            # 后续请求就能命中模型生成的内容（多轮对话）
            prev = st.hashes[-1] if st.hashes else None
            h = self._chain(prev, st.tokens[pos + 1 - bs:pos + 1])
            st.hashes.append(h)
            self._register(st.table[-1], h)
        return st.table[pos // bs] * bs + pos % bs

    # free：释放 seq 的所有 block 引用。ref 归零后：有 hash 的进入 evictable（保留缓存），无 hash 的回到空闲表。
    def free(self, seq_id: str) -> None:
        st = self._seqs.pop(seq_id)   # 删掉该序列的全部记录
        # 逆序释放：尾部 block（最不通用的前缀）先进入 LRU、成为“最旧”，因而先被驱逐；
        # 前缀开头的 block（被其他请求共享的概率最高）后进入、保留更久。易错：正序会让共享度最高的首块先死
        for b in reversed(st.table):  # tail blocks become the first eviction candidates
            self._ref[b] -= 1
            # 只有 ref 归零才真正释放；共享 block 仍被其他 seq 引用时保持不动
            if self._ref[b] == 0:
                if b in self._block_hash:
                    self._evictable[b] = None   # 放到“最新”一端
                else:
                    self._free.append(b)
