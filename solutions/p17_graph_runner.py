"""P17 reference solution · CUDA graph bucket runner（CPU 模拟）。

每个 batch bucket b 捕获一张图，所有图共享同一对 [max_bs, hidden] 静态 buffer 的前 b 行（view）。
run：选 ≥ bs 的最小 bucket，把输入拷进 static_in[:bs]、padding 行清零、replay、clone 出前 bs 行；
bs 超过最大 bucket、或 force_eager（如请求自带 RNG generator，replay 会悄悄忽略它）时走 eager。
提示页：hints/p17_overview.html（A-26）。
"""
from collections.abc import Callable

import torch


# CpuGraph：在 CPU 上模拟 CUDA graph 的关键语义——capture 时绑定静态输入/输出 buffer，
# replay 只会在“同一块” buffer 上重跑，看不到新的 Python 参数。
class CpuGraph:
    """Emulates torch.cuda.CUDAGraph semantics on CPU: replay() re-runs the recorded work on
    the *same static buffers*; it never sees new Python arguments."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor], static_in: torch.Tensor, static_out: torch.Tensor):
        self._fn, self._in, self._out = fn, static_in, static_out
        self.captured = False

    def capture(self) -> None:
        self._out.copy_(self._fn(self._in))
        self.captured = True

    def replay(self) -> None:
        assert self.captured
        self._out.copy_(self._fn(self._in))


class GraphRunner:
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor], buckets: list[int], hidden: int,
                 dtype: torch.dtype = torch.float32) -> None:
        self.fn = fn
        # 排序后线性扫描，第一个命中的就是最小可用 bucket
        self.buckets = sorted(buckets)
        self.graphs: dict[int, CpuGraph] = {}
        self.stats = {"replays": 0, "eager": 0, "last_bucket": None}
        # 静态 buffer 按最大 bucket 分配一次 [max_b, hidden]；zeros 比 empty 安全（capture 时就会跑一次 fn）
        max_b = self.buckets[-1]
        self.static_in = torch.zeros(max_b, hidden, dtype=dtype)
        self.static_out = torch.zeros(max_b, hidden, dtype=dtype)

    # 所有图都在启动期录好；运行时从不 capture（很慢、要 warmup 和同步，会造成延迟尖刺）
    def capture(self) -> None:
        # 从大到小捕获：CUDA 上各图共享一个 memory pool，先录最大的图占好内存，小图复用（CPU 模拟里不影响结果）
        for b in reversed(self.buckets):
            # static_in[:b] 是 view（共享存储），不是拷贝：之后往 static_in 的前 bs 行写，图就能“看到”
            g = CpuGraph(self.fn, self.static_in[:b], self.static_out[:b])
            g.capture()
            self.graphs[b] = g

    def run(self, x: torch.Tensor, *, force_eager: bool = False) -> torch.Tensor:
        bs = x.shape[0]
        # 向上取整到最小的 bucket；例 buckets=[1, 4, 16]：bs=3 -> 4，bs=17 -> None
        b = next((b for b in self.buckets if b >= bs), None)
        # eager 回退：没有够大的 bucket / 强制 eager / 还没 capture。eager 分支不碰静态 buffer、也不更新 last_bucket
        if b is None or force_eager or b not in self.graphs:
            self.stats["eager"] += 1
            return self.fn(x)
        # 易错：必须 copy_ 进静态 buffer，不能 self.static_in = x（图绑定的是原来那块内存）
        self.static_in[:bs].copy_(x)
        # 易错：padding 行 [bs, b) 可能残留上一次更大 batch 的数据，replay 前清零。本题 fn 逐行独立，
        # 脏行只让 padding 输出变垃圾；真实模型里有跨行交互（MoE 路由统计、NaN 扩散、整 batch 归约）会污染真实行。
        # 只清 [bs, b)：b 之后的行这张图根本不读
        self.static_in[bs:b].zero_()      # stale rows from a previous, larger batch
        self.graphs[b].replay()
        self.stats["replays"] += 1
        self.stats["last_bucket"] = b     # 语义：最近一次 replay 用的 bucket
        # 易错：必须 clone。直接返回 static_out[:bs] 这个 view，下一次 replay 会悄悄改写调用方手里的结果；
        # 也只返回前 bs 行（padding 行是“对 0 做计算”的垃圾）
        return self.static_out[:bs].clone()
