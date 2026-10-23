"""M03 reference solution · Sparse MoE（thinker）与 MoE + 门控 shared expert（talker）。

“serving 的写法”：按 expert 分组，每个 expert 只在路由给它的 token 上算一次（FusedMoE 的 CPU 版）。
提示页：hints/m03_overview.html（A-3）。
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# SharedExpert：talker 的 MoE block 额外带一个“所有 token 都走”的 SwiGLU MLP，
#   输出再乘一个逐 token 的 sigmoid 门控。thinker 没有 shared expert（shared=None）。
@dataclass
class SharedExpert:
    gate_proj: torch.Tensor   # [I_s, H]
    up_proj: torch.Tensor     # [I_s, H]
    down_proj: torch.Tensor   # [H, I_s]
    gate: torch.Tensor        # [1, H]   sigmoid 门控（每个 token 一个标量）


# MoeWeights：E 个 expert 的权重 stack 成 [E, ...]，按 expert id 下标取。
@dataclass
class MoeWeights:
    router: torch.Tensor      # [E, H]
    w_gate: torch.Tensor      # [E, I, H]
    w_up: torch.Tensor        # [E, I, H]
    w_down: torch.Tensor      # [E, H, I]
    top_k: int
    norm_topk_prob: bool
    shared: SharedExpert | None = None


def from_hf(block) -> MoeWeights:
    ex = block.experts
    shared = None
    # 只有 talker 的 block 有 shared_expert / shared_expert_gate 属性
    if hasattr(block, "shared_expert"):
        s = block.shared_expert
        shared = SharedExpert(s.gate_proj.weight.detach(), s.up_proj.weight.detach(),
                              s.down_proj.weight.detach(), block.shared_expert_gate.weight.detach())
    return MoeWeights(
        block.gate.weight.detach(),   # HF 里 router 叫 `gate`（别和 expert 的 gate_proj 混淆）
        torch.stack([e.gate_proj.weight.detach() for e in ex]),
        torch.stack([e.up_proj.weight.detach() for e in ex]),
        torch.stack([e.down_proj.weight.detach() for e in ex]),
        block.top_k, block.norm_topk_prob, shared,
    )


# ---------------------------------------------------------------------------
# route：router logits -> softmax（对全部 E 个 expert，fp32）-> top-k -> 可选重归一化 -> cast 回 x.dtype。
# 易错：顺序是“先 softmax 再 top-k”。norm_topk_prob=True 时与“对 top-k logits 做 softmax”恰好等价，
#       False 时完全不同（top-k 权重和 < 1，不能擅自归一化）。
# 例：x = [[2, 1, 0, -1]]，router = eye(4)，k=2 -> ids [0, 1]；renorm 后 [0.7311, 0.2689]，不 renorm [0.6439, 0.2369]
# ---------------------------------------------------------------------------
def route(x: torch.Tensor, w: MoeWeights) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax((x @ w.router.T).float(), dim=-1)   # [N, E]，fp32 与 HF 一致
    topw, topi = torch.topk(probs, w.top_k, dim=-1)           # [N, k] 权重 / expert id（int64）
    if w.norm_topk_prob:
        # 只在选中的 k 个之间重新分配 100% 话语权（仍在 fp32 里做）
        topw = topw / topw.sum(-1, keepdim=True)
    # 最后才 cast 回 x.dtype：路由权重要和 expert 输出（x.dtype）相乘，留在 fp32 会把结果 dtype 提升
    return topw.to(x.dtype), topi


# ---------------------------------------------------------------------------
# moe_forward：稀疏 MoE 前向（dispatch / combine）。
#   1) 把 N 个 token × k 个槽位摊平成 N*k 个 (token, expert, weight) 三元组；
#   2) 按 expert id 稳定排序，bincount 统计每个 expert 分到多少个；
#   3) 逐 expert 收集它的 token 做一次 SwiGLU，乘路由权重后 index_add_ 回各自 token；
#   4) 若有 shared expert，所有 token 再加 sigmoid(gate) * shared_mlp(x)。
# ---------------------------------------------------------------------------
def moe_forward(x: torch.Tensor, w: MoeWeights) -> torch.Tensor:
    N = x.shape[0]                 # x: [N, H]，batch/seq 已展平
    topw, topi = route(x, w)       # [N, k], [N, k]
    flat_e = topi.reshape(-1)      # [N*k] 每个 (token, slot) 对的 expert id（row-major）
    # 第 p 个三元组属于 token p // k：[0,0,1,1,2,2,...]，与 reshape(-1) 的行优先顺序一一对应。
    # 易错：arange(N).repeat(k) 得到 [0,1,2,0,1,2]，和 flat_e 错位
    flat_tok = torch.arange(N).repeat_interleave(w.top_k)
    flat_w = topw.reshape(-1)      # [N*k]
    # stable 排序把同一 expert 的对排到一起，且同一 expert 内保持 token 原顺序（结果可复现）
    order = torch.argsort(flat_e, stable=True)
    # counts[e] = 分给 expert e 的对数；minlength=E 保证没被选中的 expert 也占一项（值 0）
    counts = torch.bincount(flat_e, minlength=w.router.shape[0]).tolist()
    out = torch.zeros_like(x)      # [N, H] 累加器
    # start 指向 order 中当前 expert 那一段的起点（counts 的前缀和）；各段按 e 升序首尾相接
    start = 0
    for e, n in enumerate(counts):
        if n == 0:
            continue               # 没人挂号的 expert 完全跳过（性能要求，也避免空 matmul）
        sel = order[start:start + n]   # 属于 expert e 的对在 flat_* 中的下标
        start += n
        tok = flat_tok[sel]        # [n] token id；top-k 的 id 互不相同，所以同一 expert 内 token 不重复
        h = x[tok]                 # [n, H] gather 这批 token
        # SwiGLU：down(silu(gate(h)) * up(h))；w_gate[e] 是 [I, H]，所以要转置：[n, H] @ [H, I] -> [n, I]
        y = (F.silu(h @ w.w_gate[e].T) * (h @ w.w_up[e].T)) @ w.w_down[e].T   # [n, H]
        # combine：乘路由权重后按 token 散射累加。
        # 易错：一个 token 会被 k 个不同 expert 处理，必须 index_add_ 累加；out[tok] = ... 会被后一个 expert 覆盖
        out.index_add_(0, tok, y * flat_w[sel, None])
    if w.shared is not None:
        s = w.shared
        # shared expert 对全部 token 计算（不经过 router）
        y = (F.silu(x @ s.gate_proj.T) * (x @ s.up_proj.T)) @ s.down_proj.T   # [N, H]
        # 门控是逐 token 标量：x @ gate.T -> [N, 1]，sigmoid 后沿 H 维广播
        out = out + torch.sigmoid(x @ s.gate.T) * y
    return out
