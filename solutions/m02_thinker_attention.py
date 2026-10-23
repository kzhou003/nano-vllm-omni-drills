"""M02 reference solution · Qwen3 thinker attention：QK-norm + MRoPE + GQA + causal，带 KV cache。

同一个函数既做整段 prefill、分块 prefill，也做逐 token decode，三者结果逐行一致。
提示页：hints/m02_overview.html（A-2）。
"""
from dataclasses import dataclass, field

import torch

# 直接复用 M01：per-head QK-norm 就是对最后一维 head_dim 做 RMSNorm；RoPE 用 apply_rotary
from .m01_rmsnorm_mrope import apply_rotary, rms_norm


# AttnWeights：一层 thinker self-attention 的全部权重（从 HF 模块里摘出来的裸张量）。
# GQA：num_heads 个 query head 共享 num_kv_heads 组 K/V（num_heads % num_kv_heads == 0）。
@dataclass
class AttnWeights:
    q_proj: torch.Tensor   # [num_heads * head_dim, hidden]，行按 head 分块（前 D 行是 head 0）
    k_proj: torch.Tensor   # [num_kv_heads * head_dim, hidden]
    v_proj: torch.Tensor   # [num_kv_heads * head_dim, hidden]
    o_proj: torch.Tensor   # [hidden, num_heads * head_dim]
    q_norm: torch.Tensor   # [head_dim]   所有 head 共享的 RMSNorm 权重
    k_norm: torch.Tensor   # [head_dim]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    eps: float = 1e-6


# KVCache：一层的 KV cache，按块追加（prefill 一个大块，decode 每步一个 [1, ...] 块）。
# 不变量：len(cache) = 已缓存 token 数 = 下一批新 token 的起始绝对位置（cache offset）。
@dataclass
class KVCache:
    k: list[torch.Tensor] = field(default_factory=list)   # chunks of [t, num_kv_heads, head_dim]，已 RoPE
    v: list[torch.Tensor] = field(default_factory=list)

    def __len__(self) -> int:
        return sum(x.shape[0] for x in self.k)


def from_hf(att) -> AttnWeights:
    return AttnWeights(
        att.q_proj.weight.detach(), att.k_proj.weight.detach(), att.v_proj.weight.detach(),
        att.o_proj.weight.detach(), att.q_norm.weight.detach(), att.k_norm.weight.detach(),
        att.config.num_attention_heads, att.config.num_key_value_heads, att.head_dim, att.config.rms_norm_eps,
    )


# ---------------------------------------------------------------------------
# attention_forward：流程 = q/k/v 投影 -> 按 head 切分 -> QK-norm -> RoPE -> 追加 cache
#   -> GQA 扩展 -> 按绝对位置的因果 softmax(QK^T/√D) V（fp32）-> o_proj。
# ---------------------------------------------------------------------------
def attention_forward(
    w: AttnWeights,
    x: torch.Tensor,       # [T, hidden]       只有新 token
    cos: torch.Tensor,     # [T, head_dim]     只对应新 token 的绝对位置（调用方已切好）
    sin: torch.Tensor,
    cache: KVCache | None = None,
) -> torch.Tensor:         # [T, hidden]
    T = x.shape[0]
    H, KVH, D = w.num_heads, w.num_kv_heads, w.head_dim
    # 投影后 view 成 [T, heads, D]：q_proj 的行按 head 分块，所以最后一维连续 D 个数属于同一个 head
    q = (x @ w.q_proj.T).view(T, H, D)
    k = (x @ w.k_proj.T).view(T, KVH, D)
    v = (x @ w.v_proj.T).view(T, KVH, D)   # v 不做 norm
    # 易错：QK-norm 必须在 RoPE 之前（HF 顺序）。RMS 对旋转不变，但逐维的 q_norm 权重不是，顺序反了数值就不同。
    # rms_norm 对最后一维 D 做，[D] 的权重自动广播到 [T, H, D]（每个 head 独立归一化）
    q = apply_rotary(rms_norm(q, w.q_norm, w.eps), cos, sin)
    k = apply_rotary(rms_norm(k, w.k_norm, w.eps), cos, sin)
    # past = 本次之前已缓存的 token 数 = 新 token 的起始绝对位置
    past = 0
    if cache is not None:
        # 易错：必须在 append 之前读长度，否则 past 多算了 T，mask 整体右移，token 会“偷看”同块后面的 token
        past = len(cache)
        # cache 里存的是“RoPE 之后、GQA 扩展之前”的 k（只存 KVH 组省显存）；旧 k 永远不用重新旋转
        cache.k.append(k)
        cache.v.append(v)
        K, V = torch.cat(cache.k), torch.cat(cache.v)   # [S, KVH, D]，S = past + T
    else:
        K, V = k, v   # 普通 prefill：past = 0
    # GQA：q head h 用 kv head h // rep，即相邻 rep 个 q head 共享一个 kv head。
    # 易错：必须 repeat_interleave（[k0,k0,k1,k1]）而不是 repeat（[k0,k1,k0,k1]）；H == KVH 时两者无差，测不出来
    rep = H // KVH
    K = K.repeat_interleave(rep, dim=1)  # [S, H, D]
    V = V.repeat_interleave(rep, dim=1)
    # scores[h, t, s] = q_t · k_s / √D，在 fp32 里做 scale / mask / softmax
    scores = torch.einsum("thd,shd->hts", q, K).float() * D ** -0.5
    S = K.shape[0]
    # 因果 mask 按“绝对位置”构造：新 token t 的位置是 past + t，key s 的位置就是 s。
    # 易错：用块内下标 arange(T)（或 triu(ones(T, S), 1)）时，decode 的 token 会以为自己在位置 0，只看得见 key 0
    q_pos = past + torch.arange(T)[:, None]   # [T, 1]
    k_pos = torch.arange(S)[None, :]          # [1, S]
    scores = scores.masked_fill(k_pos > q_pos, float("-inf"))   # 每个 query 至少看得见 key 0，不会整行 -inf
    # softmax 在 key 维（dim=-1）上归一化；易错：dim=-2 会让每列和为 1
    p = torch.softmax(scores, -1).to(V.dtype)
    # 加权求和后各 head 拼回 [T, H*D]（head 在前、D 在后，与第一步的 view 对称），再投影回 hidden
    out = torch.einsum("hts,shd->thd", p, V).reshape(T, H * D)
    return out @ w.o_proj.T
