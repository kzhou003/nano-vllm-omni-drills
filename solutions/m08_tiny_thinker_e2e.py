"""M08 reference solution · 用 M01/M02/M03/M07 拼一个纯文本 thinker，并做 KV-cache greedy 解码。

embed -> N × [pre-norm attention + 残差, pre-norm MoE + 残差] -> final norm -> lm_head。
生成的 token 序列必须与 HF 每步全量重算的 greedy 完全一致。提示页：hints/m08_overview.html（A-8）。
"""
from dataclasses import dataclass

import torch

from .m01_rmsnorm_mrope import mrope_cos_sin, rms_norm, rope_inv_freq
from .m02_thinker_attention import AttnWeights, KVCache, attention_forward
from .m03_sparse_moe import MoeWeights, moe_forward
from .m07_stage_weight_loading import load_thinker_text


# Dims：模型结构超参（config 的子集）。eps 同时用于所有 RMSNorm（包括 attention 里的 q_norm/k_norm）。
@dataclass
class Dims:
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    top_k: int
    norm_topk_prob: bool
    rope_theta: float
    mrope_section: list[int]
    eps: float = 1e-6


class TinyThinker:
    def __init__(self, state: dict[str, torch.Tensor], dims: Dims) -> None:
        self.dims = dims
        g, layers = load_thinker_text(state, dims.num_layers)
        self.embed, self.norm = g["embed_tokens"], g["norm"]   # embed: [V, hidden]
        # lm_head 在 "thinker.lm_head." 下而非 "thinker.model." 下，load_thinker_text 不会返回它
        self.lm_head = state["thinker.lm_head.weight"]          # [V, hidden]
        # 逆频率表只依赖 head_dim 和 theta，初始化时算一次
        self.inv_freq = rope_inv_freq(dims.head_dim, dims.rope_theta)
        H, KVH, D = dims.num_heads, dims.num_kv_heads, dims.head_dim
        self.layers = []
        for p in layers:
            # 把 M07 的打包布局拆回 M02 / M03 需要的形式：qkv 的行按 q | k | v 切
            q, k, v = p.qkv.split([H * D, KVH * D, KVH * D], 0)
            attn = AttnWeights(q, k, v, p.o_proj, p.q_norm, p.k_norm, H, KVH, D, dims.eps)
            # w13: [E, 2I, hidden]，在 dim=1 对半：前 I 行是 gate、后 I 行是 up（与 M07 的拼接顺序对应）。
            # 易错：chunk(2, dim=0) 是沿 expert 维切
            w_gate, w_up = p.w13.chunk(2, dim=1)
            moe = MoeWeights(p.router, w_gate, w_up, p.w2, dims.top_k, dims.norm_topk_prob)
            self.layers.append((p.input_layernorm, attn, p.post_attention_layernorm, moe))

    def forward(self, ids: torch.Tensor, start: int, caches: list) -> torch.Tensor:
        """ids: [T] 新 token，绝对位置 start..start+T-1 -> logits [T, V]（返回全部行，测试会比较 prefill 的每一行）。"""
        T = ids.shape[0]
        # 纯文本：t/h/w 三个位置分量相同 -> [3, T]。此时 M-RoPE 退化为 1D RoPE，但仍走 interleaved 路径以对齐 HF。
        # 不变量：start 必须等于每层 cache 的当前长度——M02 按 len(cache) 做因果 mask，RoPE 位置由这里决定，
        # 两者不一致时不报错，输出会在若干步后才与 HF 分叉
        pos = torch.arange(start, start + T).expand(3, T)
        # cos/sin 只覆盖新 token，所有层共享（cache 里旧 token 的 K 写入时已经用自己的位置旋转过）
        cos, sin = mrope_cos_sin(self.inv_freq, pos, self.dims.mrope_section, interleaved=True)   # [T, D]
        eps = self.dims.eps
        h = self.embed[ids]   # [T, hidden] 查表
        # 每层一个独立的 KVCache；attention_forward 会就地追加 K/V
        for (ln1, attn, ln2, moe), cache in zip(self.layers, caches):
            # pre-norm 残差：norm 只作用在子层输入上，残差加的是未 norm 的 h。
            # 易错：写成 h = rms(h) + attn(...) 时 prefill logits 立刻就不对
            h = h + attention_forward(attn, rms_norm(h, ln1, eps), cos, sin, cache)
            h = h + moe_forward(rms_norm(h, ln2, eps), moe)
        # 最后一层之后还有一次 final norm（忘了它 logits 量级会明显偏大），再乘 lm_head.T
        return rms_norm(h, self.norm, eps) @ self.lm_head.T   # [T, V]

    @torch.no_grad()
    def greedy(self, prompt: list[int], max_new: int) -> list[int]:
        # 每次 greedy 都是新请求：caches 在这里新建（放 __init__ 会继承上次历史，放 forward 里等于没有 cache）
        caches = [KVCache() for _ in range(self.dims.num_layers)]
        # prefill：整段 prompt 从位置 0 开始；最后一行 logits 直接给出第一个新 token
        logits = self.forward(torch.tensor(prompt), 0, caches)
        out = []
        # 一共 max_new 次 forward：1 次 prefill + (max_new - 1) 次 decode；最后一个 token 不需要再 forward
        for _ in range(max_new):
            tok = int(logits[-1].argmax())   # 只用最后一行
            out.append(tok)
            if len(out) == max_new:
                break
            # decode：只喂刚生成的 1 个 token，它的绝对位置 = len(prompt) + len(out) - 1（== 当前 cache 长度）。
            # 易错：写成 len(prompt) + len(out) 会让 RoPE 位置整体偏 1，几步之后与 HF 分叉；
            #       每步把完整序列从 start=0 再喂进同一组 caches 会让 cache 重复增长、很快乱码
            logits = self.forward(torch.tensor([tok]), len(prompt) + len(out) - 1, caches)
        return out
