"""M02 · Qwen3 thinker attention: QK-norm + MRoPE + GQA + causal, with a KV cache.

attention_forward(w, x[T, hidden], cos[T, D], sin[T, D], cache=None) -> [T, hidden]
  1. q = x @ q_proj.T -> [T, H, D];  k, v = ... -> [T, KVH, D]
  2. RMSNorm EACH HEAD of q and k over head_dim (q_norm / k_norm weights) -- before RoPE
  3. RoPE on q and k with the given cos/sin (rows belong to the NEW tokens)
  4. if cache is given: the new tokens sit at absolute positions len(cache) .. len(cache)+T-1;
     append k, v to the cache and attend over everything cached so far
  5. causal: query at absolute position p sees keys 0..p;  GQA: query head h uses kv head
     h // (H // KVH);  scale = D ** -0.5;  softmax in fp32
  6. out [T, H*D] @ o_proj.T
Prefill-all-at-once, chunked prefill and token-by-token decode must give identical rows.
"""
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------- given types
@dataclass
class AttnWeights:
    q_proj: torch.Tensor  # [num_heads * head_dim, hidden]
    k_proj: torch.Tensor  # [num_kv_heads * head_dim, hidden]
    v_proj: torch.Tensor  # [num_kv_heads * head_dim, hidden]
    o_proj: torch.Tensor  # [hidden, num_heads * head_dim]
    q_norm: torch.Tensor  # [head_dim]  RMSNorm weight applied per head
    k_norm: torch.Tensor  # [head_dim]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    eps: float = 1e-6


@dataclass
class KVCache:
    k: list[torch.Tensor] = field(default_factory=list)  # chunks of [t, num_kv_heads, head_dim]
    v: list[torch.Tensor] = field(default_factory=list)

    def __len__(self) -> int:
        return sum(x.shape[0] for x in self.k)


def from_hf(att) -> AttnWeights:
    """Pull the weights out of a HF Qwen3OmniMoeThinkerTextAttention."""
    return AttnWeights(
        att.q_proj.weight.detach(), att.k_proj.weight.detach(), att.v_proj.weight.detach(),
        att.o_proj.weight.detach(), att.q_norm.weight.detach(), att.k_norm.weight.detach(),
        att.config.num_attention_heads, att.config.num_key_value_heads, att.head_dim, att.config.rms_norm_eps,
    )


# ------------------------------------------------------------ implement below
def attention_forward(
    w: AttnWeights,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache: KVCache | None = None,
) -> torch.Tensor:
    raise NotImplementedError
