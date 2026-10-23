"""M08 · Assemble a text-only thinker from YOUR m01/m02/m03/m07 and decode greedily.

TinyThinker(state, dims): load with your load_thinker_text; lm_head = state["thinker.lm_head.weight"].
forward(ids[T], start, caches) -> logits [T, V] for tokens at absolute positions start..start+T-1:
  text-only MRoPE positions (the same index on all 3 axes, interleaved layout),
  h = embed[ids]; per layer: h += attn(rms(h, ln1)); h += moe(rms(h, ln2)); logits = rms(h, norm) @ lm_head.T
  One KVCache per layer.
greedy(prompt, max_new): prefill the prompt once, then decode ONE token per forward using the
  caches. Must equal HF full-recompute greedy decoding.
"""
from dataclasses import dataclass

import torch

from .m02_thinker_attention import KVCache  # noqa: F401  (re-exported for the tests)


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
        raise NotImplementedError

    def forward(self, ids: torch.Tensor, start: int, caches: list) -> torch.Tensor:
        raise NotImplementedError

    def greedy(self, prompt: list[int], max_new: int) -> list[int]:
        raise NotImplementedError
