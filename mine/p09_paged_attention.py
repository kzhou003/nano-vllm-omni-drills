"""P09 · Paged KV write + decode attention (PyTorch reference; Triton is the follow-up).

Cache layout: [num_blocks, block_size, num_kv_heads, head_dim].
- write_kv: scatter token i's K/V into slot slot_mapping[i] = block*block_size + offset;
  slot -1 marks a padding token that must NOT be written.
- paged_decode_attention: one query token per sequence. Sequence b attends to its first
  context_lens[b] cached tokens, found through block_tables[b] (entries past the last
  used block are garbage). Support GQA: num_heads is a multiple of num_kv_heads and query
  head h uses kv head h // (num_heads // num_kv_heads). Softmax in fp32.
"""
import torch


def write_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens]
) -> None:
    raise NotImplementedError


def paged_decode_attention(
    q: torch.Tensor,  # [B, num_heads, head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,  # [B, max_blocks]
    context_lens: torch.Tensor,  # [B]
    scale: float,
) -> torch.Tensor:  # [B, num_heads, head_dim]
    raise NotImplementedError
