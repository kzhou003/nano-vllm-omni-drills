"""M03 · Sparse MoE block (thinker) and MoE + gated shared expert (talker).

route(x[N, H], w) -> (weights [N, k] in x.dtype, expert ids [N, k])
  softmax(x @ router.T) over ALL experts in fp32 -> top-k -> renormalize to sum 1 if
  norm_topk_prob.
moe_forward(x[N, H], w) -> [N, H]
  out[n] = sum_j weight[n, j] * Expert_{id[n, j]}(x[n]),  Expert(h) = (silu(h Wg^T) * (h Wu^T)) Wd^T
  Do it the serving way: group (token, slot) pairs by expert, run each expert ONCE on all of
  its tokens, scatter-add the weighted results; skip experts that received no token.
  If w.shared is set (talker): out += sigmoid(x @ shared.gate.T) * SharedExpert(x).
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- given types
@dataclass
class SharedExpert:
    gate_proj: torch.Tensor   # [I_s, H]
    up_proj: torch.Tensor     # [I_s, H]
    down_proj: torch.Tensor   # [H, I_s]
    gate: torch.Tensor        # [1, H]   sigmoid gate of the shared expert


@dataclass
class MoeWeights:
    router: torch.Tensor      # [E, H]
    w_gate: torch.Tensor      # [E, I, H]   stacked expert gate_proj
    w_up: torch.Tensor        # [E, I, H]
    w_down: torch.Tensor      # [E, H, I]
    top_k: int
    norm_topk_prob: bool
    shared: SharedExpert | None = None


def from_hf(block) -> MoeWeights:
    """HF Qwen3OmniMoe{Thinker,Talker}TextSparseMoeBlock -> stacked tensors."""
    ex = block.experts
    shared = None
    if hasattr(block, "shared_expert"):
        s = block.shared_expert
        shared = SharedExpert(s.gate_proj.weight.detach(), s.up_proj.weight.detach(),
                              s.down_proj.weight.detach(), block.shared_expert_gate.weight.detach())
    return MoeWeights(
        block.gate.weight.detach(),
        torch.stack([e.gate_proj.weight.detach() for e in ex]),
        torch.stack([e.up_proj.weight.detach() for e in ex]),
        torch.stack([e.down_proj.weight.detach() for e in ex]),
        block.top_k, block.norm_topk_prob, shared,
    )


# ------------------------------------------------------------ implement below
def route(x: torch.Tensor, w: MoeWeights) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def moe_forward(x: torch.Tensor, w: MoeWeights) -> torch.Tensor:
    raise NotImplementedError
