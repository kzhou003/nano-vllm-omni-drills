"""M07 · Stage-aware checkpoint loading and weight packing.

A Qwen3-Omni checkpoint holds every stage: thinker.*, talker.*, code2wav.*. Each stage process
must open only its shards and load only its tensors, then pack them for fast kernels.
shards_for_stage(weight_map, stage): sorted shard files containing at least one key of the stage.
select_stage(state, stage, strip): keep keys starting with `strip` (e.g. "thinker.model.") and
  remove that prefix.
pack_layer(sd, i) -> PackedLayer for "layers.{i}.*":
  qkv = cat(q_proj, k_proj, v_proj) along rows; w13[e] = cat(gate_proj_e, up_proj_e) along
  rows; w2[e] = down_proj_e; router = mlp.gate.weight. Expert ids must be exactly 0..E-1 and
  complete; any missing or unexpected key in the layer -> LoadError.
load_thinker_text(state, num_layers) -> ({"embed_tokens", "norm"}, [PackedLayer]); anything
  else under thinker.model. besides layers/embed_tokens/norm -> LoadError.
"""
import re
from dataclasses import dataclass

import torch

STAGE_PREFIX = {"thinker": "thinker.", "talker": "talker.", "code2wav": "code2wav."}


class LoadError(Exception):
    pass


@dataclass
class PackedLayer:
    input_layernorm: torch.Tensor
    qkv: torch.Tensor                # [(H + 2*KVH) * D, hidden]  rows: q | k | v
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o_proj: torch.Tensor
    post_attention_layernorm: torch.Tensor
    router: torch.Tensor             # [E, hidden]
    w13: torch.Tensor                # [E, 2*I, hidden]  rows: gate | up
    w2: torch.Tensor                 # [E, hidden, I]


def shards_for_stage(weight_map: dict[str, str], stage: str) -> list[str]:
    raise NotImplementedError


def select_stage(state: dict[str, torch.Tensor], stage: str, strip: str) -> dict[str, torch.Tensor]:
    raise NotImplementedError


def pack_layer(sd: dict[str, torch.Tensor], i: int) -> PackedLayer:
    raise NotImplementedError


def load_thinker_text(state: dict[str, torch.Tensor], num_layers: int) -> tuple[dict, list[PackedLayer]]:
    raise NotImplementedError
