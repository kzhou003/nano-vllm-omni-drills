"""P15 · Thinker -> Talker input construction (Qwen2.5-Omni style, pure tensor functions).

trim_thinker_payload(prompt_ids, output_ids, hidden[R, D], finish_reason)
  vLLM appends the sampled stop token BEFORE checking stop, so a "stop" finish carries one
  extra row/id the talker must not consume. Keep target = P + len(output_ids) - drop rows
  (drop = 1 for "stop", else 0); prefill = rows[:P], decode = rows[P:target]; drop the stop
  id from output_ids. Raise ValueError if fewer than P rows were captured.
build_talker_prefill(prompt_embeds[P, D], first_reply[D], spk, codec_pad, codec_bos, *ids)
  embeds = [prompt rows] + [spk + codec_pad] + [first_reply + codec_bos]   -> [P + 2, D]
  ids    = [mask_id] * P + [pad_id, start_id]
talker_decode_input(reply_queue[K, D], last_codec_embed[D], fallback[D])
  input = next thinker row (or fallback when exhausted) + embedding of the last codec id;
  return (input, remaining_queue).
"""
from dataclasses import dataclass

import torch


@dataclass
class ThinkerPayload:
    prefill: torch.Tensor
    decode: torch.Tensor
    output_ids: list[int]


def trim_thinker_payload(
    prompt_ids: list[int],
    output_ids: list[int],
    hidden: torch.Tensor,
    finish_reason: str,
) -> ThinkerPayload:
    raise NotImplementedError


def build_talker_prefill(
    prompt_embeds: torch.Tensor,
    first_reply: torch.Tensor,
    spk_embed: torch.Tensor,
    codec_pad_embed: torch.Tensor,
    codec_bos_embed: torch.Tensor,
    *,
    mask_id: int,
    pad_id: int,
    start_id: int,
) -> tuple[list[int], torch.Tensor]:
    raise NotImplementedError


def talker_decode_input(
    reply_queue: torch.Tensor,
    last_codec_embed: torch.Tensor,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError
