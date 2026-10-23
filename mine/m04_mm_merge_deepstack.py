"""M04 · Placeholder bookkeeping, multimodal embedding merge, DeepStack injection.

audio_token_count(num_mel_frames) (int or LongTensor): number of <audio> placeholders the
  Qwen3-Omni audio encoder produces. Mel frames are processed in windows of 100 frames; a
  full window yields 13 tokens; the remainder r = n % 100 passes three stride-2 convolutions:
  f = (r - 1) // 2 + 1;  tokens_r = ((f - 1) // 2 + 1 - 1) // 2 + 1.  total = tokens_r + 13 * (n // 100)
merge_multimodal(input_ids[T], text_embeds[T, H], placeholder_ids, mm_embeds)
  -> (inputs_embeds[T, H], masks{modality: bool[T]})
  Replace the rows at each modality's placeholder positions, in order, with that modality's
  encoder outputs. Count mismatch (or placeholders without embeddings) -> ValueError.
  Never mutate text_embeds.
apply_deepstack(hidden[T, H], visual_mask[T], visual_embeds[n_vis, H]) -> new tensor with
  visual_embeds ADDED at the visual rows.
run_layers(layers, hidden, visual_mask, deepstack_embeds): run the decoder layers in order;
  DeepStack level i is added right AFTER layer i (only for i < len(deepstack_embeds)).
"""
from collections.abc import Callable

import torch


def audio_token_count(num_mel_frames):
    raise NotImplementedError


def merge_multimodal(
    input_ids: torch.Tensor,
    text_embeds: torch.Tensor,
    placeholder_ids: dict[str, int],
    mm_embeds: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raise NotImplementedError


def apply_deepstack(hidden: torch.Tensor, visual_mask: torch.Tensor, visual_embeds: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def run_layers(
    layers: list[Callable[[torch.Tensor], torch.Tensor]],
    hidden: torch.Tensor,
    visual_mask: torch.Tensor,
    deepstack_embeds: list[torch.Tensor],
) -> torch.Tensor:
    raise NotImplementedError
