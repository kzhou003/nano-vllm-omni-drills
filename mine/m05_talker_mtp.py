"""M05 · Talker MTP: predict the residual codebooks of one audio frame.

`cp` is a HF Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration. You may only use:
  cp.config.num_code_groups (G), cp.model(inputs_embeds=..., [past_key_values=, use_cache=])
  .last_hidden_state, cp.lm_head[j] (j = 0..G-2), cp.model.codec_embedding[j] (j = 0..G-2).
predict_residual_codes(cp, last_talker_hidden[B, H], cb0_embed[B, H]) -> codes [B, G-1]
  Greedy. The sequence starts as [last_talker_hidden, cb0_embed]. For j = 0..G-2: run the
  transformer over the WHOLE sequence (re-prefill, no cache), take the last position, logits
  = lm_head[j], code = argmax, then append codec_embedding[j](code) to the sequence.
predict_residual_codes_kv(...): identical result with an incremental KV cache
  (transformers.DynamicCache): prefill the 2 initial positions, then feed 1 position per step.
next_talker_input(cp, cb0_embed, residual_codes, text_step) -> [B, H]
  cb0_embed + sum_j codec_embedding[j](residual_codes[:, j]) + text_step
"""
import torch


def predict_residual_codes(cp, last_talker_hidden: torch.Tensor, cb0_embed: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def predict_residual_codes_kv(cp, last_talker_hidden: torch.Tensor, cb0_embed: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def next_talker_input(cp, cb0_embed: torch.Tensor, residual_codes: torch.Tensor, text_step: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError
