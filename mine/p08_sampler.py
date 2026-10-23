"""P08 · Batch-invariant sampler with per-request seeds.

- apply_repetition_penalty: for every distinct previous token, positive logits are
  divided by the penalty and negative ones multiplied (do not mutate the input).
- sample(): per row -> repetition penalty -> temperature < 1e-5 means greedy argmax;
  otherwise divide by temperature, apply top_k (keep k largest) then top_p (keep the
  smallest prefix of the sorted distribution whose mass reaches top_p; always keep >= 1),
  then draw ONE token.
- Seeded requests own a torch.Generator created on first use from params.seed; the token
  drawn for a request must not depend on which other rows share the batch, or their order.
- free(req_id) forgets the generator (a new request with the same id restarts the stream).
"""
from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None


def apply_repetition_penalty(logits: torch.Tensor, prev_tokens: list[int], penalty: float) -> torch.Tensor:
    raise NotImplementedError


class Sampler:
    def __init__(self) -> None:
        raise NotImplementedError

    def free(self, req_id: str) -> None:
        raise NotImplementedError

    def sample(
        self,
        logits: torch.Tensor,  # [B, V]
        req_ids: list[str],
        params: list[SamplingParams],
        prev_tokens: list[list[int]],
    ) -> torch.Tensor:  # [B] long
        raise NotImplementedError
