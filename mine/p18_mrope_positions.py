"""P18 · MRoPE (Qwen2-VL style) 3-D positions for interleaved text and images.

segments: ("text", n) or ("image", (t, h, w)) with h, w in vision patches; an image
becomes t * (h // merge) * (w // merge) LLM tokens laid out t-major, then h, then w.
- text tokens: the same running index on the temporal, height and width axes;
- image tokens: (t_idx, h_idx, w_idx) + start;
- after each segment, start = max(position so far) + 1.
Return (positions[3, T] long, delta) with delta = (max position + 1) - T, so a token at
absolute index i >= T decodes at position i + delta on all three axes (decode_position).
"""
import torch

Segment = tuple[str, int | tuple[int, int, int]]


def mrope_positions(segments: list[Segment], spatial_merge: int = 2) -> tuple[torch.Tensor, int]:
    raise NotImplementedError


def decode_position(index: int, delta: int) -> torch.Tensor:
    raise NotImplementedError
