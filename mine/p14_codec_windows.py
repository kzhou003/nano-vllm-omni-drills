"""P14 · Codec window planning for streaming vocoding (pure functions).

next_window(available, emitted, finished, *, initial, chunk, left_ctx)
  available: frames produced so far; emitted: frames already shipped.
  The first window holds `initial` frames (if initial > 0), later ones `chunk` frames.
  Return (ctx_start, new_start, new_end) when a full window is available; when finished,
  flush whatever remains; otherwise None. new_start = emitted and
  ctx_start = max(0, new_start - left_ctx) (left context is re-decoded, not re-emitted).
slice_new_samples(decoded, window, upsample)
  decoded = vocoder(codes[ctx_start:new_end]) has (new_end - ctx_start) * upsample samples;
  return only the samples that belong to frames [new_start, new_end).
"""
from collections.abc import Sequence


def next_window(
    available: int,
    emitted: int,
    finished: bool,
    *,
    initial: int,
    chunk: int,
    left_ctx: int,
) -> tuple[int, int, int] | None:
    raise NotImplementedError


def slice_new_samples(decoded: Sequence[float], window: tuple[int, int, int], upsample: int) -> list[float]:
    raise NotImplementedError
