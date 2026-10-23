"""P17 · GraphRunner: batch-size buckets over captured graphs (CPU emulation of CUDA graphs).

- __init__: allocate static_in / static_out of shape [max bucket, hidden] once.
- capture(): one CpuGraph per bucket b over the views static_in[:b] / static_out[:b]
  (largest bucket first), call .capture(), keep them in self.graphs[b].
- run(x, force_eager=False):
    pick the smallest bucket >= bs; if none, or force_eager (e.g. rows carry their own RNG
    generators, which a replay would silently ignore) -> stats["eager"] += 1, return fn(x);
    else copy x into static_in[:bs], ZERO static_in[bs:b] (stale rows), replay,
    stats["replays"] += 1, stats["last_bucket"] = b, and return a result that is NOT a view
    of static_out (the next replay overwrites it).
- Never capture at run time.
"""
from collections.abc import Callable

import torch


# ---------------------------------------------------------------- given types
class CpuGraph:
    """Emulates torch.cuda.CUDAGraph semantics on CPU: replay() re-runs the recorded work on
    the *same static buffers*; it never sees new Python arguments."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor], static_in: torch.Tensor, static_out: torch.Tensor):
        self._fn, self._in, self._out = fn, static_in, static_out
        self.captured = False

    def capture(self) -> None:
        self._out.copy_(self._fn(self._in))
        self.captured = True

    def replay(self) -> None:
        assert self.captured
        self._out.copy_(self._fn(self._in))


# ------------------------------------------------------------ implement below
class GraphRunner:
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor], buckets: list[int], hidden: int,
                 dtype: torch.dtype = torch.float32) -> None:
        self.fn = fn
        self.buckets = sorted(buckets)
        self.graphs: dict[int, CpuGraph] = {}
        self.stats = {"replays": 0, "eager": 0, "last_bucket": None}
        raise NotImplementedError

    def capture(self) -> None:
        raise NotImplementedError

    def run(self, x: torch.Tensor, *, force_eager: bool = False) -> torch.Tensor:
        raise NotImplementedError
