"""M06 · Streaming versions of Code2Wav's causal Conv1d and causal ConvTranspose1d.

StreamingCausalConv1d (stride 1, dilation d, kernel K)
  The offline module left-pads the whole signal with (K-1)*d zeros. step(x[B, C, t]) must
  return exactly t outputs and, concatenated over any chunking, equal the offline output.
  Keep the last (K-1)*d inputs as state.
StreamingCausalTransConv1d (stride s, kernel K, weight [C_in, C_out, K])
  Input frame i writes output positions [i*s, i*s + K) (overlap-add). The offline module trims
  K - s samples at the START and at the END. step(x) returns every sample that is already
  final (after n frames: positions < n*s, minus the global left trim); flush() returns the rest
  minus the right trim. Add the bias once per output position, not once per contribution.
from_hf(m) builds each class from the HF module (weights .conv.weight / .conv.bias).
"""
import torch


class StreamingCausalConv1d:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, dilation: int = 1) -> None:
        raise NotImplementedError

    @classmethod
    def from_hf(cls, m) -> "StreamingCausalConv1d":  # HF Qwen3OmniMoeCausalConvNet (m.conv, m.dilation, m.stride)
        raise NotImplementedError

    def step(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StreamingCausalTransConv1d:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, stride: int) -> None:
        raise NotImplementedError

    @classmethod
    def from_hf(cls, m) -> "StreamingCausalTransConv1d":  # HF Qwen3OmniMoeCausalTransConvNet (m.conv)
        raise NotImplementedError

    def step(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def flush(self) -> torch.Tensor:
        raise NotImplementedError
