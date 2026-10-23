import random

import torch

from drill import load
from tests._tiny import M, randomize_

m = load("m06_streaming_causal_conv")


def random_chunks(total, seed):
    rnd = random.Random(seed)
    sizes, left = [], total
    while left:
        sizes.append(min(left, rnd.randint(1, 6)))
        left -= sizes[-1]
    return sizes


def test_causal_conv_streaming_equals_offline():
    for seed, (k, d) in enumerate([(3, 1), (5, 2), (7, 1), (1, 1)]):
        conv = randomize_(M.Qwen3OmniMoeCausalConvNet(4, 6, kernel_size=k, dilation=d), seed)
        x = torch.randn(2, 4, 37)
        with torch.no_grad():
            ref = conv(x)
        s = m.StreamingCausalConv1d.from_hf(conv)
        outs, pos = [], 0
        for n in random_chunks(37, seed):
            y = s.step(x[..., pos:pos + n])
            assert y.shape[-1] == n, "a causal stride-1 conv emits exactly one output per input"
            outs.append(y)
            pos += n
        torch.testing.assert_close(torch.cat(outs, -1), ref, atol=1e-5, rtol=1e-4)


def test_trans_conv_streaming_equals_offline():
    for seed, (k, st) in enumerate([(4, 2), (10, 5), (6, 3), (3, 3)]):
        conv = randomize_(M.Qwen3OmniMoeCausalTransConvNet(4, 5, kernel_size=k, stride=st), seed)
        x = torch.randn(2, 4, 23)
        with torch.no_grad():
            ref = conv(x)
        s = m.StreamingCausalTransConv1d.from_hf(conv)
        outs, pos = [], 0
        for n in random_chunks(23, seed + 10):
            outs.append(s.step(x[..., pos:pos + n]))
            pos += n
        outs.append(s.flush())
        torch.testing.assert_close(torch.cat(outs, -1), ref, atol=1e-5, rtol=1e-4)


def test_trans_conv_emits_as_early_as_possible():
    k, st = 8, 4
    conv = randomize_(M.Qwen3OmniMoeCausalTransConvNet(3, 3, kernel_size=k, stride=st), 0)
    s = m.StreamingCausalTransConv1d.from_hf(conv)
    emitted = 0
    for n in range(1, 8):
        emitted += s.step(torch.randn(1, 3, 1)).shape[-1]
        # positions < n*stride are final; the first (k - stride) are trimmed
        assert emitted == max(0, n * st - (k - st)), (n, emitted)
