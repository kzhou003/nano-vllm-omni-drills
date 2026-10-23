import torch

from drill import load
from tests._tiny import HIDDEN, M, causal_mask, randomize_, thinker_text_config

m = load("m02_thinker_attention")


def setup(T=10, seed=0):
    cfg = thinker_text_config()
    att = randomize_(M.Qwen3OmniMoeThinkerTextAttention(cfg, layer_idx=0), seed)
    rot = M.Qwen3OmniMoeThinkerTextRotaryEmbedding(cfg)
    torch.manual_seed(seed)
    x = torch.randn(T, HIDDEN)
    pos = torch.arange(T).expand(3, T).clone()
    pos[1, 3:7] += torch.tensor([0, 0, 1, 1])  # a tiny 2x2 "image" grid in the middle
    pos[2, 3:7] += torch.tensor([0, 1, 0, 1])
    cos, sin = rot(x[None], pos[:, None, :])
    with torch.no_grad():
        ref, _ = att(x[None], (cos, sin), causal_mask(T))
    return att, x, cos[0], sin[0], ref[0]


def test_prefill_matches_hf():
    att, x, cos, sin, ref = setup()
    out = m.attention_forward(m.from_hf(att), x, cos, sin)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_decode_one_token_at_a_time_with_cache():
    att, x, cos, sin, ref = setup(T=12, seed=1)
    w, cache = m.from_hf(att), m.KVCache()
    out = m.attention_forward(w, x[:5], cos[:5], sin[:5], cache)
    rows = [out]
    for t in range(5, 12):
        rows.append(m.attention_forward(w, x[t:t + 1], cos[t:t + 1], sin[t:t + 1], cache))
    torch.testing.assert_close(torch.cat(rows), ref, atol=1e-5, rtol=1e-4)
    assert len(cache) == 12


def test_chunked_prefill_with_cache():
    att, x, cos, sin, ref = setup(T=11, seed=2)
    w, cache = m.from_hf(att), m.KVCache()
    a = m.attention_forward(w, x[:4], cos[:4], sin[:4], cache)
    b = m.attention_forward(w, x[4:], cos[4:], sin[4:], cache)  # a chunk of 7 after 4 cached tokens
    torch.testing.assert_close(torch.cat([a, b]), ref, atol=1e-5, rtol=1e-4)
