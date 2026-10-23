import torch

from drill import load
from tests._tiny import HEAD_DIM, HIDDEN, MROPE_SECTION, M, thinker_text_config

m = load("m01_rmsnorm_mrope")


def mm_positions():
    # text(3) | image grid t=1,h=2,w=3 starting at 3 | text(2)
    t = [0, 1, 2, 3, 3, 3, 3, 3, 3, 5, 6]
    h = [0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 6]
    w = [0, 1, 2, 3, 4, 5, 3, 4, 5, 5, 6]
    return torch.tensor([t, h, w])


def test_rms_norm_matches_hf_fp32_and_bf16():
    norm = M.Qwen3OmniMoeThinkerTextRMSNorm(HIDDEN, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(HIDDEN))
    x = torch.randn(3, 7, HIDDEN) * 5
    torch.testing.assert_close(m.rms_norm(x, norm.weight, 1e-6), norm(x))
    xb = x.bfloat16()
    torch.testing.assert_close(m.rms_norm(xb, norm.weight.bfloat16(), 1e-6), norm.bfloat16()(xb))


def test_inv_freq():
    cfg = thinker_text_config()
    rot = M.Qwen3OmniMoeThinkerTextRotaryEmbedding(cfg)
    torch.testing.assert_close(m.rope_inv_freq(HEAD_DIM, cfg.rope_theta), rot.inv_freq)


def test_interleaved_mrope_matches_hf_qwen3_omni():
    cfg = thinker_text_config()
    rot = M.Qwen3OmniMoeThinkerTextRotaryEmbedding(cfg)
    pos = mm_positions()
    x = torch.zeros(1, pos.shape[1], HIDDEN)
    cos_hf, sin_hf = rot(x, pos[:, None, :])  # [1, T, D]
    cos, sin = m.mrope_cos_sin(rot.inv_freq, pos, MROPE_SECTION, interleaved=True)
    torch.testing.assert_close(cos, cos_hf[0])
    torch.testing.assert_close(sin, sin_hf[0])


def test_text_only_positions_reduce_to_plain_rope():
    inv = m.rope_inv_freq(HEAD_DIM, 10000.0)
    pos = torch.arange(9).expand(3, 9)
    cos_i, _ = m.mrope_cos_sin(inv, pos, MROPE_SECTION, interleaved=True)
    cos_c, _ = m.mrope_cos_sin(inv, pos, MROPE_SECTION, interleaved=False)
    freqs = torch.arange(9).float()[:, None] * inv[None]
    plain = torch.cat([freqs, freqs], -1).cos()
    torch.testing.assert_close(cos_i, plain)
    torch.testing.assert_close(cos_c, plain)


def test_apply_rotary_matches_hf_qwen3_omni():
    cfg = thinker_text_config()
    rot = M.Qwen3OmniMoeThinkerTextRotaryEmbedding(cfg)
    pos = mm_positions()
    T = pos.shape[1]
    q, k = torch.randn(T, 4, HEAD_DIM), torch.randn(T, 2, HEAD_DIM)
    cos_hf, sin_hf = rot(torch.zeros(1, T, HIDDEN), pos[:, None, :])
    q_hf, k_hf = M.apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos_hf, sin_hf)
    cos, sin = m.mrope_cos_sin(rot.inv_freq, pos, MROPE_SECTION, interleaved=True)
    torch.testing.assert_close(m.apply_rotary(q, cos, sin), q_hf[0].transpose(0, 1))
    torch.testing.assert_close(m.apply_rotary(k, cos, sin), k_hf[0].transpose(0, 1))


def test_chunked_mrope_matches_hf_qwen2_5_omni():
    from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as Q25

    pos = mm_positions()
    T = pos.shape[1]
    inv = m.rope_inv_freq(HEAD_DIM, 10000.0)
    freqs = pos.float()[:, :, None] * inv  # [3, T, half] -> full per-axis cos/sin like HF
    emb = torch.cat([freqs, freqs], -1)[:, None]  # [3, 1, T, D]
    q = torch.randn(1, 4, T, HEAD_DIM)
    q_hf, _ = Q25.apply_multimodal_rotary_pos_emb(q, q, emb.cos(), emb.sin(), MROPE_SECTION)
    cos, sin = m.mrope_cos_sin(inv, pos, MROPE_SECTION, interleaved=False)
    torch.testing.assert_close(m.apply_rotary(q[0].transpose(0, 1), cos, sin), q_hf[0].transpose(0, 1))
