"""Tiny random Qwen3-Omni configs/modules from HF transformers, used as the golden reference."""
import pytest
import torch

transformers = pytest.importorskip("transformers")
C = pytest.importorskip("transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe")
M = pytest.importorskip("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe")

HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 64, 4, 2, 16
MROPE_SECTION = [4, 2, 2]  # sums to HEAD_DIM // 2, same interleave pattern as the real [24, 20, 20]


def thinker_text_config(**kw):
    cfg = C.Qwen3OmniMoeTextConfig(
        vocab_size=97, hidden_size=HIDDEN, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
        moe_intermediate_size=32, num_experts=6, num_experts_per_tok=2, norm_topk_prob=True,
        rope_theta=10000.0, max_position_embeddings=256, rms_norm_eps=1e-6,
        rope_scaling={"rope_type": "default", "mrope_section": MROPE_SECTION, "interleaved": True},
        **kw,
    )
    cfg._attn_implementation = "eager"
    return cfg


def talker_text_config():
    cfg = C.Qwen3OmniMoeTalkerTextConfig(
        vocab_size=50, hidden_size=HIDDEN, intermediate_size=96, num_hidden_layers=1,
        num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, moe_intermediate_size=32,
        num_experts=6, num_experts_per_tok=2, norm_topk_prob=True, shared_expert_intermediate_size=48,
    )
    cfg._attn_implementation = "eager"
    return cfg


def code_predictor_config(num_code_groups=6, vocab=40):
    cfg = C.Qwen3OmniMoeTalkerCodePredictorConfig(
        vocab_size=vocab, hidden_size=HIDDEN, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
        num_code_groups=num_code_groups, max_position_embeddings=64, rms_norm_eps=1e-6,
    )
    cfg._attn_implementation = "eager"
    return cfg


def randomize_(module, seed=0, std=0.2):
    """HF inits RMSNorm weights to 1 and some linears to ~0; randomize so bugs are visible."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * std + (1.0 if p.ndim == 1 else 0.0))
    return module.eval()


def causal_mask(t, past=0):
    m = torch.full((t, past + t), float("-inf"))
    m = torch.triu(m, diagonal=past + 1)
    return m[None, None]
