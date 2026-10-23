import pytest
import torch
import torch.nn.functional as F

from drill import load
from tests._tiny import HIDDEN, M, randomize_, thinker_text_config

m = load("m07_stage_weight_loading")


def checkpoint():
    cfg = thinker_text_config()
    model = randomize_(M.Qwen3OmniMoeThinkerTextModel(cfg), seed=0)
    state = {f"thinker.model.{k}": v.detach().clone() for k, v in model.state_dict().items()}
    state["thinker.lm_head.weight"] = torch.randn(cfg.vocab_size, HIDDEN)
    state["talker.model.layers.0.self_attn.q_proj.weight"] = torch.randn(8, 8)
    state["code2wav.decoder.0.conv.weight"] = torch.randn(4, 4, 3)
    return cfg, model, state


def test_shard_planning():
    wm = {"thinker.model.a": "s1", "thinker.model.b": "s2", "talker.x": "s2", "code2wav.y": "s3", "thinker.lm_head": "s1"}
    assert m.shards_for_stage(wm, "thinker") == ["s1", "s2"]
    assert m.shards_for_stage(wm, "code2wav") == ["s3"]


def test_select_stage_strips_prefix_and_drops_other_stages():
    _, model, state = checkpoint()
    sd = m.select_stage(state, "thinker", "thinker.model.")
    assert set(sd) == set(model.state_dict())


def test_packed_tensors_reproduce_hf_projections():
    cfg, model, state = checkpoint()
    globals_, layers = m.load_thinker_text(state, cfg.num_hidden_layers)
    assert torch.equal(globals_["embed_tokens"], model.embed_tokens.weight)
    x = torch.randn(5, HIDDEN)
    D, H, KVH = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    for i, layer in enumerate(layers):
        hf = model.layers[i]
        q, k, v = (x @ layer.qkv.T).split([H * D, KVH * D, KVH * D], dim=-1)
        with torch.no_grad():
            torch.testing.assert_close(q, hf.self_attn.q_proj(x))
            torch.testing.assert_close(k, hf.self_attn.k_proj(x))
            torch.testing.assert_close(v, hf.self_attn.v_proj(x))
            for e, ex in enumerate(hf.mlp.experts):
                g, u = (x @ layer.w13[e].T).chunk(2, dim=-1)
                torch.testing.assert_close((F.silu(g) * u) @ layer.w2[e].T, ex(x))
        assert torch.equal(layer.router, hf.mlp.gate.weight)


def test_missing_and_unexpected_keys_are_errors():
    cfg, _, state = checkpoint()
    broken = dict(state)
    del broken["thinker.model.layers.1.mlp.experts.3.up_proj.weight"]
    with pytest.raises(m.LoadError):
        m.load_thinker_text(broken, cfg.num_hidden_layers)
    extra = dict(state)
    extra["thinker.model.layers.0.self_attn.rotary.inv_freq"] = torch.zeros(4)
    with pytest.raises(m.LoadError):
        m.load_thinker_text(extra, cfg.num_hidden_layers)
