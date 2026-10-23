import torch

from drill import load
from tests._tiny import HIDDEN, MROPE_SECTION, M, randomize_, thinker_text_config

m = load("m08_tiny_thinker_e2e")


def build(seed=0):
    cfg = thinker_text_config()
    model = randomize_(M.Qwen3OmniMoeThinkerTextModel(cfg), seed=seed, std=0.3)
    lm_head = torch.randn(cfg.vocab_size, HIDDEN, generator=torch.Generator().manual_seed(seed)) * 0.5
    state = {f"thinker.model.{k}": v.detach().clone() for k, v in model.state_dict().items()}
    state["thinker.lm_head.weight"] = lm_head
    dims = m.Dims(cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
                  cfg.num_experts_per_tok, cfg.norm_topk_prob, cfg.rope_theta, MROPE_SECTION, cfg.rms_norm_eps)
    return model, lm_head, state, dims


@torch.no_grad()
def hf_logits(model, lm_head, ids):
    return model(input_ids=torch.tensor([ids])).last_hidden_state[0] @ lm_head.T


@torch.no_grad()
def hf_greedy(model, lm_head, prompt, n):
    ids = list(prompt)
    for _ in range(n):
        ids.append(int(hf_logits(model, lm_head, ids)[-1].argmax()))
    return ids[len(prompt):]


def test_prefill_logits_match_hf():
    model, lm_head, state, dims = build()
    tt = m.TinyThinker(state, dims)
    ids = [3, 14, 15, 92, 65, 35, 8]
    caches = [m.KVCache() for _ in range(dims.num_layers)]
    with torch.no_grad():
        got = tt.forward(torch.tensor(ids), 0, caches)
    torch.testing.assert_close(got, hf_logits(model, lm_head, ids), atol=1e-4, rtol=1e-4)


def test_greedy_with_kv_cache_matches_hf_full_recompute():
    for seed in range(2):
        model, lm_head, state, dims = build(seed)
        tt = m.TinyThinker(state, dims)
        prompt = [1, 2, 3, 4, 5, 6][: 3 + seed * 3]
        assert tt.greedy(prompt, 12) == hf_greedy(model, lm_head, prompt, 12)
