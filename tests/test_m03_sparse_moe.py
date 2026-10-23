import torch

from drill import load
from tests._tiny import HIDDEN, M, randomize_, talker_text_config, thinker_text_config

m = load("m03_sparse_moe")


def run_hf(block, x):
    with torch.no_grad():
        y, logits = block(x[None])
    return y[0], logits


def test_routing_matches_hf_softmax_topk_renorm():
    block = randomize_(M.Qwen3OmniMoeThinkerTextSparseMoeBlock(thinker_text_config()), seed=0, std=0.5)
    x = torch.randn(17, HIDDEN)
    w = m.from_hf(block)
    topw, topi = m.route(x, w)
    probs = torch.softmax(x @ w.router.T, -1)
    ref_w, ref_i = torch.topk(probs, 2, -1)
    ref_w = ref_w / ref_w.sum(-1, keepdim=True)
    assert torch.equal(topi, ref_i)
    torch.testing.assert_close(topw, ref_w)
    torch.testing.assert_close(topw.sum(-1), torch.ones(17))


def test_thinker_moe_matches_hf():
    block = randomize_(M.Qwen3OmniMoeThinkerTextSparseMoeBlock(thinker_text_config()), seed=1)
    x = torch.randn(23, HIDDEN)
    ref, _ = run_hf(block, x)
    torch.testing.assert_close(m.moe_forward(x, m.from_hf(block)), ref, atol=1e-5, rtol=1e-4)


def test_talker_moe_with_shared_expert_matches_hf():
    block = randomize_(M.Qwen3OmniMoeTalkerTextSparseMoeBlock(talker_text_config()), seed=2)
    x = torch.randn(9, HIDDEN)
    ref, _ = run_hf(block, x)
    torch.testing.assert_close(m.moe_forward(x, m.from_hf(block)), ref, atol=1e-5, rtol=1e-4)


def test_no_renorm_variant():
    cfg = thinker_text_config()
    cfg.norm_topk_prob = False
    block = randomize_(M.Qwen3OmniMoeThinkerTextSparseMoeBlock(cfg), seed=3)
    x = torch.randn(8, HIDDEN)
    ref, _ = run_hf(block, x)
    torch.testing.assert_close(m.moe_forward(x, m.from_hf(block)), ref, atol=1e-5, rtol=1e-4)


def test_single_token_and_batch_invariance():
    block = randomize_(M.Qwen3OmniMoeThinkerTextSparseMoeBlock(thinker_text_config()), seed=4)
    w = m.from_hf(block)
    x = torch.randn(6, HIDDEN)
    full = m.moe_forward(x, w)
    for i in range(6):
        torch.testing.assert_close(m.moe_forward(x[i:i + 1], w)[0], full[i], atol=1e-5, rtol=1e-4)
