import torch

from drill import load
from tests._tiny import HIDDEN, M, code_predictor_config, randomize_

m = load("m05_talker_mtp")


def setup(G=6, B=3, seed=0):
    cp = randomize_(M.Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(code_predictor_config(G)), seed, 0.3)
    g = torch.Generator().manual_seed(seed + 100)
    hid, cb0 = torch.randn(B, HIDDEN, generator=g), torch.randn(B, HIDDEN, generator=g)
    return cp, hid, cb0


def hf_greedy(cp, hid, cb0):
    with torch.no_grad():
        res = cp.generate(inputs_embeds=torch.stack([hid, cb0], 1), max_new_tokens=cp.config.num_code_groups - 1,
                          do_sample=False, output_hidden_states=True, return_dict_in_generate=True)
    return res


def test_reprefill_matches_hf_generate():
    cp, hid, cb0 = setup()
    ref = hf_greedy(cp, hid, cb0).sequences
    codes = m.predict_residual_codes(cp, hid, cb0)
    assert codes.shape == (3, 5)
    assert torch.equal(codes, ref)


def test_kv_cache_version_matches_reprefill():
    for seed in range(3):
        cp, hid, cb0 = setup(G=7, B=2, seed=seed)
        assert torch.equal(m.predict_residual_codes_kv(cp, hid, cb0), m.predict_residual_codes(cp, hid, cb0))


def test_rows_are_independent():
    cp, hid, cb0 = setup(B=4, seed=5)
    full = m.predict_residual_codes(cp, hid, cb0)
    for i in range(4):
        assert torch.equal(m.predict_residual_codes(cp, hid[i:i + 1], cb0[i:i + 1])[0], full[i])


def test_next_input_matches_hf_summation():
    cp, hid, cb0 = setup(seed=7)
    res = hf_greedy(cp, hid, cb0)
    text_step = torch.randn(3, HIDDEN)
    # HF Qwen3OmniMoeTalkerForConditionalGeneration.prepare_inputs_for_generation
    mid = [h[0] for h in res.hidden_states[1:]]                 # embeddings of residual codes 1..G-2
    last = cp.get_input_embeddings()[-1](res.sequences[..., -1:])  # embedding of residual code G-1
    with torch.no_grad():
        ref = torch.cat([cb0[:, None]] + mid + [last], dim=1).sum(1) + text_step
        got = m.next_talker_input(cp, cb0, res.sequences, text_step)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)
