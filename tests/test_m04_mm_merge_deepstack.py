import pytest
import torch

from drill import load
from tests._tiny import HIDDEN, M, causal_mask, randomize_, thinker_text_config

m = load("m04_mm_merge_deepstack")
IMG, AUD = 90, 91


def test_audio_token_count_matches_hf():
    frames = torch.tensor([1, 2, 7, 99, 100, 101, 199, 200, 250, 1234, 3000])
    torch.testing.assert_close(m.audio_token_count(frames), M._get_feat_extract_output_lengths(frames))
    assert m.audio_token_count(100) == 13


def test_merge_places_embeddings_in_order():
    ids = torch.tensor([1, IMG, IMG, 2, AUD, AUD, AUD, 3, IMG])
    text = torch.randn(9, HIDDEN)
    img, aud = torch.randn(3, HIDDEN), torch.randn(3, HIDDEN)
    text_before = text.clone()
    out, masks = m.merge_multimodal(ids, text, {"image": IMG, "audio": AUD}, {"image": img, "audio": aud})
    torch.testing.assert_close(out[[1, 2, 8]], img)
    torch.testing.assert_close(out[[4, 5, 6]], aud)
    torch.testing.assert_close(out[[0, 3, 7]], text[[0, 3, 7]])
    assert masks["image"].tolist() == [False, True, True, False, False, False, False, False, True]
    assert torch.equal(text, text_before), "do not mutate the text embeddings in place"


def test_merge_count_mismatch_raises():
    ids = torch.tensor([IMG, IMG, 5])
    with pytest.raises(ValueError):
        m.merge_multimodal(ids, torch.randn(3, HIDDEN), {"image": IMG}, {"image": torch.randn(3, HIDDEN)})
    with pytest.raises(ValueError):
        m.merge_multimodal(ids, torch.randn(3, HIDDEN), {"image": IMG}, {})


def test_deepstack_matches_hf_text_model():
    cfg = thinker_text_config()
    model = randomize_(M.Qwen3OmniMoeThinkerTextModel(cfg), seed=0)
    T = 10
    emb = torch.randn(1, T, HIDDEN)
    vis = torch.zeros(T, dtype=torch.bool)
    vis[3:7] = True
    deep = [torch.randn(4, HIDDEN), torch.randn(4, HIDDEN)]
    pos = torch.arange(T).expand(3, 1, T)
    with torch.no_grad():
        ref = model(inputs_embeds=emb, position_ids=pos, visual_pos_masks=vis[None, :, None].expand(1, T, HIDDEN),
                    deepstack_visual_embeds=deep).last_hidden_state[0]
        cos_sin = model.rotary_emb(emb, pos)
        mask = causal_mask(T)
        layers = [lambda h, L=L: L(h[None], position_embeddings=cos_sin, attention_mask=mask)[0] for L in model.layers]
        out = model.norm(m.run_layers(layers, emb[0].clone(), vis, deep))
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_deepstack_only_touches_visual_rows():
    h = torch.randn(6, HIDDEN)
    vis = torch.tensor([0, 1, 0, 1, 0, 0], dtype=torch.bool)
    v = torch.randn(2, HIDDEN)
    out = m.apply_deepstack(h, vis, v)
    torch.testing.assert_close(out[vis], h[vis] + v)
    torch.testing.assert_close(out[~vis], h[~vis])
