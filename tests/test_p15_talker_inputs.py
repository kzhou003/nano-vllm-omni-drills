import pytest
import torch

from drill import load

m = load("p15_talker_inputs")
D = 4


def rows(n, base=0):
    return torch.arange(base, base + n, dtype=torch.float32)[:, None].repeat(1, D)


def test_stop_finish_drops_one_row_and_id():
    p = m.trim_thinker_payload([1, 2, 3], [7, 8, 151645], rows(7), "stop")  # one extra row captured
    assert p.prefill.shape == (3, D) and p.decode.shape == (2, D)
    assert p.output_ids == [7, 8]
    assert torch.equal(p.decode[:, 0], torch.tensor([3.0, 4.0]))


def test_length_finish_keeps_everything():
    p = m.trim_thinker_payload([1, 2], [5, 6, 7], rows(5), "length")
    assert p.decode.shape == (3, D) and p.output_ids == [5, 6, 7]


def test_under_captured_prefill_raises():
    with pytest.raises(ValueError):
        m.trim_thinker_payload([1, 2, 3], [4], rows(2), "length")


def test_build_prefill_layout():
    P = 3
    prompt = rows(P)
    first = torch.full((D,), 100.0)
    spk, pad, bos = torch.full((D,), 1.0), torch.full((D,), 10.0), torch.full((D,), 1000.0)
    ids, emb = m.build_talker_prefill(prompt, first, spk, pad, bos, mask_id=8296, pad_id=8292, start_id=8293)
    assert ids == [8296, 8296, 8296, 8292, 8293]
    assert emb.shape == (P + 2, D)
    assert torch.equal(emb[:P], prompt)
    assert torch.equal(emb[P], spk + pad) and torch.equal(emb[P + 1], first + bos)


def test_decode_consumes_one_row_then_falls_back():
    q = rows(2, base=50)
    last = torch.full((D,), 0.5)
    fb = torch.full((D,), -1.0)
    x, q = m.talker_decode_input(q, last, fb)
    assert torch.equal(x, torch.full((D,), 50.5)) and q.shape[0] == 1
    x, q = m.talker_decode_input(q, last, fb)
    assert torch.equal(x, torch.full((D,), 51.5)) and q.shape[0] == 0
    x, q = m.talker_decode_input(q, last, fb)
    assert torch.equal(x, torch.full((D,), -0.5)) and q.shape[0] == 0
