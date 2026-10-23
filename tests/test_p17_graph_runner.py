import torch

from drill import load

m = load("p17_graph_runner")
H = 8
W = torch.randn(H, H, generator=torch.Generator().manual_seed(0))


def fn(x):
    return torch.tanh(x @ W) + 1.0


def make():
    r = m.GraphRunner(fn, buckets=[1, 2, 4, 8], hidden=H)
    r.capture()
    return r


def test_matches_eager_for_every_batch_size():
    r = make()
    for bs in range(1, 9):
        x = torch.randn(bs, H)
        torch.testing.assert_close(r.run(x), fn(x))
    assert r.stats["eager"] == 0 and r.stats["replays"] == 8


def test_bucket_selection_and_eager_fallbacks():
    r = make()
    r.run(torch.randn(3, H))
    assert r.stats["last_bucket"] == 4
    r.run(torch.randn(9, H))  # larger than every bucket
    assert r.stats["eager"] == 1
    r.run(torch.randn(2, H), force_eager=True)  # e.g. a batch with per-row RNG generators
    assert r.stats["eager"] == 2


def test_result_does_not_alias_static_buffers():
    r = make()
    x1, x2 = torch.randn(2, H), torch.randn(2, H)
    y1 = r.run(x1)
    expected = fn(x1)
    r.run(x2)
    torch.testing.assert_close(y1, expected, msg="y1 was overwritten by the next replay")


def test_padding_rows_are_zeroed():
    r = make()
    r.run(torch.full((4, H), 7.0))
    r.run(torch.randn(3, H))
    assert torch.count_nonzero(r.static_in[3]) == 0


def test_captured_once_per_bucket():
    r = make()
    assert sorted(r.graphs) == [1, 2, 4, 8]
    graphs = dict(r.graphs)
    for bs in (1, 3, 5, 8):
        r.run(torch.randn(bs, H))
    assert all(r.graphs[b] is graphs[b] for b in graphs), "never re-capture at run time"
