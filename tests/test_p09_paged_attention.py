import random

import torch

from drill import load

m = load("p09_paged_attention")


def dense_attention(q, k, v, scale):
    # q [H,D], k/v [n,KVH,D]
    group = q.shape[0] // k.shape[1]
    k = k.repeat_interleave(group, 1)
    v = v.repeat_interleave(group, 1)
    p = torch.softmax(torch.einsum("hd,nhd->hn", q, k) * scale, -1)
    return torch.einsum("hn,nhd->hd", p, v)


def build_case(seed, lens, block_size=4, num_blocks=64, H=8, KVH=2, D=16):
    torch.manual_seed(seed)
    random.seed(seed)
    k_cache = torch.zeros(num_blocks, block_size, KVH, D)
    v_cache = torch.zeros_like(k_cache)
    free = list(range(num_blocks))
    random.shuffle(free)  # non-contiguous physical blocks
    max_blocks = max(-(-n // block_size) for n in lens)
    tables = torch.full((len(lens), max_blocks), 999, dtype=torch.int32)  # junk in unused slots
    dense = []
    for b, n in enumerate(lens):
        nb = -(-n // block_size)
        blocks = [free.pop() for _ in range(nb)]
        tables[b, :nb] = torch.tensor(blocks, dtype=torch.int32)
        k, v = torch.randn(n, KVH, D), torch.randn(n, KVH, D)
        slots = torch.tensor([blocks[p // block_size] * block_size + p % block_size for p in range(n)])
        m.write_kv(k_cache, v_cache, k, v, slots)
        dense.append((k, v))
    q = torch.randn(len(lens), H, D)
    return q, k_cache, v_cache, tables, torch.tensor(lens), dense


def test_matches_dense_attention():
    lens = [1, 4, 5, 13, 32, 7]
    q, kc, vc, tables, ctx, dense = build_case(0, lens)
    scale = 16 ** -0.5
    out = m.paged_decode_attention(q, kc, vc, tables, ctx, scale)
    for b, (k, v) in enumerate(dense):
        torch.testing.assert_close(out[b], dense_attention(q[b], k, v, scale), atol=1e-5, rtol=1e-5)


def test_padding_slots_are_not_written():
    kc = torch.zeros(4, 4, 1, 2)
    vc = torch.zeros_like(kc)
    k = torch.ones(3, 1, 2)
    m.write_kv(kc, vc, k, k, torch.tensor([5, -1, 6]))
    assert kc.sum() == 4  # two tokens x 2 dims
    assert kc[1, 1].sum() == 2 and kc[1, 2].sum() == 2


def test_random_shapes():
    for seed in range(5):
        lens = [random.randint(1, 40) for _ in range(random.randint(1, 6))]
        q, kc, vc, tables, ctx, dense = build_case(seed + 10, lens, block_size=random.choice([2, 4, 8]))
        out = m.paged_decode_attention(q, kc, vc, tables, ctx, 0.25)
        for b, (k, v) in enumerate(dense):
            torch.testing.assert_close(out[b], dense_attention(q[b], k, v, 0.25), atol=1e-5, rtol=1e-5)
