import math
import random

import torch

from drill import load

m = load("p08_sampler")
SP = m.SamplingParams
V = 50


def rand_logits(seed, n=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, V, generator=g) * 3


def test_greedy():
    s = m.Sampler()
    logits = rand_logits(0, 4)
    out = s.sample(logits, list("abcd"), [SP(temperature=0.0)] * 4, [[]] * 4)
    assert out.tolist() == logits.argmax(-1).tolist()


def test_top_k_1_and_tiny_top_p_are_greedy():
    s = m.Sampler()
    logits = rand_logits(1, 3)
    greedy = logits.argmax(-1).tolist()
    assert s.sample(logits, list("abc"), [SP(top_k=1, seed=1)] * 3, [[]] * 3).tolist() == greedy
    assert s.sample(logits, list("xyz"), [SP(top_p=1e-6, seed=2)] * 3, [[]] * 3).tolist() == greedy


def test_seeded_result_is_batch_invariant():
    row = rand_logits(7)[0]
    alone = m.Sampler()
    ref = [int(alone.sample(row[None], ["a"], [SP(temperature=1.0, seed=123)], [[]])[0]) for _ in range(30)]

    mixed = m.Sampler()
    got = []
    for step in range(30):
        others = random.randint(0, 5)
        ids = [f"o{step}_{j}" for j in range(others)] + ["a"]
        random.shuffle(ids)
        logits = torch.stack([row if r == "a" else rand_logits(step * 10 + j)[0] for j, r in enumerate(ids)])
        params = [SP(temperature=1.0, seed=123) if r == "a" else SP(temperature=1.0) for r in ids]
        out = mixed.sample(logits, ids, params, [[] for _ in ids])
        got.append(int(out[ids.index("a")]))
    assert got == ref


def test_top_k_distribution():
    s = m.Sampler()
    logits = torch.full((1, V), -1e9)
    logits[0, 3], logits[0, 9] = math.log(0.7), math.log(0.3)
    logits[0, 20] = math.log(0.2)  # removed by top_k=2
    counts = {3: 0, 9: 0}
    for _ in range(4000):
        t = int(s.sample(logits, ["r"], [SP(top_k=2, seed=5)], [[]])[0])
        assert t in counts
        counts[t] += 1
    frac = counts[3] / 4000
    assert abs(frac - 0.7) < 0.04, frac


def test_repetition_penalty_values_and_effect():
    logits = torch.tensor([2.0, -2.0, 1.5])
    pen = m.apply_repetition_penalty(logits, [0, 1, 1], 2.0)
    assert torch.allclose(pen, torch.tensor([1.0, -4.0, 1.5]))
    s = m.Sampler()
    out = s.sample(logits[None], ["r"], [SP(temperature=0.0, repetition_penalty=2.0)], [[0]])
    assert int(out[0]) == 2


def test_free_resets_generator():
    s = m.Sampler()
    row = rand_logits(3)
    first = [int(s.sample(row, ["r"], [SP(seed=9)], [[]])[0]) for _ in range(10)]
    s.free("r")
    again = [int(s.sample(row, ["r"], [SP(seed=9)], [[]])[0]) for _ in range(10)]
    assert first == again
