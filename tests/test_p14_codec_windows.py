import random

from drill import load

m = load("p14_codec_windows")
UP = 3


def toy_vocoder(codes, receptive_field):
    """Causal and local: sample j of frame i depends on codes[i-rf .. i] *within the given window*."""
    out = []
    for i in range(len(codes)):
        ctx = codes[max(0, i - receptive_field):i + 1]
        out += [sum(ctx) * 10 + j for j in range(UP)]
    return out


def stream(codes, arrivals, *, initial, chunk, left_ctx, rf):
    """Simulate the talker producing frames in bursts and the window planner reacting."""
    emitted, produced, audio, windows = 0, 0, [], []
    for k, burst in enumerate(arrivals):
        produced += burst
        finished = k == len(arrivals) - 1
        while True:
            w = m.next_window(produced, emitted, finished, initial=initial, chunk=chunk, left_ctx=left_ctx)
            if w is None:
                break
            ctx_start, new_start, new_end = w
            decoded = toy_vocoder(codes[ctx_start:new_end], rf)
            audio += m.slice_new_samples(decoded, w, UP)
            windows.append(w)
            emitted = new_end
    return audio, windows


def test_initial_then_steady_then_flush():
    assert m.next_window(3, 0, False, initial=4, chunk=25, left_ctx=25) is None
    assert m.next_window(4, 0, False, initial=4, chunk=25, left_ctx=25) == (0, 0, 4)
    assert m.next_window(20, 4, False, initial=4, chunk=25, left_ctx=25) is None
    assert m.next_window(29, 4, False, initial=4, chunk=25, left_ctx=25) == (0, 4, 29)
    assert m.next_window(60, 29, False, initial=4, chunk=25, left_ctx=25) == (4, 29, 54)
    assert m.next_window(60, 54, True, initial=4, chunk=25, left_ctx=25) == (29, 54, 60)
    assert m.next_window(60, 60, True, initial=4, chunk=25, left_ctx=25) is None


def test_no_initial_chunk():
    assert m.next_window(10, 0, False, initial=0, chunk=10, left_ctx=2) == (0, 0, 10)
    assert m.next_window(20, 10, False, initial=0, chunk=10, left_ctx=2) == (8, 10, 20)


def test_windows_tile_the_stream_exactly():
    for seed in range(20):
        random.seed(seed)
        total = random.randint(1, 200)
        arrivals, left = [], total
        while left:
            b = random.randint(1, 7)
            arrivals.append(min(b, left))
            left -= arrivals[-1]
        codes = [random.randint(0, 9) for _ in range(total)]
        _, windows = stream(codes, arrivals, initial=4, chunk=13, left_ctx=5, rf=5)
        covered = [f for (_, s, e) in windows for f in range(s, e)]
        assert covered == list(range(total)), "gap or overlap in shipped frames"


def test_streaming_equals_full_decode_iff_left_context_covers_receptive_field():
    random.seed(0)
    codes = [random.randint(0, 9) for _ in range(97)]
    arrivals = [1] * 97
    full = toy_vocoder(codes, receptive_field=4)
    ok, _ = stream(codes, arrivals, initial=4, chunk=10, left_ctx=4, rf=4)
    assert ok == full
    bad, _ = stream(codes, arrivals, initial=4, chunk=10, left_ctx=2, rf=4)
    assert bad != full, "too little left context must produce seams"
