import random
import threading
import time

from drill import load

m = load("p13_chunk_adapter")


def pair(build=lambda snap: list(snap), **conn_kw):
    conn = m.InMemoryConnector(**conn_kw)
    sender = m.ChunkAdapter(conn, stage_id=0, src_stage_id=None, build_payload=build)
    receiver = m.ChunkAdapter(conn, stage_id=1, src_stage_id=0, build_payload=build)
    sender.start()
    receiver.start()
    return conn, sender, receiver


def drain_until_finished(receiver, req_ids, timeout=5.0):
    got = {r: [] for r in req_ids}
    done = set()
    end = time.monotonic() + timeout
    while done != set(req_ids):
        assert time.monotonic() < end, f"timed out; done={done}"
        for rid, chunks in receiver.poll_ready().items():
            for c in chunks:
                assert rid not in done, "chunk after the final one"
                got[rid].append(c)
                if c["finished"]:
                    done.add(rid)
        time.sleep(0.001)
    return got


def stop(*adapters):
    for a in adapters:
        a.stop()


def test_ordered_delivery_single_request():
    conn, s, r = pair()
    r.load_async("q")
    for i in range(50):
        s.save_async("q", [i], is_final=(i == 49))
    got = drain_until_finished(r, ["q"])
    assert [c["data"] for c in got["q"]] == [[i] for i in range(50)]
    assert conn.puts == [f"q_0_{i}" for i in range(50)]
    stop(s, r)


def test_many_requests_with_random_latency():
    jitter = lambda key: time.sleep(random.random() * 0.0005)  # noqa: E731
    conn, s, r = pair(on_put=jitter, on_get=jitter)
    rids = [f"r{i}" for i in range(10)]
    for rid in rids:
        r.load_async(rid)
    for step in range(30):
        for rid in random.sample(rids, len(rids)):
            s.save_async(rid, [rid, step], is_final=(step == 29))
    got = drain_until_finished(r, rids)
    for rid in rids:
        assert [c["data"] for c in got[rid]] == [[rid, k] for k in range(30)]
    stop(s, r)


def test_save_async_never_blocks_the_caller():
    def slow_build(snap):
        time.sleep(0.05)
        return snap

    conn, s, r = pair(build=slow_build)
    t0 = time.monotonic()
    for i in range(10):
        s.save_async("q", [i], is_final=(i == 9))
    assert time.monotonic() - t0 < 0.02
    r.load_async("q")
    drain_until_finished(r, ["q"])
    stop(s, r)


def test_snapshot_isolation():
    conn, s, r = pair()
    live = [1, 2, 3]
    r.load_async("q")
    s.save_async("q", live, is_final=True)
    live.append(4)  # scheduler keeps mutating the live request
    got = drain_until_finished(r, ["q"])
    assert got["q"][0]["data"] == [1, 2, 3]
    stop(s, r)


def test_cancel_still_delivers_final_and_ids_stay_contiguous():
    gate = threading.Event()

    def gated_build(snap):
        gate.wait()
        return snap

    conn, s, r = pair(build=gated_build)
    r.load_async("q")
    s.save_async("q", ["first"], is_final=False)
    time.sleep(0.05)  # save thread is now blocked building "first"
    for i in range(20):
        s.save_async("q", [i], is_final=False)
    s.save_async("q", ["end"], is_final=True)
    s.cancel_sender("q")
    gate.set()
    got = drain_until_finished(r, ["q"])
    assert got["q"][-1]["finished"], "the final chunk is the downstream EOS: it must survive cancel"
    assert len(got["q"]) <= 2
    assert conn.puts == [f"q_0_{i}" for i in range(len(conn.puts))], "chunk ids must be contiguous"
    stop(s, r)


def test_none_payload_is_skipped_without_gap():
    conn, s, r = pair(build=lambda snap: None if snap[0] % 2 else snap)
    r.load_async("q")
    for i in range(6):
        s.save_async("q", [i], is_final=(i == 5))
    got = drain_until_finished(r, ["q"])
    assert [c["data"] for c in got["q"]] == [[0], [2], [4], None]
    assert conn.puts == ["q_0_0", "q_0_1", "q_0_2", "q_0_3"]
    stop(s, r)


def test_cleanup_is_a_commit_barrier_for_in_flight_reads():
    in_get = threading.Event()

    def slow_get(key):
        in_get.set()
        time.sleep(0.2)

    conn, s, r = pair(on_get=slow_get)
    conn.put("q_0_0", {"data": "late", "finished": False})
    r.load_async("q")
    assert in_get.wait(1.0)
    r.cleanup_receiver("q")  # request aborted while the recv thread is inside get()
    time.sleep(0.3)
    assert r.poll_ready() == {}, "a chunk read before cleanup must not be committed after it"
    stop(s, r)
