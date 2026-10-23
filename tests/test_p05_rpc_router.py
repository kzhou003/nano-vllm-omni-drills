import queue
import random
import threading
import time

import pytest

from drill import load

m = load("p05_rpc_router")


def test_concurrent_calls_out_of_order():
    router = m.RpcRouter()
    wire: queue.Queue = queue.Queue()

    def server():
        pending = []
        while True:
            item = wire.get()
            if item is None:
                break
            pending.append(item)
            if len(pending) >= 5 or wire.empty():
                random.shuffle(pending)
                for key in pending:
                    router.deliver(key, ("ok", key))
                pending.clear()

    srv = threading.Thread(target=server)
    srv.start()
    results = {}

    def client(i):
        key = ("rpc", i)
        results[i] = router.call(key, lambda: wire.put(key), timeout=2.0)

    threads = [threading.Thread(target=client, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wire.put(None)
    srv.join()
    assert results == {i: ("ok", ("rpc", i)) for i in range(50)}


def test_result_delivered_inside_send_is_not_lost():
    router = m.RpcRouter()
    assert router.call("k", lambda: router.deliver("k", 42), timeout=0.5) == 42


def test_timeout_and_late_result():
    router = m.RpcRouter()
    with pytest.raises(TimeoutError):
        router.call("slow", lambda: None, timeout=0.05)
    assert router.deliver("slow", "too late") is False
    assert router.late_results == 1
    # the key can be reused after the timed-out call cleaned up
    assert router.call("slow", lambda: router.deliver("slow", 1), timeout=0.5) == 1


def test_duplicate_key_rejected():
    router = m.RpcRouter()
    w = router.register("dup")
    with pytest.raises(ValueError):
        router.register("dup")
    router.unregister("dup", w)


def test_fail_all_wakes_waiters_and_latches():
    router = m.RpcRouter()
    errors = []

    def client(i):
        try:
            router.call(i, lambda: None, timeout=5.0)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=client, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    time.sleep(0.1)
    t0 = time.monotonic()
    router.fail_all(RuntimeError("orchestrator died"))
    for t in threads:
        t.join()
    assert time.monotonic() - t0 < 1.0
    assert len(errors) == 5 and all("orchestrator died" in str(e) for e in errors)
    with pytest.raises(RuntimeError):
        router.call("new", lambda: None, timeout=5.0)
