import asyncio
import random
import threading
import time

from drill import load

m = load("p01_output_sink")


def test_basic_fifo():
    async def main():
        s = m.OutputSink()
        s.bind_loop()
        s.put(1)
        s.put(2)
        assert await s.get_batch() == [1, 2]

    asyncio.run(main())


def test_wakeups_are_coalesced():
    async def main():
        s = m.OutputSink()
        s.bind_loop()
        t = threading.Thread(target=lambda: [s.put(i) for i in range(1000)])
        t.start()
        t.join()
        # nobody consumed yet -> exactly one wakeup was scheduled for 1000 puts
        assert s.wakeups_scheduled == 1
        got = []
        while len(got) < 1000:
            batch = await s.get_batch(max_items=256)
            assert 1 <= len(batch) <= 256
            got += batch
        assert got == list(range(1000))

    asyncio.run(main())


def test_timeout_returns_empty():
    async def main():
        s = m.OutputSink()
        s.bind_loop()
        t0 = time.monotonic()
        assert await s.get_batch(timeout=0.05) == []
        assert time.monotonic() - t0 < 1.0

    asyncio.run(main())


def test_multi_producer_per_producer_order():
    async def main():
        s = m.OutputSink()
        s.bind_loop()
        n, producers = 5000, 4

        def produce(pid):
            for i in range(n):
                s.put((pid, i))
                if i % 97 == 0:
                    time.sleep(random.random() * 0.001)

        threads = [threading.Thread(target=produce, args=(p,)) for p in range(producers)]
        for t in threads:
            t.start()
        got = []
        while len(got) < n * producers:
            batch = await s.get_batch(timeout=2.0)
            assert batch, "consumer starved -> lost wakeup?"
            got += batch
        for t in threads:
            t.join()
        for p in range(producers):
            assert [i for q, i in got if q == p] == list(range(n))

    asyncio.run(main())


def test_ping_pong_never_loses_a_wakeup():
    async def main():
        s = m.OutputSink()
        s.bind_loop()
        go = threading.Event()

        def produce():
            for i in range(300):
                go.wait()
                go.clear()
                time.sleep(random.random() * 0.0005)
                s.put(i)

        t = threading.Thread(target=produce)
        t.start()
        for i in range(300):
            go.set()
            assert await s.get_batch(timeout=2.0) == [i]
        t.join()

    asyncio.run(main())


def test_put_between_empty_drain_and_rearm_is_not_lost():
    async def main():
        s = m.OutputSink()
        s.bind_loop()
        s.put("warm")  # leaves a wakeup "in flight"
        assert await s.get_batch() == ["warm"]
        await asyncio.sleep(0.01)  # let the stale wakeup callback run
        injected = []

        def racing_put():
            if not injected:
                injected.append(True)
                t = threading.Thread(target=s.put, args=("raced",))
                t.start()
                t.join()

        s.after_empty_drain = racing_put
        t0 = time.monotonic()
        assert await s.get_batch(timeout=0.5) == ["raced"]
        assert time.monotonic() - t0 < 0.2, "the item waited for the timeout: lost wakeup"

    asyncio.run(main())
