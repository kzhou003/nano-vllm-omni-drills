import threading
import time

from drill import load

m = load("p10_engine_core_loop")
Msg = m.Msg


class FakeEngine:
    def __init__(self, steps_per_req=3, fail_on_step=None):
        self.reqs: dict[str, int] = {}
        self.calls: list = []
        self.steps = 0
        self.has_work_calls = 0
        self.steps_per_req = steps_per_req
        self.fail_on_step = fail_on_step

    def add(self, rid, data):
        self.calls.append(("add", rid))
        self.reqs[rid] = self.steps_per_req

    def abort(self, ids):
        self.calls.append(("abort", tuple(ids)))
        for r in ids:
            self.reqs.pop(r, None)

    def has_work(self):
        self.has_work_calls += 1
        return bool(self.reqs)

    def step(self):
        self.steps += 1
        if self.fail_on_step == self.steps:
            raise RuntimeError("CUDA error: illegal memory access")
        outs = []
        for r in list(self.reqs):
            self.reqs[r] -= 1
            outs.append((r, self.reqs[r] == 0))
            if self.reqs[r] == 0:
                del self.reqs[r]
        return outs

    def utility(self, name, args):
        if name == "boom":
            raise ValueError("bad utility")
        return (name, args)


def start(engine):
    emitted = []
    loop = m.EngineCoreLoop(engine, emitted.append)
    t = threading.Thread(target=loop.run_busy_loop, daemon=True)
    return loop, t, emitted


def wait_for(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_idle_loop_blocks_instead_of_spinning():
    eng = FakeEngine()
    loop, t, _ = start(eng)
    t.start()
    time.sleep(0.3)
    assert eng.steps == 0
    assert eng.has_work_calls < 10, f"busy-spinning: has_work called {eng.has_work_calls} times"
    loop.request_shutdown()
    t.join(1.0)
    assert not t.is_alive()


def test_add_runs_to_completion():
    eng = FakeEngine(steps_per_req=3)
    loop, t, emitted = start(eng)
    t.start()
    loop.input_queue.put(Msg("ADD", ("r1", None)))
    assert wait_for(lambda: ("output", ("r1", True)) in emitted)
    assert [e for e in emitted if e[0] == "output"] == [("output", ("r1", False))] * 2 + [("output", ("r1", True))]
    loop.request_shutdown()
    t.join(1.0)
    assert emitted[-1] == ("shutdown",)


def test_add_then_abort_are_applied_in_fifo_order_before_stepping():
    eng = FakeEngine()
    loop, t, emitted = start(eng)
    loop.input_queue.put(Msg("ADD", ("r1", None)))
    loop.input_queue.put(Msg("ABORT", ["r1"]))
    loop.input_queue.put(Msg("ADD", ("r2", None)))
    t.start()
    assert wait_for(lambda: ("output", ("r2", True)) in emitted)
    assert eng.calls[:3] == [("add", "r1"), ("abort", ("r1",)), ("add", "r2")]
    assert not any(e[0] == "output" and e[1][0] == "r1" for e in emitted)
    loop.request_shutdown()
    t.join(1.0)


def test_shutdown_from_another_thread_while_idle_is_fast():
    eng = FakeEngine()
    loop, t, emitted = start(eng)
    t.start()
    time.sleep(0.05)
    assert t.is_alive()
    t0 = time.monotonic()
    threading.Thread(target=loop.request_shutdown).start()
    t.join(1.0)
    assert not t.is_alive() and time.monotonic() - t0 < 0.5
    assert emitted == [("shutdown",)]


def test_step_exception_is_fatal():
    eng = FakeEngine(steps_per_req=10, fail_on_step=2)
    loop, t, emitted = start(eng)
    t.start()
    loop.input_queue.put(Msg("ADD", ("r1", None)))
    t.join(2.0)
    assert not t.is_alive()
    assert emitted[-1][0] == "fatal" and "illegal memory access" in emitted[-1][1]


def test_utility_results_and_errors():
    eng = FakeEngine()
    loop, t, emitted = start(eng)
    t.start()
    loop.input_queue.put(Msg("UTILITY", (1, "sleep", (2,))))
    loop.input_queue.put(Msg("UTILITY", (2, "boom", ())))
    assert wait_for(lambda: len([e for e in emitted if e[0] == "utility"]) == 2)
    utils = [e for e in emitted if e[0] == "utility"]
    assert utils[0] == ("utility", 1, ("sleep", (2,)), None)
    assert utils[1][1] == 2 and utils[1][2] is None and "bad utility" in utils[1][3]
    assert t.is_alive(), "a failing utility call must not kill the engine"
    loop.request_shutdown()
    t.join(1.0)
