import threading
import time

import pytest

from drill import load

m = load("p16_init_planner")
GiB = 1 << 30


def test_groups_are_connected_components():
    devs = {0: {0}, 1: {1}, 2: {1}, 3: {2, 3}, 4: {3}, 5: {4}}
    assert m.init_groups(devs) == [[0], [1, 2], [3, 4], [5]]
    assert m.init_groups({0: {0, 1}, 1: {1, 2}, 2: {2}}) == [[0, 1, 2]]


def test_groups_run_in_parallel_and_sequential_within_group():
    devs = {0: {0}, 1: {1}, 2: {1}}
    started = {s: threading.Event() for s in devs}
    active = {"g1": 0, "max_g1": 0}
    order = []
    lock = threading.Lock()

    def init(sid):
        started[sid].set()
        with lock:
            order.append(sid)
            if sid in (1, 2):
                active["g1"] += 1
                active["max_g1"] = max(active["max_g1"], active["g1"])
        if sid == 0:
            # only returns if stage 1 starts while stage 0 is still initializing
            assert started[1].wait(2.0), "groups did not run in parallel"
        time.sleep(0.05)
        with lock:
            if sid in (1, 2):
                active["g1"] -= 1
        return f"engine{sid}"

    res = m.run_init(devs, init, cleanup_fn=lambda s, r: None)
    assert res == {0: "engine0", 1: "engine1", 2: "engine2"}
    assert active["max_g1"] == 1, "stages sharing a device must not init concurrently"
    assert order.index(1) < order.index(2)


def test_failure_short_circuits_and_rolls_back():
    devs = {0: {0}, 1: {1}, 2: {1}}
    calls, cleaned = [], []
    stage0_started = threading.Event()

    def init(sid):
        calls.append(sid)
        if sid == 0:
            stage0_started.set()
        if sid == 1:
            stage0_started.wait(2.0)
            raise RuntimeError("OOM while capturing graphs")
        time.sleep(0.05)
        return sid

    with pytest.raises(m.InitError) as ei:
        m.run_init(devs, init, cleanup_fn=lambda s, r: cleaned.append(s))
    assert ei.value.stage_id == 1
    assert 2 not in calls, "stage 2 must not start after stage 1 failed in the same group"
    assert cleaned == [0], "successfully initialized stages are cleaned up"


def test_admission_check():
    totals = {0: 32 * GiB}
    ok = {0: ({0}, 0.5), 1: ({0}, 0.3)}
    assert m.admission_check(totals, ok, reserve_bytes_per_stage=GiB) == []
    bad = {0: ({0}, 0.6), 1: ({0}, 0.3), 2: ({0}, 0.1)}
    errs = m.admission_check(totals, bad, reserve_bytes_per_stage=GiB)
    assert len(errs) == 1 and "device 0" in errs[0]
