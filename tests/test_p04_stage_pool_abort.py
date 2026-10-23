import asyncio

import pytest

from drill import load

m = load("p04_stage_pool_abort")


class Rep:
    def __init__(self, name, log, fail=False):
        self.name, self.log, self.fail = name, log, fail

    async def abort(self, ids):
        self.log.append(("abort", self.name, sorted(ids)))
        if self.fail:
            raise RuntimeError("engine refused abort")


class Proc:
    def __init__(self, log):
        self.log = log
        self.state = {"x": "partial-text"}

    def collect_partial(self, rid):
        self.log.append(("collect", rid))
        return self.state.get(rid)

    def commit_abort(self, rid):
        self.log.append(("commit", rid))
        self.state.pop(rid, None)


def test_sticky_round_robin():
    pool = m.StagePool(0, [Rep("a", []), Rep("b", []), Rep("c", [])])
    picks = [pool.pick(f"r{i}") for i in range(6)]
    assert picks == [0, 1, 2, 0, 1, 2]
    assert all(pool.pick(f"r{i}") == picks[i] for i in range(6)), "binding must be sticky"
    pool.release("r0")
    assert pool.bound("r0") is None


def test_evict_and_no_live():
    pool = m.StagePool(1, [Rep("a", []), Rep("b", [])])
    for i in range(4):
        pool.pick(f"r{i}")
    affected = pool.evict(0)
    assert sorted(affected) == ["r0", "r2"]
    assert pool.live_replicas() == [1]
    assert all(pool.pick(f"n{i}") == 1 for i in range(3))
    assert pool.pick("r1") == 1
    pool.evict(1)
    with pytest.raises(m.NoLiveReplica):
        pool.pick("z")


def test_abort_groups_by_replica():
    log = []
    pool = m.StagePool(0, [Rep("a", log), Rep("b", log)])
    for i in range(4):
        pool.pick(f"r{i}")

    asyncio.run(pool.abort(["r0", "r1", "r2", "r3", "unknown"]))
    assert sorted(log) == [("abort", "a", ["r0", "r2"]), ("abort", "b", ["r1", "r3"])]
    assert all(pool.bound(f"r{i}") is None for i in range(4))


def test_abort_two_phase_order():
    log = []
    pools = [m.StagePool(0, [Rep("s0", log)]), m.StagePool(1, [Rep("s1", log)])]
    pools[0].pick("x")
    pools[1].pick("x")
    proc = Proc(log)
    out = asyncio.run(m.abort_request(pools, proc, "x"))
    assert log == [("collect", "x"), ("abort", "s0", ["x"]), ("abort", "s1", ["x"]), ("commit", "x")]
    assert out.finished and out.error == "abort" and out.data == "partial-text"


def test_abort_failure_does_not_commit():
    log = []
    pools = [m.StagePool(0, [Rep("s0", log, fail=True)])]
    pools[0].pick("x")
    proc = Proc(log)
    with pytest.raises(RuntimeError):
        asyncio.run(m.abort_request(pools, proc, "x"))
    assert ("commit", "x") not in log
    assert proc.state == {"x": "partial-text"}
    assert pools[0].bound("x") == 0, "binding kept so abort can be retried"


def test_dead_replica_only_fails_bound_requests():
    pool = m.StagePool(2, [Rep("a", []), Rep("b", [])])
    for i in range(4):
        pool.pick(f"r{i}")
    errs = m.handle_dead_replica(pool, 1)
    assert sorted(e.request_id for e in errs) == ["r1", "r3"]
    assert all(e.finished and e.error for e in errs)
    assert pool.bound("r0") == 0 and pool.bound("r2") == 0
