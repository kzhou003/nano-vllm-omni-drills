import pytest

from drill import load

m = load("p11_chunk_aware_scheduler")


class Source:
    def __init__(self):
        self.ready: set[str] = set()
        self.loads: list[str] = []

    def load_async(self, rid):
        self.loads.append(rid)

    def is_ready(self, rid):
        return rid in self.ready


def make():
    src = Source()
    return m.ChunkAwareScheduler(m.BaseScheduler(), src), src


def test_not_ready_is_never_scheduled_and_loaded_once():
    s, src = make()
    s.add(m.Req("plain", 0))
    s.add(m.Req("c1", 1, needs_chunk=True))
    for _ in range(3):
        assert "c1" not in s.schedule()
    assert src.loads == ["c1"]
    assert "c1" in s.parked and s.parked["c1"][0].status == "WAITING_FOR_CHUNK"


def test_ready_request_is_scheduled_on_next_call():
    s, src = make()
    s.add(m.Req("c1", 0, needs_chunk=True))
    assert s.schedule() == []
    src.ready.add("c1")
    assert s.schedule() == ["c1"]
    assert not s.parked


def test_running_request_waiting_for_next_chunk():
    s, src = make()
    src.ready.add("c1")
    s.add(m.Req("c1", 0, needs_chunk=True))
    assert s.schedule() == ["c1"]
    src.ready.discard("c1")  # consumed chunk k, chunk k+1 not here yet
    assert s.schedule() == []
    src.ready.add("c1")
    assert s.schedule() == ["c1"]
    assert s.base.running[0].status == "RUNNING"


class Boom(Exception):
    pass


def test_restore_happens_even_if_upstream_raises():
    src = Source()

    class Exploding(m.BaseScheduler):
        def schedule(self):
            src.ready.add("c1")  # the chunk lands while upstream is scheduling...
            raise Boom("upstream scheduler blew up")  # ...and upstream fails

    s = m.ChunkAwareScheduler(Exploding(), src)
    s.add(m.Req("c1", 0, needs_chunk=True))
    with pytest.raises(Boom):
        s.schedule()
    assert src.loads == ["c1"], "c1 must have been parked before upstream ran"
    assert "c1" not in s.parked, "the finally-restore must run even when upstream raises"
    assert [r.req_id for r in s.base.waiting] == ["c1"]


def test_abort_while_parked():
    s, src = make()
    s.add(m.Req("c1", 0, needs_chunk=True))
    s.schedule()
    s.abort("c1")
    src.ready.add("c1")
    assert s.schedule() == []
    assert "c1" not in s.parked


def test_restored_in_arrival_order():
    s, src = make()
    for i, rid in enumerate(["a", "b", "c"]):
        s.add(m.Req(rid, i, needs_chunk=True))
    s.schedule()
    src.ready.update({"c", "b", "a"})
    assert s.schedule() == ["a", "b", "c"]
