import asyncio
import random

from drill import load

m = load("p02_orchestrator")
StageOutput = m.StageOutput


class FakeStage:
    """Emits len(fn(rid, inputs)) outputs asynchronously after each submit."""

    def __init__(self, fn, max_delay=0.002, emit_first_in_submit=False):
        self.fn = fn
        self.max_delay = max_delay
        self.emit_first_in_submit = emit_first_in_submit
        self.q: asyncio.Queue = asyncio.Queue()
        self.submitted: list = []
        self.aborted: list = []

    async def submit(self, rid, inputs):
        self.submitted.append((rid, inputs))
        chunks = self.fn(rid, inputs)
        start = 0
        if self.emit_first_in_submit:
            await self.q.put([StageOutput(rid, chunks[0], len(chunks) == 1)])
            start = 1
            await asyncio.sleep(0.01)  # let readers + dispatcher run before we return
        asyncio.get_running_loop().create_task(self._emit(rid, chunks, start))

    async def _emit(self, rid, chunks, start):
        for i in range(start, len(chunks)):
            await asyncio.sleep(random.random() * self.max_delay)
            await self.q.put([StageOutput(rid, chunks[i], i == len(chunks) - 1)])

    async def get_output(self):
        return await self.q.get()

    async def abort(self, request_ids):
        self.aborted += request_ids


def thinker(rid, prompt):
    return [f"{rid}:t{i}" for i in range(5)]


def talker(rid, inputs):
    up = [x.upper() for x in inputs]
    return [up[:3], up[3:]]


def vocoder(rid, inputs):
    flat = [x for part in inputs for x in part]
    return [f"wav({x})" for x in flat]


def forward(next_stage, prompt, prev_data):
    return list(prev_data)


def make(final={0, 2}, **kw):
    stages = [FakeStage(thinker, **kw), FakeStage(talker), FakeStage(vocoder)]
    return stages, m.Orchestrator(stages, final, forward)


async def collect(orch, n_requests, timeout=5.0):
    per = {}
    done = 0
    while done < n_requests:
        o = await orch.next_output(timeout=timeout)
        per.setdefault(o.request_id, []).append(o)
        if o.finished:
            done += 1
    return per


def check_request(rid, outs, final={0, 2}):
    assert sum(o.finished for o in outs) == 1, "exactly one terminal message"
    assert outs[-1].finished, "terminal message must be the last one"
    assert {o.stage_id for o in outs} == final, "only final stages reach the client"
    if 0 in final:
        assert [o.data for o in outs if o.stage_id == 0] == thinker(rid, None)
    expected_wav = [f"wav({x.upper()})" for x in thinker(rid, None)]
    assert [o.data for o in outs if o.stage_id == 2] == expected_wav


def test_single_request_three_stages():
    async def main():
        stages, orch = make()
        await orch.start()
        await orch.add_request("r0", "hello")
        per = await collect(orch, 1)
        check_request("r0", per["r0"])
        assert orch.request_states == {}
        await orch.shutdown()

    asyncio.run(main())


def test_many_concurrent_requests():
    async def main():
        stages, orch = make()
        await orch.start()
        rids = [f"r{i}" for i in range(30)]
        for rid in rids:
            await orch.add_request(rid, "p")
        per = await collect(orch, len(rids))
        for rid in rids:
            check_request(rid, per[rid])
        # each stage got each request exactly once
        for s in stages:
            assert sorted(r for r, _ in s.submitted) == sorted(rids)
        await orch.shutdown()

    asyncio.run(main())


def test_only_last_stage_is_final():
    async def main():
        stages, orch = make(final={2})
        await orch.start()
        await orch.add_request("a", "p")
        per = await collect(orch, 1)
        check_request("a", per["a"], final={2})
        await orch.shutdown()

    asyncio.run(main())


def test_state_registered_before_submit():
    async def main():
        stages, orch = make(emit_first_in_submit=True)
        await orch.start()
        await orch.add_request("early", "p")
        per = await collect(orch, 1)
        check_request("early", per["early"])
        assert orch.dropped_outputs == 0
        await orch.shutdown()

    asyncio.run(main())


def test_unknown_request_output_is_dropped():
    async def main():
        stages, orch = make()
        await orch.start()
        await stages[0].q.put([StageOutput("ghost", "x", True)])
        await orch.add_request("r", "p")
        per = await collect(orch, 1)
        check_request("r", per["r"])
        assert orch.dropped_outputs == 1
        await orch.shutdown()

    asyncio.run(main())
