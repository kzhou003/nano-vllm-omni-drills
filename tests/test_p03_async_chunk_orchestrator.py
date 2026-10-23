import asyncio
import random

from drill import load

m = load("p03_async_chunk_orchestrator")
StageOutput = m.StageOutput


class Channel:
    """Side channel between two fake stages (stands in for the SHM connector)."""

    def __init__(self):
        self.qs: dict[str, asyncio.Queue] = {}

    def q(self, rid):
        return self.qs.setdefault(rid, asyncio.Queue())


class ChunkStage:
    """Stage that reads chunks from `inp` (or the prompt for stage 0), transforms
    each one, forwards it on `out`, and reports outputs to the orchestrator.

    late_finish: report finished=True to the orchestrator only after `late_finish` s
    (simulates a non-final stage whose terminal report arrives after the final stage).
    """

    def __init__(self, transform, inp=None, out=None, late_finish=0.0, n_tokens=6):
        self.transform, self.inp, self.out = transform, inp, out
        self.late_finish, self.n_tokens = late_finish, n_tokens
        self.q: asyncio.Queue = asyncio.Queue()
        self.submitted: list = []

    async def submit(self, rid, inputs):
        self.submitted.append((rid, inputs))
        asyncio.get_running_loop().create_task(self._run(rid))

    async def _run(self, rid):
        if self.inp is None:
            items = [(f"{rid}:t{i}", i == self.n_tokens - 1) for i in range(self.n_tokens)]
        else:
            items = None
        i = 0
        while True:
            if items is not None:
                if i == len(items):
                    break
                data, last = items[i]
                i += 1
            else:
                data, last = await self.inp.q(rid).get()
            await asyncio.sleep(random.random() * 0.002)
            y = self.transform(data)
            if self.out is not None:
                await self.out.q(rid).put((y, last))
            if last and self.late_finish:
                await self.q.put([StageOutput(rid, y, False)])
                await asyncio.sleep(self.late_finish)
                await self.q.put([StageOutput(rid, None, True)])
            else:
                await self.q.put([StageOutput(rid, y, last)])
            if last:
                break

    async def get_output(self):
        return await self.q.get()

    async def abort(self, ids):
        pass


def build(late_finish=0.0):
    c01, c12 = Channel(), Channel()
    stages = [
        ChunkStage(lambda x: x, out=c01),
        ChunkStage(lambda x: x.upper(), inp=c01, out=c12, late_finish=late_finish),
        ChunkStage(lambda x: f"wav({x})", inp=c12),
    ]
    orch = m.AsyncChunkOrchestrator(stages, {0, 2}, prewarm_fn=lambda sid, p: ("placeholder", sid))
    return stages, orch


async def collect(orch, n, timeout=5.0):
    per, done = {}, 0
    while done < n:
        o = await orch.next_output(timeout=timeout)
        per.setdefault(o.request_id, []).append(o)
        done += o.finished
    return per


def test_prewarm_and_no_forwarding():
    async def main():
        stages, orch = build()
        await orch.start()
        await orch.add_request("r", "p")
        per = await collect(orch, 1)
        outs = per["r"]
        assert [o.data for o in outs if o.stage_id == 0] == [f"r:t{i}" for i in range(6)]
        assert [o.data for o in outs if o.stage_id == 2] == [f"wav(R:T{i})" for i in range(6)]
        assert outs[-1].finished and sum(o.finished for o in outs) == 1
        # every stage submitted exactly once; downstream got the placeholder
        assert stages[0].submitted == [("r", "p")]
        assert stages[1].submitted == [("r", ("placeholder", 1))]
        assert stages[2].submitted == [("r", ("placeholder", 2))]
        await orch.shutdown()

    asyncio.run(main())


def test_terminal_waits_for_non_final_stage():
    async def main():
        stages, orch = build(late_finish=0.2)
        await orch.start()
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await orch.add_request("r", "p")
        per = await collect(orch, 1)
        elapsed = loop.time() - t0
        assert elapsed >= 0.18, "terminal was emitted before stage 1 reported finished"
        outs = per["r"]
        assert outs[-1].finished and outs[-1].stage_id == 2
        assert sum(o.finished for o in outs) == 1
        assert all(o.stage_id in (0, 2) for o in outs)
        assert orch.request_states == {}
        await orch.shutdown()

    asyncio.run(main())


def test_many_requests_with_late_finish():
    async def main():
        stages, orch = build(late_finish=0.01)
        await orch.start()
        rids = [f"r{i}" for i in range(20)]
        for rid in rids:
            await orch.add_request(rid, "p")
        per = await collect(orch, len(rids))
        for rid in rids:
            outs = per[rid]
            assert outs[-1].finished and sum(o.finished for o in outs) == 1
            assert [o.data for o in outs if o.stage_id == 2] == [f"wav({rid.upper()}:T{i})" for i in range(6)]
        await orch.shutdown()

    asyncio.run(main())


def test_late_output_after_cleanup_is_dropped():
    async def main():
        stages, orch = build()
        await orch.start()
        await orch.add_request("r", "p")
        await collect(orch, 1)
        await stages[1].q.put([StageOutput("r", "zombie", True)])
        await asyncio.sleep(0.05)
        assert orch.dropped_outputs == 1
        assert "r" not in orch.request_states
        await orch.shutdown()

    asyncio.run(main())
