"""P03 · Orchestrator v2 for async_chunk (builds on YOUR P02).

Stages now exchange data on a side channel (the connector). The orchestrator:
- on add: registers state, marks stages 0..last as submitted, submits the prompt to
  stage 0 and a placeholder prewarm_fn(sid, prompt) to every downstream stage;
- never forwards stage outputs itself;
- emits outputs of final stages, but the terminal message (finished=True) is only
  released once EVERY submitted stage has finished. If the last final stage finishes
  first, hold the message in ReqState.pending_final and emit it when the remaining
  stage(s) finish. Exactly one terminal message, always the last one.
- outputs after cleanup are dropped (self.dropped_outputs), never resurrect state.
"""
from collections.abc import Callable
from typing import Any
from .p02_orchestrator import Orchestrator, OrchOutput, ReqState, StageClient, StageOutput

__all__ = ["AsyncChunkOrchestrator", "OrchOutput", "StageOutput", "ReqState"]


class AsyncChunkOrchestrator(Orchestrator):
    def __init__(
        self,
        stages: list[StageClient],
        final_stage_ids: set[int],
        prewarm_fn: Callable[[int, Any], Any],
    ) -> None:
        super().__init__(stages, final_stage_ids, forward_fn=lambda *_: None)
        self.prewarm_fn = prewarm_fn

    # Override the add path and the per-output handling of your P02.
    async def _handle_add(self, rid: str, prompt: Any) -> None:
        raise NotImplementedError

    async def _handle_output(self, sid: int, o: StageOutput) -> None:
        raise NotImplementedError
