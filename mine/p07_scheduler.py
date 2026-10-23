"""P07 · Scheduler: continuous batching + chunked prefill + recompute preemption.

KV accounting is in blocks: a sequence holding n computed+scheduled tokens needs
ceil(n / block_size) blocks; self.blocks_used must never exceed num_blocks.

schedule() -> {seq_id: num_tokens_this_step}, total <= max_num_batched_tokens:
  1. RUNNING sequences first, in order: n = min(remaining tokens, budget)
     (1 for decode, more while a chunked prefill is still in progress). If the blocks do
     not fit, preempt the LAST running sequence (free its blocks, num_computed = 0, put it
     at the FRONT of waiting, num_preemptions += 1) and retry; if the victim is the
     sequence itself, skip it.
  2. WAITING sequences strictly FCFS while len(running) < max_num_seqs and budget > 0:
     n = min(remaining, budget); stop at the first one whose blocks do not fit.
update(sample_fn, eos): for each scheduled seq add n to num_computed; if it now covers all
  tokens, append sample_fn(seq_id) to output; finish on eos or max_tokens (free blocks).
A preempted sequence recomputes prompt + already generated tokens; outputs are kept.
"""
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Seq:
    seq_id: str
    prompt_len: int
    max_tokens: int
    output: list[int] = field(default_factory=list)
    num_computed: int = 0
    num_blocks: int = 0
    status: str = "WAITING"  # WAITING | RUNNING | FINISHED

    @property
    def num_tokens(self) -> int:
        return self.prompt_len + len(self.output)


class Scheduler:
    def __init__(self, num_blocks: int, block_size: int, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Seq] = deque()
        self.running: list[Seq] = []
        self.finished: list[Seq] = []
        self.blocks_used = 0
        self.num_preemptions = 0

    def add(self, seq: Seq) -> None:
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> dict[str, int]:
        raise NotImplementedError

    def update(self, sample_fn: Callable[[str], int], eos_token: int) -> list[Seq]:
        raise NotImplementedError
