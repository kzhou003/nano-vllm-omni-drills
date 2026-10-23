"""P06 · BlockManager: paged KV-cache allocator with ref counts and prefix caching.

- Physical blocks 0..num_blocks-1 of block_size token slots each.
- allocate(seq, tokens): all-or-nothing (raise OutOfBlocks without changing state).
  With prefix caching, FULL blocks are identified by a hash chain
  h_i = hash((h_{i-1}, tuple(block_i_tokens))). Reuse matching blocks from the start of the
  prompt, but only among the first (len(tokens)-1)//block_size blocks, so at least one
  token is always computed. num_cached_tokens(seq) = reused blocks * block_size.
- append_slot(seq, token): returns the slot (block*block_size + offset) of the new token;
  allocates a block when needed (OutOfBlocks leaves state unchanged); when a block becomes
  full it is registered in the prefix cache.
- free(seq): decrement refs (walk the table in reverse). A block whose ref hits 0 goes
  back to the free list, or -- if it carries a hash -- to an LRU "evictable" pool that
  still serves cache hits until it is reused for a new allocation.
- num_free_blocks counts free + evictable blocks.
"""


class OutOfBlocks(Exception):
    pass


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = False) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        raise NotImplementedError

    @property
    def num_free_blocks(self) -> int:
        raise NotImplementedError

    def block_table(self, seq_id: str) -> list[int]:
        raise NotImplementedError

    def num_cached_tokens(self, seq_id: str) -> int:
        raise NotImplementedError

    def slot(self, seq_id: str, pos: int) -> int:
        raise NotImplementedError

    def can_allocate(self, num_tokens: int) -> bool:
        raise NotImplementedError

    def allocate(self, seq_id: str, token_ids: list[int]) -> list[int]:
        raise NotImplementedError

    def append_slot(self, seq_id: str, token_id: int) -> int:
        raise NotImplementedError

    def free(self, seq_id: str) -> None:
        raise NotImplementedError
