import pytest

from drill import load

m = load("p06_block_manager")


def test_allocate_and_slots():
    bm = m.BlockManager(num_blocks=8, block_size=4)
    table = bm.allocate("a", list(range(10)))
    assert len(table) == 3 and len(set(table)) == 3
    assert bm.num_free_blocks == 5
    for pos in range(10):
        assert bm.slot("a", pos) == table[pos // 4] * 4 + pos % 4


def test_append_crosses_block_boundary():
    bm = m.BlockManager(num_blocks=3, block_size=4)
    bm.allocate("a", list(range(7)))  # 2 blocks, 1 free slot in the 2nd
    s7 = bm.append_slot("a", 7)
    assert bm.num_free_blocks == 1
    s8 = bm.append_slot("a", 8)  # needs a 3rd block
    assert bm.num_free_blocks == 0
    t = bm.block_table("a")
    assert s7 == t[1] * 4 + 3 and s8 == t[2] * 4 + 0


def test_out_of_blocks_is_atomic():
    bm = m.BlockManager(num_blocks=2, block_size=4)
    with pytest.raises(m.OutOfBlocks):
        bm.allocate("big", list(range(9)))
    assert bm.num_free_blocks == 2
    bm.allocate("a", list(range(8)))
    with pytest.raises(m.OutOfBlocks):
        bm.append_slot("a", 8)
    assert bm.block_table("a") and len(bm.block_table("a")) == 2
    bm.free("a")
    assert bm.num_free_blocks == 2


def test_free_returns_everything():
    bm = m.BlockManager(num_blocks=6, block_size=2)
    bm.allocate("a", [1, 2, 3])
    bm.allocate("b", [4, 5, 6, 7])
    bm.free("a")
    bm.free("b")
    assert bm.num_free_blocks == 6


def test_prefix_sharing_and_refcount():
    bm = m.BlockManager(num_blocks=10, block_size=4, enable_prefix_cache=True)
    prefix = list(range(8))
    ta = bm.allocate("a", prefix + [100, 101])
    tb = bm.allocate("b", prefix + [200])
    assert bm.num_cached_tokens("b") == 8
    assert tb[:2] == ta[:2] and tb[2] != ta[2]
    assert bm.num_free_blocks == 10 - 4
    bm.free("a")
    assert bm.num_free_blocks == 10 - 3, "shared blocks are still referenced by b"


def test_cache_survives_free_and_last_token_is_never_cached():
    bm = m.BlockManager(num_blocks=10, block_size=4, enable_prefix_cache=True)
    bm.allocate("a", list(range(8)) + [9])
    bm.free("a")
    bm.allocate("c", list(range(8)) + [42])
    assert bm.num_cached_tokens("c") == 8
    bm.free("c")
    # exactly 8 tokens: only the first block may be reused, the last token must be computed
    bm.allocate("d", list(range(8)))
    assert bm.num_cached_tokens("d") == 4


def test_eviction_drops_cached_blocks():
    bm = m.BlockManager(num_blocks=4, block_size=4, enable_prefix_cache=True)
    bm.allocate("a", list(range(9)))
    bm.free("a")
    bm.allocate("filler", list(range(1000, 1016)))  # needs all 4 blocks -> evicts
    bm.free("filler")
    bm.allocate("b", list(range(9)))
    assert bm.num_cached_tokens("b") == 0


def test_append_completing_a_block_registers_it():
    bm = m.BlockManager(num_blocks=8, block_size=4, enable_prefix_cache=True)
    bm.allocate("e", [5, 6, 7])
    bm.append_slot("e", 8)  # block [5,6,7,8] is now full
    bm.free("e")
    bm.allocate("f", [5, 6, 7, 8, 9])
    assert bm.num_cached_tokens("f") == 4
