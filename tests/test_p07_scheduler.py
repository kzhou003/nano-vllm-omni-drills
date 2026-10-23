from drill import load

m = load("p07_scheduler")
EOS = -1


def tok(seq_id, k):
    """Deterministic 'model': k-th output token of a sequence."""
    return 1000 * int(seq_id[1:]) + k


def run_step(s):
    sched = s.schedule()
    by_id = {q.seq_id: q for q in s.running}
    s.update(lambda sid: tok(sid, len(by_id[sid].output)), EOS)
    return sched


def test_token_budget_and_chunked_prefill():
    s = m.Scheduler(num_blocks=100, block_size=4, max_num_seqs=8, max_num_batched_tokens=16)
    for i in range(3):
        s.add(m.Seq(f"s{i}", prompt_len=10, max_tokens=5))
    assert run_step(s) == {"s0": 10, "s1": 6}
    assert run_step(s) == {"s0": 1, "s1": 4, "s2": 10}
    assert run_step(s) == {"s0": 1, "s1": 1, "s2": 1}


def test_max_num_seqs():
    s = m.Scheduler(num_blocks=100, block_size=4, max_num_seqs=2, max_num_batched_tokens=100)
    for i in range(4):
        s.add(m.Seq(f"s{i}", prompt_len=3, max_tokens=2))
    assert set(run_step(s)) == {"s0", "s1"}
    assert set(run_step(s)) == {"s0", "s1"}  # s0/s1 finish here (2 tokens)
    assert set(run_step(s)) == {"s2", "s3"}


def test_fcfs_head_of_line():
    s = m.Scheduler(num_blocks=4, block_size=4, max_num_seqs=8, max_num_batched_tokens=100)
    s.add(m.Seq("s0", prompt_len=12, max_tokens=50))  # 3 blocks
    s.add(m.Seq("s1", prompt_len=8, max_tokens=1))    # 2 blocks -> does not fit
    s.add(m.Seq("s2", prompt_len=2, max_tokens=1))    # would fit, but FCFS forbids skipping
    assert run_step(s) == {"s0": 12}


def test_preemption_eventually_finishes_everything_correctly():
    s = m.Scheduler(num_blocks=6, block_size=4, max_num_seqs=8, max_num_batched_tokens=64)
    seqs = [m.Seq(f"s{i}", prompt_len=5, max_tokens=8) for i in range(4)]
    for q in seqs:
        s.add(q)
    steps = 0
    while s.has_work():
        run_step(s)
        assert s.blocks_used <= s.num_blocks
        steps += 1
        assert steps < 500, "no progress"
    assert s.num_preemptions > 0
    assert s.blocks_used == 0
    for q in seqs:
        assert q.status == "FINISHED"
        assert q.output == [tok(q.seq_id, k) for k in range(8)]


def test_eos_stops_early():
    s = m.Scheduler(num_blocks=10, block_size=4, max_num_seqs=8, max_num_batched_tokens=64)
    s.add(m.Seq("s1", prompt_len=3, max_tokens=100))
    s.schedule()
    s.update(lambda sid: EOS, EOS)
    assert not s.has_work() and s.finished[0].output == [EOS] and s.blocks_used == 0
