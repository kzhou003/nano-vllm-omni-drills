import fcntl
import os
import subprocess
import sys
import time
import uuid

from drill import load

m = load("p12_shm_connector")


def _uniq(tag):
    return f"{tag}-{uuid.uuid4().hex[:8]}"


def test_roundtrip_and_consume_once():
    c = m.ShmConnector(namespace="t1")
    k = _uniq("req_0_0")
    c.put(k, b"hello")
    assert c.get(k) == b"hello"
    assert c.get(k) is None, "the reader consumes (unlinks) the segment"
    assert c.get(_uniq("missing")) is None


def test_empty_and_large_payloads():
    c = m.ShmConnector(namespace="t2")
    k1, k2 = _uniq("e"), _uniq("big")
    c.put(k1, b"")
    assert c.get(k1) == b""
    blob = os.urandom(4 * 1024 * 1024 + 3)
    c.put(k2, blob)
    assert c.get(k2) == blob


def test_independent_writer_process_exits_before_read():
    """Writer is a separate interpreter (its own resource tracker) that exits first."""
    ns = "t3"
    keys = [_uniq(f"r_0_{i}") for i in range(3)]
    code = (
        "from drill import load; c = load('p12_shm_connector').ShmConnector(namespace=%r)\n"
        "for k in %r: c.put(k, ('payload-' + k).encode())" % (ns, keys)
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root, env={**os.environ, "PYTHONPATH": root})
    time.sleep(1.0)  # give the writer's resource tracker time to "clean up"
    c = m.ShmConnector(namespace=ns)
    for k in keys:
        assert c.get(k) == f"payload-{k}".encode(), "segment vanished when the writer exited (resource tracker?)"


def test_not_ready_while_lock_is_held():
    c = m.ShmConnector(namespace="t4")
    k = _uniq("locked")
    c.put(k, b"data")
    fd = os.open(c.lock_path(k), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)  # pretend a writer is still writing
        assert c.get(k) is None
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert c.get(k) == b"data"


def test_cleanup_prefix_only_touches_unread_keys_with_prefix():
    c = m.ShmConnector(namespace="t5")
    a = _uniq("reqA")
    b = _uniq("reqB")
    c.put(f"{a}_0_0", b"1")
    c.put(f"{a}_0_1", b"2")
    c.put(f"{b}_0_0", b"3")
    assert c.get(f"{a}_0_0") == b"1"  # consumed already
    assert c.cleanup_prefix(f"{a}_") == 1
    assert c.get(f"{a}_0_1") is None
    assert c.get(f"{b}_0_0") == b"3"
