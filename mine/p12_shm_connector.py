"""P12 · ShmConnector: byte payloads through named POSIX shared memory (one consumer/key).

Protocol (no metadata channel -- both sides derive names from the key):
- shm_name(key): a short, deterministic name (macOS caps POSIX shm names at 31 bytes).
- put(key, data): take an EXCLUSIVE flock on lock_path(key) (create the file), create the
  segment (replace a stale one), write an 8-byte length header + data, then unlock.
  Remember the key so cleanup_prefix() can find it.
- get(key): non-blocking. None if the lock file is missing, the lock is held by someone
  else (LOCK_NB -> BlockingIOError = "still being written"), or the segment is missing.
  Otherwise read, close, UNLINK the segment and remove the lock file (the reader frees it).
- cleanup_prefix(prefix): producer side; unlink every key it wrote with that prefix that
  was never consumed; return how many it removed.
- The payload must survive the WRITER PROCESS EXITING before the reader reads.
  (Hint: look at multiprocessing.resource_tracker.)
"""


class ShmConnector:
    def __init__(self, lock_dir: str | None = None, namespace: str = "nvo") -> None:
        raise NotImplementedError

    def shm_name(self, key: str) -> str:
        raise NotImplementedError

    def lock_path(self, key: str) -> str:
        raise NotImplementedError

    def put(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def get(self, key: str) -> bytes | None:
        raise NotImplementedError

    def cleanup_prefix(self, prefix: str) -> int:
        raise NotImplementedError
