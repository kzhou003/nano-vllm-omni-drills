"""P12 reference solution · SharedMemoryConnector：跨进程 put/get 协议（一个 key 一个消费者）。

没有元数据通道：两边只靠 key 独立推出段名和锁文件名。协议：
  - 锁文件存在        = 有人写过、还没被读（或正在写）
  - 锁文件被 flock 持有 = writer 正在写，reader 立即返回 None（非阻塞）
  - reader 读完后负责 unlink 段 + 删除锁文件（消费即删除）
  - producer 记住自己写过的 key，请求中止时用 cleanup_prefix 清理没人读的段
参与者是进程而不是线程：threading.Lock 毫无意义，能用的只有内核提供的段、文件和 flock。
提示页：hints/p12_overview.html（A-20）。
"""
import fcntl
import hashlib
import os
import struct
import tempfile
from multiprocessing import resource_tracker, shared_memory

# 段布局：[ 8 字节 little-endian 长度 n | n 字节 payload | (页对齐填充) ]
# 易错：不能用 shm.size 当 payload 长度——段大小会被向上取整到页（macOS 上 5 字节的段 size 是 16384）
_HEADER = struct.Struct("<Q")


class ShmConnector:
    def __init__(self, lock_dir: str | None = None, namespace: str = "nvo") -> None:
        self.namespace = namespace
        # 锁文件目录：两个进程必须用同一个（默认值也只由 namespace 决定，所以两边能独立算出同一路径）
        self.lock_dir = lock_dir or os.path.join(tempfile.gettempdir(), f"{namespace}-shm-locks")
        os.makedirs(self.lock_dir, exist_ok=True)
        # producer 侧：我写过、可能还没人读的 key（只给 cleanup_prefix 用）
        self._written: set[str] = set()

    # 段名必须是 key 的确定性函数：另一个进程里的 reader 只知道 key，要能独立算出同一个名字。
    # 易错 1：用 hashlib，不能用 Python 的 hash()（每个进程有随机种子）；
    # 易错 2：macOS 的 POSIX shm 名字最长 31 字节（含库自动加的前导 "/"），所以对 key 做短哈希
    def shm_name(self, key: str) -> str:
        digest = hashlib.sha1(f"{self.namespace}/{key}".encode()).hexdigest()[:20]
        return f"{self.namespace[:8]}_{digest}"  # <= 29 bytes, fits macOS' 31

    def lock_path(self, key: str) -> str:
        return os.path.join(self.lock_dir, self.shm_name(key) + ".lock")

    # 强制删除一个段（不读数据）；段不存在时返回 False。attach 之后 unlink 会同时注销本进程 tracker 的登记，保持平衡
    @staticmethod
    def _unlink(name: str) -> bool:
        try:
            shm = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            return False
        shm.close()
        shm.unlink()
        return True

    # put：写者路径。先拿排他锁（告诉读者“正在写”），再创建段、写 header + 数据、交出所有权、close。
    # 写完后锁文件保留（表示“有数据待读”），段也保留（不 unlink，留给读者）。
    def put(self, key: str, data: bytes) -> None:
        name = self.shm_name(key)
        # O_CREAT：锁文件由 writer 创建；reader 只会 open 已存在的锁文件
        fd = os.open(self.lock_path(key), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            # 阻塞式排他锁（若 reader 正持锁读，等它读完）。
            # 易错：锁必须在创建段之前拿到——先建段后拿锁，reader 可能在两者之间拿到锁、读到写了一半的段
            fcntl.flock(fd, fcntl.LOCK_EX)
            # 同名段已存在（上一次写入没人读 / 进程崩溃残留）：先删掉再重建，不能直接复用（大小可能不够）
            self._unlink(name)  # stale segment from an earlier run
            # max(len, 1)：size=0 会 ValueError，长度头也让空 payload 有合法大小
            shm = shared_memory.SharedMemory(name=name, create=True, size=_HEADER.size + max(len(data), 1))
            _HEADER.pack_into(shm.buf, 0, len(data))
            # 切片赋值不保留 memoryview，之后可以直接 close
            shm.buf[_HEADER.size:_HEADER.size + len(data)] = data
            # hand ownership to the reader: our tracker must not unlink it when we exit
            # resource tracker 陷阱（Python < 3.13）：SharedMemory(create=True) 会把段登记到本进程的
            # resource_tracker；写进程先退出时 tracker 会把仍登记着的段当作“泄漏” unlink 掉，读者再来就读不到了。
            # 易错：漏掉这一步。_name 是带前导 "/" 的真实名字，tracker 登记用的就是它
            resource_tracker.unregister(shm._name, "shared_memory")
            shm.close()   # 只解除本进程映射，不 unlink
            self._written.add(key)
        finally:
            # 释放锁之后 reader 的 LOCK_NB 才能成功 -> 读者永远看不到写了一半的数据
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # get：读者路径，非阻塞、消费式。三种 None：锁文件不存在（从没写过 / 已被消费）/ 锁被 writer 持有（还在写）
    # / 段不存在（writer 刚创建锁文件还没建段）。每一条 return 路径都要关闭 fd（关闭即释放 flock）。
    def get(self, key: str) -> bytes | None:
        lock = self.lock_path(key)
        try:
            # 不带 O_CREAT：锁文件不存在就说明没东西可读，不能替 writer 创建
            fd = os.open(lock, os.O_RDWR)
        except FileNotFoundError:
            return None
        try:
            try:
                # LOCK_NB：拿不到立刻抛 BlockingIOError，而不是阻塞等 writer（get 由 P13 的 recv 线程稍后重试）。
                # flock 锁属于“打开的文件描述”而不是进程：同进程里另一个 fd 持锁也会冲突
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None  # still being written
            try:
                shm = shared_memory.SharedMemory(name=self.shm_name(key))
            except FileNotFoundError:
                # 锁文件在但段不在：返回 None 下次再来，不要抛异常，也不要删锁文件
                return None
            try:
                (n,) = _HEADER.unpack_from(shm.buf, 0)
                # 易错：必须先 bytes(...) 拷贝——memoryview 切片持有 buffer 导出，直接 close 会抛
                # BufferError，而且 close / unlink 之后内存可能已经失效
                data = bytes(shm.buf[_HEADER.size:_HEADER.size + n])
            finally:
                shm.close()
                shm.unlink()   # 由读者释放：只有读者知道“已经用完了”
            # 删除锁文件 = 宣告“已消费”；此时仍持有 flock，writer 插不进来（前提：key 不复用、一个消费者）
            try:
                os.unlink(lock)
            except FileNotFoundError:
                pass
            self._written.discard(key)   # 同进程读写时顺便清掉记录
            return data
        finally:
            os.close(fd)   # 关闭 fd 同时释放 flock

    # cleanup_prefix：producer 侧在请求结束 / abort 时调用（key 形如 f"{req}_{stage}_{chunk}"，前缀通常是 f"{req}_"）。
    # 只清理“我写过 + 前缀匹配 + 还没被消费”的 key，返回清理个数。
    # 易错：前缀要带分隔符，"rA_" 不会误伤 "rAx_..."，而 "rA" 会
    def cleanup_prefix(self, prefix: str) -> int:
        removed = 0
        # 先拷贝成 list：循环里要修改 self._written
        for key in [k for k in self._written if k.startswith(prefix)]:
            self._written.discard(key)
            # 段还在 = 没被 reader 消费（消费时 reader 会 unlink），这才计数；已被消费的只清记录、不计数
            if self._unlink(self.shm_name(key)):
                removed += 1
            try:
                os.unlink(self.lock_path(key))
            except FileNotFoundError:
                pass
        return removed
