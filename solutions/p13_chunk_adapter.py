"""P13 reference solution · ChunkAdapter：async-chunk 数据面里一个 stage 的发送端 + 接收端。

发送端：调度线程 save_async 只做 deepcopy + 入队；后台 save 线程 build_payload 后按
        key = f"{req}_{stage_id}_{chunk_id}" put 到 connector，chunk_id 从 0 连续递增（只在 put 之后前进）。
接收端：load_async 注册 req（发一个代际 token）；后台 recv 线程按 f"{req}_{src_stage_id}_{next_id}" 轮询 get，
        在锁内核对 token 后按序提交到 _ready，调度线程用 poll_ready 取走。
两个后台线程让慢速序列化 / 传输不阻塞调度线程；所有慢操作都在锁外。提示页：hints/p13_overview.html（A-21）。
"""
import copy
import threading
from collections import deque
from collections.abc import Callable
from typing import Any


class InMemoryConnector:
    """Transport only: put/get by key, get consumes. Thread-safe. Hooks let tests add latency."""

    def __init__(self, on_put: Callable[[str], None] | None = None, on_get: Callable[[str], None] | None = None):
        self._store: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.on_put, self.on_get = on_put, on_get
        self.puts: list[str] = []

    def put(self, key: str, obj: Any) -> None:
        # hook 在锁外调用：测试可以在这里 sleep 模拟慢传输，而不会把整个 store 锁住
        if self.on_put:
            self.on_put(key)
        with self._lock:
            self._store[key] = obj
            self.puts.append(key)

    def get(self, key: str) -> Any | None:
        with self._lock:
            present = key in self._store
        if not present:
            return None
        if self.on_get:
            self.on_get(key)
        # pop 带默认值：两次加锁之间理论上可能被别人取走，此时返回 None 而不是抛 KeyError
        with self._lock:
            return self._store.pop(key, None)


class ChunkAdapter:
    def __init__(
        self,
        connector: InMemoryConnector,
        stage_id: int,               # 作为发送方时写进 key 的编号
        src_stage_id: int | None,    # 作为接收方时上游的编号（与上游的 stage_id 拼出同一个 key）
        build_payload: Callable[[Any], Any | None],
    ) -> None:
        self.connector = connector
        self.stage_id = stage_id
        self.src_stage_id = src_stage_id
        self.build_payload = build_payload
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        # sender
        # _save_q 元素：(req_id, 深拷贝的 snapshot, is_final)；deque 的 append / popleft 本身线程安全
        self._save_q: deque = deque()
        self._save_cv = threading.Condition()   # 只用来唤醒 save 线程，不保护数据
        # 下面两个只由 save 线程写（_cancelled 由调度线程 add），单个 set/dict 操作在 CPython 下是原子的
        self._chunk_id: dict[str, int] = {}     # req_id -> 下一个要发送的 chunk id
        self._cancelled: set[str] = set()
        # receiver：三个 dict 被 recv 线程和调度线程（poll_ready / cleanup_receiver）同时读写 -> 同一把锁
        self._lock = threading.Lock()
        self._recv_cv = threading.Condition(self._lock)   # 共用 _lock：load_async 注册与唤醒是一个原子动作
        # req_id -> 代际令牌（每次注册一个新的 object()）：用 `is` 比较判断“读取期间是否被 cleanup 或重新注册过”
        self._registered: dict[str, object] = {}
        self._next_id: dict[str, int] = {}      # req_id -> 下一个期望收到的 chunk id
        self._ready: dict[str, list] = {}       # req_id -> 已按序提交、等待 poll_ready 取走的 chunk

    def start(self) -> None:
        for target in (self._save_loop, self._recv_loop):
            # daemon=True：即使 stop 没被调用，进程退出也不会被这两个线程卡住
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        # 先置位 stop，再 notify_all：线程醒来检查 _stop 即退出；wait 都带超时，错过通知也最多晚一个超时周期
        self._stop.set()
        with self._save_cv:
            self._save_cv.notify_all()
        with self._recv_cv:
            self._recv_cv.notify_all()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()

    # ---------------------------------------------------------------- sender
    # save_async：调度线程调用，必须立即返回——只做 deepcopy + 入队（+ notify）。
    # 易错：一定要 deepcopy——调用方会继续原地修改 snapshot（例如往 token 列表里 append），
    # 只存引用的话后台线程 build_payload 时看到的是“未来”的状态
    def save_async(self, req_id: str, snapshot: Any, is_final: bool) -> None:
        self._save_q.append((req_id, copy.deepcopy(snapshot), is_final))
        with self._save_cv:
            self._save_cv.notify()

    # cancel_sender：请求被 abort 时调用。之后该 req 的非 final 数据全部丢弃，但 final 标记仍然要发
    # （data=None）——它是下游唯一的 EOS，丢了下游请求就永远停在 WAITING_FOR_CHUNK
    def cancel_sender(self, req_id: str) -> None:
        self._cancelled.add(req_id)

    def _save_loop(self) -> None:
        while not self._stop.is_set():
            try:
                # 单个 save 线程按 FIFO 处理 -> 同一 req 的 chunk 顺序与 save_async 调用顺序一致
                req_id, snap, is_final = self._save_q.popleft()
            except IndexError:
                with self._save_cv:
                    # 持锁后再检查一次，避免“检查为空 -> 生产者 append+notify -> 我们才 wait”的丢通知；
                    # timeout 是丢通知时的兜底，空闲时真正睡下去（不 busy-poll 抢 GIL）
                    if not self._save_q:
                        self._save_cv.wait(timeout=0.05)
                continue
            if req_id in self._cancelled:
                if not is_final:
                    continue        # 已取消：丢弃数据……
                payload = None      # ……但 final 照发，data=None
            else:
                # build_payload 可能很慢（序列化 / 拷贝张量），在后台线程、锁外调用，不阻塞调度线程
                payload = self.build_payload(snap)  # outside any lock
                # build 期间被取消的非 final 块也丢弃（取消前已经过了这一检查的块照常发出，两种都可接受）
                if req_id in self._cancelled and not is_final:
                    continue
            # payload 为 None（本步没有新东西可发）且不是 final -> 跳过，不消耗 chunk id。
            # 例：6 次 save，其中第 3、6 次 payload=None 且第 6 次是 final -> 发出 chunk 0..4 共 5 个连续 id
            if payload is None and not is_final:
                continue
            cid = self._chunk_id.get(req_id, 0)
            self.connector.put(f"{req_id}_{self.stage_id}_{cid}", {"data": payload, "finished": is_final})
            # 易错：chunk_id 只在 put 完成之后才前进。入队时就 +1 会让被跳过的 None 块占掉一个 id，
            # 接收端按连续 id 轮询，会永远卡在那个不存在的 key 上（put 抛异常时这里也不会前进）
            self._chunk_id[req_id] = cid + 1
            if is_final:
                # 流结束：复位该 req 的发送状态（给同名的下一个请求用），也防止 dict 无限增长
                self._chunk_id.pop(req_id, None)
                self._cancelled.discard(req_id)

    # -------------------------------------------------------------- receiver
    # load_async：注册一个要接收的 req。幂等：已注册就不重置 id、不发新 token（否则两份轮询会抢同一个 chunk）。
    # chunk 在 load_async 之前就被 put 了也没关系：connector 会一直保存它，注册后照样能收到
    def load_async(self, req_id: str) -> None:
        with self._recv_cv:
            if req_id not in self._registered:
                self._registered[req_id] = object()   # generation token
                self._next_id[req_id] = 0
            self._recv_cv.notify()   # 叫醒可能正睡着的 recv 线程

    # poll_ready（调度线程）：原子地“整体换出” _ready——拿走自上次 poll 以来所有已提交的 chunk（每个 req 内按 id 有序）。
    # 易错：不要在锁外遍历 _ready，recv 线程可能正在往里 append
    def poll_ready(self) -> dict[str, list]:
        with self._lock:
            out, self._ready = self._ready, {}
        return out

    # cleanup_receiver（调度线程）：提交栅栏。返回之后，这个 req 的任何数据都不会再出现在 poll_ready 里：
    # 因为 recv 线程的“核对 token + 提交”也在同一把锁内，任何提交要么已经发生（在这里被删掉），要么会看到 token 不匹配
    def cleanup_receiver(self, req_id: str) -> None:
        with self._lock:
            self._registered.pop(req_id, None)
            self._next_id.pop(req_id, None)
            self._ready.pop(req_id, None)

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            with self._recv_cv:
                if not self._registered:
                    # 没有任何待接收的 req：睡到 load_async 的 notify（或超时兜底），不空转
                    self._recv_cv.wait(timeout=0.05)
                    continue
                # 锁内快照 (req, 令牌, 期望 chunk id)，之后在锁外做可能很慢的 get
                todo = [(r, tok, self._next_id[r]) for r, tok in self._registered.items()]
            progress = False
            for req_id, token, cid in todo:
                # 严格按 cid 取：chunk cid 没到就不会去拿 cid+1 -> 天然保证按序提交
                obj = self.connector.get(f"{req_id}_{self.src_stage_id}_{cid}")  # outside our locks
                if obj is None:
                    continue
                with self._lock:
                    # 易错：get 期间调度线程可能 cleanup 了它，甚至又 load_async 重新注册（新的 object()）。
                    # 必须用 `is` 比较令牌——只查 `req_id in self._registered` 会把旧请求的数据写进新一代的 _ready
                    if self._registered.get(req_id) is not token:
                        continue  # cleaned up while we were reading
                    self._ready.setdefault(req_id, []).append(obj)
                    progress = True
                    if obj.get("finished"):
                        # 收到流结束标记：注销，不再轮询（_ready 里的数据保留给 poll_ready）
                        del self._registered[req_id]
                        del self._next_id[req_id]
                    else:
                        self._next_id[req_id] = cid + 1
            if not progress:
                # 有 req 但本轮一块都没收到：短暂退避 ~1ms，避免 100% CPU 空转；用 Event.wait 让 stop 能立刻打断
                self._stop.wait(0.001)
