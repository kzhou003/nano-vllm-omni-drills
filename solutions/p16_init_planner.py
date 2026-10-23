"""P16 reference solution · 启动规划：设备分组、并行初始化、失败回滚、显存准入。

同一张卡上的 stage 必须串行初始化（“测空闲显存 -> 定 KV block 数”会互相干扰），不同组之间并行；
任何一个失败：不再启动新的 init，等正在跑的结束，回滚所有已成功的，再抛 InitError。
提示页：hints/p16_overview.html（A-25）。
"""
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


# InitError：携带“第一个失败的 stage_id”；原始异常通过 raise ... from cause 挂在 __cause__ 上
class InitError(Exception):
    def __init__(self, stage_id: int, cause: BaseException) -> None:
        super().__init__(f"stage {stage_id} failed to initialize: {cause!r}")
        self.stage_id = stage_id


# init_groups：把“共享 GPU 的 stage”归为一组。共享关系是传递的（A-B 共享 dev0，B-C 共享 dev1 -> A、B、C 同组），
# 用并查集求连通分量。例：{0: {2}, 1: {0}, 2: {0, 1}, 3: {1}, 4: {5}} -> [[0], [1, 2, 3], [4]]
# 输出确定：组内 stage 升序，组之间按首元素升序（与 dict 插入顺序、set 迭代顺序无关）。
def init_groups(stage_devices: dict[int, set[int]]) -> list[list[int]]:
    parent = {s: s for s in stage_devices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # 路径减半：把 x 挂到祖父节点上，树很浅
            x = parent[x]
        return x

    # owner: device -> 第一个占用它的 stage；后来者遇到已占用的 device 就和 owner 合并
    owner: dict[int, int] = {}
    # 按 stage 升序遍历，结果与输入 dict 的 key 顺序无关
    for s in sorted(stage_devices):
        for d in stage_devices[s]:
            if d in owner:
                a, b = find(owner[d]), find(s)
                if a != b:
                    parent[max(a, b)] = min(a, b)   # 较小的根当父亲 -> 每个分量的根 = 组内最小 stage id
            else:
                owner[d] = s
    groups: dict[int, list[int]] = {}
    for s in sorted(stage_devices):   # 升序 append -> 组内天然有序
        groups.setdefault(find(s), []).append(s)
    return sorted(groups.values(), key=lambda g: g[0])


# run_init：组间并行（线程池，每组一个任务）、组内串行（按 stage id 的 for 循环）。
# 用线程而不是 asyncio：init_fn 是阻塞的同步调用（加载权重、等子进程就绪）。
def run_init(
    stage_devices: dict[int, set[int]],
    init_fn: Callable[[int], Any],
    cleanup_fn: Callable[[int, Any], None],
    max_workers: int = 4,
) -> dict[int, Any]:
    # lock 保护 results 与 failure（多个 worker 线程并发读写）
    lock = threading.Lock()
    results: dict[int, Any] = {}                     # 成功的 stage -> init_fn 返回值（cleanup 时要用）
    failure: list[tuple[int, BaseException]] = []    # 最多一个元素：第一次失败（“已出事”的旗子）

    def run_group(group: list[int]) -> None:
        for sid in group:
            # 短路：每次开始新的 init 之前看一眼旗子；已有失败就不再启动新工作（已在跑的 init 无法打断，会跑完）。
            # max_workers 小于组数时，排队的组轮到时发现已失败会直接返回
            with lock:
                if failure:
                    return
            # 易错：init_fn 很慢，必须在锁外调用，否则组间并行就退化成全局串行
            try:
                r = init_fn(sid)
            except BaseException as e:  # noqa: BLE001
                # 捕获 BaseException：KeyboardInterrupt / SystemExit 也要触发回滚，不能让已起来的 stage 泄漏。
                # 异常不留在 Future 里——那样主线程会丢失“哪个 stage”以及“先回滚再抛”的控制
                with lock:
                    if not failure:   # 多个组几乎同时失败时只记第一条
                        failure.append((sid, e))
                return                # 本组后续 stage 不再初始化
            with lock:
                results[sid] = r

    # with 块退出 = 所有组都结束（join），之后 results 不再变化
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(run_group, init_groups(stage_devices)))   # list(...) 消费迭代器；run_group 自己吞掉了异常
    # 易错：回滚必须在线程池退出之后做——在失败那一刻就 cleanup，会漏掉另一组里稍后才成功的 stage。
    # 失败的 stage 本身没有 result，不 cleanup
    if failure:
        for sid in sorted(results):
            cleanup_fn(sid, results[sid])   # roll back everything that did come up
        sid, cause = failure[0]
        raise InitError(sid, cause) from cause
    return results


# admission_check：启动前的显存准入（只读）。对每张卡：
#   need = Σ_{该卡上的 stage} (gpu_memory_utilization × 该卡 total + reserve_bytes_per_stage)
# need > total 就报一条含 "device <id>" 的错误；恰好相等允许；没有 stage 的卡 need=0，不输出。
# 跨卡 stage（如 TP=2）在它占用的每张卡上都各按该卡的 total 计一份。
# 例（80 GiB 卡，reserve=2 GiB）：dev1 上两个 stage util=0.6/0.3 -> (48+2)+(24+2)=76 ≤ 80 -> 通过
def admission_check(
    device_total_bytes: dict[int, int],
    stages: dict[int, tuple[set[int], float]],   # stage -> (devices, gpu_memory_utilization)
    reserve_bytes_per_stage: int,
) -> list[str]:
    errors = []
    for dev, total in sorted(device_total_bytes.items()):   # 按 device 升序，输出顺序确定
        on_dev = [(sid, util) for sid, (devs, util) in stages.items() if dev in devs]
        # utilization 是相对“这张卡的总量”的比例
        need = sum(util * total + reserve_bytes_per_stage for _, util in on_dev)
        if need > total:
            errors.append(f"device {dev}: stages {[s for s, _ in on_dev]} need {need / 2**30:.1f} GiB "
                          f"> {total / 2**30:.1f} GiB")
    return errors
