"""P14 reference solution · Codec 窗口规划：首块小（低 TTFP）、后块大（高吞吐）、左上下文重算（无接缝）。

窗口 = [ctx_start, new_start) 左上下文（只用来让 vocoder 平滑衔接，输出会被丢弃）
     + [new_start, new_end)  本次真正新增的帧。三个坐标都是整条流的全局帧下标。
左上下文 ≥ vocoder 的（因果）感受野时，流式解码与整段解码逐样本相同。提示页：hints/p14_overview.html（A-23）。
"""
from collections.abc import Sequence


# 例（initial=2, chunk=5, left_ctx=3）：
#   (available=2,  emitted=0,  F) -> (0, 0, 2)    首块 2 帧
#   (available=6,  emitted=2,  F) -> None         只有 4 帧新帧，不够一块
#   (available=12, emitted=7,  F) -> (4, 7, 12)   左上下文 4,5,6
#   (available=20, emitted=12, T) -> (9, 12, 17)  已结束但完整的一块还够 -> 仍按 5 帧切
#   (available=20, emitted=17, T) -> (14, 17, 20) 尾巴 3 帧 flush
#   (available=20, emitted=20, T) -> None         全部发完
def next_window(
    available: int,     # talker 已产出的帧数
    emitted: int,       # 已发出的帧数（= 上一个窗口的 new_end）
    finished: bool,
    *,
    initial: int,
    chunk: int,
    left_ctx: int,
) -> tuple[int, int, int] | None:
    # 函数无状态，“第一块”只能从参数判断：还没发过任何帧（emitted == 0）且配置了 initial。
    # 易错：别用 available 的大小去猜首块——帧可能一次来很多（available=30, emitted=0 时首块仍是 initial）
    size = initial if emitted == 0 and initial > 0 else chunk
    # 判断顺序很重要：先看“够不够一个完整窗口”，再看“是否 finished 且有剩余”。
    # 把 finished 放前面会把多块合并成一个大窗口（不丢帧，但违反“完整窗口可用时返回完整窗口”）
    if available - emitted >= size:
        new_end = emitted + size            # 只取一块，多出来的帧留给下一次调用
    elif finished and available > emitted:
        # 已结束且还有不足一块的尾巴 -> 全部发出。例：极短的流 (1, 0, T) -> (0, 0, 1)
        new_end = available
    else:
        # 帧不够且没结束（继续等），或已结束且全部发完。
        # 易错：结束时返回空窗口 (…, 20, 20) 会让 emitted 不再前进，调用方的 while 循环永远停不下来
        return None
    # new_start 永远等于 emitted（窗口首尾相接、无重叠无空洞）；ctx_start 在流开头被 0 截断（帧下标没有负数）
    return max(0, emitted - left_ctx), emitted, new_end


# decoded = vocoder(codes[ctx_start:new_end])，每帧对应 upsample 个样本，长度 (new_end - ctx_start) * upsample。
# 丢掉前面属于上下文帧的样本，只保留新帧的样本，所有窗口的输出首尾相接就是连续音频。
# 例：window=(4, 7, 12), upsample=2 -> decoded 长度 16；上下文帧 4..6 占样本 [0, 6)，返回 decoded[6:16]
def slice_new_samples(decoded: Sequence[float], window: tuple[int, int, int], upsample: int) -> list[float]:
    ctx_start, new_start, new_end = window
    # 易错：decoded 的下标是相对 ctx_start 的局部坐标（不是全局帧号），所以都要减 ctx_start。
    # 直接用 new_start * upsample 切只在 ctx_start=0 时碰巧正确（上例会得到 decoded[14:24]，只剩 2 个样本）
    return list(decoded[(new_start - ctx_start) * upsample:(new_end - ctx_start) * upsample])
