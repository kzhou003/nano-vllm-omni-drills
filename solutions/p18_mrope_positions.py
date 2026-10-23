"""P18 reference solution · Qwen2-VL 风格的 MRoPE 三维位置 (temporal, height, width)。

M01 是“给定位置怎么旋转”，本题是“位置本身怎么算”：
  文本：三个轴取同一个递增下标（退化为普通 1D RoPE）
  图像：LLM 网格 (t, h//merge, w//merge)，三个轴分别是网格坐标 + 段起点 start
  每段之后 start = 当前最大位置 + 1（不是“已有 token 数”）
图像段的位置跨度 max(t, gh, gw) 通常小于 token 数，所以最终 max+1 < T，差值 delta 用于 decode：
  decode 位置 = 绝对序列下标 index + delta（三个轴相同）。提示页：hints/p18_overview.html（A-24）。
"""
import torch

# ("text", token 数) 或 ("image", (t, h, w))，图像的 t/h/w 以 patch 为单位（merge 之前）
Segment = tuple[str, int | tuple[int, int, int]]


# 例：[("text",2), ("image",(1,4,4)), ("text",3)]
#   t: 0 1 | 2 2 2 2 | 4 5 6
#   h: 0 1 | 2 2 3 3 | 4 5 6
#   w: 0 1 | 2 3 2 3 | 4 5 6      T=9, max=6, delta = 7 - 9 = -2
def mrope_positions(segments: list[Segment], spatial_merge: int = 2) -> tuple[torch.Tensor, int]:
    parts = []      # 每段一个 [3, n_seg] 张量，最后沿 dim=1 拼接
    start = 0       # 下一段的起始位置
    for kind, spec in segments:
        if kind == "text":
            # [n] -> expand 成 [3, n]（三行相同）；expand 是 view，加 start 时会生成新张量
            p = torch.arange(spec).expand(3, spec) + start
        elif kind == "image":
            t, h, w = spec
            # 视觉 encoder 把 merge×merge 个 patch 合成一个 LLM token；t 方向不合并。
            # 例 (2, 4, 6), merge=2 -> 网格 (2, 2, 3) = 12 个 token
            gh, gw = h // spatial_merge, w // spatial_merge
            # token 按 t 最外、h 其次、w 最内（row-major）展开，三个坐标各自 flatten 成 [t*gh*gw]：
            #   (t, gh, gw) = (2, 2, 3)：ti = 0 0 0 0 0 0 1 1 1 1 1 1
            #                            hi = 0 0 0 1 1 1 0 0 0 1 1 1
            #                            wi = 0 1 2 0 1 2 0 1 2 0 1 2
            # 易错：h、w 互换 / 展开顺序错在方形网格上看不出来，一定要用矩形网格和 t > 1 自测
            ti = torch.arange(t).view(-1, 1).expand(t, gh * gw).flatten()
            hi = torch.arange(gh).view(1, -1, 1).expand(t, gh, gw).flatten()
            wi = torch.arange(gw).view(1, 1, -1).expand(t, gh, gw).flatten()
            p = torch.stack([ti, hi, wi]) + start   # [3, t*gh*gw]，三个轴统一加段起点
        else:
            raise ValueError(f"unknown segment kind {kind!r}")
        parts.append(p)
        # 易错：下一段 start = 本段最大位置 + 1（三个轴一起取 max——最大值可能来自 t（视频）、h（高图）或 w（宽图）），
        # 而不是 start + token 数（纯文本时两者相同，有图像就错）。例：上面 12 个 token 的图像段 max=2 -> 下一段 start=3。
        # 空段（n=0 或网格为 0）没有 max，不推进 start
        if p.numel():
            start = int(p.max()) + 1
    # 位置是整数下标，用 long；没有任何段时返回 [3, 0] 的 long 张量，保证形状/类型一致
    pos = torch.cat(parts, 1).long() if parts else torch.zeros(3, 0, dtype=torch.long)
    # delta = (prefill 之后下一个位置) - T。此时 start == max + 1（空输入时为 0）。
    # 有图像时通常为负；为 0 不代表没有图像，只代表位置跨度恰好等于 token 数；不可能为正。
    # 易错：少了 +1，或 decode 时直接用 index 当位置
    delta = start - pos.shape[1]
    return pos, delta


# decode 阶段每个新 token 都是文本，三个轴位置相同 = index + delta。
# 例：T=14, delta=-9 -> 第一个 decode token（index=14）位置 [5, 5, 5]，紧接 prefill 的 max=4
def decode_position(index: int, delta: int) -> torch.Tensor:
    return torch.full((3,), index + delta, dtype=torch.long)
