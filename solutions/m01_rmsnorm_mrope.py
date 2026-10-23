"""M01 reference solution · RMSNorm + 多模态 RoPE（Qwen3 interleaved / Qwen2.5 chunked）。

提示页：hints/m01_overview.html（A-1）。注释里的“易错：”标出最容易写错的地方。
"""
import torch


# ---------------------------------------------------------------------------
# rms_norm：Qwen3 系列所有 RMSNorm（input_layernorm / post_attention_layernorm / q_norm / k_norm /
#   final norm）的公共实现。y = weight * x / sqrt(mean(x², -1) + eps)。
#   顺序与 HF Qwen3OmniMoe*RMSNorm 完全一致：先 upcast 到 fp32 归一化，再 cast 回输入 dtype，最后乘 weight。
# ---------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    # fp32 upcast：bf16/fp16 下 x² 的求和容易丢精度甚至溢出，统计量必须在 fp32 里算
    h = x.float()
    # mean 只对最后一维（hidden / head_dim）求，每个 token（每个 head）独立归一化；keepdim 便于广播
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    # 易错：必须先 .to(dtype) 再乘 weight（cast-back-before-weight）。
    # 若在 fp32 里先乘 weight 再 cast，bf16 下会与 HF 差 1 ulp，逐元素对拍失败。
    return weight * h.to(dtype)


# ---------------------------------------------------------------------------
# rope_inv_freq：标准 RoPE 的逆频率表 inv_freq[j] = theta^(-2j / head_dim)，j = 0..head_dim/2-1。
#   每个频率负责一对维度（rotate_half 布局下是第 j 维和第 j + D/2 维）。
#   例：rope_inv_freq(8, 10000.0) -> [1.0, 0.1, 0.01, 0.001]
# ---------------------------------------------------------------------------
def rope_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    # arange(0, D, 2) = [0, 2, 4, ...]，长度 D/2；先建 int64 再转 float32（与 HF 写法一致），
    # 除以 head_dim 后指数在 [0, 1)，inv_freq 从 1 单调递减到接近 1/theta
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))


# ---------------------------------------------------------------------------
# mrope_cos_sin：Multimodal RoPE。每个 token 有 3 个位置分量 (t, h, w)，
#   把 D/2 个频率按 mrope_section 分给三个轴：频率 j 的角度 = pos[该频率所属的轴] * inv_freq[j]。
#   interleaved=True （Qwen3-Omni / Qwen3-VL）：T H W T H W ... 循环交错，H/W 份额用完后剩下全给 T
#   interleaved=False（Qwen2-VL / Qwen2.5-Omni）：连续分块 [T...T | H...H | W...W]
#   纯文本时 t = h = w，两种布局都退化为普通 1D RoPE（三条 freqs 相同，怎么挑列都一样）。
# ---------------------------------------------------------------------------
def mrope_cos_sin(
    inv_freq: torch.Tensor,        # [head_dim // 2]
    positions: torch.Tensor,       # [3, T]   (temporal, height, width)
    mrope_section: list[int],      # sums to head_dim // 2，是“每个轴分到几个频率”（不是维度数）
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:   # cos, sin: [T, head_dim]
    # 一次广播把 3 个轴在全部 D/2 个频率上的角度都算出来；后面只是“挑列”，不做新运算
    freqs = positions.float()[:, :, None] * inv_freq.float()[None, None, :]  # [3, T, half]
    if interleaved:
        # 以 T 轴为底。易错：必须 clone，否则下面的切片赋值会改到 freqs[0] 本身
        f = freqs[0].clone()  # [T, half]
        # axis=1(H) 占下标 1, 4, 7, ...；axis=2(W) 占下标 2, 5, 8, ...（offset 恰好等于轴号）
        for axis, offset in ((1, 1), (2, 2)):
            # 易错：stop 是 mrope_section[axis] * 3 而不是 half——每个轴只拿 section[axis] 个频率，
            # 交错区之后的尾巴保持 T 轴。例 section=[6,2,2]：slice(1, 6, 3) -> 1, 4；j ≥ 6 全是 T
            idx = slice(offset, mrope_section[axis] * 3, 3)
            f[..., idx] = freqs[axis][..., idx]
    else:
        # chunked：第 axis 段取该轴在“同一个列区间” [start, start + n) 上的角度。
        # 易错：H 段取的是 freqs[1][:, s_t:s_t+s_h]，不是 freqs[1][:, 0:s_h]
        parts, start = [], 0
        for axis, n in enumerate(mrope_section):
            parts.append(freqs[axis][..., start:start + n])
            start += n
        f = torch.cat(parts, -1)  # [T, half]
    # 前后两半用同一组角度拼成 [T, D]：对应 rotate_half 的“前半/后半配对”（neox 风格）。
    # 易错：不是 repeat_interleave（相邻两维配对的 GPT-J 风格），两者与 rotate_half 不兼容
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


# rotate_half：把 [a, b]（各 D/2）变成 [-b, a]，即二维旋转中与 sin 相乘的那一项。
# 例：rotate_half([1, 2, 3, 4]) = [-3, -4, 1, 2]。易错：写成 [b, -a] 旋转方向就反了
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)  # 沿 head_dim 对半切（不是按奇偶位切）
    return torch.cat([-b, a], -1)


# apply_rotary：x' = x * cos + rotate_half(x) * sin，即对每对 (x_j, x_{j+D/2}) 做角度 θ_j 的二维旋转。
def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [T, heads, D]；cos/sin: [T, D]。先 cast 到 x.dtype（HF 的 rotary embedding 返回的就是 x.dtype），
    # 再插入 head 维变成 [T, 1, D] 以便在 head 维广播。
    # 易错：直接与 [T, heads, D] 相乘时 PyTorch 右对齐，会把 cos 的 T 维对到 heads 维——
    # T == heads 时不报错但结果全错
    cos = cos.to(x.dtype)[:, None, :]
    sin = sin.to(x.dtype)[:, None, :]
    return x * cos + rotate_half(x) * sin
