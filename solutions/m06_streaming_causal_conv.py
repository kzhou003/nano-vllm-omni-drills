"""M06 reference solution · Code2Wav 因果 Conv1d / 因果 ConvTranspose1d 的流式（带状态）版本。

任意分块喂入、拼起来与离线整段处理逐样本一致，且每个样本一旦“确定”就立刻输出。
提示页：hints/m06_overview.html（A-6）。
"""
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# StreamingCausalConv1d：因果 Conv1d（stride 1，任意 dilation）的流式版本。
#   离线版 = 整段左补 (K-1)*d 个 0 再做 valid 卷积。流式版把“左补零”换成一个 state：
#   永远保存最近 ctx = (K-1)*d 个历史输入；每来一块 x 就拼成 [state | x] 做 valid 卷积，
#   输出长度恰好等于本块长度 t。
# ---------------------------------------------------------------------------
class StreamingCausalConv1d:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, dilation: int = 1) -> None:
        # weight: [C_out, C_in, K]（nn.Conv1d 布局）
        self.weight, self.bias, self.dilation = weight, bias, dilation
        # 感受野跨度：输出位置 t 需要输入 t-(K-1)*d .. t
        self.ctx = (weight.shape[-1] - 1) * dilation
        # [B, C_in, ctx]；B / dtype / device 第一次 step 才知道，所以懒创建
        self.state: torch.Tensor | None = None

    @classmethod
    def from_hf(cls, m) -> "StreamingCausalConv1d":   # HF Qwen3OmniMoeCausalConvNet
        assert m.stride == 1   # 这里的流式状态推导只对 stride 1 成立
        bias = m.conv.bias.detach() if m.conv.bias is not None else None
        return cls(m.conv.weight.detach(), bias, m.dilation)

    def step(self, x: torch.Tensor) -> torch.Tensor:   # [B, C_in, t] -> [B, C_out, t]
        if self.state is None:
            # 首块之前的“历史”就是离线版的左补零
            self.state = x.new_zeros(x.shape[0], x.shape[1], self.ctx)
        full = torch.cat([self.state, x], -1)          # “便签 + 新页”，长度 ctx + t
        # 新 state = full 的最后 ctx 个样本。
        # 易错 1：要从 full（旧 state + 新块）末尾取，而不是从 x 末尾取——t < ctx 时 x 不够长，
        #         state 里仍需保留一部分旧样本（可能还有补出来的 0）。
        # 易错 2：K=1（ctx=0）时 full[..., -0:] 是整个 full，state 会无限增长；
        #         这里用 shape[-1] - ctx 作起点，ctx=0 时得到长度 0 的切片，正好避开这个坑
        self.state = full[..., full.shape[-1] - self.ctx:]
        # 无 padding 的 valid 卷积：输出长度 (ctx + t) - ctx = t
        return F.conv1d(full, self.weight, self.bias, dilation=self.dilation)


# ---------------------------------------------------------------------------
# StreamingCausalTransConv1d：上采样用的因果 ConvTranspose1d 的流式版本（overlap-add）。
#   输入帧 i 写入输出区间 [i*s, i*s + K)，相邻帧重叠 K-s 个样本；离线模块会裁掉开头和结尾各 K-s 个。
#   维护累加缓冲 acc（从全局位置 base 开始、尚未输出的样本，未加 bias）：
#   - step：本块的转置卷积结果加到 acc 对应位置；收到 n 帧后位置 < n*s 的样本不会再被写，可以输出；
#   - 全局左裁剪 [0, trim) 和 bias 都在 _emit 里处理；右裁剪在 flush 里处理。
#   “尽早输出”：喂入 n 帧后累计输出恰为 max(0, n*s - (K-s)) 个。
# ---------------------------------------------------------------------------
class StreamingCausalTransConv1d:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, stride: int) -> None:
        # weight: [C_in, C_out, K]（nn.ConvTranspose1d 布局，与 Conv1d 相反）
        self.weight, self.bias, self.stride = weight, bias, stride
        self.K = weight.shape[-1]
        self.trim = self.K - stride   # 离线版两端各裁掉的样本数
        self.frames = 0               # 已收到的输入帧总数
        self.base = 0          # global position of acc[..., 0]（= 已经处理掉的样本数，含被左裁剪的）
        self.acc: torch.Tensor | None = None

    @classmethod
    def from_hf(cls, m) -> "StreamingCausalTransConv1d":   # HF Qwen3OmniMoeCausalTransConvNet
        bias = m.conv.bias.detach() if m.conv.bias is not None else None
        # 易错：nn.ConvTranspose1d.stride 是 tuple，要取 [0]
        return cls(m.conv.weight.detach(), bias, m.conv.stride[0])

    # _emit：输出全局区间 [max(base, trim), end) 的已定稿样本，并把 acc / base 推进到 end。
    def _emit(self, end: int) -> torch.Tensor:
        # 全局左裁剪：位置 < trim 的样本属于离线版裁掉的开头。用“全局位置”判断，而不是每块都裁——
        # 按块裁会让后面每块都少 K-s 个样本
        start = max(self.base, self.trim)
        if self.acc is None or end <= start:
            # 没有可输出的（例如第一帧定稿的样本全在左裁剪区里）：返回长度 0 的张量，
            # 注意此时 base 不前进——这些样本留在 acc 里，下次 _emit 时 start 仍会跳过它们
            if self.acc is None:
                return torch.zeros(0)
            return self.acc[..., :0]
        out = self.acc[..., start - self.base:end - self.base]
        # 易错：bias 只能在定稿时每个位置加一次。若在 conv_transpose1d 里带 bias，
        # 重叠区会被多个帧各加一次（例：所有值多 0.5）
        if self.bias is not None:
            out = out + self.bias[None, :, None]
        # acc 只保留尚未定稿的尾巴（长度始终 ≤ K-s：还在等后续帧叠加的部分）
        self.acc = self.acc[..., end - self.base:]
        self.base = end
        return out

    def step(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, t] 新到的 t 帧。本块单独做转置卷积（不带 bias），长度 (t-1)*s + K，
        # 块内的重叠相加已经由这一次卷积完成；块与块之间的重叠由 acc 负责
        y = F.conv_transpose1d(x, self.weight, None, stride=self.stride)
        off = self.frames * self.stride    # 本块第一帧写入的全局起点
        end = off + y.shape[-1]
        if self.acc is None:
            self.acc = y.new_zeros(y.shape[0], y.shape[1], 0)
        # 右侧补 0，把 acc 延长到能放下本块全部写入（下标都要减 base 换成 acc 的局部坐标）
        need = end - self.base
        if self.acc.shape[-1] < need:
            pad = need - self.acc.shape[-1]
            self.acc = torch.cat([self.acc, self.acc.new_zeros(*self.acc.shape[:2], pad)], -1)
        # overlap-add：与上一块尾巴重叠的部分相加
        self.acc[..., off - self.base:end - self.base] += y
        self.frames += x.shape[-1]
        # 收到 frames 帧后，位置 < frames*s 的样本不会再被未来帧（起点 ≥ frames*s）写到，可以输出。
        # 易错：输出 ≥ frames*s 的位置——那些值之后还会被加，结果与离线不一致
        return self._emit(self.frames * self.stride)

    def flush(self) -> torch.Tensor:
        if self.frames == 0:
            return torch.zeros(0)
        # 离线未裁剪时总长 (L-1)*s + K；去掉结尾 trim 个（右裁剪只能在流结束时做——中途不知道哪里是结尾）。
        # 正常情况下剩余 pending 恰好是 trim 个，被全部裁掉，flush 返回长度 0；trim == 0（K == s）时也正确
        total = (self.frames - 1) * self.stride + self.K
        return self._emit(total - self.trim)
