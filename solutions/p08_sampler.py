"""P08 reference solution · 批不变的采样器（per-request seed）。

每行独立：repetition penalty -> (temperature≈0 则 greedy) -> /T -> top_k -> top_p -> 用该请求自己的
generator 抽一个 token。带 seed 的请求拥有私有 torch.Generator（首次使用时创建、跨 step 复用、free 时删除），
所以它抽到的 token 只取决于它自己的 logits / 参数 / 随机流，与同 batch 的其他行及其顺序无关。
提示页：hints/p08_overview.html（A-18）。
"""
from dataclasses import dataclass

import torch


# 每个请求独立的采样参数。seed=None 表示不要求可复现
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1           # <= 0 或 >= V 表示不启用
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None


# repetition penalty（HF/vLLM 约定）：正 logit 除以 penalty、负 logit 乘以 penalty，
# 两种情况都让该 token 概率下降（penalty > 1 时）。
# 例：logits=[3.0, -1.0, 0.5, -0.2]，prev=[1, 3, 3]，penalty=1.5 -> [3.0, -1.5, 0.5, -0.3]
def apply_repetition_penalty(logits: torch.Tensor, prev_tokens: list[int], penalty: float) -> torch.Tensor:
    # 易错：不能原地修改调用方传入的 logits（调用方可能还要用它算 logprobs）
    out = logits.clone()
    if penalty == 1.0 or not prev_tokens:
        return out
    # 去重：同一个 token 在历史中出现多次也只惩罚一次（不随出现次数叠加）
    idx = torch.tensor(sorted(set(prev_tokens)), dtype=torch.long)
    vals = out[..., idx]
    # 易错：负 logit 也做除法会让它变大（-2/2 = -1），反而成了奖励
    out[..., idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return out


class Sampler:
    def __init__(self) -> None:
        # req_id -> 该请求私有的 torch.Generator（仅带 seed 的请求才有）
        self._gens: dict[str, torch.Generator] = {}

    # 请求结束时丢弃 generator；之后同 id 的新请求会从 seed 重新开始随机流（不 free 会接着旧随机流）
    def free(self, req_id: str) -> None:
        self._gens.pop(req_id, None)

    # 带 seed 的请求：第一次使用时用 seed 创建，之后在多步 decode 间持续推进同一个随机流。
    # 不带 seed 的请求返回 None -> multinomial 用全局 RNG（本来就不要求可复现）
    def _generator(self, req_id: str, p: SamplingParams) -> torch.Generator | None:
        if p.seed is None:
            return None
        g = self._gens.get(req_id)
        # 易错：不能每步都用 seed 重新创建 generator，否则每步抽到的随机数都相同，分布明显偏
        if g is None:
            g = torch.Generator().manual_seed(p.seed)
            self._gens[req_id] = g
        return g

    def _sample_row(self, row: torch.Tensor, req_id: str, p: SamplingParams, prev: list[int]) -> int:
        # 转 fp32 计算：bf16/fp16 下 softmax / cumsum 与参考实现会差到 1e-2
        row = apply_repetition_penalty(row.float(), prev, p.repetition_penalty)
        # temperature≈0 视为 greedy：直接 argmax，不消耗随机数。
        # 易错：判断必须在除以 temperature 之前（否则 /0 得 inf/NaN）；penalty 在它之前，所以对 greedy 行仍生效
        if p.temperature < 1e-5:
            return int(row.argmax())
        row = row / p.temperature   # T<1 放大分差，T>1 压平
        if p.top_k > 0 and p.top_k < row.shape[-1]:
            # 第 k 大的值作为阈值；严格小于它的置 -inf（并列第 k 名会多保留几个，可以接受）
            kth = torch.topk(row, p.top_k).values[-1]
            row = row.masked_fill(row < kth, float("-inf"))
        if p.top_p < 1.0:
            # 在 top_k 过滤之后的分布上做 nucleus：按概率降序排序后计算累积概率
            probs, order = torch.softmax(row, -1).sort(descending=True)
            # cumsum - probs 是“该 token 之前”的累积质量；只有之前的质量已超过 top_p 才删，
            # 这样跨过阈值的那个 token 被保留，保留的是“质量达到 top_p 的最短前缀”。
            # 例：probs [0.5, 0.25, 0.15, 0.1]，top_p=0.7 -> 保留 {0, 1}
            remove = probs.cumsum(-1) - probs > p.top_p
            # 易错：至少保留概率最高的 1 个，避免极小 top_p 时全部被置 -inf、softmax 出 NaN
            remove[0] = False
            row = row.clone()
            # order 把排序后的位置映射回原始 vocab 下标
            row[order[remove]] = float("-inf")
        # 被过滤的位置 softmax 后概率恰好为 0，其余自动重新归一化
        probs = torch.softmax(row, -1)
        # 每行单独抽 1 个样本，用这个请求自己的 generator——这是批不变性的关键：
        # 它的随机流只被它自己消费，消费次数只取决于它被调度过几次
        return int(torch.multinomial(probs, 1, generator=self._generator(req_id, p)))

    # logits: [B, V] -> [B] long，第 i 个元素对应第 i 行 / req_ids[i]。
    # 易错：对整个 [B, V] 调一次 multinomial（哪怕 generator 按 seed 建）——各行会按顺序消费同一个 generator，
    # 请求拿到的随机数取决于前面有几行，批不变性就没了
    def sample(
        self,
        logits: torch.Tensor,
        req_ids: list[str],
        params: list[SamplingParams],
        prev_tokens: list[list[int]],
    ) -> torch.Tensor:
        out = [self._sample_row(logits[i], req_ids[i], params[i], prev_tokens[i]) for i in range(len(req_ids))]
        return torch.tensor(out, dtype=torch.long)
