"""M05 reference solution · Talker MTP：一帧内预测 G-1 个残差 codebook。

每个音频帧有 G 个 codebook（CB0 + G-1 个残差）。talker 主干只采样 CB0；code predictor（小 transformer）
以 [talker 最后隐状态, E(CB0)] 开头自回归地预测 CB1..CB{G-1}：第 j 步用 lm_head[j] 预测 CB{j+1}，
再用 codec_embedding[j] 把它嵌入、拼到序列末尾。re-prefill 与 KV 版结果必须逐元素相同。
提示页：hints/m05_overview.html（A-5）。
"""
import torch
from transformers import DynamicCache


# re-prefill（vllm-omni 默认）：每步把整个序列重新送进模型，不用 KV cache。
# 序列最长 G+1，O(T²) 重算几乎免费；没有 cache 状态 -> 形状固定，可以整体包成一张 CUDA graph。
@torch.no_grad()
def predict_residual_codes(cp, last_talker_hidden: torch.Tensor, cb0_embed: torch.Tensor) -> torch.Tensor:
    G = cp.config.num_code_groups            # 每帧 codebook 总数（含 CB0）
    # 初始序列长度 2：位置 0 = talker 隐状态，位置 1 = CB0 的 embedding（各 [B, H]）
    seq = [last_talker_hidden, cb0_embed]
    codes = []
    # 易错：循环 G-1 次（range(G) 会越界访问 lm_head[G-1]，range(G-2) 会少一个 code）
    for j in range(G - 1):                   # 第 j 步预测 codebook j+1
        # stack(dim=1) 得到 [B, L, H]（cat 会得到 [2B, H] 或 [B, 2H]）；只取最后一个位置的隐状态
        h = cp.model(inputs_embeds=torch.stack(seq, 1)).last_hidden_state[:, -1]   # [B, H]
        # 易错：第 j 步用 lm_head[j]，每个残差 codebook 有自己的 head；全都用 lm_head[0] 时只有第一列是对的
        code = cp.lm_head[j](h).argmax(-1)   # [B] greedy（真实服务用 top-k/top-p 采样）
        codes.append(code)
        # 嵌入刚预测的 code 也用同一个下标 j（两张表都只覆盖残差 codebook，下标 0 指 CB1）
        seq.append(cp.model.codec_embedding[j](code))
    return torch.stack(codes, 1)             # 每步一个 [B]，按列拼成 [B, G-1]（第 0 列是 CB1）


# KV 版：第一步 prefill 2 个位置，之后每步只送 1 个新位置；cache 长度依次 2 -> 3 -> ...，
# 新位置的 position 由 DynamicCache 记录的长度自动推出，所以与 re-prefill 数学上等价。
@torch.no_grad()
def predict_residual_codes_kv(cp, last_talker_hidden: torch.Tensor, cb0_embed: torch.Tensor) -> torch.Tensor:
    G = cp.config.num_code_groups
    # 每次调用新建 cache：每个音频帧的残差预测是一段独立的短序列，不能复用上一帧的 cache
    cache = DynamicCache()
    step = torch.stack([last_talker_hidden, cb0_embed], 1)   # 首步 [B, 2, H]
    codes = []
    for j in range(G - 1):
        # HF 在 forward 内原地把本次输入的 K/V 追加进 cache
        h = cp.model(inputs_embeds=step, past_key_values=cache, use_cache=True).last_hidden_state[:, -1]
        code = cp.lm_head[j](h).argmax(-1)
        codes.append(code)
        # 易错：下一步只喂“新 code”的 embedding [B, 1, H]；若再喂整段，cache 里会重复存旧位置、位置被推远，
        # 结果悄悄偏离（有的行对、有的行错）
        step = cp.model.codec_embedding[j](code)[:, None]
    return torch.stack(codes, 1)


# next_talker_input：talker 下一步的输入 embedding =
#   本帧全部 G 个 codebook embedding 之和（CB0 用 talker 自己的表、由调用方传入；CB1.. 用 code predictor 的表）
#   + 当前步对齐的 thinker 文本向量 text_step。结果 [B, H]，没有序列维。
def next_talker_input(cp, cb0_embed: torch.Tensor, residual_codes: torch.Tensor, text_step: torch.Tensor) -> torch.Tensor:
    # 用 a = a + b 构造新张量，不原地修改调用方的 cb0_embed
    out = cb0_embed + text_step
    # residual_codes: [B, G-1]；第 j 列（CB{j+1}）用 codec_embedding[j] 查表（与预测时同一个 j）。
    # 易错：不要把整个 [B, G-1] 丢进同一张表（会得到 [B, G-1, H] 且所有列用了同一张表）
    for j in range(residual_codes.shape[1]):
        out = out + cp.model.codec_embedding[j](residual_codes[:, j])
    return out
