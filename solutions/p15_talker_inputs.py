"""P15 reference solution · Thinker -> Talker 输入构造（Qwen2.5-Omni 布局，纯张量函数）。

“差一行”：vLLM 先把采样出的 stop token 追加到 output 再判停，所以 finish_reason == "stop" 的请求
多累积了一行 hidden（产生 stop token 那一步），talker 吃进去音频末尾就会多出杂音。
提示页：hints/p15_overview.html（A-22）。
"""
from dataclasses import dataclass

import torch


# ThinkerPayload：thinker stage 交给 talker stage 的数据（已裁掉 stop token 那一行）。
# prefill 行对应 prompt token，decode 行与 output_ids 一一对应。
@dataclass
class ThinkerPayload:
    prefill: torch.Tensor     # [P, D]
    decode: torch.Tensor      # [K, D]   生成 token 的行（不含 stop 行）
    output_ids: list[int]     # 生成的 id（不含 stop id），新 list


# 例：P=4, output_ids=[21, 22, <stop>], hidden 7 行, finish="stop"
#     -> drop=1, target=4+3-1=6；prefill=hidden[0:4]，decode=hidden[4:6]，output_ids=[21, 22]
def trim_thinker_payload(
    prompt_ids: list[int],
    output_ids: list[int],
    hidden: torch.Tensor,       # [R, D] prefill + 每个 decode 步累积的行（可能还多捕获了行）
    finish_reason: str,         # "stop" | "length"
) -> ThinkerPayload:
    P = len(prompt_ids)   # prefill / decode 的分界由 prompt 长度决定，hidden 的行数只用于检查和截断
    # 先检查再切片：切片越界不报错，只会悄悄返回短 tensor。连 prompt 行都没捕获全时 talker 的 prefill 无法构造
    if hidden.shape[0] < P:
        raise ValueError(f"captured {hidden.shape[0]} hidden rows for a {P}-token prompt")
    # "length"（到达 max_tokens）结束时最后一个 token 是正常内容，一个都不能丢
    drop = 1 if finish_reason == "stop" else 0
    # 易错：按“应有行数” target 截断，而不是盲目 hidden[:-1]——多捕获了行时 [:-1] 砍掉的是多余行，stop 行还在。
    # （stop 但 output_ids 为空时 target = P-1：decode 切成空，prefill 仍按 P 切，不会误删 prompt 行）
    target = P + len(output_ids) - drop
    # 构造新 list，不原地修改调用方的 output_ids（它可能还要拿去 detokenize）
    ids = list(output_ids[: len(output_ids) - drop])
    return ThinkerPayload(hidden[:P], hidden[P:target], ids)


# build_talker_prefill：Qwen2.5-Omni talker 的 prefill 输入（ids 与 embeds 两路并行，共 P+2 个位置）：
#   [0, P)  prompt 行：thinker 的 prompt embedding，id 统一填 mask_id（talker 只看 embedding，不按 id 查表）
#   P       spk_embed + codec_pad_embed，id = pad_id
#   P+1     first_reply + codec_bos_embed，id = start_id（talker 从这里开始说出第一帧 codec）
def build_talker_prefill(
    prompt_embeds: torch.Tensor,     # [P, D]
    first_reply: torch.Tensor,       # [D] 第一个生成 token 的 thinker 行
    spk_embed: torch.Tensor,         # [D] 说话人 embedding
    codec_pad_embed: torch.Tensor,
    codec_bos_embed: torch.Tensor,
    *,
    mask_id: int,
    pad_id: int,
    start_id: int,
) -> tuple[list[int], torch.Tensor]:
    P = prompt_embeds.shape[0]
    # 每个特殊位置 = “文本侧” embedding + “codec 侧” embedding。
    # 易错：[D] 要先 [None] 变成 [1, D] 才能与 [P, D] 沿 dim 0 拼接
    embeds = torch.cat([
        prompt_embeds,
        (spk_embed + codec_pad_embed)[None],
        (first_reply + codec_bos_embed)[None],
    ])
    # 返回顺序是 (ids, embeds)，len(ids) == embeds.shape[0] == P + 2
    return [mask_id] * P + [pad_id, start_id], embeds


# talker_decode_input：talker 每个 decode step 的输入 = 下一行 thinker 回复（文本语义）+ 上一步采样的 codec embedding
#   （音频自回归）。thinker 行是 FIFO 队列，每步消费一行；用完后（音频通常比文本长）一直用 fallback（tts_pad）。
#   返回 (本步输入 [D], 剩余队列)。
def talker_decode_input(
    reply_queue: torch.Tensor,        # [K, D] 剩余的 thinker 回复行
    last_codec_embed: torch.Tensor,   # [D]
    fallback: torch.Tensor,           # [D]
) -> tuple[torch.Tensor, torch.Tensor]:
    if reply_queue.shape[0] == 0:
        # 队列已空：用 fallback，并原样返回 [0, D] 的空队列（易错：返回 None，调用方下一步还要拿它继续调）
        return fallback + last_codec_embed, reply_queue
    # reply_queue[0] 是 [D]；剩余 [K-1, D]（K=1 时是 [0, D]）
    return reply_queue[0] + last_codec_embed, reply_queue[1:]
