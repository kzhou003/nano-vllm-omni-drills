"""M04 reference solution · placeholder 计数、多模态 embedding 合并、DeepStack 注入。

提示页：hints/m04_overview.html（A-4）。
"""
from collections.abc import Callable

import torch


# ---------------------------------------------------------------------------
# audio_token_count：前端要在 prompt 里放多少个 <audio> placeholder——必须与 AuT 音频编码器
#   实际输出的 token 数严格相等，否则 merge_multimodal 会对不上。
# 规则：每满 100 个 mel 帧 -> 13 个 token（100 -> 50 -> 25 -> 13）；余下 r 帧单独过三次
#   stride-2 卷积，每次长度 L -> (L-1)//2 + 1（= ceil(L/2)）。
# 只用 % // + - * 运算，所以 int 和 LongTensor（逐元素）都能直接用。
# 例：n=99 -> 13，n=100 -> 13，n=101 -> 14，n=330 -> 43
# ---------------------------------------------------------------------------
def audio_token_count(num_mel_frames):
    n = num_mel_frames
    r = n % 100                    # 不足一个窗口的尾巴
    f = (r - 1) // 2 + 1           # 第 1 次 stride-2 卷积后的长度
    # 第 2、3 次卷积写在一个表达式里：((f-1)//2+1) 是第 2 次结果，外层再做第 3 次。
    # 边界：r == 0 时 f = (-1)//2 + 1 = 0（floor 除法向负无穷取整），后两次也是 0。
    # 易错：不要写 if r == 0 / int() / math.floor——对多元素 tensor 不可用；
    #       换成向零截断的除法时 r == 0 会得到 1，100 帧被错算成 14
    tokens_r = ((f - 1) // 2 + 1 - 1) // 2 + 1
    return tokens_r + 13 * (n // 100)


# ---------------------------------------------------------------------------
# merge_multimodal：把文本 embedding 中各模态 placeholder 位置的行，按顺序替换成编码器输出，
#   得到 thinker 的 inputs_embeds；同时返回每个模态的 bool mask（DeepStack 要用 visual mask）。
# ---------------------------------------------------------------------------
def merge_multimodal(
    input_ids: torch.Tensor,               # [T]
    text_embeds: torch.Tensor,             # [T, H]   embed_tokens(input_ids)
    placeholder_ids: dict[str, int],       # modality -> placeholder token id
    mm_embeds: dict[str, torch.Tensor],    # modality -> [n_tokens, H] 编码器输出，按出现顺序
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # 易错：必须先 clone——不能原地修改调用方的 text_embeds（它可能被复用）
    out = text_embeds.clone()
    masks = {}
    # 以 placeholder_ids 为主循环（而不是 mm_embeds），“有占位符却缺 embedding”的情况才会被覆盖到
    for modality, pid in placeholder_ids.items():
        mask = input_ids == pid            # [T] bool，该模态的占位位置
        masks[modality] = mask             # 即使这次没出现也返回（全 False），调用方可统一按模态取
        n = int(mask.sum())
        if n == 0 and modality not in mm_embeds:
            continue                       # 没占位也没 embedding：正常跳过
        if modality not in mm_embeds:
            # 有占位符却没给 embedding：线上通常意味着前后端 processor 版本不一致
            raise ValueError(f"{n} {modality} placeholders but no {modality} embeddings")
        emb = mm_embeds[modality]
        # 数量必须严格相等；易错：截断 / 补零会让模型“看错图”而不报错
        if emb.shape[0] != n:
            raise ValueError(f"{modality}: {n} placeholders vs {emb.shape[0]} embeddings")
        # bool mask 赋值按 T 维从前往后依次填入 emb 的第 0, 1, 2... 行（这就是 “in order”）；
        # 编码器输出 dtype 可能与文本 embedding 不同（如 ViT fp32、LLM bf16），先 cast
        out[mask] = emb.to(out.dtype)
    return out, masks


# apply_deepstack：把视觉编码器中间层的特征“加”（不是替换）到视觉 token 的行上；返回新张量，不改输入。
def apply_deepstack(hidden: torch.Tensor, visual_mask: torch.Tensor, visual_embeds: torch.Tensor) -> torch.Tensor:
    out = hidden.clone()
    # hidden[visual_mask] 是 [n_vis, H]，与 visual_embeds 行数一致；文本行不变。
    # 易错：hidden[visual_mask] += ... 会原地改输入；out += visual_embeds 形状对不上（n_vis 行 vs T 行）
    out[visual_mask] = out[visual_mask] + visual_embeds.to(out.dtype)
    return out


# run_layers：带 DeepStack 的 decoder 层循环。level i 在 layer i “之后”注入；
#   级数通常少于层数，只有前 len(deepstack_embeds) 层之后有注入。
def run_layers(
    layers: list[Callable[[torch.Tensor], torch.Tensor]],
    hidden: torch.Tensor,
    visual_mask: torch.Tensor,
    deepstack_embeds: list[torch.Tensor],
) -> torch.Tensor:
    for i, layer in enumerate(layers):
        hidden = layer(hidden)
        # 易错：注入放在 layer 调用之前是 off-by-one，只有视觉行会变，只看文本行永远发现不了；
        # 也不要用 zip(layers, deepstack)，那会把多出来的 layer 跳过
        if i < len(deepstack_embeds):
            hidden = apply_deepstack(hidden, visual_mask, deepstack_embeds[i])
    return hidden
