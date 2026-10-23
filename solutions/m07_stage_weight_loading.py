"""M07 reference solution · 按 stage 选择 shard / 张量，并把一层权重打包成 kernel 友好的布局。

“少加载一个张量”在 PyTorch 里不报错（参数保持随机初始化，输出悄悄变噪声），所以缺 key、多 key、
expert 编号不连续、expert 不完整都必须抛 LoadError。提示页：hints/m07_overview.html（A-7）。
"""
import re
from dataclasses import dataclass

import torch

# 一个 Qwen3-Omni checkpoint 同时包含三个 stage 的权重，靠 key 的顶层前缀区分。
# 易错：前缀带点号 "thinker."，否则 "thinkers.xxx" 这类相似前缀也会被误匹配
STAGE_PREFIX = {"thinker": "thinker.", "talker": "talker.", "code2wav": "code2wav."}


class LoadError(Exception):
    pass


# PackedLayer：推理引擎常用的“打包”权重布局——q/k/v 拼成一个 qkv、每个 expert 的 gate/up 拼成 w13，
#   这样一次 GEMM 就能算出多个投影，之后再按行 split 回去。
@dataclass
class PackedLayer:
    input_layernorm: torch.Tensor
    qkv: torch.Tensor                  # [(H + 2*KVH) * D, hidden]   rows: q | k | v
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o_proj: torch.Tensor
    post_attention_layernorm: torch.Tensor
    router: torch.Tensor               # [E, hidden]（HF 里叫 mlp.gate，但它是 router，不是 gate_proj）
    w13: torch.Tensor                  # [E, 2*I, hidden]   rows: gate | up（FusedMoE 的 w1/w3）
    w2: torch.Tensor                   # [E, hidden, I]


# 一层里除 expert 之外必须恰好出现的 9 个 key（层内相对名）
_FIXED = {
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate.weight",
}
# 层内 expert 权重名，如 "mlp.experts.12.up_proj.weight" -> group(1)="12", group(2)="up_proj"；
# 带 ^...$ 锚点，拼写错误或多余的名字不会被误收
_EXPERT = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


# shards_for_stage：每个 stage 进程只打开含有本 stage key 的 shard
#   （来自 model.safetensors.index.json 的 weight_map: param key -> shard 文件名）。
def shards_for_stage(weight_map: dict[str, str], stage: str) -> list[str]:
    prefix = STAGE_PREFIX[stage]
    # set 去重（一个 shard 可能同时属于多个 stage），sorted 保证结果确定。
    # 注意 thinker.lm_head.* 也属于 thinker（它在 "thinker." 下，只是不在 "thinker.model." 下）
    return sorted({f for k, f in weight_map.items() if k.startswith(prefix)})


# select_stage：从全量 state dict 里筛出某个子模块（如 "thinker.model."）的权重并去掉前缀，
#   得到子模块本地命名（"layers.0.xxx"、"norm.weight"...）。
def select_stage(state: dict[str, torch.Tensor], stage: str, strip: str) -> dict[str, torch.Tensor]:
    # strip 必须落在该 stage 之下，防止调用方误用别的 stage 的前缀
    if not strip.startswith(STAGE_PREFIX[stage]):
        raise LoadError(f"{strip!r} is not under stage {stage!r}")
    # startswith + 切片去前缀（易错：str.replace 可能替换掉中间的子串）。
    # 不在 strip 下的 key（如 thinker.lm_head.weight）被丢掉，需要时由调用方另取；值原样保留
    return {k[len(strip):]: v for k, v in state.items() if k.startswith(strip)}


# ---------------------------------------------------------------------------
# pack_layer：把第 i 层的 HF 散装权重打包成 PackedLayer，并严格校验：
#   缺 _FIXED 中的 key、出现不认识的 key、expert id 不是 0..E-1、某 expert 缺 gate/up/down -> LoadError。
# ---------------------------------------------------------------------------
def pack_layer(sd: dict[str, torch.Tensor], i: int) -> PackedLayer:
    # 易错：前缀带结尾的点。"layers.1" 还会匹配 "layers.10."、"layers.11." ...
    prefix = f"layers.{i}."
    local = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}   # 层内相对 key
    missing = _FIXED - set(local)
    if missing:
        raise LoadError(f"layer {i}: missing {sorted(missing)}")
    # 收集 expert 权重：experts[e]["gate_proj" | "up_proj" | "down_proj"] = tensor。
    # 不在 _FIXED 里、又不匹配 expert 正则的 key 就是“多余 key”（如 rotary_emb.inv_freq、shared_expert.*）
    experts: dict[int, dict[str, torch.Tensor]] = {}
    for k, v in local.items():
        if k in _FIXED:
            continue
        m = _EXPERT.match(k)
        if not m:
            raise LoadError(f"layer {i}: unexpected key {k}")
        experts.setdefault(int(m.group(1)), {})[m.group(2)] = v
    # expert id 必须恰好是 0..E-1（不能为空、不能有空洞）；E 就是收集到的不同 id 数。
    # sorted(experts) 按整数排序（易错：若用字符串排序会得到 0, 1, 10, 11, 2 ...）
    if not experts or sorted(experts) != list(range(len(experts))):
        raise LoadError(f"layer {i}: expert ids {sorted(experts)} are not 0..E-1")
    # 正则只允许三种 proj，所以“恰好 3 个”就等价于三种齐全
    for e, parts in experts.items():
        if len(parts) != 3:
            raise LoadError(f"layer {i}: expert {e} has only {sorted(parts)}")
    E = len(experts)
    return PackedLayer(
        input_layernorm=local["input_layernorm.weight"],
        # nn.Linear 的 weight 是 [out, in]：共享输入的三个投影沿 dim 0（输出维 / 行）拼接，
        # 之后 (x @ qkv.T).split([H*D, KVH*D, KVH*D], -1) 就能还原 q / k / v
        qkv=torch.cat([local["self_attn.q_proj.weight"], local["self_attn.k_proj.weight"],
                       local["self_attn.v_proj.weight"]], 0),
        q_norm=local["self_attn.q_norm.weight"],
        k_norm=local["self_attn.k_norm.weight"],
        o_proj=local["self_attn.o_proj.weight"],
        post_attention_layernorm=local["post_attention_layernorm.weight"],
        router=local["mlp.gate.weight"],
        # 每个 expert 先把 gate [I, hidden] 和 up [I, hidden] 沿行拼成 [2I, hidden]，再按 e = 0..E-1 stack。
        # 易错 1：顺序是 gate | up，拼反了形状照样对，只是 SiLU 用错了那一半；
        # 易错 2：必须按 expert id 顺序（range(E)），不能按 dict 遍历顺序
        w13=torch.stack([torch.cat([experts[e]["gate_proj"], experts[e]["up_proj"]], 0) for e in range(E)]),
        w2=torch.stack([experts[e]["down_proj"] for e in range(E)]),   # [E, hidden, I]
    )


# load_thinker_text：加载 thinker 文本主干。先校验 thinker.model. 下只有 embed_tokens / norm /
#   layers.{0..num_layers-1}.*，再逐层 pack。返回 ({"embed_tokens", "norm"}, [PackedLayer] * num_layers)。
#   lm_head 不在 thinker.model. 下（是 thinker.lm_head.weight），由调用方另取；talker.* 也会被 select_stage 过滤掉。
def load_thinker_text(state: dict[str, torch.Tensor], num_layers: int) -> tuple[dict, list[PackedLayer]]:
    sd = select_stage(state, "thinker", "thinker.model.")
    for k in sd:
        if k in ("embed_tokens.weight", "norm.weight"):
            continue
        # 层外的其他 key（如 rotary_emb.inv_freq）或层号 ≥ num_layers 的多余层都是错误
        m = re.match(r"^layers\.(\d+)\.", k)
        if not m or int(m.group(1)) >= num_layers:
            raise LoadError(f"unexpected key thinker.model.{k}")
    for k in ("embed_tokens.weight", "norm.weight"):
        if k not in sd:
            raise LoadError(f"missing thinker.model.{k}")
    # num_layers 比实际多时，缺的那层会在 pack_layer 里因 missing 报错
    layers = [pack_layer(sd, i) for i in range(num_layers)]
    return {"embed_tokens": sd["embed_tokens.weight"], "norm": sd["norm.weight"]}, layers
