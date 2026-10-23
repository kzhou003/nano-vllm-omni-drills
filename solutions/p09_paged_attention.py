"""P09 reference solution · Paged KV 写入 + decode attention（PyTorch 参考实现，Triton 版的对拍标准）。

cache 布局 [num_blocks, block_size, num_kv_heads, head_dim]：前两维合起来就是 slot 维，
slot = block * block_size + offset（与 P06 BlockManager.slot 一致）。提示页：hints/p09_overview.html（A-19）。
"""
import torch


# write_kv：把本步新算出的 K/V 按 slot_mapping 散写（scatter）进 paged KV cache，原地写、返回 None。
def write_kv(
    k_cache: torch.Tensor,        # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,
    k: torch.Tensor,              # [num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,
    slot_mapping: torch.Tensor,   # [num_tokens]，-1 = padding token（CUDA graph 把 batch 补齐出来的），不写
) -> None:
    bs = k_cache.shape[1]
    # 易错：必须先过滤 slot == -1。PyTorch 的负下标把 -1 解释成“最后一格”，padding token 会安静地覆盖
    # 最后一个物理块的最后一个 slot——不报错、不告警
    keep = slot_mapping >= 0
    slots = slot_mapping[keep].long()   # 转 long：int32 索引在某些后端 / 旧版本不被接受
    # 扁平 slot 拆成 (block, offset) 两个高级索引同时给出：cache[blocks[j], offsets[j]] = k[j]（每个元素 [KVH, D]）。
    # 易错：写成 k_cache[blk][off] = ... 会先产生一份拷贝再写，改不到原 cache
    k_cache[slots // bs, slots % bs] = k[keep].to(k_cache.dtype)
    v_cache[slots // bs, slots % bs] = v[keep].to(v_cache.dtype)


# paged_decode_attention：decode 阶段每条序列只有 1 个 query token（最新的那个，它的 KV 已先 write_kv 写入），
# 它能看见全部 context，不需要 causal mask。先写对：逐序列循环。
def paged_decode_attention(
    q: torch.Tensor,              # [B, num_heads, head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,   # [B, max_blocks]，每行尾部未使用的表项是垃圾值（甚至超出 num_blocks）
    context_lens: torch.Tensor,   # [B]
    scale: float,
) -> torch.Tensor:                # [B, num_heads, head_dim]
    B, H, D = q.shape
    bs, KVH = k_cache.shape[1], k_cache.shape[2]
    group = H // KVH   # GQA：每 group 个 query head 共享一个 KV head
    out = torch.empty_like(q)
    for b in range(B):
        n = int(context_lens[b])   # 该序列的上下文长度（含当前 token）
        # 只取前 ceil(n / bs) 个表项。易错 1：用 n // bs 会把最后一个不满的块整块丢掉；
        # 易错 2：取整行会读到垃圾表项（IndexError，或悄悄读到别人的 KV）
        blocks = block_tables[b, : -(-n // bs)].long()
        # k_cache[blocks]: [nb, bs, KVH, D]，按 table 的顺序 gather（table 顺序就是逻辑顺序，不要排序）
        # -> 展平成 [nb*bs, KVH, D] -> 截断到前 n 个。
        # 易错 3：不截断时最后一块多出来的槽位也参与 attention——它们的 score 是 0 而不是 -inf，
        #         在 softmax 里仍占一份权重，结果悄悄错
        # GQA：repeat_interleave 让 query head h 对上 kv head h // group（repeat 会变成 h % KVH，KVH=1 时看不出来）
        k = k_cache[blocks].reshape(-1, KVH, D)[:n].repeat_interleave(group, 1)   # [n, H, D]
        v = v_cache[blocks].reshape(-1, KVH, D)[:n].repeat_interleave(group, 1)
        # scores[h, j] = q[h] · k[j, h] * scale -> [H, n]，fp32 计算；scale 乘在 softmax 之前
        s = torch.einsum("hd,nhd->hn", q[b].float(), k.float()) * scale
        p = torch.softmax(s, -1)   # 在 key 维（n）上归一化
        # out[h] = sum_j p[h, j] * v[j, h] -> [H, D]，再转回输入 dtype
        out[b] = torch.einsum("hn,nhd->hd", p, v.float()).to(q.dtype)
    return out
