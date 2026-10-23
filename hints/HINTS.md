# 附录 A · 各题提示（卡住 15 分钟以上再看）

摘自《nano-vllm-omni 自学手册》附录 A（PDF 第 74–76 页）。

## Track M · Qwen-Omni 模型实现

**A-1 · M01 RMSNorm 与多模态 RoPE**
1. interleaved：先拿 T 轴的 freqs 拷贝一份，再用 `slice(offset, section*3, 3)` 覆盖 H(offset=1)、W(offset=2)。
2. `cat([f, f])` 而不是交错复制——这是 HF/vLLM 的 neox 风格 rotate_half。

**A-2 · M02 Thinker Attention**
1. QK-norm 在 RoPE 之前；norm 的权重形状是 [D]，对 [T, H, D] 的最后一维广播。
2. 因果 mask：`k_pos > q_pos` 处置 -inf，其中 `q_pos = past + arange(T)`。

**A-3 · M03 Sparse MoE**
1. `flat_tok = arange(N).repeat_interleave(k)`，和 `topi.reshape(-1)` 一一对应。
2. 权重先在 fp32 算，最后转回 x.dtype（与 HF 一致）。

**A-4 · M04 多模态拼接**
1. DeepStack 是在 layer 之后注入，不是之前；顺序错了 HF 对拍会失败。

**A-5 · M05 Talker MTP**
1. lm_head 和 codec_embedding 的下标都是 j（预测第 j+1 个 codebook）；最后一个 code 也要用 `codec_embedding[G-2]` 嵌入。
2. KV 版：第一次喂 2 个位置，之后每次只喂 1 个新 embedding。

**A-6 · M06 流式因果卷积与转置卷积**
1. 转置卷积：维护一个从 base 开始的累加缓冲区；新块的卷积结果加到 `[frames·s, frames·s+len)`。
2. 全局左裁剪只影响最开始输出的 K-s 个样本：用全局位置判断，而不是每块都裁。

**A-7 · M07 按 stage 加载与打包权重**
1. 用正则 `^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$` 收集 expert。

**A-8 · M08 tiny thinker 的 KV-cache greedy 解码**
1. MoeWeights 的 w_gate/w_up 就是 w13 的上下两半。
2. logits 只需要最后一行用于采样，但测试会比较 prefill 的全部行。

## Track A · 前端与 Orchestrator

**A-9 · P02 Orchestrator v1**
1. 结构：每个 stage 一个 reader task → 一个 ready 队列 → 唯一 dispatcher。只有 dispatcher（和注册）改 request_states。
2. 终态条件是 `final_stage_ids ⊆ finished`，而不是“某个 final stage 完成”。
3. dispatcher 的临界区里不要 await 别的东西再回来改状态；需要 await 时先把状态改完。

**A-10 · P03 Orchestrator v2**
1. add 时把所有要 prewarm 的 stage 先标记为 submitted，再逐个 await submit。
2. 非 final stage 的完成事件也要检查：它可能是“最后一块拼图”，此时放行 pending_final。

**A-11 · P01 OutputSink**
1. deque 的 append/popleft 在 CPython 下是线程安全的；标志位的“检查并设置”不是，要锁。
2. put 里先 append 再看标志；get 里先清标志再第二次 drain——顺序决定正确性。

**A-12 · P04 StagePool**
1. 绑定表 `dict[req → replica]`；evict 时同时删除这些绑定。
2. abort 成功后才释放该 replica 上这些请求的绑定。

**A-13 · P05 RpcRouter**
1. deliver 在锁内 pop waiter，在锁外 put（避免持锁阻塞）。

## Track B · Stage 进程与调度

**A-14 · P10 EngineCore busy loop**
1. 无工作时阻塞在 `input_queue.get()`；有工作时 `get_nowait` 把队列抽干再 step。
2. shutdown = 设置 Event + 往队列塞 WAKEUP。

**A-15 · P11 WAITING_FOR_CHUNK**
1. 停靠表：id → (req, origin, previous_status)。

**A-16 · P07 Scheduler**
1. block 记账只需要每个序列一个 num_blocks 计数器，`_try_grow(seq, n_tokens)` 原子地增加。

**A-17 · P06 BlockManager**
1. 先算命中数和所需新块数，够了再动手。
2. 每个序列保存自己的 token 列表和哈希链，append 时才能算新块的哈希。

**A-18 · P08 采样器**
1. top_p：`remove = cumsum - prob > top_p`，并强制保留第一个。

**A-19 · P09 Paged attention**
1. 这题不追求速度：先写对，用循环也可以。

## Track C · 数据面

**A-20 · P12 SharedMemoryConnector**
1. `from multiprocessing import resource_tracker; resource_tracker.unregister(shm._name, "shared_memory")`

**A-21 · P13 ChunkAdapter**
1. deque 无锁 append + `Condition(timeout)` 作为丢通知的兜底。
2. 注册时生成一个 token 对象 `object()`；get 回来后检查 `registered[req] is token`。

## Track D · 模型胶水

**A-22 · P15 Thinker→Talker 输入构造**
1. `target_rows = P + len(output_ids) - drop`。

**A-23 · P14 Codec 窗口规划**
1. 切片起点 = `(new_start - ctx_start) * upsample`。

**A-24 · P18 MRoPE 三维位置**
1. `arange(t).view(-1,1).expand(t, gh*gw).flatten()` 之类的写法生成三个下标。

## Track E · 启动与 GPU

**A-25 · P16 启动规划**
1. 并查集；线程池按组提交；用锁保护“第一次失败”。

**A-26 · P17 CUDA graph bucket runner**
1. 从大到小 capture（CUDA 上共享一个 memory pool 时更省显存）。
