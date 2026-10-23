# nano-vllm-omni-drills

Hands-on drills for **Part I of the nano-vllm-omni self-study book**: 26 one-hour exercises that each implement one core mechanism of an omni-modal (Qwen3-Omni style) serving system — model pieces (RMSNorm/MRoPE, attention, MoE, talker MTP, streaming conv) and serving infrastructure (orchestrator, scheduler, paged KV block manager, sampler, shared-memory connector, chunk adapter, CUDA graph runner).

- `mine/` holds skeletons for you to fill in, `tests/` holds the acceptance tests, and `solutions/` holds reference solutions.
- Everything runs on CPU; no GPU required.
- The book (problem statements, background, follow-up interview questions) is in [`book/`](book/): [English](book/nano-vllm-omni-book-en.pdf) · [中文](book/nano-vllm-omni-book-cn.pdf).

Quick start: `./check.sh m01` runs the tests for drill M01 against your code in `mine/`. The rest of this README is in Chinese.

---

《nano-vllm-omni 自学手册》Part I 的 26 道一小时核心实现题（练习区）。
题面、背景和“面试官追问”见 `book/` 下的 PDF：[中文版](book/nano-vllm-omni-book-cn.pdf) · [English](book/nano-vllm-omni-book-en.pdf)。下表的“页”是 PDF 页码（中文版 / 英文版）。

## 目录结构

```
mine/        ← 你要写的代码（接口骨架，全部是 raise NotImplementedError）
tests/       ← 验收测试（附录 B 原样提取）
solutions/   ← 参考解（全部测试通过；做完再看）
hints/HINTS.md ← 附录 A 提示（卡住 15 分钟以上再看）
hints/       ← 交互版提示：浏览器打开 hints/index.html（源文件在 hints/_src/，python3 hints/_src/build.py 重新生成）
check.sh     ← 跑某一题的测试
drill.py     ← 测试用它加载 mine/ 或 solutions/
.venv/       ← 专用环境：transformers==4.57.1 + pytest（torch 用系统里的）
```

## 怎么做一道题

1. 读题 5 分钟（PDF 对应页），列出你想到的边界情况。
2. 在 `mine/<文件>.py` 里实现，限时 45 分钟。
3. 跑测试、修 bug，15 分钟：

```bash
cd ~/development/nano-vllm-omni-drills
./check.sh m01                # 只跑 M01
./check.sh p07 -x             # 遇到第一个失败就停
./check.sh all                # 全部
./check.sh m01 --solutions    # 看参考解是否通过（等价于 DRILL_IMPL=solutions）
```

想直接用 pytest：`.venv/bin/pytest tests/test_m01_rmsnorm_mrope.py`

建议顺序：Track M → A → B → C → D → E。
依赖：M08 会 import 你写的 M01、M02、M03、M07；P03 继承你写的 P02；其余各题互相独立。

## 进度表（共 26 题，约 23 小时）

| ✓ | # | ID | 题目 | 分钟 | 难度 | 页（中/英） | 文件 |
|---|---|----|------|-----|-----|----|------|
| ☐ | 1 | M01 | RMSNorm 与多模态 RoPE（interleaved / chunked） | 45 | 中 | 24 / 27 | `m01_rmsnorm_mrope` |
| ☐ | 2 | M02 | Thinker Attention：QK-norm + GQA + KV cache | 60 | 中 | 28 / 31 | `m02_thinker_attention` |
| ☐ | 3 | M03 | Sparse MoE：路由、按 expert 分组计算、shared expert | 60 | 中 | 32 / 35 | `m03_sparse_moe` |
| ☐ | 4 | M04 | 多模态拼接：placeholder 计数、embedding 合并、DeepStack | 45 | 中 | 36 / 39 | `m04_mm_merge_deepstack` |
| ☐ | 5 | M05 | Talker MTP：一帧内预测 15 个残差 codebook | 60 | 难 | 40 / 43 | `m05_talker_mtp` |
| ☐ | 6 | M06 | Code2Wav 的流式因果卷积与转置卷积 | 60 | 难 | 43 / 46 | `m06_streaming_causal_conv` |
| ☐ | 7 | M07 | 按 stage 加载与打包权重 | 45 | 中 | 46 / 49 | `m07_stage_weight_loading` |
| ☐ | 8 | M08 | 串起来：tiny thinker 的 KV-cache greedy 解码 | 60 | 中 | 49 / 52 | `m08_tiny_thinker_e2e` |
| ☐ | 9 | P02 | Orchestrator v1：线性多 stage 路由 | 60 | 中 | 53 / 56 | `p02_orchestrator` |
| ☐ | 10 | P03 | Orchestrator v2：async_chunk 预热 + 终态守门 | 45 | 中 | 56 / 59 | `p03_async_chunk_orchestrator` |
| ☐ | 11 | P01 | OutputSink：线程 → asyncio 的批量唤醒 | 45 | 中 | 59 / 62 | `p01_output_sink` |
| ☐ | 12 | P04 | StagePool：粘性绑定、两阶段 abort、故障隔离 | 60 | 中 | 62 / 65 | `p04_stage_pool_abort` |
| ☐ | 13 | P05 | RpcRouter：共享通道上的 RPC 关联 | 40 | 易 | 66 / 69 | `p05_rpc_router` |
| ☐ | 14 | P10 | EngineCore busy loop | 45 | 中 | 70 / 73 | `p10_engine_core_loop` |
| ☐ | 15 | P11 | WAITING_FOR_CHUNK：停靠与归还 | 45 | 中 | 73 / 76 | `p11_chunk_aware_scheduler` |
| ☐ | 16 | P07 | Scheduler：continuous batching + chunked prefill + 抢占 | 75 | 难 | 77 / 80 | `p07_scheduler` |
| ☐ | 17 | P06 | BlockManager：paged KV + 前缀缓存 | 75 | 难 | 81 / 85 | `p06_block_manager` |
| ☐ | 18 | P08 | 批不变的采样器（per-request seed） | 45 | 中 | 85 / 90 | `p08_sampler` |
| ☐ | 19 | P09 | Paged KV 写入 + decode attention（参考实现） | 45 | 中 | 89 / 94 | `p09_paged_attention` |
| ☐ | 20 | P12 | SharedMemoryConnector：跨进程 put/get 协议 | 60 | 难 | 93 / 99 | `p12_shm_connector` |
| ☐ | 21 | P13 | ChunkAdapter：后台收发线程与顺序/清理语义 | 75 | 难 | 96 / 103 | `p13_chunk_adapter` |
| ☐ | 22 | P15 | Thinker→Talker 输入构造（含“差一行”） | 40 | 易 | 101 / 109 | `p15_talker_inputs` |
| ☐ | 23 | P14 | Codec 窗口规划：首块小、后块大、左上下文 | 40 | 易 | 104 / 113 | `p14_codec_windows` |
| ☐ | 24 | P18 | MRoPE 三维位置 | 40 | 中 | 107 / 116 | `p18_mrope_positions` |
| ☐ | 25 | P16 | 启动规划：设备分组、并行初始化、回滚、准入 | 45 | 中 | 111 / 120 | `p16_init_planner` |
| ☐ | 26 | P17 | CUDA graph bucket runner（CPU 模拟） | 40 | 易 | 114 / 124 | `p17_graph_runner` |

做完 26 题之后是 Capstone（中文版第 116 页 / 英文版第 127 页）：用自己的实现在 CPU 上拼一个 3 stage 的 mini omni 服务。

## 环境说明

- 全部测试都在 CPU 上跑，不需要 GPU。Track M 的测试拿 HF transformers 4.57.1 的 Qwen3-Omni 实现当标准答案（tiny 随机权重，几秒钟）。
- `.venv` 是用 `--system-site-packages` 建的，torch 直接用 base 环境里的；base 环境里的 transformers 5.x 不受影响。
- 系统 `PYTHONPATH` 里有 ROS 的 pytest 插件，会导致 pytest 启动失败，`pytest.ini` 里已经用 `-p no:...` 关掉了。
- 想重做某一题：从 PDF 重新抄骨架，或者用 `git checkout mine/<文件>` 还原。
