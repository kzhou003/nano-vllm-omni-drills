lede: 每题一页：题面速览、直觉类比、shape trace、易错点、可逐步播放的交互图，以及按剧透程度递增的 L1 → L2 → L3 三级提示（卡住 15 分钟以上再看）。

## 怎么用这些提示

1. **先读题 5 分钟**（PDF 对应页），列出你想到的边界情况；然后在 `mine/<文件>.py` 里限时 45 分钟实现；最后 15 分钟用 `./check.sh <id>` 跑测试、修 bug。
2. 卡住 15 分钟以上，再打开对应题的提示页，**按级别逐级展开**：

<div class="legend">
<span><span class="lvl" style="background:var(--l1);color:#fff">L1</span> I/O 示例 + 关键问题 + 细节：只告诉你“应该得到什么”</span>
<span><span class="lvl" style="background:var(--l2);color:#fff">L2</span> 思路拆分与常见坑：告诉你“怎么拆、哪里会错”</span>
<span><span class="lvl" style="background:var(--l3);color:#fff">L3</span> 详解：关键问题答案、逐步状态追踪、骨架要点</span>
</div>

3. 每页的 **▶ 逐步图** 可以用“下一步”按钮（或聚焦后按 ← →）一步一步看状态如何变化；**虚线框里的“显示答案”** 是题面里留给你填的 `??`。
4. 做完一题，勾选页面顶部的“标记完成”，这里的进度条会同步（只存在本浏览器里）。
5. 带完整中文注释的参考解在 `solutions/`（易错点用 `易错：` 标出），做完再看：`./check.sh m01 --solutions`。

::: callout info 推荐顺序与依赖
Track **M → A → B → C → D → E**。题目之间的依赖：**M08** 会 import 你写的 M01、M02、M03、M07；**P03** 继承你写的 P02；其余各题互相独立。
全部在 CPU 上跑；Track M 的测试拿 HF transformers 4.57.1 的 Qwen3-Omni 作为对拍标准答案。
:::

## 并发程序设计框架：T · SBWC · LIFE

写 asyncio / 多线程并发程序时按顺序走的思考清单（Track A/B/C/E 的提示页都会用它来分析题目）。**先定场地 T，再搭骨架 SBWC，再让它活起来 LIFE。**

| 步骤 | 问自己 | 规则 |
|---|---|---|
| **T** Topology 执行拓扑 | 代码跑在哪些进程 / 线程 / 事件循环里？边界在哪？ | 每跨一层边界，能用的通信方式和同步原语都不一样；先定 T |
| **S** State 账本 | 哪些可变共享状态？每一份由谁写？ | 尽量单写者（single writer），做不到才上锁 |
| **B** Blocking 阻塞点 | 程序会在哪里“干等”？哪些互相独立？ | 每个独立阻塞源配一个专门守着它的 worker |
| **W** Workers 岗位 | 每个 worker 等什么、做什么、**不准做什么**？ | 职责越单一越好：搬运工只搬运，决策者只决策 |
| **C** Channels 通道 | 岗位之间用什么传消息？消息格式？ | 多生产者一消费者 → fan-in 到一个队列，消息带来源标识 |
| **L** Lifecycle 生命周期 | 怎么启动？怎么干净地停？ | `create_task` 保存句柄；停时 `cancel()` + `gather(return_exceptions=True)` |
| **I** Interface 对外接口 | 公开 API 只做什么？跑在谁的上下文？ | 接口要薄：只往通道 put / 从通道 get |
| **F** Flow + Invariants | 必须永远成立的规则是什么？ | `await` = 交出控制权；await 之前状态必须自洽 |
| **E** Edges + Errors | 迟到 / 乱序 / 重复 / 未知 / 异常怎么办？ | worker 不能因一条坏消息退出；“不该发生但可能发生”的要有明确处理并计数 |

**跨越边界用什么：**

| 边界 | 通信方式 | 同步原语 |
|---|---|---|
| 进程 ↔ 进程 | pipe / zmq / 共享内存（`multiprocessing.shared_memory`） | 文件锁、协议约定（谁创建、谁释放） |
| 线程 ↔ 线程 | `queue.Queue`、`collections.deque` + 锁 | `threading.Lock / Event / Condition` |
| 线程 → 事件循环 | `loop.call_soon_threadsafe(...)`、`asyncio.run_coroutine_threadsafe(...)` | 只能通过这两座“桥”进入循环 |
| task ↔ task（同一循环） | `asyncio.Queue` | `asyncio.Event`，或干脆单写者不加锁 |

::: callout ok 三条贯穿始终的法则
1. **await 就是交出控制权。** 看到 `await` 就问：“如果此刻别的 worker 来跑，它看到的状态对吗？”状态修改要在 await 之前完成。
2. **跨线程只能用专门的桥。** 从普通线程进入 asyncio 用 `loop.call_soon_threadsafe`，不要在线程里直接调 `asyncio.Queue` / `Event` 的方法（P01 练的就是这个）。
3. **能用“队列 + 单写者”，就别用锁。** 队列把并发问题变成串行处理，“谁在什么时候改了什么”一目了然。
:::

| 常见模式 | 什么时候用 | 本套题里 |
|---|---|---|
| fan-in（多读者 → 一个队列 → 单消费者） | 多个独立阻塞源，结果要一处统一处理 | P02 / P03 的 readers + dispatcher |
| single writer | 共享状态需要复杂的读-改-写 | P02 dispatcher、P10 EngineCore loop |
| 线程 → 事件循环的桥（批量唤醒） | 生产者在线程、消费者在 asyncio | P01 OutputSink |
| 请求关联（key → future） | 共享通道上的 RPC，结果乱序返回 | P05 RpcRouter |
| 后台收发线程 + 顺序号 | 跨进程数据要按顺序、可取消地传递 | P13 ChunkAdapter |

## 26 题

::: callout analogy Capstone（做完 26 题之后）
用你自己的实现在 CPU 上拼一个 3 stage 的 mini omni 服务：三个 spawn 子进程各跑 P10 busy loop（内部 P07 scheduler，可选 P06）；父进程 orchestrator 线程用 P03 编排、结果经 P01 交给前端、abort 用 P04、RPC 用 P05；stage 之间用 P12 + P13 传数据，接收端用 P11 停靠、codec 切窗用 P14，启动用 P16。验收：100 个并发请求、随机 10% abort、随机 kill 一次 stage2 —— 每个请求恰好一个终态、`/dev/shm` 无残留、没有僵尸进程。
:::
