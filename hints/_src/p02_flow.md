lede: 一步一步看例 1（S0→S1→S2，final={0,2}）里每个 worker 在做什么、消息走哪条通道、账本怎么变——再对比两个经典反例：先 submit 后注册、dispatcher 自己轮流等。

## 架构图（SBWC 填完之后）

橙色虚线 = 向 stage 提交（submit）；灰色实线 = 队列 put / get。只有 dispatcher 和（注册时的）request_handler 会碰账本 `request_states`。点下面“下一步”，图上正在动的部分会高亮。

::: steps 例 1：add_request("r1", "hi there") 的一生
::: raw
<svg viewBox="0 0 780 330" role="img" aria-label="P02 orchestrator 架构图">
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path class="arrowhead" d="M0,0 L10,5 L0,10 z"/>
    </marker>
  </defs>
  <text x="12" y="16" font-size="12" class="muted-text">I · 公开 API</text>
  <text x="150" y="16" font-size="12" class="muted-text">C · 通道</text>
  <text x="270" y="16" font-size="12" class="muted-text">W · worker / S · 账本</text>
  <text x="530" y="16" font-size="12" class="muted-text">W · reader</text>
  <text x="660" y="16" font-size="12" class="muted-text">stage 进程（黑盒）</text>

  <g class="node" data-node="add"><rect x="10" y="30" width="120" height="38" rx="8"/><text x="70" y="54" font-size="13" text-anchor="middle">add_request()</text></g>
  <g class="node" data-node="req_q"><rect x="150" y="30" width="90" height="38" rx="8"/><text x="195" y="54" font-size="13" text-anchor="middle">_req_q</text></g>
  <g class="node" data-node="handler"><rect x="265" y="30" width="140" height="38" rx="8"/><text x="335" y="54" font-size="13" text-anchor="middle">_request_handler</text></g>
  <g class="node state" data-node="states"><rect x="265" y="135" width="140" height="44" rx="8"/><text x="335" y="155" font-size="13" text-anchor="middle">request_states</text><text x="335" y="171" font-size="11" text-anchor="middle" class="muted-text">账本（单写者）</text></g>
  <g class="node" data-node="next"><rect x="10" y="262" width="120" height="38" rx="8"/><text x="70" y="286" font-size="13" text-anchor="middle">next_output()</text></g>
  <g class="node" data-node="out_q"><rect x="150" y="262" width="90" height="38" rx="8"/><text x="195" y="286" font-size="13" text-anchor="middle">_out_q</text></g>
  <g class="node" data-node="dispatcher"><rect x="265" y="262" width="140" height="38" rx="8"/><text x="335" y="286" font-size="13" text-anchor="middle">_dispatcher</text></g>
  <g class="node" data-node="ready_q"><rect x="425" y="262" width="90" height="38" rx="8"/><text x="470" y="286" font-size="13" text-anchor="middle">_ready_q</text></g>

  <g class="node" data-node="r0"><rect x="535" y="40" width="90" height="34" rx="8"/><text x="580" y="62" font-size="13" text-anchor="middle">reader 0</text></g>
  <g class="node" data-node="r1"><rect x="535" y="120" width="90" height="34" rx="8"/><text x="580" y="142" font-size="13" text-anchor="middle">reader 1</text></g>
  <g class="node" data-node="r2"><rect x="535" y="200" width="90" height="34" rx="8"/><text x="580" y="222" font-size="13" text-anchor="middle">reader 2</text></g>
  <g class="node" data-node="s0"><rect x="665" y="40" width="105" height="34" rx="8"/><text x="717" y="62" font-size="13" text-anchor="middle">S0 thinker</text></g>
  <g class="node" data-node="s1"><rect x="665" y="120" width="105" height="34" rx="8"/><text x="717" y="142" font-size="13" text-anchor="middle">S1 talker</text></g>
  <g class="node" data-node="s2"><rect x="665" y="200" width="105" height="34" rx="8"/><text x="717" y="222" font-size="13" text-anchor="middle">S2 code2wav</text></g>

  <path class="edge" data-node="e_add" d="M130,49 L148,49" marker-end="url(#ah)"/>
  <path class="edge" data-node="e_req" d="M240,49 L263,49" marker-end="url(#ah)"/>
  <path class="edge" data-node="e_reg" d="M335,68 L335,133" marker-end="url(#ah)"/>
  <path class="edge submit" data-node="e_sub0" d="M405,40 C520,22 600,22 663,50" marker-end="url(#ah)"/>
  <path class="edge" data-node="e_g0" d="M665,57 L627,57" marker-end="url(#ah)"/>
  <path class="edge" data-node="e_g1" d="M665,137 L627,137" marker-end="url(#ah)"/>
  <path class="edge" data-node="e_g2" d="M665,217 L627,217" marker-end="url(#ah)"/>
  <path class="edge" data-node="e_p0" d="M560,74 L480,260" marker-end="url(#ah)"/>
  <path class="edge" data-node="e_p1" d="M570,154 L490,260" marker-end="url(#ah)"/>
  <path class="edge" data-node="e_p2" d="M575,234 L505,260" marker-end="url(#ah)"/>
  <path class="edge" data-node="e_rd" d="M425,281 L407,281" marker-end="url(#ah)"/>
  <path class="edge" data-node="e_st" d="M335,262 L335,181" marker-end="url(#ah)"/>
  <path class="edge" data-node="e_out" d="M265,281 L242,281" marker-end="url(#ah)"/>
  <path class="edge" data-node="e_next" d="M150,281 L132,281" marker-end="url(#ah)"/>
  <path class="edge submit" data-node="e_sub1" d="M405,270 C470,200 560,160 663,145" marker-end="url(#ah)"/>
  <path class="edge submit" data-node="e_sub2" d="M405,290 C500,320 620,280 663,226" marker-end="url(#ah)"/>
</svg>
:::
--- step 用户调用 add_request {hl=add,e_add,req_q}
`add_request` 只做一件事：`await _req_q.put(("add", "r1", "hi there"))`。它不碰账本、不 submit——公开 API 要薄（I）。
--- step handler 先注册，再 submit S0 {hl=handler,e_req,e_reg,states,e_sub0,s0}
`_handle_add`：`request_states["r1"] = ReqState(...)`、`submitted = {0}` ——**全是同步代码，没有 await，不会被打断**；然后才 `await S0.submit("r1", "hi there")`。

从这一刻起，无论 S0 多快吐输出、dispatcher 何时运行，都能查到 r1。
--- step S0 吐出 "hi"，reader 0 搬运 {hl=s0,e_g0,r0,e_p0,ready_q}
reader 0 阻塞在 `S0.get_output()`，拿到 `[StageOutput("r1","hi",False)]`，打上 sid 放进 `_ready_q`：`(0, [..])`。reader **不看内容、不碰账本**。
--- step dispatcher 处理 "hi" {hl=ready_q,e_rd,dispatcher,e_st,states,e_out,out_q}
`stage_data = {0: ['hi']}`；0 ∈ final → `done = {0,2} ⊆ {}` 为 False → put `OrchOutput("r1", 0, "hi", False)`。S0 未完成，不转发。
--- step 用户拿到第一条 {hl=out_q,e_next,next}
`next_output()` = `wait_for(_out_q.get(), timeout)`：`(0, 'hi', False)`。
--- step S0 吐出 "there"(fin) → 发给用户 + 转发 S1 {hl=s0,e_g0,r0,e_p0,ready_q,e_rd,dispatcher,states,out_q,e_out,e_sub1,s1}
finished = {0}，`{0,2} ⊄ {0}` → 发 `(0, 'there', False)`。

同一条输出又满足 “o.finished 且 0 < last_stage” → `forward_fn(1, prompt, ['hi','there'])`，**先** `submitted.add(1)` **再** `await S1.submit(...)`。两件事是两个独立的 if。
--- step S1 完成 → 只转发，不给用户 {hl=s1,e_g1,r1,e_p1,ready_q,e_rd,dispatcher,states,e_sub2,s2}
S1 ∉ final：不发给用户。1 完成且 1 < 2 → `forward_fn(2, …, [['HI','THERE']])` → submit S2。
--- step S2 吐出 "wav:HI" {hl=s2,e_g2,r2,e_p2,ready_q,e_rd,dispatcher,out_q,e_out,next,e_next}
发 `(2, 'wav:HI', False)`。
--- step S2 吐出 "wav:THERE"(fin) → 终态 + 清理 {hl=s2,e_g2,r2,e_p2,ready_q,e_rd,dispatcher,e_st,states,out_q,e_out}
finished = {0,1,2} ⊇ {0,2} → 发 `(2, 'wav:THERE', True)` → `_cleanup("r1")` → **return**。账本里 r1 没了。
--- step 迟到的 "wav:LATE" {hl=s2,r2,ready_q,dispatcher,states}
dispatcher 查不到 r1 → `dropped_outputs += 1` → return。**绝不复活状态**。
:::

## 反例 1：先 submit 后注册

| 顺序 | 时间线 | 结果 |
|---|---|---|
| ❌ 先 submit 后注册 | `await submit()` 交出控制权 → S0 很快吐出 "hi" → reader 搬进 `_ready_q` → dispatcher 查 request_states：查不到 r1 → 当成未知请求丢弃 → 回到 handler，这时才注册 | 用户永远收不到 "hi"，`dropped_outputs = 1` |
| ✅ 先注册后 submit | 注册（无 await，不会被打断）→ `await submit()` → 之后无论谁先跑，dispatcher 都能查到 r1 | 正确 |

## 反例 2：dispatcher 自己轮流等（违反 B 原则）

```python
while True:
    for sid, client in enumerate(self.stages):
        outs = await client.get_output()     # ← 一次只能等一个
        ...
```

| 时刻 | dispatcher 在做什么 | S0 | S1 |
|---|---|---|---|
| t1 | `await S0.get_output()`，拿到 "hi" | 已吐出 "hi" | 还没有任务 |
| t2 | `await S1.get_output()` | "there" 已吐出，没人读 | 没有任务，永远不会有输出 |
| t3… | 一直卡在 S1 | 还在等 | S1 的任务要等 S0 的 "there" 被处理后才会提交，可 "there" 没人读 |

结果是**死锁**：dispatcher 等 S1，S1 等 dispatcher 转发，而转发需要先读到 S0 的 "there"。规则：**每个独立的阻塞源配一个专门守着它的 worker**。

## 生命周期（L）

```python
async def start(self):
    for sid, client in enumerate(self.stages):          # 每个阻塞源一个 worker（B → W）
        self._tasks.append(asyncio.create_task(self._reader(sid, client)))
    self._tasks.append(asyncio.create_task(self._request_handler()))
    self._tasks.append(asyncio.create_task(self._dispatcher()))

async def shutdown(self):
    for t in self._tasks:
        t.cancel()                                       # 在各自的 await 处抛 CancelledError
    await asyncio.gather(*self._tasks, return_exceptions=True)   # 等它们真正结束
```

常见坑：① 不保存 task 引用 → 可能被 GC，shutdown 也找不到；② 只 cancel 不 gather → “Task was destroyed but it is pending”；③ start() 里 await 了永远不返回的东西。
