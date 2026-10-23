lede: P02 的骨架和真实代码一模一样——request handler 处理控制消息，每个 stage replica 一个 reader 汇入同一个 ready_q，一个 dispatcher 串行路由，终态由 final ⊆ finished 决定；最大差别在 T：真实 orchestrator 跑在单独线程里，经 janus 队列和 OutputQueueSink（P01）与主线程通信。

## 一句话结论与拓扑差别

练习：`mine/p02_orchestrator.py` · 源码：vllm-project/vllm-omni @ fa8b419 · `vllm_omni/engine/orchestrator.py`（行号固定在该 commit）。

```
主线程（serving loop）              "orchestrator" 线程 · 自己的事件循环              stage 进程 × N（可多 replica）
add_request() ─sync_q.put─► janus request_queue ─► _request_handler ─submit_initial─► StagePool / client
                                                     request_states                       │ get_output
output handler ◄─call_soon_threadsafe─ OutputQueueSink ◄─ dispatcher loop ◄─ ready_q ◄─ reader(s,r) × k
                                                     forward: process_engine_inputs → submit ─► EngineCore (P10)
```

## 逐项对照表

| P02 | vllm-omni | 源码 |
|---|---|---|
| `class Orchestrator` | `OrchestratorBase`（通用骨架）＋ `Orchestrator(OrchestratorBase)`（turn-based 子类，负责 add） | `orchestrator.py:322` · `:3033` |
| `ReqState` | `OrchestratorRequestState`：`final_output_stage_ids`、`finished_final_output_stage_ids`、`finished_stage_ids`、`pending_final_output` | `orchestrator.py:233-272` |
| `request_states` | `self.request_states: dict[str, OrchestratorRequestState]` | `orchestrator.py:390` |
| `_req_q` | `request_async_queue`：janus 队列，主线程 `sync_q.put`，orchestrator 线程 `async_q.get` | `omni_engine_base.py:308` |
| `_out_q` | `output_async_queue = OutputQueueSink`：一批输出只做一次 `call_soon_threadsafe`（就是 P01） | `omni_engine_base.py:309-310` · `async_engine_utils.py:40-50` |
| `start()` | `run()`：create_task 出 request handler 和 output handler，然后 gather | `orchestrator.py:479-512` |
| `shutdown()` | `run()` 的 finally：cancel ＋ `gather(*tasks, return_exceptions=True)`，最后再扫一遍 all_tasks | `orchestrator.py:530-555` |
| `_request_handler` | `_request_handler`：按 msg.type 分派 add / abort / collective_rpc / register / shutdown | `orchestrator.py:557-592` |
| `_handle_add` | `_handle_add_request`：先 `self.request_states[request_id] = req_state`，再 submit_initial | `orchestrator.py:3062`（注册 `:3117` · submit `:3128`） |
| `_reader(sid)` | `_llm_replica_reader(stage_id, replica_id, client)`：每个 (stage, replica) 一个；出错也作为 `("error", …)` 消息交给 dispatcher | `orchestrator.py:1117-1136` |
| `_ready_q` | `ready_q: asyncio.Queue[(kind, stage_id, replica_id, payload)]` | `orchestrator.py:1112` |
| `_dispatcher` | `_orchestration_loop_event_driven` 的主循环：串行消费 ready_q | 主循环 `:1264` |
| `_handle_output` | `_handle_processed_outputs → _route_output` | `orchestrator.py:1381-1412` · `:1912-2043` |
| 未知 id：`dropped_outputs += 1` | `req_state is None`：打 warning 并 continue（没有计数器） | `orchestrator.py:1385-1393` |
| 终态 `final_stage_ids ⊆ finished` | `request_finished = final_output_stage_ids.issubset(req_state.finished_final_output_stage_ids)` | `orchestrator.py:1936-1940` |
| 发给用户 `_out_q.put(OrchOutput)` | `OutputMessage(…, finished=request_finished)` → `output_async_queue.put` | `orchestrator.py:1944-1967` |
| 转发 `forward_fn(s+1, prompt, data)` | `_forward_to_next_stage` → `next_client.process_engine_inputs(source_outputs, req_state.prompt)` → `custom_process_input_func`（如 `thinker2talker_full_payload`） | 触发条件 `:1988` · `:2602` · `stage_engine_core_client.py:455-471` |
| `_cleanup(rid)` | `_cleanup_request_ids`：释放 binding、传输资源，`request_states.pop(...)` | `orchestrator.py:1699-1762`（pop `:1756`） |
| `pending_final`（P03） | `pending_final_output`：async_chunk 下暂扣终态，等所有已提交 stage 都结束再放行 | `orchestrator.py:1954-1965` · `:2036-2040` |
| abort（P04） | `_handle_abort` → `_cleanup_request_ids(abort=True)` → `_abort_request_ids` 对每个 pool `abort_requests` | `orchestrator.py:645-683` · `:685-731` · `stage_pool.py:1279` |
| dead replica（P04） | reader 发 `("error", …EngineDeadError)` → dispatcher `_handle_dead_replica` 驱逐并继续服务 | `orchestrator.py:1297-1312` · `:1464` |

## 建议阅读路线（约 300 行）

### ① 线程与事件循环的边界（T）

主线程（HTTP / serving loop）调用 `add_request()`，把消息 `sync_q.put` 进 janus 队列；一个名叫 `"orchestrator"` 的后台线程里 `new_event_loop()`，在那里跑 `orchestrator.run()`。这是 P02 替你省掉的一层：P02 的调用方和 orchestrator 在同一个事件循环里。

```python
# omni_engine_base.py:303-330 · janus 队列 + orchestrator 线程
self.request_queue: janus.Queue[EngineQueueMessage] = janus.Queue(maxsize=_REQUEST_QUEUE_MAXSIZE)
self.output_queue: janus.Queue[EngineQueueMessage] = janus.Queue()
self._output_sink = OutputQueueSink(self.output_queue.sync_q)
self.rpc_output_queue: janus.Queue[EngineQueueMessage] = janus.Queue()
...
self.orchestrator_thread = threading.Thread(
    target=self._bootstrap_orchestrator,
    args=(stage_init_timeout, startup_future),
    daemon=True,
    name="orchestrator",

# omni_engine_base.py:440-452 · 线程里新建事件循环
def _bootstrap_orchestrator(self, stage_init_timeout, startup_future) -> None:
    """Create loop, initialize stages, then run Orchestrator."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
```

### ② 先注册，再 submit

在 `_handle_add_request` 里找这两句的先后：注册 `request_states` 发生在任何 await 之前。

```python
# orchestrator.py:3107-3138
req_state = OrchestratorRequestState(request_id=request_id, prompt=original_prompt, ...)
self.request_states[request_id] = req_state                     # 3117：注册
self._maybe_attach_native_kv_transfer_params(req_state, prompt)
...
if not await self._dispatch_or_fail_request(                     # 3128：第一个 await
    lambda: self.stage_pools[stage_id].submit_initial(request_id, req_state, prompt, ...),
    req_id=request_id, stage_id=stage_id, operation="add_request",
):
```

### ③ reader：只搬运，连错误也只是搬运

注意 `except BaseException`：reader 自己出错时不改任何状态，而是把错误包装成一条消息交给 dispatcher——单写者原则的延续。

```python
# orchestrator.py:1117-1136
async def _llm_replica_reader(stage_id: int, replica_id: int, client: Any) -> None:
    try:
        pool = self.stage_pools[stage_id]
        while not self._shutdown_event.is_set():
            raw_outputs = await pool.read_llm_raw_output(client)
            if raw_outputs is None:
                await asyncio.sleep(0.001)
                continue
            await ready_q.put(("llm", stage_id, replica_id, raw_outputs))
            self._orch_monitor.set_dispatch_queue_size(ready_q.qsize())
    except asyncio.CancelledError:
        raise
    except BaseException as e:  # noqa: BLE001 - routed to the dispatcher
        await ready_q.put(("error", stage_id, replica_id, e))
```

### ④ dispatcher：串行消费 ready_q

真实代码同时等三件事：新输出、关闭信号、replica 变化，所以用 `asyncio.wait(FIRST_COMPLETED)`，每轮重建 `pending_get`。

```python
# orchestrator.py:1264-1272, 1291-1293
while not self._shutdown_event.is_set():
    if pending_get is None:
        pending_get = asyncio.create_task(ready_q.get(), name="orch-dispatch-get")
    done, _ = await asyncio.wait(
        {pending_get, shutdown_task, reconcile_task},
        timeout=max(0.0, next_reconcile - _time.monotonic()),
        return_when=asyncio.FIRST_COMPLETED,
    )
    if shutdown_task in done:
        ...
    kind, stage_id, replica_id, payload = pending_get.result()
    pending_get = None
```

### ⑤ `_route_output`：P02 `_handle_output` 的真身

顺序：标记 finished → 计算 request_finished → final stage 就发 OutputMessage → 若 finished 且 stage_id < final_stage_id 就转发 → request_finished 则清理。

```python
# orchestrator.py:1933-1967 · 终态判断 + 发给用户
if finished and not segment_finished:
    req_state.finished_stage_ids.add(stage_id)
request_finished = False
if finished and self.stage_pools[stage_id].final_output and not segment_finished:
    req_state.finished_final_output_stage_ids.add(stage_id)
    final_output_stage_ids = req_state.final_output_stage_ids or {req_state.final_stage_id}
    request_finished = final_output_stage_ids.issubset(req_state.finished_final_output_stage_ids)
...
elif self.stage_pools[stage_id].final_output:
    message = OutputMessage(request_id=req_id, stage_id=stage_id, ..., finished=request_finished)
    if (request_finished and self.async_chunk and not req_state.streaming.enabled
            and not set(req_state.stage_submit_ts).issubset(req_state.finished_stage_ids)):
        # Direct chunk delivery can finish the decoder before the upstream engine's
        # terminal output/metrics reaches this loop. Keep request state until routed.
        req_state.pending_final_output = message
        request_finished = False
    else:
        await self.output_async_queue.put(message)

# orchestrator.py:1988-1993 · 转发条件
if ((finished or segment_finished)
        and stage_id < req_state.final_stage_id
        and not req_state.skip_legacy_stage_forward
        and (not self.async_chunk or not self._stage_receives_async_chunks(stage_id + 1))
        and (not self._next_stage_already_submitted(stage_id, req_state) or req_state.streaming.enabled)

# orchestrator.py:2036-2043 · pending_final 放行 + 清理
pending = req_state.pending_final_output
if pending is not None and set(req_state.stage_submit_ts).issubset(req_state.finished_stage_ids):
    req_state.pending_final_output = None
    await self.output_async_queue.put(pending)
    request_finished = True
if request_finished and not req_state.session_owned:
    await self._cleanup_request_ids([req_id, *self._cfg_tracker.cleanup_parent(req_id)])
```

### ⑥ `run()` 的 finally：cancel + gather

外加最后一遍兜底：扫 `asyncio.all_tasks(loop)`，把漏网的 task 也取消并等待。

```python
# orchestrator.py:530-555
finally:
    self._shutdown_event.set()
    for task in tasks:
        if not task.done():
            task.cancel()
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception:
        pass
    await self._shutdown_extensions()
    ...
    self._shutdown_stages()
    loop = asyncio.get_running_loop()
    pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task() and not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
```

## 真实代码比 P02 多出来的东西

| 多出来的 | 为什么需要 | 对应题 | 源码 |
|---|---|---|---|
| 每个 stage 多个 replica；`_reconcile_readers` 动态增删 reader | replica 会注册、死掉、被替换 | P04 | `orchestrator.py:1159-1233` |
| dispatcher 用 `asyncio.wait({ready_q.get(), shutdown, reconcile})` | 除了输出，还要响应关闭和成员变化 | — | `orchestrator.py:1264-1289` |
| reader 出错发 `("error", …)`，dispatcher 驱逐该 replica 并继续 | 故障隔离：一个 replica 挂了不拖垮整体 | P04 | `orchestrator.py:1297-1312` |
| 转发时 `source_outputs = [output]`，不自己攒 chunk | stage 的 output processor 已累积好输出；P02 用 stage_data 模拟 | — | `orchestrator.py:2366` |
| orchestrator 在独立线程，经 janus 和 OutputQueueSink 与主线程通信 | 让 HTTP 主循环不被路由逻辑拖慢 | P01 | `async_engine_utils.py:40-50` |
| diffusion stage 用 poller 轮询 | diffusion 客户端只有非阻塞 poll 接口 | — | `orchestrator.py:1138-1157` |
| CFG companion、PD 分离、streaming、session、metrics | 业务功能 | — | `orchestrator.py:1912-1931` |

## 读完之后想一想

1. 真实 orchestrator 在另一个线程里。主线程调用 `add_request` 时，为什么必须走 janus 的 `sync_q.put`，而不能直接调用 orchestrator 循环里某个 `asyncio.Queue.put_nowait`？
2. `_route_output` 里，发给用户的 OutputMessage 是在转发给下一个 stage **之前**放进队列的。如果反过来，先 `await _forward_to_next_stage`（一次跨进程通信）再发给用户，用户体感上会有什么差别？
3. reader 出错时为什么不直接调用 `_handle_dead_replica`，而要把错误包成消息交给 dispatcher？

::: answer 参考思路（本页补充，非书中原文）
1. `asyncio.Queue` 不是线程安全的：在别的线程里 `put_nowait` 既可能破坏内部状态，也不会唤醒阻塞在 selector 上的事件循环（对应“三条法则”第 2 条）。janus 的 `sync_q.put` 内部加锁，并通过 `call_soon_threadsafe` 把唤醒送进循环。
2. 用户看到的文本 token 会被延迟一次跨进程 submit 的时间（以及 talker 侧可能的排队），首 token / 流式延迟变差；先发给用户则文本输出与下游转发解耦。
3. 单写者：`_handle_dead_replica` 要修改 request_states、绑定表并发错误终态；只有 dispatcher 能改这些状态。reader 直接改会和 dispatcher 在 await 点交错，破坏“恰好一条终态”等不变量。
:::
