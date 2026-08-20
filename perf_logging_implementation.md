# DramStore / DramPool 性能日志实现详解

## 1. 总体架构

性能日志覆盖 DramStore 和 DramPool 两个模块，采用"本地时间点 + 完成汇总 + 关键阶段切换"的记录策略。

### 1.1 三类日志

| 类型 | event | 用途 |
|---|---|---|
| **完成汇总日志** | `request_done` / `task_done` | 请求完成后打一条，包含全部阶段耗时，是性能分析的主要数据源 |
| **阶段切换日志** | `stage` | 仅在异步状态变化时打，用于定位卡住的请求 |

### 1.2 计时方式

- **`SteadyNowUs()`**：基于 `std::chrono::steady_clock`，单调递增，用于计算阶段耗时差值（`*_us`）
- **`UnixNowUs()`**：基于 `std::chrono::system_clock`，Unix 微秒时间戳，用于跨进程日志粗粒度关联（`*_ts_us`）

### 1.3 日志格式

所有 PERF 日志使用 `UC_INFO` 级别，以 `[PERF]` 前缀开头，`key=value` 格式：

```text
[PERF] component=drampool event=request_done request_id=1024 opcode=2 peer=xxx
  batch_size=32 data_bytes=67108864 failed_items=0 status=SUCCESS failed_stage=NONE
  request_queue_us=18 metadata_prepare_us=22 taskworker_prepare_us=45 poller_queue_us=7
  data_transfer_us=610 metadata_settle_us=15 response_slot_wait_us=0
  response_submit_us=19 response_transfer_us=61 total_us=752
```

---

## 2. DramPool 侧

### 2.1 数据结构：`RequestTiming`

定义在 `drampool_types.h:69-90`，承载一个请求在 DramPool 内部的全部时间点。

```cpp
struct RequestTiming {
    // steady 时间点（用于计算阶段耗时）
    std::uint64_t received_us{0};                    // TCP 收到请求
    std::uint64_t worker_started_us{0};              // TaskWorker 开始处理
    std::uint64_t metadata_prepare_started_us{0};    // 元数据准备开始
    std::uint64_t metadata_prepare_completed_us{0};  // 元数据准备完成
    std::uint64_t completion_queued_us{0};            // CompletionRecord 入队
    std::uint64_t poller_admitted_us{0};             // CompletionPoller 取出
    std::uint64_t data_transfer_submitted_us{0};     // HIXL 数据传输提交
    std::uint64_t data_transfer_completed_us{0};     // HIXL 数据传输完成
    std::uint64_t metadata_settle_completed_us{0};   // 元数据 settle 完成
    std::uint64_t response_ready_us{0};              // 准备响应
    std::uint64_t response_slot_acquired_us{0};      // flag buffer slot 获取
    std::uint64_t response_submitted_us{0};          // 响应传输提交
    std::uint64_t response_completed_us{0};          // 响应传输完成
    std::uint64_t request_completed_us{0};           // 请求完成

    // Unix 时间戳（用于跨进程关联）
    std::uint64_t received_ts_us{0};
    std::uint64_t worker_started_ts_us{0};
    std::uint64_t data_transfer_submitted_ts_us{0};
    std::uint64_t data_transfer_completed_ts_us{0};
    std::uint64_t response_submitted_ts_us{0};
    std::uint64_t request_completed_ts_us{0};
};
```

### 2.2 打点追踪

#### ① `received_us` — TCP 收到请求并入队

```cpp
// drampool_server.cc:511-512  RequestReceiveLoop
task->timing.received_us = SteadyNowUs();
task->timing.received_ts_us = UnixNowUs();
```

`RequestReceiveLoop` 线程从 TCP MessageChannel 收到消息，`UnpackRequest` 解析成功后记录。随后推入 `requestQueue_`。

#### ② `worker_started_us` — TaskWorker 开始处理

```cpp
// task_worker.cc:72-73  ProcessOneRequest
task->timing.worker_started_us = SteadyNowUs();
task->timing.worker_started_ts_us = UnixNowUs();
```

TaskWorker 从 `requestQueue_` 取出请求，开始处理前记录。

#### ③ `metadata_prepare_started_us` / `metadata_prepare_completed_us` — 元数据准备

以 DUMP 为例（LOAD 和 LOOKUP 逻辑相同）：

```cpp
// task_worker.cc:126  ProcessDump
timing.metadata_prepare_started_us = SteadyNowUs();
for (std::uint16_t index = 0; index < request.batch_size; ++index) {
    // DUMP: StoreBegin 为每个 entry 分配元数据槽位和 buffer
    // LOAD: LoadBegin 查找元数据并 pin 住 buffer
    // LOOKUP: Exist 检查 key 是否存在
    const auto storeStatus = runtime_.metadata.StoreBegin(entry.key, metadataEntry);
    ...
}
// task_worker.cc:156
timing.metadata_prepare_completed_us = SteadyNowUs();
```

各操作的具体行为：

| 操作 | 元数据准备内容 | 代码位置 |
|---|---|---|
| DUMP | `StoreBegin`：分配 buffer、创建元数据条目 | task_worker.cc:126-156 |
| LOAD | `LoadBegin`：查找元数据、pin 住 buffer | task_worker.cc:223-252 |
| LOOKUP | `Exist`：检查 key 是否存在（无数据传输） | task_worker.cc:310-316 |

#### ④ `data_transfer_submitted_us` — HIXL 数据传输提交

```cpp
// task_worker.cc:195-196  ProcessDump（DUMP 用 RDMA Read）
// task_worker.cc:291-292  ProcessLoad（LOAD 用 RDMA Write）
record.timing.data_transfer_submitted_us = SteadyNowUs();
record.timing.data_transfer_submitted_ts_us = UnixNowUs();
```

在 `runtime_.transport.ExecuteAsync(operation, handle)` 之后记录。LOOKUP 没有数据传输，不记录此时间点。

#### ⑤ `completion_queued_us` — CompletionRecord 入队

```cpp
// task_worker.cc:367  SubmitCompletion
record.timing.completion_queued_us = SteadyNowUs();
runtime_.completionQueue.Push(std::move(record));
```

TaskWorker 处理完（数据传输已提交或无需传输），把 `CompletionRecord` 推入 `completionQueue_`。

#### ⑥ `poller_admitted_us` — CompletionPoller 取出

```cpp
// completion_poller.cc:106  FillPendingWindow
record.timing.poller_admitted_us = SteadyNowUs();
pending_.emplace_back(std::move(record));
```

CompletionPoller 从 `completionQueue_` 取出记录，加入 `pending_` 列表。

#### ⑦ `data_transfer_completed_us` — HIXL 数据传输完成

```cpp
// completion_poller.cc:200-201  PollDataTransfer
record.timing.data_transfer_completed_us = SteadyNowUs();
record.timing.data_transfer_completed_ts_us = UnixNowUs();
```

`runtime_.transport.GetStatus(handle)` 返回非 `Waiting` 状态（Completed 或 Failed）时记录。

#### ⑧ `metadata_settle_completed_us` — 元数据 settle 完成

```cpp
// completion_poller.cc:208-209  PollDataTransfer
SettleDataTransfer(record, transportStatus);  // DUMP: StoreEnd / Delete; LOAD: LoadEnd
record.timing.metadata_settle_completed_us = SteadyNowUs();
```

数据传输完成后，调用 `SettleDataTransfer` 做元数据收尾：

| 操作 | settle 内容 |
|---|---|
| DUMP 成功 | `StoreEnd`：确认元数据条目可用 |
| DUMP 失败 | `Delete`：删除预留的元数据条目 |
| LOAD | `LoadEnd`：释放 pin 住的 buffer |

#### ⑨ `response_ready_us` — 准备响应

```cpp
// completion_poller.cc:210  PollDataTransfer（有数据传输的情况）
record.timing.response_ready_us = record.timing.metadata_settle_completed_us;

// task_worker.cc:355  QueueResponse（无数据传输的情况，如 LOOKUP）
timing.response_ready_us = SteadyNowUs();
```

有数据传输时直接等于 `metadata_settle_completed_us`；无数据传输时（LOOKUP 或全部命中/跳过）在 `QueueResponse` 中记录。

#### ⑩ `response_slot_acquired_us` — flag buffer slot 获取

```cpp
// completion_poller.cc:246  SubmitResponse
auto allocateStatus = runtime_.flagBufferPool.Allocate(record.local_resp_slot);
...
record.timing.response_slot_acquired_us = SteadyNowUs();
```

从 flag buffer pool（host-pinned memory）分配一个 slot，用于存放响应数据。如果 pool 满（`NoSpace`），会 retry 下一轮。

#### ⑪ `response_submitted_us` — 响应传输提交

```cpp
// completion_poller.cc:286-287  SubmitResponse
record.timing.response_submitted_us = SteadyNowUs();
record.timing.response_submitted_ts_us = UnixNowUs();
```

`PackResponse` 序列化响应到 flag buffer slot，然后 `ExecuteAsync` 提交 RDMA Write（把响应写到 DramStore 的 reply slot 地址）。

#### ⑫ `response_completed_us` — 响应传输完成

```cpp
// completion_poller.cc:334  PollResponseTransfer
record.timing.response_completed_us = SteadyNowUs();
```

`runtime_.transport.GetStatus(response_handle)` 返回 Completed 时记录。随后释放 flag buffer slot。

#### ⑬ `request_completed_us` — 请求完成

```cpp
// completion_poller.cc:51-52  LogRequestDone
record.timing.request_completed_us = SteadyNowUs();
record.timing.request_completed_ts_us = UnixNowUs();
```

在 `LogRequestDone` 函数入口处记录，是整个请求的终点。

### 2.3 完成汇总日志：`LogRequestDone`

定义在 `completion_poller.cc:49-80`，每个请求完成（成功、失败、超时）后打印一条。

```text
[PERF] component=drampool event=request_done request_id=1024 opcode=2 peer=xxx
  batch_size=32 data_bytes=67108864 failed_items=0 status=SUCCESS failed_stage=NONE
  received_ts_us=... worker_started_ts_us=... data_transfer_submitted_ts_us=...
  data_transfer_completed_ts_us=... response_submitted_ts_us=... completed_ts_us=...
  data_tm_execute_async_ts_us=... data_hixl_execute_async_ts_us=...
  data_tm_get_status_ts_us=... data_hixl_query_handle_ts_us=...
  response_tm_execute_async_ts_us=... response_hixl_execute_async_ts_us=...
  response_tm_get_status_ts_us=... response_hixl_query_handle_ts_us=...
  request_queue_us=18 metadata_prepare_us=22 taskworker_prepare_us=45 poller_queue_us=7
  data_transfer_us=610 metadata_settle_us=15 response_slot_wait_us=0
  response_submit_us=19 response_transfer_us=61 data_tm_to_hixl_execute_async_us=...
  data_tm_to_hixl_query_handle_us=... response_tm_to_hixl_execute_async_us=...
  response_tm_to_hixl_query_handle_us=... total_us=752
```

#### 耗时字段计算

| 日志字段 | 计算 | 含义 |
|---|---|---|
| `request_queue_us` | `received_us → worker_started_us` | 请求在 requestQueue 中排队等待 TaskWorker 取出 |
| `metadata_prepare_us` | `metadata_prepare_started_us → metadata_prepare_completed_us` | 遍历 entries 做 StoreBegin / LoadBegin / Exist |
| `taskworker_prepare_us` | `worker_started_us → completion_queued_us` | TaskWorker 从开始到把 CompletionRecord 入队（含 metadata_prepare + 数据传输提交 + 构建 record） |
| `poller_queue_us` | `completion_queued_us → poller_admitted_us` | CompletionRecord 在 completionQueue 中排队等待 CompletionPoller 取出 |
| `data_transfer_us` | `data_transfer_submitted_us → data_transfer_completed_us` | HIXL RDMA Read/Write 实际执行时间 |
| `metadata_settle_us` | `data_transfer_completed_us → metadata_settle_completed_us` | 数据传输完成后做 StoreEnd / LoadEnd / Delete |
| `response_slot_wait_us` | `response_ready_us → response_slot_acquired_us` | 等待 flag buffer pool 分配 slot（pool 满时会卡住） |
| `response_submit_us` | `response_slot_acquired_us → response_submitted_us` | PackResponse + ExecuteAsync 提交响应传输 |
| `response_transfer_us` | `response_submitted_us → response_completed_us` | RDMA Write 响应实际传输时间 |
| `total_us` | `received_us → request_completed_us` | 从 TCP 收到请求到响应传输完成的总耗时 |

#### TransportManager → HIXL 调用字段

下列字段分别为数据传输（`data_`）和响应传输（`response_`）记录。绝对时间使用
Unix epoch 微秒；时间差使用 steady clock 计算。

| 字段后缀 | 含义 |
|---|---|
| `tm_execute_async_ts_us` | 进入 `TransportManager::ExecuteAsync` 的绝对时间 |
| `hixl_execute_async_ts_us` | HIXL worker 实际调用 `engine.TransferAsync` 的绝对时间 |
| `tm_to_hixl_execute_async_us` | 从进入 TransportManager 到实际调用 HIXL ExecuteAsync 的耗时 |
| `tm_get_status_ts_us` | 最终一次进入 `TransportManager::GetStatus` 的绝对时间 |
| `hixl_query_handle_ts_us` | HIXL worker 最终一次实际调用 `engine.GetTransferStatus` 的绝对时间 |
| `tm_to_hixl_query_handle_us` | 从进入 TransportManager 到实际调用 HIXL query handle 的耗时 |

`GetStatus` 在轮询期间可能多次返回 `Waiting`；`request_done` 保存的是使传输进入终态的
最后一次 query。LOOKUP 没有数据传输，因此对应的 `data_*` 字段为 `0`。

> **注意**：`taskworker_prepare_us` 包含了 `metadata_prepare_us`，不是互斥的。`metadata_prepare_us` 是单独拎出来看的子阶段。

#### 通用字段

| 字段 | 含义 |
|---|---|
| `request_id` | DramStore 分配的下游请求 ID |
| `opcode` | 0=LOOKUP, 1=DUMP, 2=LOAD |
| `peer` | DramStore 的 one-sided transport manager ID |
| `batch_size` | 请求中的 entry 数量 |
| `data_bytes` | 数据传输总字节数 |
| `failed_items` | 失败的 entry 数量 |
| `status` | SUCCESS / DATA_TRANSFER_FAILED / ITEM_FAILURE / FLAG_BUFFER_ALLOCATION_FAILED 等 |
| `failed_stage` | 发生失败的阶段，成功时为 NONE |

### 2.4 阶段切换日志

仅在异步状态变化时打印，用于判断未完成请求卡在哪个阶段。

| stage | 代码位置 | 触发时机 |
|---|---|---|
| `REQUEST_RECEIVED` | drampool_server.cc:513 | TCP 收到并解析请求成功 |
| `WORKER_STARTED` | task_worker.cc:74 | TaskWorker 取出请求开始处理 |
| `DATA_TRANSFER_SUBMITTED` | task_worker.cc:197 (DUMP), 293 (LOAD) | HIXL RDMA 传输已提交 |
| `COMPLETION_QUEUED` | task_worker.cc:368 | CompletionRecord 推入 completionQueue |
| `POLLER_ADMITTED` | completion_poller.cc:107 | CompletionPoller 取出 record 加入 pending |
| `DATA_TRANSFER_COMPLETED` | completion_poller.cc:176 (失败), 203 (成功) | HIXL 传输到达终态 |
| `SUBMIT_RESPONSE` | completion_poller.cc:213 | 数据传输完成，准备提交响应 |
| `RESPONSE_TRANSFER_SUBMITTED` | completion_poller.cc:291 | 响应 RDMA Write 已提交 |
| `RESPONSE_TRANSFER_COMPLETED` | completion_poller.cc:336 | 响应 RDMA Write 完成 |

### 2.5 操作差异

| 操作 | 数据传输 | 元数据准备 | 元数据 settle |
|---|---|---|---|
| **DUMP** | RDMA Read（从 DramStore GPU 读数据到 DramPool buffer） | `StoreBegin`：分配 buffer + 创建元数据 | `StoreEnd`（成功）/ `Delete`（失败） |
| **LOAD** | RDMA Write（从 DramPool buffer 写数据到 DramStore GPU） | `LoadBegin`：查找元数据 + pin buffer | `LoadEnd` |
| **LOOKUP** | 无（`data_transfer_us` = 0） | `Exist`：检查 key 是否存在 | 无 |

---

## 3. DramStore Task 级

### 3.1 数据结构：`TaskTiming`

定义在 `task_manager.h:91-99`，承载一个 task 在 TaskManager 内部的时间点。

```cpp
struct TaskTiming {
    std::uint64_t enqueuedUs{0};                    // 任务入队
    std::uint64_t processSubmissionStartedUs{0};    // ProcessSubmission 开始
    std::uint64_t requestsStartedUs{0};             // 子请求开始提交
    std::uint64_t completedUs{0};                   // 任务完成
    std::uint64_t enqueuedTsUs{0};                  // Unix 时间戳
    std::uint64_t requestsStartedTsUs{0};
    std::uint64_t completedTsUs{0};
};
```

### 3.2 打点追踪

#### ① `enqueuedUs` — 任务入队

```cpp
// task_manager.cc:97-98  EnqueueTask
timing.enqueuedUs = SteadyNowUs();
timing.enqueuedTsUs = UnixNowUs();
```

Python 调 `store.load_data()` / `store.dump_data()` 进入 C++ 后，`EnqueueTask` 分配 `taskId`，记录时间，然后把 `Submission` 推入 `submissions_` 队列。

> **注意**：时间记录在 `workMutex_` 加锁之前，所以如果 worker 线程正在持锁处理 completion，等锁耗时会被算进 `queue_us`。

#### ② `processSubmissionStartedUs` — ProcessSubmission 开始

```cpp
// task_manager.cc:237  ProcessSubmission
submission.timing.processSubmissionStartedUs = SteadyNowUs();
```

worker 线程从 `submissions_` 队列取出 `Submission`，开始处理前记录。

#### ③ `requestsStartedUs` — 子请求开始提交

```cpp
// task_manager.cc:273-274  ProcessSubmission
task.timing.requestsStartedUs = SteadyNowUs();
task.timing.requestsStartedTsUs = UnixNowUs();
```

在 `NormalizeTransfer`（展平 entries）+ `BuildRequests`（路由拆分）完成后，开始逐个提交子请求前记录。

`ProcessSubmission` 内部流程：

```
processSubmissionStartedUs
  ├── NormalizeTransfer: 把 TaskDesc 展平为 IoEntry 数组
  ├── BuildRequests: RouteKeys 路由 + 按节点和批次拆分
  ├── 分配 requestId
  └── requestsStartedUs → 逐个 submitRequest 到 NodeScheduler
```

#### ④ `completedUs` — 任务完成

```cpp
// task_manager.cc:337-338  LogTaskDone
timing.completedUs = SteadyNowUs();
timing.completedTsUs = UnixNowUs();
```

所有子请求都返回后（`remainingRequests == 0`），在 `CompleteRequest` 中调用 `LogTaskDone`，入口处记录。

### 3.3 完成汇总日志：`LogTaskDone`

定义在 `task_manager.cc:334-353`，一个 task 的全部子请求完成后打印一条。

```text
[PERF] component=dramstore event=task_done task_id=18 opcode=2 entries=64
  request_count=2 status=SUCCESS status_code=0
  enqueued_ts_us=... requests_started_ts_us=... completed_ts_us=...
  queue_us=20 route_us=35 requests_inflight_us=1800 total_us=1855
```

#### 耗时字段计算

| 日志字段 | 计算 | 含义 |
|---|---|---|
| `queue_us` | `enqueuedUs → processSubmissionStartedUs` | 任务在 submissions_ 队列中等待 worker 取出（含 workMutex_ 等锁时间） |
| `route_us` | `processSubmissionStartedUs → requestsStartedUs` | NormalizeTransfer + RouteKeys + BuildRequests（路由拆分） |
| `requests_inflight_us` | `requestsStartedUs → completedUs` | 从第一个子请求提交到所有子请求全部完成 |
| `total_us` | `enqueuedUs → completedUs` | 端到端总耗时 |

#### 通用字段

| 字段 | 含义 |
|---|---|
| `task_id` | DramStore 分配的任务 ID（单调递增） |
| `opcode` | 0=LOOKUP, 1=DUMP, 2=LOAD |
| `entries` | 展平后的 IoEntry 总数 |
| `request_count` | 拆分后的子请求数量 |
| `status` | SUCCESS / FAILED |
| `status_code` | Status 底层错误码 |

---

## 4. DramStore Request 级

### 4.1 数据结构：`RequestTiming`

定义在 `types.h:101-117`，承载一个子请求在 DramStore 内部的时间点。

```cpp
struct RequestTiming {
    // steady 时间点
    std::uint64_t nodeQueuedUs{0};                       // 进入 NodeActor 队列
    std::uint64_t nodeActorStartedUs{0};                 // StartRequest 开始
    std::uint64_t replySlotAcquiredUs{0};                // reply slot 申请成功
    std::uint64_t requestEncodedUs{0};                   // 请求编码完成
    std::uint64_t controlTransportSubmitStartedUs{0};    // 开始调 submitTransport
    std::uint64_t controlTransportSubmittedUs{0};        // submitTransport 返回
    std::uint64_t controlTransportCompletedUs{0};        // TCP TransmitCompleted 事件
    std::uint64_t replyObservedUs{0};                    // ReplyService 观察到响应
    std::uint64_t replyProcessedUs{0};                   // 响应处理完成
    std::uint64_t completedUs{0};                        // 请求完成

    // Unix 时间戳
    std::uint64_t nodeQueuedTsUs{0};
    std::uint64_t controlTransportSubmittedTsUs{0};
    std::uint64_t controlTransportCompletedTsUs{0};
    std::uint64_t replyObservedTsUs{0};
    std::uint64_t completedTsUs{0};
};
```

### 4.2 打点追踪

#### ① `nodeQueuedUs` — 进入 NodeActor 队列

```cpp
// node_scheduler.cc:119-120  Post
request.timing.nodeQueuedUs = SteadyNowUs();
request.timing.nodeQueuedTsUs = UnixNowUs();
```

`TaskManager` 调 `dependencies_.submitRequest(request)` → `NodeScheduler::Post`，把 request 推入 Runner 的命令队列前记录。

#### ② `nodeActorStartedUs` — StartRequest 开始

```cpp
// node_actor.cc:303  StartRequest
active.request.timing.nodeActorStartedUs = SteadyNowUs();
```

Runner 线程唤醒后，`NodeActor::DispatchPendingRequests` 从 `pendingRequests_` 取出 request，调 `StartRequest` 时记录。

`nodeQueuedUs → nodeActorStartedUs` 就是请求在 NodeActor pending 队列中等待的时间（包含 Runner 线程唤醒延迟 + 容量限制等待）。

#### ③ `replySlotAcquiredUs` — reply slot 申请成功

```cpp
// node_actor.cc:315-328  StartRequest
auto acquired = dependencies_.acquireReplySlot(active.token, active.request.op,
                                               active.request.entries.size());
...
active.replySlot = std::move(acquired).Value();
active.request.timing.replySlotAcquiredUs = SteadyNowUs();
```

从 `ReplyService` 的 host-pinned buffer pool 中分配一个 slot，DramPool 会把响应 RDMA Write 到这个地址。

#### ④ `requestEncodedUs` — 请求编码完成

```cpp
// node_actor.cc:331-343  StartRequest
auto status = EncodeRequest(active.replySlot, active.request.requestId,
                            active.request.op, active.request.entries, payload);
...
active.request.timing.requestEncodedUs = SteadyNowUs();
```

把 `requestId`、`opcode`、`entries`（block key + GPU 地址 + 长度）序列化到 `payload` 字节数组。

#### ⑤ `controlTransportSubmitStartedUs` / `controlTransportSubmittedUs` — TCP 控制消息提交

```cpp
// node_actor.cc:349-352  StartRequest
active.request.timing.controlTransportSubmitStartedUs = SteadyNowUs();
status = dependencies_.submitTransport(command);
active.request.timing.controlTransportSubmittedUs = SteadyNowUs();
active.request.timing.controlTransportSubmittedTsUs = UnixNowUs();
```

`submitTransport` 把编码好的 payload 通过 TCP 发给 DramPool。这是一个同步调用（`control_.Send`），所以 `controlTransportSubmitStartedUs → controlTransportSubmittedUs` 就是 TCP send 的耗时。

#### ⑥ `controlTransportCompletedUs` — TCP TransmitCompleted

```cpp
// node_actor.cc:553-554  Handle(TransmitCompleted)
found->second.request.timing.controlTransportCompletedUs = SteadyNowUs();
found->second.request.timing.controlTransportCompletedTsUs = UnixNowUs();
```

`TransportManagerBackend::Transmit` 返回后，通过 `TransmitCompleted` 事件通知 NodeActor。成功后 request 状态从 `TRANSMITTING` 变为 `INFLIGHT`。

#### ⑦ `replyObservedUs` — ReplyService 观察到响应

```cpp
// node_actor.cc:498-499  Handle(ReplyObserved)
found->second.request.timing.replyObservedUs = SteadyNowUs();
found->second.request.timing.replyObservedTsUs = UnixNowUs();
```

`ReplyService` 轮询 reply slot，检测到 DramPool 通过 RDMA Write 写入了响应数据，解码后发送 `ReplyObserved` 事件给 NodeActor。

#### ⑧ `replyProcessedUs` — 响应处理完成

```cpp
// node_actor.cc:537  Handle(ReplyObserved)
found->second.request.timing.replyProcessedUs = SteadyNowUs();
```

在 `Handle(ReplyObserved)` 末尾，检查 entry results、处理 item failures 后记录。

#### ⑨ `completedUs` — 请求完成

```cpp
// node_actor.cc:131-132  QueueCompletion
request.timing.completedUs = SteadyNowUs();
request.timing.completedTsUs = UnixNowUs();
```

`Handle(ReplyObserved)` 最后调 `Complete(status, entryResults)` → `RetireRequest` → `QueueCompletion`，入口处记录。

### 4.3 完成汇总日志：`QueueCompletion`

定义在 `node_actor.cc:128-165`，每个子请求完成后打印一条。

```text
[PERF] component=dramstore event=request_done task_id=18 request_id=1024 opcode=2
  node_id=1 entries=32 status=SUCCESS status_code=0
  node_queued_ts_us=... transport_submitted_ts_us=... transmit_completed_ts_us=...
  reply_observed_ts_us=... completed_ts_us=...
  node_queue_us=14 reply_slot_wait_us=28 encode_us=12 control_submit_us=45
  control_transfer_us=8 remote_wait_us=765 reply_process_us=8 total_us=860
```

#### 耗时字段计算

| 日志字段 | 计算 | 含义 |
|---|---|---|
| `node_queue_us` | `nodeQueuedUs → nodeActorStartedUs` | 在 NodeActor pending 队列中等待（含 Runner 线程唤醒延迟 + 容量限制等待） |
| `reply_slot_wait_us` | `nodeActorStartedUs → replySlotAcquiredUs` | 从 ReplyService 申请 host-pinned reply slot |
| `encode_us` | `replySlotAcquiredUs → requestEncodedUs` | 序列化 KvRequest 到 payload |
| `control_submit_us` | `controlTransportSubmitStartedUs → controlTransportSubmittedUs` | TCP Send 调用耗时 |
| `control_transfer_us` | `controlTransportSubmittedUs → controlTransportCompletedUs` | TCP 传输完成确认 |
| `remote_wait_us` | `controlTransportCompletedUs → replyObservedUs` | 等 DramPool 处理 + RDMA 响应回写 + ReplyService 轮询 |
| `reply_process_us` | `replyObservedUs → replyProcessedUs` | 解码响应、检查 entry results |
| `total_us` | `nodeQueuedUs → completedUs` | 端到端总耗时 |

#### `remote_wait_us` 的 fallback

```cpp
// node_actor.cc:136-138
const auto remoteWaitStarted = request.timing.controlTransportCompletedUs != 0
                                   ? request.timing.controlTransportCompletedUs
                                   : request.timing.controlTransportSubmittedUs;
```

如果 TCP 传输失败（没收到 `TransmitCompleted`），`controlTransportCompletedUs` 为 0，回退到 `controlTransportSubmittedUs` 作为起点，避免算出离谱的负数或超大值。

#### 通用字段

| 字段 | 含义 |
|---|---|
| `task_id` | 所属 task 的 ID |
| `request_id` | 子请求 ID（单调递增） |
| `opcode` | 0=LOOKUP, 1=DUMP, 2=LOAD |
| `node_id` | 目标 DramPool 节点 ID |
| `entries` | 该子请求包含的 entry 数量 |
| `status` | SUCCESS / FAILED |
| `status_code` | Status 底层错误码 |

### 4.4 阶段切换日志

| stage | 代码位置 | 触发时机 |
|---|---|---|
| `NODE_QUEUED` | node_actor.cc:384 | 请求进入 NodeActor pending 队列 |
| `REQUEST_DISPATCHED` | node_actor.cc:305 | StartRequest 开始处理 |
| `CONTROL_TRANSFER_SUBMITTED` | node_actor.cc:355 | TCP 控制消息已提交 |
| `CONTROL_TRANSFER_COMPLETED` | node_actor.cc:556 | TCP 控制消息传输完成 |
| `REPLY_OBSERVED` | node_actor.cc:501 | ReplyService 观察到 DramPool 响应 |

---

## 5. 跨模块关联分析

### 5.1 通过 `request_id` 关联

DramStore 和 DramPool 使用相同的 `request_id`，可以通过它关联两侧的 `request_done` 日志：

```text
# DramStore 侧
[PERF] component=dramstore event=request_done request_id=1024 ...
  remote_wait_us=765 total_us=860

# DramPool 侧
[PERF] component=drampool event=request_done request_id=1024 ...
  data_transfer_us=610 total_us=752
```

### 5.2 可计算的跨模块指标

| 想分析什么 | 怎么算 |
|---|---|
| DramPool 数据传输耗时 | 直接读 DramPool `data_transfer_us` |
| DramPool 元数据操作耗时 | 直接读 DramPool `metadata_prepare_us` + `metadata_settle_us` |
| DramPool 排队耗时 | 直接读 DramPool `request_queue_us` |
| DramStore 本地调度+准备 | DramStore `node_queue_us` + `reply_slot_wait_us` + `encode_us` |
| TCP 控制消息发送耗时 | 直接读 DramStore `control_submit_us` + `control_transfer_us` |
| 响应回传耗时 | 直接读 DramPool `response_transfer_us` |
| 网络传输 + 轮询开销 | `DramStore remote_wait_us - DramPool total_us - DramPool response_transfer_us` |
| 全链路外围开销 | `DramStore total_us - DramPool total_us` |

### 5.3 限制

- `remote_wait_us` 包含网络传输 + DramPool 处理 + ReplyService 轮询延迟，不能直接解释为网络耗时
- `steady_clock` 只能用于本进程内的时间差，不能跨进程直接比较
- `*_ts_us`（Unix 时间戳）可用于粗粒度关联，但不能用于精确耗时计算（NTP 可能调整）

---

## 6. 请求完整生命周期时间线

```
DramStore TaskManager                    DramStore NodeActor                    DramPool
─────────────────────                    ───────────────────                    ────────
EnqueueTask
  enqueuedUs ──────────────────────────────────────────────────────────────────────────
  │
ProcessSubmission
  processSubmissionStartedUs
  ├── NormalizeTransfer
  ├── BuildRequests (RouteKeys)
  └── requestsStartedUs
      │
      ├── submitRequest → Post
      │     nodeQueuedUs ──────────────────────────────────────────────────────────────
      │     │
      │     │  (Runner 线程唤醒)
      │     │
      │     StartRequest
      │       nodeActorStartedUs
      │       ├── acquireReplySlot → replySlotAcquiredUs
      │       ├── EncodeRequest    → requestEncodedUs
      │       ├── submitTransport  → controlTransportSubmitStartedUs
      │       │                      controlTransportSubmittedUs
      │       │                        │
      │       │                        │  ──── TCP ────→  RequestReceiveLoop
      │       │                        │                    received_us
      │       │                        │                    │
      │       │                        │                    requestQueue_.TryPush
      │       │                        │                    │
      │       │                        │                    ProcessOneRequest
      │       │                        │                      worker_started_us
      │       │                        │                      ├── metadata_prepare_started_us
      │       │                        │                      ├── metadata_prepare_completed_us
      │       │                        │                      ├── ExecuteAsync (HIXL RDMA)
      │       │                        │                      │   data_transfer_submitted_us
      │       │                        │                      └── SubmitCompletion
      │       │                        │                          completion_queued_us
      │       │                        │
      │       │                        │  ← TransmitCompleted
      │       │                        │  controlTransportCompletedUs
      │       │                        │
      │       │                        │                    CompletionPoller
      │       │                        │                      poller_admitted_us
      │       │                        │                      ├── PollDataTransfer
      │       │                        │                      │   data_transfer_completed_us
      │       │                        │                      │   metadata_settle_completed_us
      │       │                        │                      ├── SubmitResponse
      │       │                        │                      │   response_slot_acquired_us
      │       │                        │                      │   response_submitted_us
      │       │                        │                      └── PollResponseTransfer
      │       │                        │                          response_completed_us
      │       │                        │
      │       │                        │  ← RDMA Write (response)
      │       │
      │       ReplyService 轮询
      │         replyObservedUs
      │         replyProcessedUs
      │
      │     QueueCompletion
      │       completedUs
      │       ──→ request_done 日志
      │
  CompleteRequest (remainingRequests == 0)
    completedUs
    ──→ task_done 日志
                                                                 request_completed_us
                                                                 ──→ request_done 日志
```
