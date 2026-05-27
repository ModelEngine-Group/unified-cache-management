# ASU Transport 业务处理逻辑

## 1. 整体架构

ASU Transport 是一个高性能的 KV 存储传输层，采用多线程异步架构，通过 TransProvider 抽象层支持多种底层传输实现（当前为 AICPU + HCOMM）。

### 1.1 线程与队列架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户线程 (多个)                                 │
│                                                                             │
│  SubmitAsync() ──┐                                                          │
│  Wait()          │ 操作 TransportTaskManager                                │
│  Check()         │ 等待 ctx->cv (protected by ctx->waitMu)                 │
│  Cancel()        │                                                          │
└──────────────────┼──────────────────────────────────────────────────────────┘
                   │
                   │ lock(producerMu_)
                   ▼
        ┌──────────────────────┐
        │   executeQueue       │  SPSC 无锁队列
        │   (用户线程 → Worker) │  容量: maxInflightTasks + 1
        └──────────┬───────────┘
                   │ TryPop()
                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WorkerLoop 线程                                    │
│                                                                             │
│  CompleteTask(ctx):                                                         │
│    1. ConnectionManager.SelectConnection()                                  │
│    2. TransProvider.Send({batch})                                           │
│    3. SubmitPending({ctx, channel, flagBuffer, deadline})                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   pendingQueue       │  SPSC 无锁队列
        │  (Worker → Poller)   │  容量: maxInflightTasks
        └──────────┬───────────┘
                   │ TryPop() 批量取出
                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PollLoop 线程                                      │
│                                                                             │
│  循环处理:                                                                   │
│    1. 读取 flagBuffer 检查任务是否完成                                        │
│    2. *flagBuffer != 0 → Finalize(req, OK)                                  │
│       超时 → ReportFailure + Finalize(req, TIMEOUT)                         │
│    3. Finalize: lock(ctx->waitMu), state=COMPLETED, cv.notify_all()        │
│    4. ReleaseInflight(), 释放 flagBuffer slot                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         RecoverLoop 线程 (独立运行)                          │
│                                                                             │
│  每 100ms 执行:                                                              │
│    1. lock(drainMu_) → swap(drainList_, to_recover)                        │
│    2. 对每个 channel:                                                        │
│       - createFn_() 创建新连接                                               │
│       - lock(structureMu_) → RemoveChannel + AddChannel                    │
│       - cacheDirty = true                                                   │
│    注意: 仅当 inflight==0 时回收，无超时强制回收                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AsuTransportImpl                                  │
│                                                                         │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐  │
│  │TransportTaskManager │    │      ConnectionManager               │  │
│  │                     │    │                                      │  │
│  │ - tasks_ (map)      │    │  ┌────────────────────────────┐    │  │
│  │ - mutex_            │    │  │ groups_ (vector<Group>)    │    │  │
│  │                     │    │  │   └─ channels (shared_ptr) │    │  │
│  │ 管理所有             │    │  │      └─ handle_ (void*)    │    │  │
│  │ TransportTaskContext │    │  │                            │    │  │
│  └─────────────────────┘    │  │ channelCache_ (shared_ptr) │    │  │
│                             │  │ drainList_ (shared_ptr)    │    │  │
│  ┌─────────────────────┐    │  │                            │    │  │
│  │ CompletionPoller    │    │  │ 锁:                        │    │  │
│  │                     │    │  │ - structureMu_ (shared)    │    │  │
│  │ - pendingQueue_     │    │  │ - drainMu_ (shared)        │    │  │
│  │ - pollerThread_     │    │  └────────────────────────────┘    │  │
│  │ - reportFailureFn_  │    │                                      │  │
│  │ - releaseFlagBufFn_ │    └──────────────────────────────────────┘  │
│  └─────────────────────┘                                              │
│                                                                         │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐  │
│  │   executeQueue_     │    │         其他成员                      │  │
│  │   (SPSC)            │    │                                      │  │
│  └─────────────────────┘    │  - transProvider_ (TransProvider)    │  │
│                             │  - producerMu_ (保护 Push)           │  │
│  ┌─────────────────────┐    │  - worker_ (WorkerLoop 线程)         │  │
│  │ flagBuffer 内存池    │    │  - config_ (配置)                    │  │
│  │ - flagBufferPool_   │    │  - GetTimeoutMs(opType)             │  │
│  │ - freeFlagSlots_    │    └──────────────────────────────────────┘  │
│  │ - flagBufferMu_     │                                              │
│  └─────────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 TransProvider 抽象层

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TransProvider (抽象基类)                           │
│                                                                     │
│  using ConnectionHandle = void*;                                    │
│  using ThreadHandle = void*;                                        │
│  using MemHandle = void*;                                           │
│                                                                     │
│  virtual CreateConnection(localIp, remoteIp, port, qpNum, ...)     │
│  virtual DeleteConnections(handles)                                 │
│  virtual Send(ioBatches, kernelCount, quietCount)                   │
│  virtual RegisterMemory(handle, memDescs, memHandles)               │
│  virtual UnregisterMemory(unregDescs)                               │
│  virtual AllocThread(threadNum, notifyNumPerThread, threads)        │
│  virtual FreeThread(threads)                                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                AICPUTransProvider (AICPU + HCOMM 实现)               │
│                                                                     │
│  内部结构:                                                            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ LinkContext:  phyDev, channel, thread, remoteIp, remotePort  │  │
│  │ EndpointContext: endpoint (void*), refCount                   │  │
│  │ endpointMap_: phyDev → EndpointContext (按物理设备共享)         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  引擎: COMM_ENGINE_AICPU_TS                                         │
│  端点: ENDPOINT_LOC_TYPE_DEVICE                                     │
│  Socket: HCOMM_SOCKET_ROLE_RESERVED                                 │
│  notifyNum: 0 (AICPU_TS 不需要 notify)                              │
│  exchangeAllMems: true (交换所有注册内存)                              │
│                                                                     │
│  Send: 仅设置 flagBuffer，实际数据传输由 AICPU kernel 完成            │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 锁与同步机制

| 锁 | 保护对象 | 使用线程 | 类型 |
|---|---------|---------|------|
| `producerMu_` | executeQueue Push | 用户线程 | std::mutex |
| `ctx->waitMu` | ctx 状态、finalStatus | 用户/Worker/Poller | std::mutex |
| `structureMu_` | groups_, channelCache_ | Worker/Recover | std::shared_mutex |
| `drainMu_` | drainList_ | Worker/Poller/Recover | std::shared_mutex |
| `TaskManager::mutex_` | tasks_ map | 所有线程 | std::mutex |
| `flagBufferMu_` | flagBuffer 内存池分配 | 用户/Poller | std::mutex |
| `AICPUTransProvider::mutex_` | linkContexts_, endpointMap_ | 所有线程 | std::mutex |

### 1.5 数据流向

```
用户请求
    │
    ├─► 分配 flagBuffer slot (flagBufferMu_)
    │
    ├─► TransportTaskManager.Submit() ──► 分配 TaskId, 存储 ctx
    │
    ├─► executeQueue.Push(ctx) ──► WorkerLoop.TryPop()
    │                                    │
    │                                    ├─► ConnectionManager.SelectConnection()
    │                                    │       │
    │                                    │       ├─► RebuildChannelCache() (if dirty)
    │                                    │       └─► 返回 shared_ptr<Channel>
    │                                    │
    │                                    ├─► TransProvider.Send({batch})
    │                                    │       └─► 设置 flagBuffer = 1
    │                                    │
    │                                    └─► pendingQueue.Push({ctx, channel, flagBuffer, deadline})
    │                                              │
    │                                              └─► PollLoop.TryPop() 批量
    │                                                      │
    │                                                      ├─► 读取 *flagBuffer
    │                                                      │
    │                                                      └─► Finalize()
    │                                                              │
    │                                                              ├─► lock(ctx->waitMu)
    │                                                              ├─► state = COMPLETED
    │                                                              ├─► cv.notify_all()
    │                                                              ├─► ReleaseInflight()
    │                                                              └─► 释放 flagBuffer slot
    │
    └─► Wait(taskId) ──► cv.wait() ◄────────────────────────────────┘
                            │
                            └─► 返回 TaskResult


故障恢复 (异步):
    PollLoop/WorkerLoop ──► ReportFailure(channel)
                                │
                                ├─► errorCount++ (atomic)
                                ├─► MarkForDrain() (CAS: ACTIVE→DRAINING)
                                └─► drainList_.push_back(channel)
                                         │
                                         └─► RecoverLoop (每100ms)
                                                 │
                                                 ├─► 仅当 inflight==0 时回收
                                                 ├─► createFn_() 创建新连接
                                                 ├─► RemoveChannel(old)
                                                 ├─► AddChannel(new)
                                                 └─► cacheDirty = true
```

## 2. 核心组件

### 2.1 AsuTransportImpl

传输层主入口，协调各组件工作。

**职责**：
- 接收用户请求（Load/Store/Query/Delete）
- 管理任务生命周期
- 管理 flagBuffer 内存池（NPU 注册内存，初始化时申请，任务完成时释放 slot）
- 协调 WorkerLoop 和 CompletionPoller
- 提供同步/异步接口

**关键方法**：
```cpp
Status LoadAsync(entries, taskId)      // 异步加载
Status StoreAsync(entries, taskId)     // 异步存储
Status QueryAsync(keys, taskId)        // 异步查询
Status Wait(taskId, timeout, result)   // 等待任务完成
Status Check(taskId, result)           // 检查任务状态
uint64_t GetTimeoutMs(opType)          // 根据操作类型获取超时时间
```

### 2.2 TransportTaskManager

管理所有传输任务的生命周期。

**职责**：
- 分配 TaskId
- 存储 TransportTaskContext
- 提供任务查询接口

**任务状态机**：
```
PENDING → INFLIGHT → COMPLETED
   ↓         ↓
FAILED    FAILED
   ↓
CANCELED
```

**各状态作用**：
- **PENDING**：任务已提交但尚未发送。CompleteTask 检查此状态，若已被 Cancel 则跳过发送。
- **INFLIGHT**：任务已发送到网络，等待硬件完成。用户可在此状态调用 Cancel。
- **COMPLETED/FAILED/CANCELED**：终态，Done() 返回 true，唤醒等待线程。

### 2.3 ConnectionManager

管理所有网络连接，提供连接选择、故障检测和自动恢复。

**职责**：
- 管理 ConnectionGroup（连接组）
- 提供连接选择策略（Round Robin / Least Loaded）
- 检测连接故障并触发恢复
- 维护 channelCache 加速连接选择

**关键组件**：
- `ConnectionGroup`：管理同一 endpoint 的多个 channel
- `ConnectionChannel`：单个连接通道，持有 `TransProvider::ConnectionHandle` (void*)
- `channelCache`：活跃 channel 的扁平缓存
- `drainList`：待恢复的 channel 列表

### 2.4 CompletionPoller

轮询任务完成状态，通知等待线程。

**职责**：
- 轮询 flagBuffer 检查任务是否完成（替代 HcommWaitCompletion）
- 任务完成时调用 Finalize
- 超时触发 ReportFailure
- 使用 SPSC 队列接收 PendingRequest
- 任务完成后释放 flagBuffer slot

**关键方法**：
```cpp
void SubmitPending(PendingRequest)  // WorkerLoop 提交待轮询请求
void PollLoop()                      // 轮询循环
void Finalize(req, status)          // 完成任务，通知等待线程，释放 flagBuffer
```

### 2.5 TransProvider / AICPUTransProvider

传输提供者抽象层，隔离底层通信实现。

**TransProvider（基类）**：
- 定义通用接口：CreateConnection、DeleteConnections、Send、RegisterMemory、UnregisterMemory、AllocThread、FreeThread
- 使用 void* 作为通用句柄类型

**AICPUTransProvider（子类）**：
- 基于 HCOMM 实现 AICPU 传输
- 按物理设备（phyDev）共享 endpoint
- 使用 COMM_ENGINE_AICPU_TS 引擎
- Send 仅设置 flagBuffer，实际数据传输由 AICPU kernel 完成

## 3. 线程模型

系统包含 4 个主要线程，通过 2 个 SPSC 队列通信：

```
用户线程 ──(executeQueue)──> WorkerLoop ──(pendingQueue)──> PollLoop
                                                              │
                                                              ▼
                                                         用户线程 (Wait)

RecoverLoop (独立运行，定期检查 drainList)
```

### 3.1 用户线程

- 调用 `SubmitAsync` 提交任务（分配 flagBuffer slot）
- 调用 `Wait` 等待任务完成
- 通过 `ctx->cv` 被唤醒

### 3.2 WorkerLoop

- 从 `executeQueue` 消费任务
- 调用 `CompleteTask` 处理任务
- 选择 channel，通过 TransProvider.Send 发送
- 成功后提交到 `pendingQueue`

### 3.3 PollLoop

- 从 `pendingQueue` 消费 PendingRequest
- 轮询 flagBuffer 检查完成状态
- 完成时调用 `Finalize`，唤醒用户线程，释放 flagBuffer slot
- 超时时触发 `ReportFailure`

### 3.4 RecoverLoop

- 定期检查 `drainList`
- 仅当 inflight==0 时回收（无超时强制回收）
- 移除故障 channel，创建新 channel
- 更新 `channelCache`

## 4. 任务处理流程

### 4.1 任务提交流程

```
用户调用 LoadAsync/StoreAsync/QueryAsync
    ↓
分配 flagBuffer slot (flagBufferMu_)
    ↓
初始化 flagBufferPool[slot] = 0
    ↓
ctx->flagBuffer = &flagBufferPool[slot]
    ↓
TaskManager.Submit() 分配 TaskId
    ↓
executeQueue.TryPush(ctx)
    ↓
返回 TaskId 给用户
```

### 4.2 任务执行流程 (WorkerLoop)

```
CompleteTask(ctx):
    ↓
SelectConnection() 选择 channel
    ↓
检查 ctx->state == PENDING
    ↓
TransProvider.Send({batch})
    ├─ 失败 → ReleaseInflight, ReportFailure, 重试 (最多2次)
    └─ 成功 ↓
         ↓
    CAS: PENDING → INFLIGHT
    ├─ 失败 → ReleaseInflight, return (任务已被取消)
    └─ 成功 ↓
         ↓
    timeoutMs = GetTimeoutMs(ctx->opType)
    SubmitPending({ctx, channel, flagBuffer, deadlineMs})
    ↓
    return
```

### 4.3 任务完成流程 (PollLoop)

```
PollLoop:
    ↓
从 pendingQueue 批量取出 PendingRequest
    ↓
对每个 request:
    读取 *flagBuffer
    ├─ != 0 (完成) → Finalize(req, OK)
    ├─ 超时 → ReportFailure, Finalize(req, TIMEOUT)
    └─ == 0 (未完成) → 放回 carry 列表

Finalize(req, status):
    ↓
    lock(ctx->waitMu)
    ↓
    检查 state != CANCELED
    ↓
    设置 finalStatus
    ↓
    state = COMPLETED/FAILED
    ↓
    释放 flagBuffer slot (releaseFlagBufferFn_)
    ↓
    cv.notify_all()  // 唤醒用户线程
    ↓
    ReleaseInflight()
```

### 4.4 用户等待流程

```
Wait(taskId, timeout):
    ↓
TaskManager.Get(taskId) 获取 ctx
    ↓
lock(ctx->waitMu)
    ↓
cv.wait_for(timeout, [ctx]{ return ctx->Done(); })
    ↓
BuildResult(ctx, result)
    ↓
unlock
    ↓
TaskManager.Remove(taskId)
    ↓
返回 result
```

## 5. 连接管理

### 5.1 连接选择策略

**Round Robin**：
```cpp
idx = rrIndex_.fetch_add(1)
start = idx % total
for i in [0, total):
    pos = (start + i) % total
    channel = channelCache_[pos]
    if channel->state == ACTIVE && inflight < MAX:
        IncrementInflight()
        return channel
```

**Least Loaded**：
```cpp
min_inflight = MAX
for channel in channelCache_:
    if channel->state == ACTIVE && inflight < min_inflight:
        min_inflight = inflight
        selected = channel
        if min_inflight == 0: break
IncrementInflight(selected)
return selected
```

### 5.2 故障检测与恢复

**故障检测**：
```
ReportFailure(channel):
    ↓
errorCount = FetchAddErrorCount(1)
    ↓
if errorCount + 1 < THRESHOLD (2):
    return  // 未达到阈值
    ↓
MarkForDrain()  // CAS: ACTIVE → DRAINING
    ↓
cacheDirty = true
    ↓
drainList.push_back(channel)
```

**故障恢复 (RecoverLoop)**：
```
每 100ms 执行一次:
    ↓
swap(drainList_, to_recover)
    ↓
对每个 channel (仅当 inflight==0 时回收):
    createFn_(endpoint, 1) 创建新连接
    ├─ 失败 → 放回 drainList
    └─ 成功 ↓
         ↓
    RemoveChannel(old_channel)
    AddChannel(new_channel)
    cacheDirty = true
```

### 5.3 channelCache 重建

```
RebuildChannelCache():
    ↓
if !cacheDirty: return
    ↓
lock(structureMu_, shared)
    ↓
遍历所有 groups 和 channels
    ↓
过滤 state == ACTIVE 的 channel
    ↓
channelCache_ = new_cache
    ↓
cacheDirty = false
```

### 5.4 Endpoint 共享管理 (AICPUTransProvider)

```
同一物理设备 (phyDev) 共享一个 endpoint:

endpointMap_[phyDev] = { endpoint, refCount }

CreateConnection(phyDev=0):
    GetOrCreateEndpoint(0) → refCount++
    HcommChannelCreate(endpoint, AICPU_TS, ...)
    HcommThreadAlloc(AICPU_TS, ...)

DeleteConnection(handle):
    HcommThreadFree(...)
    HcommChannelDestroy(...)
    ReleaseEndpoint(0) → refCount--
       └─ refCount == 0 时销毁 endpoint
```

## 6. 智能指针生命周期管理

### 6.1 ConnectionChannel

**持有者**：
- `ConnectionGroup::channels` (shared_ptr)
- `ConnectionManager::channelCache_` (shared_ptr)
- `ConnectionManager::drainList_` (shared_ptr)
- `PendingRequest::channel` (shared_ptr)
- `CompleteTask` 局部变量 (shared_ptr)

**生命周期**：
```
创建: AddChannel() 创建 shared_ptr
销毁: 所有引用释放后自动析构
      析构时释放 handle_ (TransProvider::ConnectionHandle = void*)
```

### 6.2 TransProvider 连接资源

**生命周期**：
```
创建: AICPUTransProvider.CreateConnection()
      - GetOrCreateEndpoint(phyDev) 共享 endpoint
      - HcommChannelCreate() 创建 channel
      - HcommThreadAlloc() 分配线程

销毁: AICPUTransProvider.DeleteConnections()
      - HcommThreadFree() 释放线程
      - HcommChannelDestroy() 销毁 channel
      - ReleaseEndpoint(phyDev) 减少引用计数
        └─ refCount == 0 时 HcommEndpointDestroy()
```

### 6.3 Shutdown 清理顺序

```
AsuTransportImpl::Shutdown():
    ↓
1. 取消所有未完成任务 (state=CANCELED, cv.notify_all)
2. stop_ = true → WorkerLoop 退出
3. worker_.join()
4. completionPoller_.Stop()
5. 清理 taskManager
6. connManager_->Shutdown()
7. 释放 flagBuffer 内存 (UnregisterMemory)

ConnectionManager::Shutdown():
    ↓
1. channelCache_.clear()
2. drainList_.clear()
3. groups_.clear()  // 最后销毁 ConnectionGroup
```

## 7. 错误处理

### 7.1 任务级错误

| 错误类型 | 触发条件 | 处理方式 |
|---------|---------|---------|
| NO_ACTIVE_CONNECTION | 无法选择可用 channel | 设置 finalStatus，唤醒等待线程 |
| CONNECTION_ERROR | TransProvider.Send 失败 | ReleaseInflight，ReportFailure，重试 |
| TIMEOUT | flagBuffer 轮询超时 | ReportFailure，Finalize(TIMEOUT) |
| CANCELED | 用户调用 Cancel | 设置 state=CANCELED，唤醒等待线程 |
| RESOURCE_BUSY | flagBuffer slot 耗尽或队列满 | 返回错误给用户 |

### 7.2 连接级错误

| 错误类型 | 触发条件 | 处理方式 |
|---------|---------|---------|
| 单次发送失败 | TransProvider.Send 返回失败 | ReportFailure，重试其他 channel |
| 累计故障 | errorCount >= 2 | MarkForDrain，加入 drainList |
| 恢复失败 | createFn_ 返回空 | 保留在 drainList，下次重试 |

## 8. 性能优化

### 8.1 无锁设计

- `executeQueue` / `pendingQueue`：SPSC 无锁队列
- `inflightCount` / `errorCount`：原子操作
- `channelCache`：仅 WorkerLoop 访问，无需加锁
- `cacheDirty`：原子标志，避免不必要的重建

### 8.2 缓存优化

- `ConnectionChannel` 热路径字段 cache line 对齐
- `channelCache` 扁平化存储，提高缓存命中率
- `PendingRequest` 使用 carry 列表避免重复入队

### 8.3 批量处理

- PollLoop 批量取出 PendingRequest
- 减少队列操作次数
- 提高 CPU 缓存利用率

### 8.4 Endpoint 共享

- 同一物理设备共享一个 endpoint
- 避免端口冲突和资源浪费
- 内存注册一次，所有 channel 共享

## 9. 文件结构

```
trans/src/
├── trans_provider.h              # TransProvider 抽象基类
├── aicpu_trans_provider.h/cpp    # AICPU + HCOMM 实现
├── asu_transport_impl.h/cpp      # 传输层主实现
├── connection_manager.h/cpp      # 连接管理
├── connection_internal.h/cpp     # ConnectionChannel/ConnectionGroup
├── completion_poller.h/cpp       # 完成轮询
├── transport_task_manager.h/cpp  # 任务管理
├── transport_config_parser.h/cpp # 配置解析
├── sqe.h/cpp                     # SQE 打包
└── link_proto.h/cpp              # 链路协议

test/case/
├── connection_test.cc            # 连接相关测试 (合并后)
├── asu_smoke_test.cc             # 冒烟测试
├── test_helper.h                 # 测试辅助
├── sqe_pack_test.cc              # SQE 打包测试
└── link_proto_pack_test.cc       # 链路协议测试
```

## 10. 测试覆盖

### 10.1 单元测试

- `ConnectionManagerTest`：连接管理基础功能（AddGroup、SelectConnection、ReportFailure、RecoverLoop、Shutdown）
- `ConnectionTransportTest`：传输层端到端功能（Init/Shutdown、LoadAsync、QueryAsync、Check、Wait、MultipleTasksSequential）
- `ConnectionConcurrentTest`：并发场景验证（InflightConsistency、MarkForDrainCAS、ConcurrentSubmit、ConcurrentDrain、ConcurrentRecovery）
- `AsuSmokeTest`：端到端冒烟测试（ClientAsyncTasks、ConcurrentAll8Interfaces、SequentialDrain、DrainUnderHeavyLoad）
- `AsuClientImplTest`：客户端层测试（42 个用例）
- `ViewServerTest`：视图服务测试

### 10.2 关键验证点

- 任务状态机正确性
- 连接选择策略公平性
- 故障检测与恢复及时性
- 智能指针生命周期安全性
- 多线程并发安全性
- Shutdown 清理完整性
- flagBuffer 内存池分配与释放

## 11. 配置参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| queryQpNum | 1 | Query 操作的 QP 数量 |
| loadQpNum | 2 | Load 操作的 QP 数量 |
| storeQpNum | 1 | Store 操作的 QP 数量 |
| maxInflightTasks | 64 | 最大并发任务数（同时决定 flagBuffer 池大小） |
| queryTimeoutMs | 5000 | Query 超时时间 |
| loadTimeoutMs | 5000 | Load 超时时间 |
| storeTimeoutMs | 5000 | Store 超时时间 |
| kFailureThreshold | 2 | 触发 drain 的故障阈值 |
| kRecoverIntervalMs | 100 | RecoverLoop 检查间隔 |
| kMaxInflightPerChannel | 256 | 单 channel 最大 inflight 数 |
