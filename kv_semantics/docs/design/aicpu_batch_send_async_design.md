# AICPU 跨 Send 异步 Launch/Wait 方案

## 1. 目标

实际运行中一次 `Send` 的 `groups.size()` 通常为 1，因此在单次 `Send` 内并行 connection group 不能解决主要瓶颈。目标调整为：让同一个 transport worker 连续发起多个 `Send`，把多个 `HixlBatchSend` 排入 ACL stream 后，再统一进入等待阶段。

本方案不创建 Host 线程池。所有 ACL/HIXL launch 和 wait 仍由原来的单个 transport worker 顺序调用，以规避多个 Host 线程同时操作 ACL context、stream 和 HIXL function handle 的风险。

## 2. 接口与兼容性

原同步接口和实现完整保留：

```cpp
std::vector<Status> TransProvider::Send(...);
Status AICPUTransProvider::Impl::LaunchBatchSendLocked(...);
```

新增 provider 两阶段接口：

```cpp
class TransProvider::SendOperation;
using SendOperationPtr = std::unique_ptr<SendOperation>;

std::vector<Status> AsyncSend(..., SendOperationPtr& operation);
std::vector<Status> WaitSend(SendOperationPtr& operation);
```

- `AsyncSend` 只准备参数并调用 `aclrtLaunchKernelWithConfig`，成功后返回 move-only operation。
- `WaitSend` 调用 `aclrtSynchronizeStreamWithTimeout`、读取 HIXL status array，并消费 operation。
- launch 成功只表示任务已进入 stream，不表示设备执行成功；AICPU exception 和 HIXL entry error 仍在 `WaitSend` 返回。
- 不支持异步发送的 provider 保持原同步行为；配置为 async 时 transport 初始化会明确失败。

## 3. Transport 调度时序

`aicpu_send_mode=async` 时，transport worker 从现有 SPSC queue 中最多取 `max(1, TransportConfig.maxInflightTasks)` 个任务，复用 Transport 配置，不再设置独立的 AICPU 窗口：

```text
task A: prepare -> AsyncSend -> operation A
task B: prepare -> AsyncSend -> operation B
task C: prepare -> AsyncSend -> operation C

task A: WaitSend -> launch result/status array
task B: WaitSend -> launch result/status array
task C: WaitSend -> launch result/status array

remote completion仍由 flag buffer + CompletionLoop 处理
```

这里的并发窗口是“transport task / Send 数”，不是单次 `Send` 的 connection group 数，也不是线程数。窗口为 1 时用于功能回归；建议按 2、4、8 逐步验证吞吐和稳定性。

如果 worker 取到第一个任务时队列里暂时没有更多任务，本轮只发一个，不额外等待凑批，避免人为增加空载延迟。

## 4. Workspace 和资源生命周期

同步路径继续复用 `ConnectionRecord::mappedBatchWorkspace`。

异步路径不能让多个 in-flight operation 同时复用同一个 workspace，因为同一 connection 上的下一个 `AsyncSend` 会覆盖前一个 kernel 尚在访问的 `io_batches` 和 `status_array`。因此每个未完成的 operation/ticket 独占：

- mapped batch workspace；
- `shared_ptr<ConnectionRecord>`；
- batch 数量和原始下标映射；
- 绝对 deadline。

connection 维护异步 workspace 池。`AsyncSend` 优先取能够容纳当前 batch 的最小空闲 workspace；没有足够容量时扩容最大的空闲 workspace，池为空时才申请并注册新的 mapped 内存。`WaitSend` 在 stream 同步成功后将 workspace 归还池中，所以稳定状态下 workspace 数量约等于该 connection 实际达到的最大 in-flight 数，而不是每次 Send 都执行 `aclrtMallocHost/aclrtHostRegister/aclrtHostUnregister/aclrtFreeHost`。如果 stream 同步失败，设备是否仍在访问内存无法确定，该 workspace 只进入隔离区，不再复用，直到 connection 清理。

connection 记录未完成异步 send 数量。`DeleteConnections` 在数量非零时返回 `RESOURCE_BUSY`，防止 stream/channel/thread 被提前销毁。MR register/bind/unregister 与异步 launch 共用资源门禁；遇到已 launch 的 ticket 时等待其 drain，而不是把启动阶段正常的并发注册误报成失败。若 operation 无法 drain，则返回连接错误并保留隔离资源。调用方仍必须在释放 send/flag buffer 或销毁 provider 之前消费所有 operation。

## 5. Completion、取消和超时

- `CompletionLoop` 只有在 `sendReturned=true` 后才轮询远端 flag，避免 kernel 尚未 drain 时释放 buffer。
- 异步 send 尚未完成时收到 cancel，只记录 `cancelRequested`；worker 仍先调用 `WaitSend`，随后再释放 sub-batch 资源并完成取消。
- 每个 operation 使用 launch 时计算的绝对 deadline，wait 使用剩余时间，避免重复获得完整 timeout。
- 某个 operation wait 失败后仍继续 drain 同一批中其他已经 launch 的 operation。
- AICPU exception/timeout 后 connection 标记为 faulted，禁止继续异步 launch，交由连接恢复或 shutdown 处理。

## 6. 配置传递

推理进程 YAML 配置：

```yaml
asu_aicpu_send_mode: "async"
asu_transport_max_inflight_tasks: 1024
```

传递链路：

```text
ucm_config_asu_aicpu.yaml
  -> AsuStore::ParseConfig
  -> AsuStore::BuildTransportConfig
  -> TransportConfig.attrs (send mode) / TransportConfig.maxInflightTasks (窗口)
  -> AsuTransportImpl (选择同步/异步调度)
  -> AICPUTransProvider::AsyncSend/WaitSend
```

kv-test 的对应配置为 `transport.aicpu_send_mode=async` 和 `transport.max_inflight_tasks=1024`。窗口默认 1024；旧的独立 AICPU 窗口配置已移除，原先填 2 的配置需要将 Transport 限制设为 2 才能保留实验规模。原生 Transport 配置为 0 时，发送窗口取 1，以保证 worker 推进。

该配置同时决定 Transport 执行队列容量和每轮异步下发上限，不是所有未完成业务请求的严格总上限。Client 的 `asu_client_max_inflight_tasks` 是另一层队列配置，不控制此窗口。同步模式仍调用原 `Send`/`LaunchBatchSendLocked`。

## 7. 验证建议

1. `send_mode=sync` 对照原有功能和性能。
2. `send_mode=async, max_inflight=1` 验证两阶段接口结果与同步路径一致。
3. 在足够并发的上层 workload 下测试 2、4、8；确认日志中的每轮 `tasks/launched` 大于 1。
4. 同时记录 launch、wait、端到端耗时和吞吐；仅看单请求延迟可能无法反映排队收益。
5. 观察 AICPU/HCCP 日志、SQ 深度、timeout 和 `507018`。若 2 路即出现设备异常，不再扩大窗口，并确认 HIXL/HCOMM 对同 stream 多 kernel 排队的约束。
6. 覆盖 launch 失败、wait 失败、cancel、timeout、shutdown 和 connection delete 的资源生命周期测试。

## 8. 完成事件与失败边界

每个异步 ticket 在自己的 HixlBatchSend 后记录一个 ACL event。WaitSend 等待该 event，后续提交的 kernel 不会延长前一个 ticket 的等待。event 记录失败时，连接停止接受新发送，已有 ticket 回退到 stream 同步以确认设备完成。

WaitSend 只有在 operation 被清空时才允许调用方应用最终状态和释放缓冲区。event/stream 同步失败或 ACL context 绑定失败时，operation、ticket 计数、workspace、发送槽位及响应槽位均保留；worker 间隔 10 ms 重试等待，完成线程和取消操作不能提前释放这些资源。MR 注销继续受未完成 ticket 计数保护。

本实现不自动 abort 整个设备，以免影响共用设备的其他任务。若设备永久故障且无法确认完成，worker 和 Shutdown 会继续等待，必须由运维停止进程并恢复设备；不能将超时当成 DMA 已停止的证明。发送数据内存的所有者仍须在业务任务真正完成前保持内存有效。

## 9. 已淘汰实现

- Host thread pool 并行调用完整同步 launch 函数：已淘汰。
- 仅在一次 `Send` 内对 connection group 做 fan-out/fan-in：已淘汰，因为 `groups.size()==1` 是正常主路径，无法提升多个 Send 串行提交的性能。
