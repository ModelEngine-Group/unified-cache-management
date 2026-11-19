# AICPU Batch Send 并发化方案 1（Host 线程池）

## 状态

该方案已完成原型验证，代码已撤回，不再作为当前实现。

历史原型位于个人分支提交 `9986a4b`（`send by sync`）。当前实现改用
[`aicpu_batch_send_async_design.md`](./aicpu_batch_send_async_design.md) 中的单 Host 线程
async-launch + wait 方案。

## 原方案

`AICPUTransProvider::Send` 将输入按 connection group 分组，把每个 group 的完整同步调用
提交到固定 Host 线程池：

```text
worker 1: bind ACL context -> launch group A -> synchronize stream A
worker 2: bind ACL context -> launch group B -> synchronize stream B
worker 3: bind ACL context -> launch group C -> synchronize stream C
caller: wait all futures -> merge statuses -> return
```

原型保持 `TransProvider::Send` 的同步语义不变，并通过
`transport.aicpu_send_worker_num` 控制 worker 数。任务持有 connection shared ownership，
每个 connection 继续由 `sendMu` 串行化，状态由调用线程统一归并。

## 验证结果与放弃原因

在确认新 SO 生效并将 Host worker 数设为 16 后，设备侧 `HixlBatchSend` 出现 AICPU
exception：

```text
aclrtSynchronizeStreamWithTimeout failed ret=507018
kernelName=HixlBatchSend errorCode=0x2a
```

随后出现 `HcommChannelUpdateStagedLocalMemInfo` 和 `HcommChannelDestroy` 的次生错误。该结果
不能单独证明设备侧完全不支持并发，但说明“多个 Host worker 同时进入 ACL/HIXL launch 和
synchronize”引入了额外的不确定性，包括 ACL current context、共享 function handle、Runtime
线程安全性及设备侧并发上限。

因此当前版本移除以下内容：

- Host send thread pool 及 future；
- `aicpu_send_worker_num` 配置；
- thread-pool 单元测试和构建项；
- worker 内独立绑定 ACL context 的路径。

## 保留结论

- 不同 connection 的 stream、HCOMM channel/thread 和 mapped workspace 必须保持独立。
- 某个 group 失败后仍要等待所有已提交 operation 到达终态。
- `Send` 对外继续保持同步，远端 KV 完成仍由现有 flag buffer/CompletionLoop 处理。
- 后续并发度应控制“设备侧 in-flight kernel 数”，而不是增加 Host launch 线程数。
- 并发档位应按 1、2、4、8、16 逐级验证；若低档位重现 `507018`，需确认 HIXL/HCOMM 的
  设备侧可重入约束。
