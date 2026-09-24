# 存储健康检查与熔断

推理服务能够响应，并不代表缓存后端可用。Store 健康状态说明 Pipeline 阶段是否接收新的缓存操作，不能证明某个 KV 块存在、模型回答正确，或部署达到了延迟目标。

## 后端故障时会发生什么

启用健康检查后，Pipeline 为每个加载的阶段包装 `HealthBreakerStore`。包装器初始允许操作，并保留一个滚动探测结果窗口。默认情况下，窗口内出现两次失败就阻止新的操作；恢复则要求完整窗口中的八次结果全部成功。

Posix 和 Mooncake 还会根据实际 I/O 失败触发被动熔断，与主动探测独立统计。恢复须等待冷却结束；恢复后短期内再次故障会延长冷却时间，持续稳定后重置退避。日志记录包含失败的被动窗口，以及探测已正常但仍在冷却时的剩余时间。

| 被阻止期间的操作 | 结果 |
| --- | --- |
| `Lookup` | 返回所有请求块均未命中 |
| `LookupOnPrefix` / `LookupOnReverse` | 返回未命中（`-1`） |
| `Prefetch` | 不提交新的预取 |
| `Load` / `Dump` | 返回 `StoreUnhealthy` |
| 已有任务的 `Check` / `Wait` | 继续转发给底层 Store |

阻止新操作不会取消已经提交的传输。缓存未命中后重计算还是返回错误，由引擎集成和请求路径决定；熔断器本身不会重试推理请求。

基类 `StoreV1.CheckHealth()` 直接返回成功。Posix 和 Mooncake 覆盖该方法并执行后端操作。因此，没有实现具体探测的阶段，即使包装器处于允许状态，也不能据此证明其存储服务经过了独立检查。

## 设置探测策略

`store_health` 放在 `ucm_connector_config` 内。下面的`Cache|Posix` 示例显式填写了默认健康策略：

```yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: /mnt/ucm-cache
      store_health:
        enabled: true
        health_check_interval_s: 10
        health_check_timeout_s: 3
        health_window_size: 8
        failure_threshold: 2
enable_metrics: true
```

模型、挂载和缓存参数仍须符合实际部署，详见[存储流水线](../../developer-guide/cache-configuration/pipeline.md)。

| 字段 | 含义 |
| --- | --- |
| `enabled` | 创建健康包装器和探测线程；默认 `true` |
| `health_check_interval_s` | 目标探测间隔；默认 10 秒 |
| `health_check_timeout_s` | 探测执行期限；默认 3 秒 |
| `health_window_size` | 保留的最近结果数量；默认 8 |
| `failure_threshold` | 阻止新操作所需的失败次数；默认 2 |

数值必须为正数，失败阈值不能大于窗口长度，超时必须短于探测间隔。首次探测会等待一个间隔，再加不超过一个间隔的随机延迟。首次探测前就会发布初始允许状态，因此初始状态不能作为 I/O 成功的证据。

设置 `enabled: false` 会撤掉这层保护，不会修复后端故障，也不会关闭后端自身的错误处理。

## 理解探测实际检查了什么

**Posix** 遍历存储布局选定的健康检查路径，创建临时文件、写入 4096 字节、读回比较并删除。Buffered I/O 还会调用 `Sync`，Direct I/O 则使用已配置的直接打开标志。打开、传输、同步、比较或删除失败都会使探测失败。路径使用 NFS 时，检查的是该 UCM 进程实际看到的文件系统挂载。

**Mooncake** 在专用 key 下写入八字节测试值，读回比较后删除。传输路径使用实际客户端，scheduler 路径使用 RPC 客户端。这检查的是已配置的客户端通路，不会验证所有模型的 KV 布局或每个远端副本。

健康检查会产生自己的小文件或 key，不能把这些对象当成请求已保存可复用 KV 数据的证据。

## 先检查单条状态，再考虑聚合

默认 vLLM connector 导出以下指标：

| 后端 | 状态 Gauge | 探测 Counter |
| --- | --- | --- |
| Posix | `ucm:posix_store_health` | `ucm:posix_healthy_count_total`、`ucm:posix_unhealthy_count_total` |
| Mooncake | `ucm:mooncake_store_health` | `ucm:mooncake_healthy_count_total`、`ucm:mooncake_unhealthy_count_total` |

Gauge 为 1 表示包装器允许操作，为 0 表示阻止操作。Counter 记录探测成功、失败及超时，不记录状态迁移次数。一次成功探测不一定会使被阻止的 Gauge 恢复为 1。

`ucm:posix_passive_failures_total` 和 `ucm:mooncake_passive_failures_total` 单独统计观察到的 I/O 失败次数，不与主动探测混算。

先查看保留完整标签的各条序列：

```promql
ucm:posix_store_health{job="vllm"}
```

再查看最近窗口中的失败：

```promql
increase(ucm:posix_unhealthy_count_total{job="vllm"}[5m])
```

这些查询假设抓取任务名为 `vllm`。定位问题时保留 `instance`、模型、engine 和 `worker_rank` 标签。Scheduler 的标签值为 `worker_rank="scheduler"`。多个进程可能探测同一个后端，导出的标签也不能标识每个物理挂载或底层流水线对象，因此序列数量不是故障磁盘数量。

原生探测线程和 Prometheus 抓取采用不同的调度节奏。在 vLLM connector 路径中，原生统计通过 `get_kv_connector_stats()` 进入导出器。把重复数值或缺失序列解释为后端当前状态之前，应先确认统计采集仍在推进。

## 定位故障并确认恢复

1. 检查抓取目标，根据标签找到对应进程。
2. 在其日志中查找 `Store health check` 失败和 `transitioned to UNHEALTHY`；日志会给出流水线阶段标识、触发原因及冷却时间。
3. Posix 检查该进程的挂载、权限、可用容量及读写删除错误；Mooncake 检查客户端、metadata/master 连接和具体操作错误。
4. 修复依赖后，观察成功探测逐步替换失败窗口，在冷却结束后确认 `transitioned to HEALTHY` 及对应 Gauge 更新。
5. 单独重做[外部缓存验证](../quick_start/index.md#vllm-verify-the-service-and-external-cache)。探测恢复不能证明某个请求的缓存恢复。

指标单位和导出路径见[指标参考](metrics-reference.md)。策略和操作行为分别定义在
[`StoreHealthConfig`](https://github.com/ModelEngine-Group/unified-cache-management/blob/a336d69bc03a550d44bee3df9da7664e9edfe3a7/ucm/store/pipeline/cc/store_health_config.h)
和 [`HealthBreakerStore`](https://github.com/ModelEngine-Group/unified-cache-management/blob/a336d69bc03a550d44bee3df9da7664e9edfe3a7/ucm/store/pipeline/cc/health_breaker_store.cc)。
