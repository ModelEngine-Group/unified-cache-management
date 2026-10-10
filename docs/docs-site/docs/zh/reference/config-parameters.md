# 配置参数

本页按读取配置的组件说明字段归属、类型、默认值和取值约束。后端选择与配置方法见开发者指南的[缓存配置](../developer-guide/cache-configuration/index.md)。

下面的结构用于 vLLM UCM YAML；SGLang 的入口格式见 [SGLang 快速开始](../user-guide/quick_start/index.md#sglang)。

```yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: /mnt/ucm-cache
enable_metrics: true
```

根级字段控制引擎 Connector，`ucm_connector_config` 内字段由所选 Store 读取。只为实际使用的阶段设置参数，未列出的后端选项参见各后端指南。

## 引擎 Connector

以下字段写在 YAML 根级，模型或引擎特定限制在说明中标出。

| 配置项 | 是否必填 | 取值类型 | 取值范围 | 配置说明 |
| --- | --- | --- | --- | --- |
| `use_layerwise` | 选填 | bool | 默认 `true` | 是否启用分层（逐层）加载和保存模式。推荐设置为 `true`，DeepSeek v4 系列模型推荐设置为 `false`。 |
| `enable_event_sync` | 选填 | bool | 默认 `true` | 性能优化开关，推荐开启。 |
| `persist_token_threshold` | 选填 | int | `0` | 当请求长度小于 `persist_token_threshold` 时，UCM 软件不对该请求进行处理。 |
| `wa_dump_block_wise` | 选填 | bool | `true` | 仅在 FAWA connector 中使用。`true`: 每个 block 的 WA cache 都会被 dump（高频 dump）；`false`: 只 dump 每个 chunk prefill 最后 block 的 WA cache（低频 dump）。 |
| `load_tokens_threshold` | 选填 | int | `0` | 设置触发 KV cache 加载的最小 token 阈值，仅在 DeepSeek V4 系列模型生效。当外部命中 tokens 数 > `load_tokens_threshold` 时触发 KV Cache 加载。 |
| `enable_record_traces` | 选填 | bool | `false` | 用来记录请求信息（时间戳，输入长度，输出长度等信息）。 |
| `enable_metrics` | 选填 | bool | 默认 `true` | 是否开启 metrics 收集。 |
| `use_lite` | 选填 | bool | `false` | 是否启用 UCM Lite 功能。不对 KV Cache 数据进行保存和加载，仅对元数据进行保存和查询。仅可用于评估 KV Cache 命中率情况，无加速效果。 |
| `metrics_config_path` | 选填 | string | 自行配置 | 指定监控指标配置文件路径，启用后可通过 toolkit 进行 UCM 在线监控。参考配置文件：`examples/metrics/metrics_configs.yaml`。 |

## Store 选择与任务设置

在 `ucm_connectors[]` 中用 `ucm_connector_name` 选择 Store，以下字段放在其 `ucm_connector_config` 内。`store_pipeline` 仅适用于 `UcmPipelineStore`；NFS Connector 的独立任务设置见[对应指南](../developer-guide/cache-configuration/nfs.md)。

| 配置项 | 是否必填 | 取值类型 | 取值范围 | 配置说明 |
| --- | --- | --- | --- | --- |
| `store_pipeline` | **Pipeline Store 必填** | string | 见下方可选值 | 已注册的管线名称；在 vLLM YAML 中显式设置。SGLang 适配器会选择 `Posix`。 |
| `timeout_ms` | 选填 | int | 默认 `30000` | 缓存传输与 Posix I/O 任务的超时时间（毫秒）。配置在 `ucm_connector_config` 内，不能放在 YAML 根节点。 |

## Posix 文件系统阶段

以下为 Pipeline 中 Posix 阶段的参数；包含 Posix 时需提供 `storage_backends`。直接 NFS Connector 接受部分同名选项，但默认值不同，不套用本表。

| 配置项 | 是否必填 | 取值类型 | 取值范围 | 配置说明 |
| --- | --- | --- | --- | --- |
| `storage_backends` | **必填** | string | 自行配置，多个挂载点用冒号隔开 | 填写本地目录或者挂载点，如果有多个挂载点需要用冒号隔开。 |
| `io_direct` | 选填 | bool | 默认 `true` | 是否启用直接 IO 模式（绕过操作系统页缓存）。`false`: 使用 PageCache；`true`: 跳过 PageCache，直接 IO。 |
| `posix_io_engine` | 选填 | string | 默认 `psync` | 文件 IO 模式。`psync`：同步 io；`aio`：异步 io，要求 `io_direct` 配置为 `true`。 |
| `posix_data_trans_concurrency` | 选填 | int | 默认 `128` | `psync` 模式下单卡对存储的读写线程数。NFS over RDMA 下推荐单卡 128 线程。`aio` 下不读取该值。 |
| `posix_open_concurrency` | 选填 | int | 默认 `32` | `aio` 下对文件进行 open 操作的线程数。`psync` 下不感知。 |
| `posix_commit_concurrency` | 选填 | int | 默认 `4` | `aio` 下对文件进行 rename 操作的线程数。`psync` 下不感知。 |
| `posix_lookup_concurrency` | 选填 | int | 默认 `16` | 在挂载点中查找文件是否存在的线程数。 |

## Cache 主机缓冲阶段

仅适用于包含 Cache 的流水线。共享缓冲按共享域规划，非共享缓冲按同机 worker 总量规划。

| 配置项 | 是否必填 | 取值类型 | 取值范围 | 配置说明 |
| --- | --- | --- | --- | --- |
| `cache_buffer_capacity_gb` | 选填 | int | 见配置说明 | 原生 Cache Store 启用共享 buffer 时默认为 256 GiB，`share_buffer_enable` 为 false 时默认为每个 worker 32 GiB。启用共享 buffer 且省略容量时，vLLM connector 会提供 128 GiB。MLA 默认启用共享 buffer。显式设置正数容量会覆盖原生默认值；非共享模式需汇总主机上所有 worker 的预算。 |
| `cache_sdma_direct` | 选填 | bool | 依据编译环境变量决定，`PLATFORM=ascend-a3` 时默认 `true`，其他默认 `false` | 启用 SDMA H2D/D2H 传输路径，仅在 A3 设备生效，推荐关闭。 |
| `cache_load_backend_only` | 选填 | bool | 默认 `false` | 即使在 cache 层命中还是会强制从 SSD 上加载，仅供测试使用。 |
| `cache_io_aggregation` | 选填 | bool | 默认 `false`，仅在 `PLATFORM=ascend` 且模型为 V4 时自动开启 | 启用 IO 聚合 h2d 传输，仅在 A2 设备生效。 |
| `share_buffer_enable` | 选填 | bool | MLA 默认启用，GQA 默认不启用 | 是否启用共享内存。MLA 如果不用 shm 或 GQA 用 shm 都会导致性能下降。 |

## Posix 容量回收

回收配置不预留磁盘空间。共享存储命名空间必须协调 GC 归属，不能由多个独立实例同时按自身预算回收。

| 配置项 | 是否必填 | 取值类型 | 取值范围 | 配置说明 |
| --- | --- | --- | --- | --- |
| `posix_capacity_gb` | 选填 | int | 默认 `0`，表示不启用 GC；不可超过挂载文件系统可用容量 | 设置磁盘存储的最大容量（GB），当已用容量 >= `posix_capacity_gb * posix_gc_trigger_threshold_ratio` 时触发 GC。多实例部署共享同一文件系统时，只能有一个实例开启 GC，其他实例不要开启。 |
| `posix_gc_trigger_threshold_ratio` | 条件选填 | float | 默认 `0.7`，范围：0~1。`posix_capacity_gb` 未配置时不填写 | GC 阈值比例，配合 `posix_capacity_gb` 使用。 |
| `posix_gc_recycle_percent` | 选填 | float | 默认 `0.1`，范围：0~1。`posix_capacity_gb` 未配置时不填写 | 每轮 GC 删除当前容量的比值。 |
| `posix_gc_max_recycle_count_per_shard` | 选填 | int | 默认 `50000`，>0。不建议修改。`posix_capacity_gb` 未配置时不填写 | 每轮 GC 单目录允许删除的文件数上限。 |
| `posix_gc_shard_sample_ratio` | 选填 | float | 默认 `0.1`，范围：0~1。`posix_capacity_gb` 未配置时不填写 | 采样 10% 目录估算总容量。 |
| `posix_gc_check_interval_sec` | 选填 | int | 默认 `30`，>0。`posix_capacity_gb` 未配置时不填写 | GC 采样及触发间隔。 |
| `posix_gc_concurrency` | 选填 | int | 默认 `16`，>0。`posix_capacity_gb` 未配置时不填写 | GC 线程池 worker 数。 |
| `posix_gc_task_timeout_ms` | 选填 | int | 默认 `300000`，>0。`posix_capacity_gb` 未配置时不填写 | 单目录任务超时 watchdog。`0`=禁用。 |
| `posix_gc_precise_mode` | 选填 | bool | 默认 `true`。`posix_capacity_gb` 未配置时不填写 | `true`: 精准模式（全局最冷）；`false`: 性能模式（每目录最冷）。 |

## Pipeline 健康检查

字段放在 `ucm_connector_config.store_health` 内。探测和恢复行为见[存储健康检查](../user-guide/observability/health-metrics.md)。

| 配置项 | 是否必填 | 取值类型 | 取值范围 | 配置说明 |
| --- | --- | --- | --- | --- |
| `enabled` | 选填 | bool | 默认 `true` | 存储隔离机制总开关。开启后给磁盘 KV 缓存增加故障熔断器，磁盘读写频繁超时/报错时自动切断存储。 |
| `health_check_interval_s` | 选填 | number | 默认 `10` | 缓存磁盘健康巡检周期（秒）。必须 >0 且 > `health_check_timeout_s`。 |
| `health_check_timeout_s` | 选填 | number | 默认 `3` | 单次探测超时时间（秒）。必须 >0 且 < `health_check_interval_s`。 |
| `health_window_size` | 选填 | int | 默认 `8` | 故障统计窗口长度。必须为正整数且 >= `failure_threshold`。 |
| `failure_threshold` | 选填 | int | 默认 `2` | 故障触发阈值。必须为正整数且 <= `health_window_size`。 |

## 已注册管线

选择名称前确认构建包含所需阶段。`Empty`、`Fake` 和包含这些阶段的组合用于测试，不作为持久化配置。

| 管线 | 用途 |
| --- | --- |
| `Cache|Posix` | 正常使用场景 |
| `Cache|Empty` | MLA 纯 Cache 测试 |
| `Cache|Fake` | MLA/GQA 纯 Cache 测试（有精度问题） |
| `Empty` | Store 全部接口空实现，引擎侧永远不命中 |
| `Fake` | 不实际存取，仅存元数据 lookup，测试极限性能 |
| `Mooncake` | 对接 Mooncake 内存池，仅支持 vllm-ascend |
| `Mooncake|Posix` | 对接 Mooncake 内存池并支持落盘 |
| `YuanRong` | 对接 YuanRong 内存池 |
| `YuanRong|Posix` | 对接 YuanRong 内存池并支持落盘 |
| `Cache|Ds3fs` | 3FS 客户端读写 |
| `Cache|Compress|Posix` | 压缩后写入文件系统 |

其他后端设置见 [3FS](../developer-guide/cache-configuration/ds3fs.md)、[Mooncake](../developer-guide/cache-configuration/mooncake.md)、[压缩](../developer-guide/cache-configuration/compress.md)和 [NFS Connector](../developer-guide/cache-configuration/nfs.md)。

默认值分别来自引擎 Connector、各阶段 `global_config.h` 和 `store_health_config.h`。容量覆盖与配置归属见[缓存工作原理](../developer-guide/capability-principles.md)；最小配置与操作步骤见[文件系统流水线](../developer-guide/cache-configuration/pipeline.md)。
