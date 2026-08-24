# UCM 指标参考

## 1. 默认导出的指标

下表使用默认的 `ucm:` 前缀。默认配置包含 78 个 Counter、14 个 Gauge 和 67 个 Histogram。

Store 健康指标及推荐的聚合方式请参见 [UCM 健康指标](health_metrics.md)。

### 1.1 Connector

#### Counter

| 指标 | 说明 |
| --- | --- |
| `ucm:load_bytes_total` | 所有 `start_load_kv` 调用累计加载的字节数 |
| `ucm:save_bytes_total` | 所有 `wait_for_save` 调用累计保存的字节数 |
| `ucm:total_prefix_query_tokens_total` | UCM Connector 观测到的前缀缓存查询 token 总数 |
| `ucm:gpu_hbm_hit_tokens_total` | UCM Lookup 前已经在 GPU/HBM 中命中的前缀 token 数 |
| `ucm:ucm_hit_tokens_total` | UCM Connector 命中的前缀 token 数 |
| `ucm:total_prefix_query_blocks_total` | UCM Connector 查询的完整前缀块总数 |
| `ucm:gpu_hbm_hit_blocks_total` | UCM Lookup 前已经在 GPU/HBM 中命中的完整前缀块数 |

#### Gauge

默认不导出 Connector 专属的 Gauge。

#### Histogram

| 指标 | 说明 |
| --- | --- |
| `ucm:save_duration` | 从进入 `wait_for_save` 到异步 Dump 完成的耗时 |
| `ucm:save_completion_wait_duration` | 确认异步 Dump 完成时实际发生阻塞的耗时 |
| `ucm:interval_lookup_hit_rates` | 单次 UCM Lookup 请求命中率的分布 |
| `ucm:connector_get_block_size_duration_ms` | Connector 接口 `get_block_size` 的耗时 |
| `ucm:connector_get_kv_connector_stats_duration_ms` | Connector 接口 `get_kv_connector_stats` 的耗时 |
| `ucm:connector_get_num_new_matched_tokens_duration_ms` | Connector 接口 `get_num_new_matched_tokens` 的耗时 |
| `ucm:connector_update_state_after_alloc_duration_ms` | Connector 接口 `update_state_after_alloc` 的耗时 |
| `ucm:connector_register_kv_caches_duration_ms` | Connector 接口 `register_kv_caches` 的耗时 |
| `ucm:connector_build_connector_meta_duration_ms` | Connector 接口 `build_connector_meta` 的耗时 |
| `ucm:connector_bind_connector_metadata_duration_ms` | Connector 接口 `bind_connector_metadata` 的耗时 |
| `ucm:connector_handle_preemptions_duration_ms` | Connector 接口 `handle_preemptions` 的耗时 |
| `ucm:connector_has_connector_metadata_duration_ms` | Connector 接口 `has_connector_metadata` 的耗时 |
| `ucm:connector_start_load_kv_duration_ms` | Connector 接口 `start_load_kv` 的耗时 |
| `ucm:connector_wait_for_layer_load_duration_ms` | Connector 接口 `wait_for_layer_load` 的耗时 |
| `ucm:connector_save_kv_layer_duration_ms` | Connector 接口 `save_kv_layer` 的耗时 |
| `ucm:connector_wait_for_save_duration_ms` | Connector 接口 `wait_for_save` 的耗时 |
| `ucm:connector_request_finished_all_groups_duration_ms` | Connector 接口 `request_finished_all_groups` 的耗时 |
| `ucm:connector_request_finished_duration_ms` | Connector 接口 `request_finished` 的耗时 |
| `ucm:connector_get_finished_duration_ms` | Connector 接口 `get_finished` 的耗时 |
| `ucm:connector_build_connector_worker_meta_duration_ms` | Connector 接口 `build_connector_worker_meta` 的耗时 |
| `ucm:connector_update_connector_output_duration_ms` | Connector 接口 `update_connector_output` 的耗时 |
| `ucm:connector_clear_connector_metadata_duration_ms` | Connector 接口 `clear_connector_metadata` 的耗时 |
| `ucm:layerwise_batch_total_load_only_ms` | 仅包含 Load 的 Layerwise 批次总墙钟耗时 |
| `ucm:layerwise_batch_total_save_only_ms` | 仅包含 Save 的 Layerwise 批次总墙钟耗时 |
| `ucm:layerwise_batch_total_load_save_ms` | 同时包含 Load 和 Save 的 Layerwise 批次总墙钟耗时 |
| `ucm:layerwise_batch_load_wait_total_load_only_ms` | 仅包含 Load 的批次中所有 `wait_for_layer_load` 阻塞时间之和 |
| `ucm:layerwise_batch_load_wait_total_load_save_ms` | 同时包含 Load 和 Save 的批次中所有 `wait_for_layer_load` 阻塞时间之和 |

### 1.2 Cache Store

#### Counter

| 指标 | 说明 |
| --- | --- |
| `ucm:cache_lookup_hit_blocks_total` | Cache Lookup 直接返回、未下探后端的块数 |
| `ucm:cache_lookup_miss_blocks_total` | Cache Lookup 未命中并传递给后端的块数 |
| `ucm:cache_load_shards_total` | Load 期间检查过 Cache buffer 状态的分片总数 |
| `ucm:cache_load_wait_shards_total` | 获取 Cache buffer 时尚未 Ready、需要等待的分片数 |
| `ucm:cache_load_backend_shards_total` | 分配 Cache buffer 期间下探后端的分片数 |
| `ucm:cache_load_success_shards_total` | 从已 Ready 的 Cache buffer 成功加载到设备的分片数 |
| `ucm:cache_posix_load_success_shards_total` | 等待 Posix 填充 Cache 后成功加载到设备的分片数 |
| `ucm:cache_dump_shards_total` | Cache Dump 处理的分片描述符总数，包括失败任务 |
| `ucm:cache_dump_backend_shards_total` | 实际写入后端的 owner 分片数 |
| `ucm:cache_load_bytes_total` | 经过 Cache 层累计加载的字节数 |
| `ucm:cache_dump_bytes_total` | 经过 Cache 层累计 Dump 的字节数 |

对于 Cache | Posix pipeline，Cache 加载占比为 `(总分片数 - 等待分片数) / 总分片数`，Posix 加载占比为 `等待分片数 / 总分片数`。Grafana 和 Metrics View 使用这两个占比拆分外存命中率。

#### Gauge

默认不导出 Cache Store 专属的 Gauge。

#### Histogram

| 指标 | 说明 |
| --- | --- |
| `ucm:cache_lookup_duration_ms` | 一次 Cache buffer `Lookup`/`LookupOnPrefix` 调用的墙钟耗时 |
| `ucm:cache_lookup_backend_duration_ms` | 不存在 buffer 或 buffer 未命中时，后端 Lookup 的墙钟耗时 |
| `ucm:cache_load_duration_ms` | Cache 层 Load 任务的端到端耗时 |
| `ucm:cache_dump_duration_ms` | Cache 层 Dump 任务的端到端耗时 |
| `ucm:cache_load_bandwidth_gbps` | 完整 Cache Load 任务生命周期内的有效带宽 |
| `ucm:cache_dump_bandwidth_gbps` | 完整 Cache Dump 任务生命周期内的有效带宽 |
| `ucm:cache_load_queue_wait_duration_ms` | Cache Load 任务等待 dispatch worker 获取的时间 |
| `ucm:cache_dump_queue_wait_duration_ms` | Cache Dump 任务等待 dispatch worker 获取的时间 |
| `ucm:cache_load_backend_submit_duration_ms` | 分配 Cache buffer 并同步提交后端 Load 的耗时 |
| `ucm:cache_shard_backend_wait_ms` | 单个分片在提交 H2D 前等待后端 Ready 的时间 |
| `ucm:cache_h2d_submit_ms` | 单个分片异步提交 H2D 的 CPU 开销，不包含传输时间 |
| `ucm:cache_h2d_sync_ms` | 最后一个分片提交后，等待 H2D stream 排空的剩余时间 |
| `ucm:cache_dump_mkbuf_duration_ms` | Cache Dump buffer 分配/复用及异步提交 D2H 的耗时 |
| `ucm:cache_dump_prereq_wait_ms` | D2H 开始前等待当前层 KV Ready 计算事件的时间 |
| `ucm:cache_d2h_duration_ms` | Cache Dump stream 同步耗时，包括前置计算等待和 D2H copy |
| `ucm:cache_dump_backend_submit_duration_ms` | 向下层 Store 同步提交 buffer 的耗时 |
| `ucm:cache_dump_backend_wait_duration_ms` | 等待下层 Store 完成写入的时间 |

### 1.3 Posix Store

#### Counter

| 指标 | 说明 |
| --- | --- |
| `ucm:posix_s2h_bytes_total` | 从 Posix 存储累计读取到 host buffer 的字节数 |
| `ucm:posix_h2s_bytes_total` | 从 host buffer 累计写入 Posix 存储的字节数 |
| `ucm:posix_lookup_query_blocks_total` | 提交给 Posix Lookup 的块总数 |
| `ucm:posix_lookup_hit_blocks_total` | Posix Lookup 命中的块数 |

#### Gauge

| 指标 | 说明 |
| --- | --- |
| `ucm:posix_store_used_bytes` | GC 采样估算的 Posix Store 逻辑已用空间，单位为字节 |
| `ucm:posix_store_capacity_bytes` | 配置的 Posix Store 逻辑容量，单位为字节 |
| `ucm:posix_store_usage_ratio` | 估算的 Posix Store 逻辑空间使用率 |
| `ucm:posix_store_health` | 熔断器的实际状态：1 表示可用，0 表示已熔断 |
| `ucm:posix_gc_running` | GC 状态：1 表示正在运行，0 表示空闲 |

#### Histogram

| 指标 | 说明 |
| --- | --- |
| `ucm:posix_load_task_duration_ms` | Posix Load 任务从提交到最后一个分片完成的端到端耗时 |
| `ucm:posix_dump_task_duration_ms` | Posix Dump 任务从提交到最后一个分片完成的端到端耗时 |
| `ucm:posix_s2h_bandwidth_gbps` | 单个 Posix 读任务的带宽 |
| `ucm:posix_h2s_bandwidth_gbps` | 单个 Posix 写任务的带宽 |
| `ucm:posix_load_queue_wait_duration_ms` | Posix Load 任务等待第一个 worker 获取的时间 |
| `ucm:posix_dump_queue_wait_duration_ms` | Posix Dump 任务等待第一个 worker 获取的时间 |

### 1.4 YuanRong Store

#### Counter

| 指标 | 说明 |
| --- | --- |
| `ucm:yuanrong_load_success_shards_total` | 从 YuanRong 成功加载到设备的分片数 |
| `ucm:yuanrong_lookup_miss_posix_load_success_shards_total` | YuanRong Lookup miss 后从 Posix 成功加载的分片数 |
| `ucm:yuanrong_load_fallback_posix_load_success_shards_total` | YuanRong Load 失败并回退后，从 Posix 成功加载的分片数 |
| `ucm:yuanrong_local_dram_load_hits_total` | 从 `kv_resource.log` 转发的 YuanRong 本地 DRAM Get 命中估算值 |
| `ucm:yuanrong_remote_load_hits_total` | 从 `kv_resource.log` 转发的 YuanRong 远端 worker Get 命中估算值 |
| `ucm:yuanrong_local_ssd_load_hits_total` | 从 `kv_resource.log` 转发的 YuanRong 本地 spill SSD Get 命中估算值 |
| `ucm:yuanrong_l2_load_hits_total` | 从 `kv_resource.log` 转发的 YuanRong L2 持久化 Get 命中数 |

#### Gauge

| 指标 | 说明 |
| --- | --- |
| `ucm:yuanrong_dram_used_bytes` | YuanRong 物理共享内存使用量，单位为字节 |
| `ucm:yuanrong_dram_capacity_bytes` | YuanRong 共享内存容量，单位为字节 |
| `ucm:yuanrong_dram_usage_ratio` | YuanRong 物理共享内存使用率 |
| `ucm:yuanrong_ssd_used_bytes` | YuanRong 物理 spill 磁盘使用量，单位为字节 |
| `ucm:yuanrong_ssd_capacity_bytes` | YuanRong spill 磁盘容量，单位为字节 |
| `ucm:yuanrong_ssd_usage_ratio` | YuanRong 物理 spill 磁盘使用率 |
| `ucm:yuanrong_resource_log_last_update_timestamp_seconds` | UCM 最近一次解析 YuanRong 资源快照的 Unix 时间戳 |
| `ucm:yuanrong_resource_log_reporter_leader` | 当前 UCM 进程是否为本机 YuanRong 资源上报 leader |

#### Histogram

默认不导出 YuanRong 专属的 Histogram。

### 1.5 Mooncake Store

#### Counter

| 指标 | 说明 |
| --- | --- |
| `ucm:mooncake_load_blocks_total` | Mooncake Load 层处理的块总数 |
| `ucm:mooncake_dump_blocks_total` | Mooncake Dump 层处理的块总数 |
| `ucm:mooncake_lookup_hit_blocks_total` | 下探后端前由 Mooncake Lookup 直接命中的块数 |
| `ucm:mooncake_load_bytes_total` | 经过 Mooncake 层累计加载的字节数 |
| `ucm:mooncake_dump_bytes_total` | 经过 Mooncake 层累计 Dump 的字节数 |
| `ucm:mooncake_load_hit_shards_total` | 由 Mooncake 直接提供的 Load 分片数 |
| `ucm:mooncake_load_miss_shards_total` | Mooncake miss 后下探后端或重新计算的 Load 分片数 |
| `ucm:mooncake_load_backend_shards_total` | Mooncake miss 后提交给后端加载的分片数 |
| `ucm:mooncake_dump_existing_shards_total` | Mooncake 中已经存在的 Dump 分片数 |
| `ucm:mooncake_dump_missing_shards_total` | 写入 Mooncake 的缺失 Dump 分片数 |
| `ucm:mooncake_dump_backend_shards_total` | 归档到后端的 Dump 分片数 |
| `ucm:mooncake_h2d_bytes_total` | Mooncake 从 host 累计复制到设备的字节数 |
| `ucm:mooncake_d2h_bytes_total` | Mooncake 从设备累计复制到 host 的字节数 |

#### Gauge

| 指标 | 说明 |
| --- | --- |
| `ucm:mooncake_store_health` | 熔断器的实际状态：1 表示可用，0 表示已熔断 |

#### Histogram

| 指标 | 说明 |
| --- | --- |
| `ucm:mooncake_load_duration_ms` | Mooncake Load 任务的端到端耗时 |
| `ucm:mooncake_dump_duration_ms` | Mooncake Dump 任务的端到端耗时 |
| `ucm:mooncake_load_bandwidth_gbps` | Mooncake 层 Load 的有效带宽 |
| `ucm:mooncake_dump_bandwidth_gbps` | Mooncake 层 Dump 的有效带宽 |
| `ucm:mooncake_load_queue_wait_duration_ms` | Mooncake Load 任务等待 dispatch worker 获取的时间 |
| `ucm:mooncake_dump_queue_wait_duration_ms` | Mooncake Dump 任务等待 dispatch worker 获取的时间 |
| `ucm:mooncake_get_duration_ms` | Load 路径中 Mooncake batch get 的耗时 |
| `ucm:mooncake_exists_duration_ms` | Dump 路径中 Mooncake batch exists 检查的耗时 |
| `ucm:mooncake_put_duration_ms` | Dump 路径中 Mooncake batch put 的耗时 |
| `ucm:mooncake_load_backend_submit_duration_ms` | Mooncake miss 后提交后端 Load 的耗时 |
| `ucm:mooncake_backend_load_wait_duration_ms` | 等待后端加载缺失分片的时间 |
| `ucm:mooncake_h2d_duration_ms` | Mooncake Load H2D stream 排空耗时 |
| `ucm:mooncake_dump_prereq_wait_ms` | Mooncake put 前等待前置计算事件的时间 |
| `ucm:mooncake_d2h_duration_ms` | 后端归档所需的 Mooncake D2H stream 排空耗时 |
| `ucm:mooncake_dump_backend_submit_duration_ms` | D2H 归档复制后提交后端 Dump 的耗时 |
| `ucm:mooncake_dump_backend_wait_duration_ms` | 等待后端归档完成的时间 |

## 2. 原始指标使用方式

### 2.1 命中率

建议使用分层计算，不要单独计算每个 Store 的命中率。首先计算相邻两个缓存层边界上的总命中率，再根据这两个层实际加载来源的比例拆分总命中率。这样可以确保整个层次结构中分子和分母的统计语义一致，得到最准确的分层命中率估算结果。

例如，可以使用 vLLM 的 token Counter 计算外存总命中率：

```text
external_cache_hit_rate = external_hit / external_query
                          * (1 - hbm_hit / hbm_query)
```

对于 Cache | Posix pipeline，使用分片 Counter 拆分外存总命中率：

```text
cache_share = (cache_load_shards - cache_load_wait_shards)
              / cache_load_shards
posix_share = cache_load_wait_shards / cache_load_shards

cache_hit_rate = external_cache_hit_rate * cache_share
posix_hit_rate = external_cache_hit_rate * posix_share
```

其他分层 Store 也采用相同方式：先计算这些层的总命中率，再使用能表示实际加载来源的 Counter 进行拆分。同一公式中的所有 Counter 应在相同时间范围内使用 `rate()` 或 `increase()` 计算；同一个比例中不要混用 token、block 和 shard 数量。

### 2.2 带宽

UCM 提供两种含义不同的带宽视图：

- `*_bandwidth_gbps` Histogram 记录单个任务的有效带宽，可以观察任务的瞬时速度以及 p50、p90、p99 等分布，但不能表示系统整体吞吐。
- 系统平均带宽应使用总传输字节数除以经过的时间计算。这个结果包含并发和空闲时间，因此能表示所选时间范围内系统的整体吞吐。

应根据目标 Store 和传输方向选择对应的累计字节 Counter。例如：

```promql
sum(rate(ucm:cache_load_bytes_total[$__rate_interval])) / 1e9
```

对于固定时间范围，也可以使用等价公式：

```text
average_bandwidth_GBps = increase(transferred_bytes_total) / elapsed_seconds / 1e9
```

应先汇总各 worker 的字节速率，再换算为 GB/s。Load 与 Dump、读与写应分别展示。不要通过平均单任务带宽来估算系统整体吞吐。
