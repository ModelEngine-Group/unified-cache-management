# 指标清单

UCM 指标记录 KV 缓存的查找、加载、保存和存储状态，用来判断**缓存有没有被复用、数据搬运有多快、时间花在哪里、存储是否可用**。先根据要回答的问题选择指标，再查看下面的完整清单。

| 想知道什么 | 先看什么 | 怎么理解 |
| --- | --- | --- |
| 缓存有没有帮上忙？ | Connector 的查询与命中 token 数、加载字节数 | 同一时间窗口内，命中 token 占查询 token 的比例反映可复用前缀的贡献；再看加载量确认传输活动。是否改善首 token 延迟，还要对照引擎的 TTFT |
| 数据搬运有多快？ | 所用 Store 的 `*_bytes_total` 和 `*_bandwidth_gbps` | 字节 Counter 的每秒速率反映整体吞吐；带宽 Histogram 反映单个任务的速度。负载少时整体吞吐低，不一定是存储慢 |
| 缓存操作慢在哪里？ | `*_duration_ms`，再看排队、后端等待和 H2D/D2H 耗时 | 先看整个任务，再区分排队、下层存储等待、主机与设备间复制；在相近任务大小和并发下比较 |
| 存储快满了或不可用了？ | `*_usage_ratio`、`*_store_health`，Posix 再看 `posix_gc_running` | 使用率看容量压力，健康状态看是否允许新操作，GC 看是否在回收。使用率高和后端故障是不同问题 |

**先分清数值的类型。** 清单中的 Counter、Gauge 和 Histogram 需要用不同方式查看：

| 类型 | 记录什么 | 读法与例子 |
| --- | --- | --- |
| Counter（累计量） | 从进程启动以来累计发生的数量，通常以 `_total` 结尾；重启后可能归零 | 用 `rate(...[5m])` 看最近 5 分钟的平均每秒速率，用 `increase(...[5m])` 看这段时间的增量。`load_bytes_total` 本身是累计加载字节数，不是当前带宽 |
| Gauge（当前值） | 最近一次上报的容量、比例或状态，可以升降 | 直接看数值及趋势。例如 `posix_store_usage_ratio = 0.8` 表示估算的逻辑容量使用率为 80%；`posix_store_health = 0` 表示已熔断 |
| Histogram（分布） | 多次操作的耗时或带宽样本 | 用平均值看整体水平，用分位数看分布。耗时 p95 为 20 ms，表示约 95% 的采样操作耗时不超过 20 ms；它不是每个用户请求的耗时 |

单位也要一起看：`*_bytes*` 使用字节，`*_duration_ms` 使用毫秒；`save_duration` 和 `save_completion_wait_duration` 虽然没有 `_ms` 后缀，单位也是**毫秒**。`*_bandwidth_gbps` 实际单位为 **GB/s（每秒十亿字节）**。H2D 表示主机内存到设备内存，D2H 表示反方向；Posix 的 S2H/H2S 表示存储与主机内存之间的读写。

比较时使用同一模型、服务和时间窗口。定位问题时保留 `instance`、`engine`、`worker_rank` 标签；同一个 KV 块经过 Connector 和多个 Store 时会在不同阶段分别计数，不能把这些阶段的字节数相加当作唯一数据量。

指标名称默认使用 `ucm:` 前缀。只有启用的指标且相应路径执行后才可能看到序列；缺失不等于零。vLLM connector 路径还依赖推理期间同步统计，无请求时数值可能停留在上次上报结果。

采集方式见[指标设置](metrics.md)，导出链路见[指标开发](../../developer-guide/add-metrics.md)，健康状态解释见[存储健康检查](health-metrics.md)。

## 1. 指标目录

### 1.1 Connector

Connector 连接推理引擎与 UCM，适合先判断**前缀能复用多少，以及引擎调用缓存接口花了多久**。查询 token 是统计基数，HBM 命中表示已有设备侧前缀，UCM 命中表示额外找到的外部前缀；这些计数不能当作请求数。

保存变慢时同时看 `save_duration` 和 `save_completion_wait_duration`：前者覆盖异步保存的整体时段，后者只记录确认完成时实际阻塞的部分。`connector_*_duration_ms` 记录接口调用耗时，异步接口返回后传输可能仍在继续，因此不能用接口耗时替代完整 I/O 耗时。

#### Counters

| 指标 | 说明 |
| --------------------------------------- | ----------------------------------------------------------------- |
| `ucm:load_bytes_total` | Connector 传输路径成功加载的累计字节数 |
| `ucm:save_bytes_total` | Connector 保存路径成功提交的累计字节数 |
| `ucm:total_prefix_query_tokens_total` | UCM Connector 观测到的前缀缓存查询 token 总数 |
| `ucm:gpu_hbm_hit_tokens_total` | UCM Lookup 前已在 GPU/HBM 中命中的前缀 token 数 |
| `ucm:ucm_hit_tokens_total` | UCM Connector 命中的前缀 token 数 |
| `ucm:total_prefix_query_blocks_total` | UCM Connector 查询的完整前缀块总数 |
| `ucm:gpu_hbm_hit_blocks_total` | UCM Lookup 前已在 GPU/HBM 中命中的完整前缀块数 |

#### Gauges

默认不导出 Connector 专用 Gauge。

#### Histograms

| 指标 | 说明 |
| --------------------------------------------------------- | -------------------------------------------------------------------------- |
| `ucm:save_duration` | 从进入 wait_for_save 到异步 Dump 完成的时长，单位 ms；观察保存的整体耗时 |
| `ucm:save_completion_wait_duration` | 确认异步 Dump 完成时实际阻塞等待的时长，单位 ms；观察保存留给调用方的等待开销 |
| `ucm:interval_lookup_hit_rates` | 每个请求的 UCM Lookup 命中率分布[^export-scope] |
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
| `ucm:layerwise_layer_load_duration_ms` | 从某层开始加载到 wait_for_layer_load 返回的墙钟时长 |
| `ucm:layerwise_batch_load_duration_sum_ms` | 一个 Layerwise 批次内各层加载时长之和 |

### 1.2 Cache Store

Cache Store 利用主机缓冲衔接设备与下层存储，适合判断**加载是否直接拿到就绪数据，还是要等待后端**。先看 `cache_load_wait_shards_total` 相对 `cache_load_shards_total` 的增量占比，再看成功加载分片数确认完成情况。等待占比上升表示更多分片没有立即就绪，但不直接等于磁盘读取增加。

加载耗时上升时，对照排队、后端就绪等待和 H2D 同步耗时；保存则对照前置计算等待、D2H 和后端写入等待。这些时段可能重叠，且有分片与任务粒度之分，不能简单相加还原总耗时。

#### Counters

| 指标 | 说明 |
| --------------------------------------------- | -------------------------------------------------------------------------- |
| `ucm:cache_lookup_hit_blocks_total` | Cache Lookup 直接命中、未向后端查询的块数 |
| `ucm:cache_lookup_miss_blocks_total` | Cache Lookup 未命中并继续查询后端的块数 |
| `ucm:cache_load_shards_total` | Load 时检查过 Cache 缓冲状态的分片总数 |
| `ucm:cache_load_wait_shards_total` | 获取缓冲时尚未 Ready、需要等待的分片数 |
| `ucm:cache_load_backend_shards_total` | 分配 Cache 缓冲时向后端发起加载的分片数 |
| `ucm:cache_load_success_shards_total` | 从已就绪的 Cache 缓冲成功加载到设备的分片数 |
| `ucm:cache_posix_load_success_shards_total` | 等待 Posix 填充 Cache 后成功加载到设备的分片数[^cache-source] |
| `ucm:cache_dump_shards_total` | Cache Dump 处理的分片描述符总数，包括失败任务 |
| `ucm:cache_dump_backend_shards_total` | 实际写入后端的 owner 分片数 |
| `ucm:cache_load_bytes_total` | 经 Cache 阶段加载的累计字节数 |
| `ucm:cache_dump_bytes_total` | 经 Cache 阶段保存的累计字节数 |

对于 `Cache|Posix`，Cache 加载占比为 `(total shards - wait shards) / total shards`，Posix 加载占比为 `wait shards / total shards`。Grafana 和 Metrics View 将这些占比应用到外部缓存命中率。这里按等待来源估算分层占比，具体含义见下方注记。

#### Gauges

默认不导出 Cache Store 专用 Gauge。

#### Histograms

| 指标 | 说明 |
| --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `ucm:cache_lookup_duration_ms` | Lookup、LookupOnPrefix 或 LookupOnReverse 扫描 Cache 缓冲的时长；关闭共享内存时直接查询下层 Store，不产生该采样 |
| `ucm:cache_lookup_backend_duration_ms` | 没有缓冲或缓冲未命中时，后端 Lookup 的墙钟时长 |
| `ucm:cache_load_duration_ms` | Cache 阶段 Load 任务的端到端时长 |
| `ucm:cache_dump_duration_ms` | Cache 阶段 Dump 任务的端到端时长 |
| `ucm:cache_load_bandwidth_gbps` | 完整 Cache Load 任务生命周期内的有效带宽 |
| `ucm:cache_dump_bandwidth_gbps` | 完整 Cache Dump 任务生命周期内的有效带宽 |
| `ucm:cache_load_queue_wait_duration_ms` | Cache Load 任务被分发线程取出前的排队时长 |
| `ucm:cache_dump_queue_wait_duration_ms` | Cache Dump 任务被分发线程取出前的排队时长 |
| `ucm:cache_load_backend_submit_duration_ms` | 分配 Cache 缓冲并同步提交后端 Load 的时长 |
| `ucm:cache_shard_backend_wait_ms` | 单个分片在提交 H2D 前等待后端就绪的时长 |
| `ucm:cache_h2d_submit_ms` | 异步提交一个分片 H2D 的 CPU 开销，不含传输时间 |
| `ucm:cache_h2d_sync_ms` | 最后一个分片提交后，等待 H2D 流排空的剩余时长 |
| `ucm:cache_dump_mkbuf_duration_ms` | Cache Dump 分配或复用缓冲并异步提交 D2H 的时长 |
| `ucm:cache_dump_prereq_wait_ms` | D2H 开始前等待该层 KV 就绪计算事件的时长 |
| `ucm:cache_d2h_duration_ms` | Cache Dump 的流同步时长，包括前置计算等待和 D2H 复制 |
| `ucm:cache_dump_backend_submit_duration_ms` | 将缓冲同步提交给下层 Store 的时长 |
| `ucm:cache_dump_backend_wait_duration_ms` | 等待下层 Store 完成写入的时长 |

### 1.3 Posix Store

Posix Store 通过文件系统读写 KV 数据，适合判断**文件后端的命中、读写速度和容量压力**。`posix_lookup_hit_blocks_total` 与 `posix_lookup_query_blocks_total` 的增量比值，只表示已经查询到 Posix 这一层的块命中率。

读写变慢时，对照任务耗时与排队耗时；容量方面把逻辑使用率与 `posix_gc_running` 放在一起看。GC 运行本身是正常回收行为；`posix_store_health = 0` 表示熔断器阻止新操作。逻辑使用率来自 GC 采样估算，不能替代宿主机文件系统的剩余空间检查。

#### Counters

| 指标 | 说明 |
| --------------------------------------- | ----------------------------------------------------------- |
| `ucm:posix_s2h_bytes_total` | 从 Posix 存储读入主机缓冲的累计字节数[^posix-bytes] |
| `ucm:posix_h2s_bytes_total` | 从主机缓冲写入 Posix 存储的累计字节数[^posix-bytes] |
| `ucm:posix_lookup_query_blocks_total` | 提交给 Posix Lookup 的块总数 |
| `ucm:posix_lookup_hit_blocks_total` | Posix Lookup 找到的块数 |

#### Gauges

| 指标 | 说明 |
| ---------------------------------- | -------------------------------------------------------------- |
| `ucm:posix_store_used_bytes` | 根据 GC 采样估算的 Posix 逻辑已用容量，单位字节 |
| `ucm:posix_store_capacity_bytes` | 配置的 Posix 逻辑容量，单位字节 |
| `ucm:posix_store_usage_ratio` | 估算的 Posix 逻辑容量使用率 |
| `ucm:posix_store_health` | 生效的熔断器状态：1 表示可用，0 表示熔断 |
| `ucm:posix_gc_running` | 垃圾回收状态：1 表示运行中，0 表示空闲 |

#### Histograms

| 指标 | 说明 |
| ----------------------------------------- | ----------------------------------------------------------------------------------- |
| `ucm:posix_load_task_duration_ms` | 从提交 Posix Load 到最后一个分片完成的端到端时长 |
| `ucm:posix_dump_task_duration_ms` | 从提交 Posix Dump 到最后一个分片完成的端到端时长 |
| `ucm:posix_s2h_bandwidth_gbps` | 单个 Posix 读取任务的带宽 |
| `ucm:posix_h2s_bandwidth_gbps` | 单个 Posix 写入任务的带宽 |
| `ucm:posix_load_queue_wait_duration_ms` | Posix Load 任务被首个 worker 取出前的等待时长 |
| `ucm:posix_dump_queue_wait_duration_ms` | Posix Dump 任务被首个 worker 取出前的等待时长 |

### 1.4 YuanRong Store

YuanRong 指标用于判断**加载由哪层提供、是否退回 Posix，以及共享内存和溢写磁盘的占用情况**。`lookup_miss_posix_load_success` 增长表示查询未命中后由 Posix 提供数据；`load_fallback_posix_load_success` 增长则表示加载失败后回退成功，应结合错误日志排查。

DRAM、远端 worker 和 SSD 命中估算来自资源日志，不能直接与 Connector 的 token 命中数相除。读容量和来源分布前，先检查 `yuanrong_resource_log_last_update_timestamp_seconds` 是否在推进；`reporter_leader = 1` 只表示报告进程身份，不表示后端健康。

#### Counters

| 指标 | 说明 |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `ucm:yuanrong_load_success_shards_total` | 从 YuanRong 成功加载到设备的分片数 |
| `ucm:yuanrong_lookup_miss_posix_load_success_shards_total` | YuanRong Lookup 未命中后从 Posix 成功加载的分片数 |
| `ucm:yuanrong_load_fallback_posix_load_success_shards_total` | YuanRong Load 失败后从 Posix 成功加载的分片数 |
| `ucm:yuanrong_local_dram_load_hits_total` | 从 kv_resource.log 转发的 YuanRong 本地 DRAM Get 命中估算值 |
| `ucm:yuanrong_remote_load_hits_total` | 从 kv_resource.log 转发的 YuanRong 远端 worker Get 命中估算值 |
| `ucm:yuanrong_local_ssd_load_hits_total` | 从 kv_resource.log 转发的 YuanRong 本地溢写 SSD Get 命中估算值 |
| `ucm:yuanrong_l2_load_hits_total` | 从 kv_resource.log 转发的 YuanRong L2 持久化 Get 命中数 |

#### Gauges

| 指标 | 说明 |
| ----------------------------------------------------------- | ---------------------------------------------------------------------- |
| `ucm:yuanrong_dram_used_bytes` | YuanRong 物理共享内存已用字节数 |
| `ucm:yuanrong_dram_capacity_bytes` | YuanRong 共享内存容量，单位字节 |
| `ucm:yuanrong_dram_usage_ratio` | YuanRong 物理共享内存使用率 |
| `ucm:yuanrong_ssd_used_bytes` | YuanRong 物理溢写磁盘已用字节数 |
| `ucm:yuanrong_ssd_capacity_bytes` | YuanRong 溢写磁盘容量，单位字节 |
| `ucm:yuanrong_ssd_usage_ratio` | YuanRong 物理溢写磁盘使用率 |
| `ucm:yuanrong_resource_log_last_update_timestamp_seconds` | UCM 最近解析的 YuanRong 资源快照 Unix 时间戳 |
| `ucm:yuanrong_resource_log_reporter_leader` | 当前 UCM 进程是否为本机 YuanRong 资源报告 leader |

#### Histograms

默认不导出 YuanRong 专用 Histogram。

### 1.5 Mooncake Store

Mooncake 指标用于判断**数据是否由 Mooncake 提供、是否下探后端，以及哪一步拖慢传输**。先比较 `mooncake_load_hit_shards_total` 与 `mooncake_load_miss_shards_total` 的增量，再看 `mooncake_load_backend_shards_total` 确认下层加载活动。

加载慢时看排队、batch-get、后端等待与 H2D；保存慢时看前置计算等待、batch-put 和后端归档。`mooncake_dump_existing_shards_total` 增长表示保存时数据已经存在，本身不是失败。健康 Gauge 为 0 时先按[存储健康检查](health-metrics.md)处理熔断。

#### Counters

| 指标 | 说明 |
| ------------------------------------------- | --------------------------------------------------------------------------- |
| `ucm:mooncake_load_blocks_total` | Mooncake Load 阶段处理的块总数 |
| `ucm:mooncake_dump_blocks_total` | Mooncake Dump 阶段处理的块总数 |
| `ucm:mooncake_lookup_hit_blocks_total` | 向后端查询前由 Mooncake Lookup 直接找到的块数 |
| `ucm:mooncake_load_bytes_total` | 经 Mooncake 阶段加载的累计字节数 |
| `ucm:mooncake_dump_bytes_total` | 经 Mooncake 阶段保存的累计字节数 |
| `ucm:mooncake_load_hit_shards_total` | 由 Mooncake 直接提供数据的加载分片数 |
| `ucm:mooncake_load_miss_shards_total` | Mooncake 未命中、继续查询后端或重新计算的加载分片数 |
| `ucm:mooncake_load_backend_shards_total` | Mooncake 未命中后提交给后端加载的分片数 |
| `ucm:mooncake_dump_existing_shards_total` | Dump 时已存在于 Mooncake 中的分片数 |
| `ucm:mooncake_dump_missing_shards_total` | Dump 时不存在、随后写入 Mooncake 的分片数 |
| `ucm:mooncake_dump_backend_shards_total` | 归档到后端的 Dump 分片数 |
| `ucm:mooncake_h2d_bytes_total` | Mooncake 从主机复制到设备的累计字节数 |
| `ucm:mooncake_d2h_bytes_total` | Mooncake 从设备复制到主机的累计字节数 |

#### Gauges

| 指标 | 说明 |
| ----------------------------- | -------------------------------------------------------------- |
| `ucm:mooncake_store_health` | 生效的熔断器状态：1 表示可用，0 表示熔断 |

#### Histograms

| 指标 | 说明 |
| ------------------------------------------------ | --------------------------------------------------------------------- |
| `ucm:mooncake_load_duration_ms` | Mooncake 阶段 Load 任务的端到端时长 |
| `ucm:mooncake_dump_duration_ms` | Mooncake 阶段 Dump 任务的端到端时长 |
| `ucm:mooncake_load_bandwidth_gbps` | Mooncake 阶段 Load 的有效带宽 |
| `ucm:mooncake_dump_bandwidth_gbps` | Mooncake 阶段 Dump 的有效带宽 |
| `ucm:mooncake_load_queue_wait_duration_ms` | Mooncake Load 任务被分发线程取出前的排队时长 |
| `ucm:mooncake_dump_queue_wait_duration_ms` | Mooncake Dump 任务被分发线程取出前的排队时长 |
| `ucm:mooncake_get_duration_ms` | Load 路径中 Mooncake batch-get 的时长 |
| `ucm:mooncake_exists_duration_ms` | Dump 路径中 Mooncake batch-exists 检查的时长 |
| `ucm:mooncake_put_duration_ms` | Dump 路径中 Mooncake batch-put 的时长 |
| `ucm:mooncake_load_backend_submit_duration_ms` | Mooncake 未命中后向后端提交 Load 的时长 |
| `ucm:mooncake_backend_load_wait_duration_ms` | 等待后端加载缺失分片的时长 |
| `ucm:mooncake_h2d_duration_ms` | Mooncake Load 等待 H2D 流排空的时长 |
| `ucm:mooncake_dump_prereq_wait_ms` | Mooncake put 前等待前置计算事件的时长 |
| `ucm:mooncake_d2h_duration_ms` | 后端归档所需的 Mooncake D2H 流排空时长 |
| `ucm:mooncake_dump_backend_submit_duration_ms` | D2H 归档复制后向后端提交 Dump 的时长 |
| `ucm:mooncake_dump_backend_wait_duration_ms` | 等待后端归档完成的时长 |

## 2. 原始指标的使用

下面的 PromQL 可在 Prometheus 查询页或 Grafana 的 Prometheus 数据源中执行，假设抓取任务名为 `vllm`。按实际部署替换 `job`；一个任务包含多个模型或服务时，先加标签筛选再聚合。示例统一使用最近 5 分钟，Grafana 中可将 `[5m]` 换成 `[$__rate_interval]`。

### 2.1 命中率

先回答最直接的问题：**Connector 查询的前缀 token 中，有多少由 UCM 找到了可复用数据？**

```promql
sum(rate(ucm:ucm_hit_tokens_total{job="vllm"}[5m]))
/
sum(rate(ucm:total_prefix_query_tokens_total{job="vllm"}[5m]))
```

结果为 `0.3` 表示该窗口内 UCM 命中 token 占 Connector 查询 token 的 30%，不表示 30% 的请求完全命中，也不表示延迟降低 30%。分母包含 HBM 已覆盖的 token，所以这是 UCM 对全部查询前缀的命中贡献。窗口内没有查询时，比例没有意义，应显示为无数据。

采用分层计算：先计算相邻缓存层边界上的总体命中率，再按这些层报告的加载来源占比拆分。这样可以保持分子、分母的口径一致，得到各层命中贡献的估算。

例如，可以使用 vLLM 的 token Counter 计算外部缓存命中贡献：

```text
external_cache_hit_rate = external_hit / external_query
                          * (1 - hbm_hit / hbm_query)
```

这里的 `external_query` 指 HBM 未命中后向外部层查询的 token 数。公式中的名称是计算符号，不是实际导出的指标名；不能用包含 HBM 已覆盖部分的 `total_prefix_query_tokens_total` 直接替代这个分母。

对于 `Cache|Posix`，使用分片 Counter 拆分外部缓存命中率：

```text
cache_share = (cache_load_shards - cache_load_wait_shards)
              / cache_load_shards
posix_share = cache_load_wait_shards / cache_load_shards

cache_hit_rate = external_cache_hit_rate * cache_share
posix_hit_rate = external_cache_hit_rate * posix_share
```

其他分层 Store 采用相同思路：先计算合并命中率，再按加载来源 Counter 拆分。同一个公式中的 Counter 使用相同时间范围的 `rate()` 或 `increase()`，同一比例中不混用 token、block 和 shard。

### 2.2 带宽

UCM 提供两种含义不同的带宽视角：

- `*_bandwidth_gbps` Histogram 记录单任务有效带宽及其分布，例如 p50、p90、p99；这些值不等于系统总吞吐。
- 系统平均带宽使用传输字节总量除以经过的时间，包含并发与空闲阶段，反映所选时间窗口内的整体吞吐。

按 Store 和传输方向选择累计字节 Counter，例如：

```promql
sum(rate(ucm:cache_load_bytes_total{job="vllm"}[5m])) / 1e9
```

固定时间范围下的等价计算为：

```text
average_bandwidth_GBps = increase(transferred_bytes_total) / elapsed_seconds / 1e9
```

先汇总各 worker 的字节速率，再转换为 GB/s。Load 与 Dump、读取与写入分别展示，不通过平均单任务带宽来估算系统吞吐。

结果为 `2` 表示最近 5 分钟经过 Cache 加载阶段的平均计数字节速率为 2 GB/s。对比单任务带宽时，低分位数（如 p10）更适合找慢任务；带宽 p99 反映高速一端，不能按“尾部延迟”来理解。

### 2.3 耗时与慢任务

以 Cache Load 为例，Histogram 的 `_count` 是采样任务数，`_sum` 是这些任务的累计耗时，`_bucket{le="20"}` 是耗时不超过 20 ms 的累计样本数。最近 5 分钟的平均任务耗时为：

```promql
sum(rate(ucm:cache_load_duration_ms_sum{job="vllm"}[5m]))
/
sum(rate(ucm:cache_load_duration_ms_count{job="vllm"}[5m]))
```

要观察较慢的那部分任务，查看 p95：

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(ucm:cache_load_duration_ms_bucket{job="vllm"}[5m]))
)
```

两者的结果都以 ms 为单位。平均值平稳但 p95 上升，说明较慢一端变差；结合任务大小与负载变化，再查看该阶段的排队、后端等待和复制耗时。分位数由桶估算；汇总多个 worker 时先合并桶，不平均各 worker 的 p95。计算方法见 [Prometheus Histogram 说明](https://prometheus.io/docs/practices/histograms/)。

### 2.4 容量与健康

Gauge 直接查询，保留每条序列以定位对应进程。例如查看 Posix 的逻辑容量使用百分比：

```promql
100 * ucm:posix_store_usage_ratio{job="vllm"}
```

结果为 `80` 表示该序列最近上报的逻辑使用率为 80%。持续接近容量上限时，结合 `ucm:posix_gc_running` 与文件系统可用空间检查回收是否跟得上写入。

```promql
ucm:posix_store_health{job="vllm"}
```

`1` 表示允许新的缓存操作，`0` 表示熔断；不要把多个进程的值相加后当作一个健康状态。还要确认指标仍在更新，并结合探测结果判断，完整解释见[存储健康检查](health-metrics.md)。

## 3. 运行口径注记

[^export-scope]: 当前 `ucm/metrics_config.py` 中的 `vllm_connector` consumer 排除了 `interval_lookup_hit_rates`。本目录保留原生指标条目，但默认 vLLM 端点不会导出它。
[^cache-source]: `cache_posix_load_success_shards_total` 按分派时 Cache 缓冲尚未就绪的状态分类，下层 Store 不一定是 Posix。Cache/Posix 的来源比例用于估算各层贡献，应结合后端活动和已完成传输，不将等待直接解释为物理磁盘读取。
[^posix-bytes]: 当前 Posix 字节计数在任务完成时累计名义任务字节数，包含错误和中止路径。查看字节速率时同时检查错误；物理存储吞吐需要结合文件系统或设备观测。
