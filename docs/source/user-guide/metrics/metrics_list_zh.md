# UCM metrics列表

## 3. 默认暴露指标

以下表格使用默认的 `ucm:` 前缀。默认配置包含 67 个 Counter、4 个 Gauge 和 70 个 Histogram。除非指标名称或说明另有说明，耗时指标单位为毫秒，带宽指标单位为 GB/s，累计数据量单位为字节。

### 3.1 Counter

#### Cache Store

| 指标 | 说明 |
| --- | --- |
| `ucm:cache_lookup_hit_blocks_total` | Cache Lookup 直接命中、未下沉到后端的 block 数 |
| `ucm:cache_lookup_miss_blocks_total` | Cache Lookup 未命中并下沉到后端的 block 数 |
| `ucm:cache_load_blocks_total` | Cache Load 处理的 block 总数 |
| `ucm:cache_dump_blocks_total` | Cache Dump 处理的 block 总数 |
| `ucm:cache_load_shards_total` | Cache Load 分发的 shard 总数 |
| `ucm:cache_load_backend_shards_total` | Cache buffer 分配期间实际下沉到后端的 shard 数 |
| `ucm:cache_dump_shards_total` | Cache Dump 分发的 shard 总数 |
| `ucm:cache_dump_backend_shards_total` | 实际写入后端的 owner shard 数 |
| `ucm:cache_load_queue_full_total` | 等待队列已满导致 Cache Load 提交被拒绝的次数 |
| `ucm:cache_dump_queue_full_total` | 等待队列已满导致 Cache Dump 提交被拒绝的次数 |
| `ucm:cache_backend_load_submit_errors_total` | Cache 向后端提交 Load 失败的次数 |
| `ucm:cache_backend_load_wait_errors_total` | Cache 等待后端 Load 失败的次数 |
| `ucm:cache_backend_dump_submit_errors_total` | Cache 向后端提交 Dump 失败的次数 |
| `ucm:cache_backend_dump_wait_errors_total` | Cache 等待后端 Dump 失败的次数 |
| `ucm:cache_h2d_errors_total` | Cache H2D 传输或 stream 同步失败次数 |
| `ucm:cache_d2h_errors_total` | Cache D2H 传输、event 等待或 stream 同步失败次数 |
| `ucm:cache_load_bytes_total` | 经过 Cache 层加载的累计字节数 |
| `ucm:cache_dump_bytes_total` | 经过 Cache 层写出的累计字节数 |

#### Posix Store

| 指标 | 说明 |
| --- | --- |
| `ucm:posix_s2h_bytes_total` | 从 Posix 存储读取到 host buffer 的累计字节数 |
| `ucm:posix_h2s_bytes_total` | 从 host buffer 写入 Posix 存储的累计字节数 |
| `ucm:posix_lookup_query_blocks_total` | 提交给 Posix Lookup 的 block 总数 |
| `ucm:posix_lookup_hit_blocks_total` | Posix Lookup 命中的 block 数 |
| `ucm:posix_healthy_count_total` | Posix 健康探测成功次数 |
| `ucm:posix_unhealthy_count_total` | Posix 健康探测失败次数 |
| `ucm:posix_aio_timeout_total` | Posix AIO 任务或提交超时次数 |
| `ucm:posix_io_timeout_total` | Posix 同步 I/O worker 任务超时次数 |
| `ucm:posix_open_errors_total` | Posix 文件打开失败次数 |
| `ucm:posix_io_errors_total` | Posix 读、写或 AIO 完成失败次数 |

#### Mooncake Store

| 指标 | 说明 |
| --- | --- |
| `ucm:mooncake_load_blocks_total` | Mooncake Load 层处理的 block 总数 |
| `ucm:mooncake_dump_blocks_total` | Mooncake Dump 层处理的 block 总数 |
| `ucm:mooncake_lookup_hit_blocks_total` | Mooncake Lookup 直接命中、未下沉到后端的 block 数 |
| `ucm:mooncake_healthy_count_total` | Mooncake 健康探测成功次数 |
| `ucm:mooncake_unhealthy_count_total` | Mooncake 健康探测失败次数 |
| `ucm:mooncake_load_bytes_total` | 经过 Mooncake 层加载的累计字节数 |
| `ucm:mooncake_dump_bytes_total` | 经过 Mooncake 层写出的累计字节数 |
| `ucm:mooncake_load_hit_shards_total` | Mooncake 直接完成加载的 shard 数 |
| `ucm:mooncake_load_miss_shards_total` | Mooncake 未命中并下沉后端或重新计算的 shard 数 |
| `ucm:mooncake_load_backend_shards_total` | Mooncake miss 后提交给后端加载的 shard 数 |
| `ucm:mooncake_dump_existing_shards_total` | Mooncake 中已存在的 Dump shard 数 |
| `ucm:mooncake_dump_missing_shards_total` | 缺失并写入 Mooncake 的 Dump shard 数 |
| `ucm:mooncake_dump_backend_shards_total` | 归档到后端的 Dump shard 数 |
| `ucm:mooncake_load_queue_full_total` | 等待队列已满导致 Mooncake Load 提交被拒绝的次数 |
| `ucm:mooncake_dump_queue_full_total` | 等待队列已满导致 Mooncake Dump 提交被拒绝的次数 |
| `ucm:mooncake_get_errors_total` | Mooncake batch-get 失败次数 |
| `ucm:mooncake_put_errors_total` | Mooncake batch-put 失败次数 |
| `ucm:mooncake_backend_load_submit_errors_total` | Mooncake 后端 Load 提交失败次数 |
| `ucm:mooncake_backend_load_wait_errors_total` | 等待 Mooncake 后端 Load 失败的次数 |
| `ucm:mooncake_backend_dump_submit_errors_total` | Mooncake 后端 Dump 提交失败次数 |
| `ucm:mooncake_backend_dump_wait_errors_total` | 等待 Mooncake 后端 Dump 失败的次数 |
| `ucm:mooncake_h2d_errors_total` | Mooncake H2D 传输或同步失败次数 |
| `ucm:mooncake_d2h_errors_total` | Mooncake D2H 传输、event 等待或同步失败次数 |
| `ucm:mooncake_h2d_bytes_total` | Mooncake 从 host 复制到 device 的累计字节数 |
| `ucm:mooncake_d2h_bytes_total` | Mooncake 从 device 复制到 host 的累计字节数 |

#### Connector

| 指标 | 说明 |
| --- | --- |
| `ucm:load_bytes_total` | 所有 `start_load_kv` 调用加载的累计字节数 |
| `ucm:save_bytes_total` | 所有 `wait_for_save` 调用保存的累计字节数 |
| `ucm:total_prefix_query_tokens_total` | UCM connector 观察到的 prefix cache 查询 token 总数 |
| `ucm:gpu_hbm_hit_tokens_total` | UCM Lookup 前已在 GPU/HBM 中命中的 prefix token 数 |
| `ucm:ucm_hit_tokens_total` | UCM connector 命中的 prefix token 数 |
| `ucm:total_prefix_query_blocks_total` | UCM connector 查询的完整 prefix block 总数 |
| `ucm:gpu_hbm_hit_blocks_total` | UCM Lookup 前已在 GPU/HBM 中命中的完整 prefix block 数 |
| `ucm:connector_lookup_errors_total` | 作为 cache miss 处理的 Connector Lookup 错误次数 |
| `ucm:connector_load_submit_errors_total` | Connector Load 提交失败次数 |
| `ucm:connector_load_wait_errors_total` | Connector Load 等待失败次数 |
| `ucm:connector_load_invalid_requests_total` | Load 失败导致请求 block 失效的事件次数 |
| `ucm:connector_load_invalid_blocks_total` | Load 失败导致新失效的 vLLM block ID 数 |
| `ucm:connector_dump_submit_errors_total` | Connector Dump 提交失败次数 |
| `ucm:connector_dump_wait_errors_total` | Connector Dump 等待失败次数 |

### 3.2 Gauge

Store 健康 Counter、Gauge 及推荐聚合方法见 [UCM 健康指标](health_metrics_zh.md)。

| 指标 | 说明 |
| --- | --- |
| `ucm:cache_lookup_hit_rate` | 最近一次 Cache Lookup 的即时命中率 |
| `ucm:posix_store_health` | Posix 熔断器有效状态：1 表示可用，0 表示已熔断 |
| `ucm:mooncake_store_health` | Mooncake 熔断器有效状态：1 表示可用，0 表示已熔断 |
| `ucm:posix_gc_running` | Posix GC 状态：1 表示正在运行，0 表示空闲 |

### 3.3 Histogram

#### Connector 基础指标

| 指标 | 说明 |
| --- | --- |
| `ucm:load_requests_num` | 一次 UCM Load 涉及的请求数 |
| `ucm:load_blocks_num` | 一次 UCM Load 涉及的 block 数 |
| `ucm:load_duration` | UCM Connector Load 耗时 |
| `ucm:load_speed` | UCM Connector Load 吞吐，单位 GB/s |
| `ucm:save_requests_num` | 一次 UCM Save 涉及的请求数 |
| `ucm:save_blocks_num` | 一次 UCM Save 涉及的 block 数 |
| `ucm:save_duration` | 从进入 `wait_for_save` 到异步 Dump 完成的耗时 |
| `ucm:save_completion_wait_duration` | 确认异步 Dump 完成时实际阻塞的时间 |
| `ucm:interval_lookup_hit_rates` | 每请求 UCM Lookup 命中率分布 |

#### Cache Store

| 指标 | 说明 |
| --- | --- |
| `ucm:cache_lookup_duration_ms` | 一次 Cache buffer `Lookup`/`LookupOnPrefix` 调用的墙钟时间 |
| `ucm:cache_lookup_backend_duration_ms` | 无 buffer 或 buffer miss 时后端 Lookup 的墙钟时间 |
| `ucm:cache_load_duration_ms` | Cache 层 Load 任务端到端耗时 |
| `ucm:cache_dump_duration_ms` | Cache 层 Dump 任务端到端耗时 |
| `ucm:cache_load_bandwidth_gbps` | 完整 Cache Load 任务生命周期的有效带宽 |
| `ucm:cache_dump_bandwidth_gbps` | 完整 Cache Dump 任务生命周期的有效带宽 |
| `ucm:cache_load_queue_wait_duration_ms` | Cache Load 任务等待 dispatch worker 取走的时间 |
| `ucm:cache_dump_queue_wait_duration_ms` | Cache Dump 任务等待 dispatch worker 取走的时间 |
| `ucm:cache_load_backend_submit_duration_ms` | 分配 Cache buffer 并同步提交后端 Load 的时间 |
| `ucm:cache_shard_backend_wait_ms` | 单个 shard 在提交 H2D 前等待后端就绪的时间 |
| `ucm:cache_h2d_submit_ms` | 单个 shard 异步 H2D 提交的 CPU 开销，不含传输时间 |
| `ucm:cache_h2d_sync_ms` | 最后一个 shard 提交后剩余的 H2D stream 排空时间 |
| `ucm:cache_dump_mkbuf_duration_ms` | Cache Dump buffer 分配/复用及异步 D2H 提交时间 |
| `ucm:cache_dump_prereq_wait_ms` | D2H 开始前等待该层 KV-ready 计算 event 的时间 |
| `ucm:cache_d2h_duration_ms` | Cache Dump stream 同步时间，包含前置计算等待和 D2H 复制 |
| `ucm:cache_dump_backend_submit_duration_ms` | 向下层 Store 同步提交 buffer 的时间 |
| `ucm:cache_dump_backend_wait_duration_ms` | 等待下层 Store 完成写入的时间 |

#### Posix Store

| 指标 | 说明 |
| --- | --- |
| `ucm:posix_load_task_duration_ms` | Posix Load 任务从提交到最后一个 shard 完成的端到端耗时 |
| `ucm:posix_dump_task_duration_ms` | Posix Dump 任务从提交到最后一个 shard 完成的端到端耗时 |
| `ucm:posix_s2h_bandwidth_gbps` | 单任务 Posix 读取带宽 |
| `ucm:posix_h2s_bandwidth_gbps` | 单任务 Posix 写入带宽 |
| `ucm:posix_load_queue_wait_duration_ms` | Posix Load 任务等待首个 worker 取走的时间 |
| `ucm:posix_dump_queue_wait_duration_ms` | Posix Dump 任务等待首个 worker 取走的时间 |

#### Mooncake Store

| 指标 | 说明 |
| --- | --- |
| `ucm:mooncake_load_duration_ms` | Mooncake Load 任务端到端耗时 |
| `ucm:mooncake_dump_duration_ms` | Mooncake Dump 任务端到端耗时 |
| `ucm:mooncake_load_bandwidth_gbps` | Mooncake 层 Load 有效带宽 |
| `ucm:mooncake_dump_bandwidth_gbps` | Mooncake 层 Dump 有效带宽 |
| `ucm:mooncake_load_queue_wait_duration_ms` | Mooncake Load 任务等待 dispatch worker 取走的时间 |
| `ucm:mooncake_dump_queue_wait_duration_ms` | Mooncake Dump 任务等待 dispatch worker 取走的时间 |
| `ucm:mooncake_get_duration_ms` | Load 路径上的 Mooncake batch-get 耗时 |
| `ucm:mooncake_exists_duration_ms` | Dump 路径上的 Mooncake batch-exists 检查耗时 |
| `ucm:mooncake_put_duration_ms` | Dump 路径上的 Mooncake batch-put 耗时 |
| `ucm:mooncake_load_backend_submit_duration_ms` | Mooncake miss 后提交后端 Load 的时间 |
| `ucm:mooncake_backend_load_wait_duration_ms` | 等待后端加载缺失 shard 的时间 |
| `ucm:mooncake_h2d_duration_ms` | Mooncake Load H2D stream 排空时间 |
| `ucm:mooncake_dump_prereq_wait_ms` | Mooncake put 前等待前置计算 event 的时间 |
| `ucm:mooncake_d2h_duration_ms` | 后端归档所需的 Mooncake D2H stream 排空时间 |
| `ucm:mooncake_dump_backend_submit_duration_ms` | D2H 归档复制后提交后端 Dump 的时间 |
| `ucm:mooncake_dump_backend_wait_duration_ms` | 等待后端归档完成的时间 |

#### Layerwise

| 指标 | 说明 |
| --- | --- |
| `ucm:layerwise_batch_total_ms` | 从进入 `start_load_kv` 到 `wait_for_save` 返回的 batch 总墙钟时间 |
| `ucm:layerwise_batch_total_load_only_ms` | 仅加载 layerwise batch 的总墙钟时间 |
| `ucm:layerwise_batch_total_save_only_ms` | 仅保存 layerwise batch 的总墙钟时间 |
| `ucm:layerwise_batch_total_load_save_ms` | 同时包含 Load 和 Save 的 layerwise batch 总墙钟时间 |
| `ucm:layerwise_batch_total_no_transfer_ms` | 无 Load 或 Save 传输的 layerwise batch 总墙钟时间 |
| `ucm:layerwise_batch_load_wait_total_load_only_ms` | 仅加载 batch 中所有 `wait_for_layer_load` 阻塞时间之和 |
| `ucm:layerwise_batch_load_wait_total_load_save_ms` | 加载并保存 batch 中所有 `wait_for_layer_load` 阻塞时间之和 |
| `ucm:layerwise_batch_save_tail_save_only_ms` | 仅保存 batch 中 `wait_for_save` 的尾部耗时 |
| `ucm:layerwise_batch_save_tail_load_save_ms` | 加载并保存 batch 中 `wait_for_save` 的尾部耗时 |
| `ucm:layerwise_wait_blocking_ms` | 单次 `wait_for_layer_load` 阻塞时间；接近零表示重叠良好 |
| `ucm:layerwise_wait_tasks_count` | 单次 layer wait 等待的请求 Load 任务数 |
| `ucm:layerwise_inter_wait_interval_ms` | 连续两次 `wait_for_layer_load` 调用的间隔 |
| `ucm:layerwise_next_layer_submit_ms` | 在 `wait_for_layer_load` 中提交下一层 Load 任务的时间 |
| `ucm:layerwise_first_layer_submit_ms` | 在 `start_load_kv` 中提交第一层 Load 任务的时间 |
| `ucm:layerwise_first_layer_requests` | 在 `start_load_kv` 中提交第一层 Load 的请求数 |
| `ucm:layerwise_save_submit_ms` | 在 `save_kv_layer` 中提交单层 Dump 任务的时间 |
| `ucm:layerwise_save_tail_total_ms` | 兼容性指标；Layerwise 已不再在 `wait_for_save` 中等待 Dump 完成 |

#### FAWA

| 指标 | 说明 |
| --- | --- |
| `ucm:fawa_scheduler_lookup_external_hit_blocks_ms` | Scheduler Store Lookup 耗时 |
| `ucm:fawa_scheduler_get_num_new_matched_tokens_ms` | Store Lookup 与 block hash 生成总耗时 |
| `ucm:fawa_worker_wait_wait_all_load_task_ms` | Worker Store Load 等待耗时 |
| `ucm:fawa_worker_start_load_kv_ms` | Worker Store Load 任务构造和提交耗时 |
| `ucm:fawa_worker_wait_for_save_ms` | Worker Store Dump 耗时 |
