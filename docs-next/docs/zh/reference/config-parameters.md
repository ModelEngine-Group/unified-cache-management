# 原生UCM方案参数

---

## 顶层参数

> 直接写在 YAML 根级别的全局参数。

| 配置项 | 是否必填 | 取值类型 | 取值范围 | 配置说明 |
|---|---|---|---|---|
| `use_layerwise` | 选填 | bool | 默认 `false` | 是否启用分层（逐层）加载和保存模式。推荐设置为 `true`，DeepSeek v4 系列模型推荐设置为 `false`。 |
| `enable_event_sync` | 选填 | bool | 默认 `true` | 性能优化开关，推荐开启。 |
| `persist_token_threshold` | 选填 | int | `0` | 当请求长度小于 `persist_token_threshold` 时，UCM 软件不对该请求进行处理。 |
| `timeout_ms` | 选填 | int | >0，默认 `30000` | 显存和 DRAM/SHM 之间的拷贝以及读写盘的超时时间（单位：ms）。 |
| `wa_dump_block_wise` | 选填 | bool | `true` | 仅在 FAWA connector 中使用。`true`: 每个 block 的 WA cache 都会被 dump（高频 dump）；`false`: 只 dump 每个 chunk prefill 最后 block 的 WA cache（低频 dump）。 |
| `load_tokens_threshold` | 选填 | int | `0` | 设置触发 KV cache 加载的最小 token 阈值，仅在 DeepSeek V4 系列模型生效。当外部命中 tokens 数 > `load_tokens_threshold` 时触发 KV Cache 加载。 |
| `enable_record_traces` | 选填 | bool | `false` | 用来记录请求信息（时间戳，输入长度，输出长度等信息）。 |
| `enable_metrics` | 选填 | bool | 默认 `true` | 是否开启 metrics 收集。 |
| `use_lite` | 选填 | bool | `false` | 是否启用 UCM Lite 功能。不对 KV Cache 数据进行保存和加载，仅对元数据进行保存和查询。仅可用于评估 KV Cache 命中率情况，无加速效果。 |
| `metrics_config_path` | 选填 | string | 自行配置 | 指定监控指标配置文件路径，启用后可通过 toolkit 进行 UCM 在线监控。详见 [Toolkit 使用指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100582752/e66a9eed#ZH-CN_TOPIC_0000002657360684)。 |

---

## ucm_connector_config（存储后端配置）

> 写在 `ucm_connectors[0].ucm_connector_config` 下的参数。

| 配置项 | 是否必填 | 取值类型 | 取值范围 | 配置说明 |
|---|---|---|---|---|
| `store_pipeline` | 选填 | string | 见下方可选值 | 管线名称，决定 Cache 与 Store 的组合方式。默认使用 `Cache\|Posix`。 |
| `storage_backends` | **必填** | string | 自行配置，多个挂载点用冒号隔开 | 填写创库使用章节中在计算主机的挂载目录。详见 [创建文件系统挂载](https://support.huawei.com/enterprise/zh/doc/EDOC1100582752/47f029f5#ZH-CN_TOPIC_0000002501883936)。 |
| `io_direct` | 选填 | bool | 默认 `true` | 是否启用直接 IO 模式（绕过操作系统页缓存）。`false`: 使用 PageCache；`true`: 跳过 PageCache，直接 IO。 |
| `posix_io_engine` | 选填 | string | 默认 `psync` | 文件 IO 模式。`psync`：同步 io；`aio`：异步 io，要求 `io_direct` 配置为 `true`。 |
| `posix_data_trans_concurrency` | 选填 | int | 默认 `128` | `psync` 模式下单卡对存储的读写线程数。nfs over rdma 下推荐单卡 128 线程，dpc 下单机不超过 256 线程。`aio` 下不读取该值。 |
| `posix_open_concurrency` | 选填 | int | 默认 `32` | `aio` 下对文件进行 open 操作的线程数。`psync` 下不感知。 |
| `posix_commit_concurrency` | 选填 | int | 默认 `4` | `aio` 下对文件进行 rename 操作的线程数。`psync` 下不感知。 |
| `posix_lookup_concurrency` | 选填 | int | 默认 `16` | 在挂载点中查找文件是否存在的线程数。 |
| `cache_buffer_capacity_gb` | 选填 | int | 见下方推荐表 | GQA 模型不配置时默认每卡 32GB；MLA 模型为单个 DP 组所占的 `/dev/shm` 空间。Ascend 环境下单台不超过 200GB。 |
| `cache_sdma_direct` | 选填 | bool | 默认 `true` | 启用 SDMA H2D/D2H 传输路径，仅在 A3 设备生效，推荐关闭。 |
| `cache_load_backend_only` | 选填 | bool | 默认 `false` | 即使在 cache 层命中还是会强制从 SSD 上加载，仅供测试使用。 |
| `cache_io_aggregation` | 选填 | bool | 默认 `false`，仅在 `PLATFORM=ascend` 且模型为 V4 时自动开启 | 启用 IO 聚合 h2d 传输，仅在 A2 设备生效。 |
| `share_buffer_enable` | 选填 | bool | MLA 默认启用，GQA 默认不启用 | 是否启用共享内存。MLA 如果不用 shm 或 GQA 用 shm 都会导致性能下降。 |
| `posix_capacity_gb` | 选填 | int | 默认 `0`，表示不启用 GC；不可超过挂载文件系统可用容量 | 设置磁盘存储的最大容量（GB），当已用容量 >= `posix_capacity_gb * posix_gc_trigger_threshold_ratio` 时触发 GC。 |
| `posix_gc_trigger_threshold_ratio` | 条件选填 | float | 默认 `0.7`，范围：0~1。`posix_capacity_gb` 未配置时不填写 | GC 阈值比例，配合 `posix_capacity_gb` 使用。 |
| `posix_gc_recycle_percent` | 选填 | float | 默认 `0.1`，范围：0~1 | 每轮 GC 删除当前容量的比值。 |
| `posix_gc_max_recycle_count_per_shard` | 选填 | int | 默认 `50000`，>0 | 每轮 GC 单目录允许删除的文件数上限。 |
| `posix_gc_shard_sample_ratio` | 选填 | float | 默认 `0.1`，范围：0~1 | 采样 10% 目录估算总容量。 |
| `posix_gc_check_interval_sec` | 选填 | int | 默认 `30`，>0 | GC 采样及触发间隔。 |
| `posix_gc_concurrency` | 选填 | int | 默认 `16`，>0 | GC 线程池 worker 数。 |
| `posix_gc_task_timeout_ms` | 选填 | int | 默认 `300000`，>0 | 单目录任务超时 watchdog。`0`=禁用。 |
| `posix_gc_precise_mode` | 选填 | bool | 默认 `true` | `true`: 精准模式（全局最冷）；`false`: 性能模式（每目录最冷）。 |

???+ tip "cache_buffer_capacity_gb 推荐配置"
    | 模型类型 | 部署方式 | 推荐值 |
    |---|---|---|
    | GQA（Qwen3/GLM-4.7/MiniMax M2） | — | 默认每卡 32GB（不配置） |
    | MLA（DS V3/R1, Kimi K2, GLM-5） | 单台 A3, TP8DP2 | `96` |
    | MLA（DS V3/R1, Kimi K2, GLM-5） | 两台 A2, TP8DP2 | `192` |
    | DeepSeek V4 系列 | 单台 A3, TP8DP2 | `48` |
    | DeepSeek V4 系列 | 两台 A2, TP8DP2 | `96` |

    !!! warning "上限"
        Ascend 环境下，单台 `cache_buffer_capacity_gb` 不超过 200GB。

### store_pipeline 可选值

| 值 | 说明 |
|---|---|
| `Cache\|Posix` | 正常使用场景 |
| `Cache\|Empty` | MLA 纯 Cache 测试 |
| `Cache\|Fake` | MLA/GQA 纯 Cache 测试（有精度问题） |
| `Empty` | Store 全部接口空实现，引擎侧永远不命中 |
| `Fake` | 不实际存取，仅存元数据 lookup，测试极限性能 |
| `Mooncake` | 对接 Mooncake 内存池，仅支持 vllm-ascend |
| `Mooncake\|Posix` | 对接 Mooncake 内存池并支持落盘 |
| `YuanRong` | 对接 YuanRong 内存池 |
| `YuanRong\|Posix` | 对接 YuanRong 内存池并支持落盘 |

!!! note "storage_backends 须知"
    若 `storage_backends` 参数填写创库使用章节挂载的文件系统，则不可设置 `posix_capacity_gb`。

---

## store_health（ucm_connector_config.store_health）

> 写在 `ucm_connector_config.store_health` 下的参数。

| 配置项 | 是否必填 | 取值类型 | 取值范围 | 配置说明 |
|---|---|---|---|---|
| `enabled` | 选填 | bool | 默认 `false` | 存储隔离机制总开关。开启后给磁盘 KV 缓存增加故障熔断器，磁盘读写频繁超时/报错时自动切断存储。 |
| `health_check_interval_s` | 选填 | int | 默认 `5` | 缓存磁盘健康巡检周期（秒）。必须 >0 且 > `health_check_timeout_s`。 |
| `health_check_timeout_s` | 选填 | int | 默认 `3` | 单次探测超时时间（秒）。必须 >0 且 < `health_check_interval_s`。 |
| `health_window_size` | 选填 | int | 默认 `8` | 故障统计窗口长度。必须为正整数且 >= `failure_threshold`。 |
| `failure_threshold` | 选填 | int | 默认 `2` | 故障触发阈值。必须为正整数且 <= `health_window_size`。 |

???+ info "参数约束关系"
    - `health_check_interval_s` > `health_check_timeout_s` > 0
    - `health_window_size` >= `failure_threshold` > 0

---

## pmr_config（PMR 配置）

!!! warning
    PMR 默认不开启，`use_pmr=false`。PMR 仅在部分模型、框架版本支持。

???+ abstract "适合开启 PMR 的典型任务场景"
    1. 面向长文本的总结摘要生成，能够显著加速对重复或相似信息片段的提炼过程。
    2. 在检索增强生成（RAG）中，可对检索返回的相关文档片段进行快速前缀匹配与候选路径预判。

| 配置项 | 是否必填 | 取值类型 | 取值范围 | 配置说明 |
|---|---|---|---|---|
| `use_pmr` | **必填** | bool | `false` | PMR 算法的开关。为 `true` 时需填写 `pmr_config` 下方参数，否则不填写。 |
| `searchN` | 选填 | int | `5` | 查询时最大前缀匹配长度。 |
| `speculator_length` | 选填 | int | `12` | 单请求最长投机长度。 |
| `num_speculative_tokens` | 选填 | int | `3` | pmr 投机个数。开启 FLASHCOMM1 后需满足 `max(tp, 1+mtp+pmr)` 整除 `min(tp, 1+mtp+pmr)`。 |
| `maxSeqLen` | 选填 | int | `16` | pmr 树的最大深度。 |
| `use_history` | 选填 | bool | `true` | 是否从历史请求中查询。 |
| `token_len_list` | 选填 | list | `[5, 5, 5]` | 单并发时不同 prefix 的序列投机长度：prefix > searchN/2, 1 < prefix < searchN/2, prefix = 1。 |
| `large_bs_token_len_list` | 选填 | list | `[3, 2, 1]` | 大并发时不同 prefix 的序列投机长度：prefix > searchN/2, 1 < prefix < searchN/2, prefix = 1。 |
| `storage_path` | **必填** | string | `/mnt/storages/pmr` | pmr 树的落盘路径。 |
| `model_path` | **必填** | string | `/home/models/Qwen3-32B/` | 模型权重存放目录，同 `vllm serve` 的模型路径保持一致。 |
| `base_core_id` | 选填 | int | `0` | 算法调用的 core ids 的起始 core id。 |
| `thread_num` | 选填 | int | `16` | 算法调用的 core ids 个数，每个 core 上开启一个 thread。 |
| `interval_minutes` | 选填 | int | `400` | 落盘时间，单位分钟。 |
| `memSoftLimitInGB` | 选填 | int | `30` | 开始触发淘汰的阈值，单位 GB。 |
| `memHardLimitInGB` | 选填 | int | `40` | pmr-tree 停止写入的阈值，单位 GB。 |

---

## KVCS 相关配置

!!! note
    KVCS 参数默认不配置。仅在使用 KV Cache Service 时填写。

| 配置项 | 是否必填 | 取值类型 | 取值范围 | 配置说明 |
|---|---|---|---|---|
| `kvcs_store_id` | 默认不配置 | int | — | 获取已创建的 KV Cache 库 ID。填写创库使用章节中存储上已创建的 KV Cache 库 ID。详见 [创建 KV Cache 库](https://support.huawei.com/enterprise/zh/doc/EDOC1100582752/47f029f5#ZH-CN_TOPIC_0000002501883936)。 |
| `kvcs_instance_name` | 默认不配置 | string | 默认 `default_instance` | 推理实例名称，Unified Cache 推理加速服务异常告警上报使用，用于区分推理实例，由用户自定义。 |
| `kvcs_ucm_over_tcp_ip_list` | 默认不配置 | string | — | 创建 UCM over TCP 逻辑端口章节创建的 UCM Over TCP 逻辑端口 IP 列表，支持多 IP 以冒号 ":" 分割，如 `192.168.0.1:192.168.0.2`。详见 [创建 UCM over TCP 逻辑端口](https://support.huawei.com/enterprise/zh/doc/EDOC1100582752/30f71963#ZH-CN_TOPIC_0000002533643827)。 |
| `kvcs_block_size` | 默认不配置 | int | 默认 `128` | 与推理引擎拉起配置参数中的 `block_size` 保持一致，拉起 DeepSeek V4 系列模型时填 `4 * block_size`。 |
| `kvcs_tls_enable` | 默认不配置 | bool | 默认 `false`（关闭 TLS 身份验证） | 推理引擎和存储 GRPC 通信时是否开启 TLS 身份验证。 |
| `kvcs_sliding_window_size` | 默认不配置 | int | 默认 `100`，参数范围 10-1000 | Unified Cache 推理加速服务告警配置：滑动窗口大小，指代统计失败率时的推理请求总数（最近 N 次）。 |
| `kvcs_failure_rate_threshold` | 默认不配置 | int | 默认 `10`，参数范围 0-100 | Unified Cache 推理加速服务告警配置：推理请求失败率阈值，最近 N 次请求中请求失败率大于该阈值时，会隔离该 KV Cache 库，所有推理请求均会完全重算，同时会上报重要级别告警。 |
| `kvcs_consecutive_fail_limit` | 默认不配置 | int | 默认 `5`，参数范围 1-100 | Unified Cache 推理加速服务告警配置：推理请求连续失败次数超过该阈值，此时会隔离该 KV Cache 库，所有推理请求均会完全重算，同时会上报重要级别告警。 |
