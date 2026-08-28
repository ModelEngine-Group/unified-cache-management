# UCM Configuration Parameters

---

## Top-Level Parameters

> Global parameters written directly at YAML root level.

| Parameter | Required | Type | Value Range | Description |
|---|---|---|---|---|
| `use_layerwise` | Optional | bool | Default: `true` | Enable layer-wise (per-layer) load/save mode. Recommended `true`; DeepSeek V4 series recommends `false`. |
| `enable_event_sync` | Optional | bool | Default: `true` | Performance optimization switch. Recommended to enable. |
| `persist_token_threshold` | Optional | int | `0` | When request length < `persist_token_threshold`, UCM does not process the request. |
| `timeout_ms` | Optional | int | >0, default `30000` | Timeout for memory/DRAM/SHM copies and disk read/write (ms). |
| `wa_dump_block_wise` | Optional | bool | `true` | Only used in FAWA connector. `true`: every block's WA cache is dumped (high frequency); `false`: only dump last block's WA cache of each chunk prefill (low frequency). |
| `load_tokens_threshold` | Optional | int | `0` | Minimum token threshold for triggering KV cache loading. Only effective for DeepSeek V4 series. When external hit tokens > `load_tokens_threshold`, triggers KV Cache loading. |
| `enable_record_traces` | Optional | bool | `false` | Record request information (timestamps, input length, output length, etc.). |
| `enable_metrics` | Optional | bool | Default: `true` | Whether to enable metrics collection. |
| `use_lite` | Optional | bool | `false` | Enable UCM Lite. Does not save/load KV Cache data, only saves and queries metadata. Used to evaluate KV Cache hit rate — no acceleration effect. |
| `metrics_config_path` | Optional | string | User-configured | Custom metrics config file path. Enables UCM online monitoring via toolkit. See [Toolkit Guide](https://support.huawei.com/enterprise/zh/doc/EDOC1100582752/e66a9eed#ZH-CN_TOPIC_0000002657360684). |

---

## ucm_connector_config (Storage Backend Configuration)

> Parameters under `ucm_connectors[0].ucm_connector_config`.

| Parameter | Required | Type | Value Range | Description |
|---|---|---|---|---|
| `store_pipeline` | Optional | string | See valid values below | Pipeline name. Default: `Cache\|Posix`. |
| `storage_backends` | **Required** | string | User-configured, multiple mount points separated by `:` | Storage mount path. See [Create Filesystem Mount](https://support.huawei.com/enterprise/zh/doc/EDOC1100582752/47f029f5#ZH-CN_TOPIC_0000002501883936). |
| `io_direct` | Optional | bool | Default: `true` | Enable Direct I/O (bypass OS page cache). `false`: uses PageCache; `true`: skips PageCache. |
| `posix_io_engine` | Optional | string | Default: `psync` | File I/O mode. `psync`: synchronous; `aio`: asynchronous, requires `io_direct=true`. |
| `posix_data_trans_concurrency` | Optional | int | Default: `128` | Read/write threads per card in `psync` mode. NFS over RDMA: 128/card. Not used in `aio` mode. |
| `posix_open_concurrency` | Optional | int | Default: `32` | File open threads in `aio` mode. Not applicable in `psync`. |
| `posix_commit_concurrency` | Optional | int | Default: `4` | File rename threads in `aio` mode. Not applicable in `psync`. |
| `posix_lookup_concurrency` | Optional | int | Default: `16` | Threads for checking file existence at mount point. |
| `cache_buffer_capacity_gb` | Optional | int | See recommendation table | GQA: default 32GB/card; MLA: `/dev/shm` space per DP group. Ascend: max 200GB per node. |
| `cache_sdma_direct` | Optional | bool | Default: `true` | Enable SDMA H2D/D2H transfer. Only effective on A3 devices. Recommended to disable. |
| `cache_load_backend_only` | Optional | bool | Default: `false` | Force load from SSD even on cache hit. Test only. |
| `cache_io_aggregation` | Optional | bool | Default: `false`, auto-enabled when `PLATFORM=ascend` and model is V4 | Enable IO aggregation H2D transfer. Only effective on A2 devices. |
| `share_buffer_enable` | Optional | bool | MLA: default enabled; GQA: default disabled | Enable shared memory. MLA without shm or GQA with shm causes performance degradation. |
| `posix_capacity_gb` | Optional | int | Default: `0` (no GC) | Max disk storage capacity (GB). Triggers GC when used >= `posix_capacity_gb * posix_gc_trigger_threshold_ratio`. |
| `posix_gc_trigger_threshold_ratio` | Conditional | float | Default: `0.7`, 0~1 | GC trigger threshold ratio. Used with `posix_capacity_gb`. |
| `posix_gc_recycle_percent` | Optional | float | Default: `0.1`, 0~1 | Ratio of current capacity deleted per GC round. |
| `posix_gc_max_recycle_count_per_shard` | Optional | int | Default: `50000`, >0 | Max file deletion count per directory per GC round. |
| `posix_gc_shard_sample_ratio` | Optional | float | Default: `0.1`, 0~1 | Sample 10% directories to estimate total capacity. |
| `posix_gc_check_interval_sec` | Optional | int | Default: `30`, >0 | GC sampling and trigger interval. |
| `posix_gc_concurrency` | Optional | int | Default: `16`, >0 | GC thread pool worker count. |
| `posix_gc_task_timeout_ms` | Optional | int | Default: `300000`, >0 | Single directory task timeout watchdog. `0` = disabled. |
| `posix_gc_precise_mode` | Optional | bool | Default: `true` | `true`: precise mode (global coldest); `false`: performance mode (per-directory coldest). |

???+ tip "cache_buffer_capacity_gb Recommendations"
    | Model | Deployment | Recommended |
    |---|---|---|
    | GQA (Qwen3/GLM-4.7/MiniMax M2) | — | Default 32GB/card (not configured) |
    | MLA (DS V3/R1, Kimi K2, GLM-5) | Single A3, TP8DP2 | `96` |
    | MLA (DS V3/R1, Kimi K2, GLM-5) | Two A2, TP8DP2 | `192` |
    | DeepSeek V4 series | Single A3, TP8DP2 | `48` |
    | DeepSeek V4 series | Two A2, TP8DP2 | `96` |

    !!! warning "Limit"
        Ascend: max 200GB per node.

### store_pipeline Valid Values

| Value | Description |
|---|---|
| `Cache\|Posix` | Normal use case |
| `Cache\|Empty` | MLA pure Cache test |
| `Cache\|Fake` | MLA/GQA pure Cache test (precision risk) |
| `Empty` | All Store interfaces empty; engine never hits |
| `Fake` | No actual load/dump, stores metadata only; tests peak performance |
| `Mooncake` | Mooncake memory pool; vllm-ascend only |
| `Mooncake\|Posix` | Mooncake memory pool with disk persistence |
| `YuanRong` | YuanRong memory pool |
| `YuanRong\|Posix` | YuanRong memory pool with disk persistence |

!!! note "storage_backends Note"
    If `storage_backends` uses the filesystem mounted per "Create Repository" chapter, do not set `posix_capacity_gb`.

---

## store_health (ucm_connector_config.store_health)

> Parameters under `ucm_connector_config.store_health`.

| Parameter | Required | Type | Value Range | Description |
|---|---|---|---|---|
| `enabled` | Optional | bool | Default: `false` | Master switch for storage isolation. Adds circuit breaker for disk KV cache. |
| `health_check_interval_s` | Optional | int | Default: `5` | Disk health check interval (sec). Must be >0 and > `health_check_timeout_s`. |
| `health_check_timeout_s` | Optional | int | Default: `3` | Single probe timeout (sec). Must be >0 and < `health_check_interval_s`. |
| `health_window_size` | Optional | int | Default: `8` | Fault statistics window. Must be positive and >= `failure_threshold`. |
| `failure_threshold` | Optional | int | Default: `2` | Fault trigger threshold. Must be positive and <= `health_window_size`. |

???+ info "Parameter Constraints"
    - `health_check_interval_s` > `health_check_timeout_s` > 0
    - `health_window_size` >= `failure_threshold` > 0


