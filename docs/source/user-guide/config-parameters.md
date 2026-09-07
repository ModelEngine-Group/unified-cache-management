# UCM Configuration Parameters

The configuration file follows a strict YAML hierarchy. The annotated example
below is a complete reference: write the parameters at exactly the indent level
shown, so that each parameter lands in the right YAML section.

- Top-level parameters are written at the YAML root (column 0), beside `ucm_connectors`.
- Connector parameters are written under `ucm_connectors[0].ucm_connector_config` (6-space indent).
- `store_health` is a nested dict under `ucm_connector_config` (8-space indent).
- Mooncake / YuanRong parameters sit at the same level as Cache / Posix parameters.

:::{dropdown} Full annotated YAML reference
:animate: fade-in

```yaml
# ==================== Top-Level Parameters ====================

# Enable layer-wise (per-layer) load/save mode. Recommended true;
# DeepSeek V4 series recommends false.
# Optional | bool | Default: true
use_layerwise: true

# Performance optimization switch. Recommended to enable.
# Optional | bool | Default: true
enable_event_sync: true

# When request length < persist_token_threshold, UCM does not process the request.
# Optional | int | Default: 0
persist_token_threshold: 0

# Only used in FAWA connector.
# true:  every block's WA cache is dumped (high frequency)
# false: only the last block's WA cache of each chunk prefill is dumped (low frequency)
# Optional | bool | Default: true
wa_dump_block_wise: true

# Minimum token threshold for triggering KV cache loading. Only effective for DeepSeek V4.
# When external hit tokens > load_tokens_threshold, triggers KV Cache loading.
# Optional | int | Default: 2048
load_tokens_threshold: 2048

# Record request information (timestamps, input length, output length, etc.).
# Optional | bool | Default: false
enable_record_traces: false

# Whether to enable metrics collection.
# Optional | bool | Default: true
enable_metrics: true

# Custom metrics config file path. Enables UCM online monitoring via toolkit.
# Reference config: examples/metrics/metrics_configs.yaml
# Optional | string | Default: built-in config
metrics_config_path: "/path/to/metrics_configs.yaml"

# Enable UCM Lite: does not save/load KV Cache data, only saves and queries metadata.
# Used to evaluate KV Cache hit rate - no acceleration effect.
# Optional | bool | Default: false
use_lite: false

# ==================== Connector Config ====================

ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:

      # ---------- Pipeline & Storage ----------

      # Pipeline name. See "store_pipeline Valid Values" below.
      # Required | string | No default
      store_pipeline: "Cache|Posix"

      # Local directory or mount point. Multiple mount points are separated by colons.
      # Note: if using a mounted filesystem, do not set posix_capacity_gb.
      # Required | string | User-configured
      storage_backends: "/mnt/test"

      # Timeout for memory/DRAM/SHM copies and disk read/write (ms).
      # Optional | int | Default: 30000, >0
      timeout_ms: 30000

      # ---------- Cache Store ----------

      # Cache buffer capacity (GB).
      # GQA: 32GB DRAM per card. MLA: 128GB shm space per node.
      # Recommended to use the default values.
      # Optional | int
      cache_buffer_capacity_gb: 32

      # Enable IO aggregation H2D transfer. Only effective on A2 devices.
      # Auto-enabled when PLATFORM=ascend and model is V4.
      # Optional | bool | Default: false
      cache_io_aggregation: false

      # Enable SDMA H2D/D2H transfer. Only effective on A3 devices. Recommended to disable.
      # Depends on build env: true when PLATFORM=ascend-a3, false otherwise.
      # Optional | bool
      cache_sdma_direct: false

      # Force load from SSD even on cache hit. Test only.
      # Optional | bool | Default: false
      cache_load_backend_only: false

      # Enable shared memory.
      # MLA: enabled by default. GQA: disabled by default.
      # MLA without shm or GQA with shm causes performance degradation.
      # Optional | bool
      share_buffer_enable: false

      # ---------- Posix Store ----------

      # Enable Direct I/O (bypass OS page cache).
      # false: uses PageCache. true: skips PageCache, for large sequential I/O.
      # Optional | bool | Default: true
      io_direct: true

      # File I/O mode. psync: synchronous; aio: asynchronous, requires io_direct=true.
      # Optional | string | Default: psync
      posix_io_engine: "psync"

      # Read/write threads per card in psync mode. NFS over RDMA: 128/card. Not used in aio mode.
      # Optional | int | Default: 128
      posix_data_trans_concurrency: 128

      # File open threads in aio mode. Not applicable in psync mode.
      # Optional | int | Default: 32
      posix_open_concurrency: 32

      # File rename threads in aio mode. Not applicable in psync mode.
      # Optional | int | Default: 4
      posix_commit_concurrency: 4

      # Threads for checking file existence at the mount point.
      # Optional | int | Default: 16
      posix_lookup_concurrency: 16

      # ---------- GC (Garbage Collection) ----------

      # Max disk storage capacity (GB). Triggers GC when used >=
      # posix_capacity_gb * posix_gc_trigger_threshold_ratio.
      # 0 = GC disabled. Must not exceed mounted filesystem available capacity.
      # In multi-instance deployments sharing the same filesystem,
      # only one instance should enable GC; others should not.
      # Optional | int | Default: 0
      posix_capacity_gb: 10240

      # GC trigger threshold ratio. Used with posix_capacity_gb.
      # Not set when posix_capacity_gb is not configured.
      # Conditional | float | Default: 0.7, 0-1
      posix_gc_trigger_threshold_ratio: 0.7

      # Ratio of current capacity deleted per GC round.
      # Not set when posix_capacity_gb is not configured.
      # Optional | float | Default: 0.1, 0-1
      posix_gc_recycle_percent: 0.1

      # Max file deletion count per directory per GC round. Not recommended to modify.
      # Not set when posix_capacity_gb is not configured.
      # Optional | int | Default: 50000, >0
      posix_gc_max_recycle_count_per_shard: 50000

      # Sample 10% directories to estimate total capacity.
      # Not set when posix_capacity_gb is not configured.
      # Optional | float | Default: 0.1, 0-1
      posix_gc_shard_sample_ratio: 0.1

      # GC sampling and trigger interval.
      # Not set when posix_capacity_gb is not configured.
      # Optional | int | Default: 30, >0
      posix_gc_check_interval_sec: 30

      # GC thread pool worker count.
      # Not set when posix_capacity_gb is not configured.
      # Optional | int | Default: 16, >0
      posix_gc_concurrency: 16

      # Single directory task timeout watchdog. 0 = disabled.
      # Not set when posix_capacity_gb is not configured.
      # Optional | int | Default: 300000, >0
      posix_gc_task_timeout_ms: 300000

      # true: precise mode (global coldest). false: performance mode (per-directory coldest).
      # Not set when posix_capacity_gb is not configured.
      # Optional | bool | Default: true
      posix_gc_precise_mode: true

      # ---------- store_health (nested dict) ----------

      store_health:
        # Master switch for storage isolation. Adds circuit breaker for disk KV cache.
        # Optional | bool | Default: true
        enabled: true

        # Disk health check interval (sec). Must be >0 and > health_check_timeout_s.
        # Optional | int | Default: 10
        health_check_interval_s: 10

        # Single probe timeout (sec). Must be >0 and < health_check_interval_s.
        # Optional | int | Default: 3
        health_check_timeout_s: 3

        # Fault statistics window. Must be positive and >= failure_threshold.
        # Optional | int | Default: 8
        health_window_size: 8

        # Fault trigger threshold. Must be positive and <= health_window_size.
        # Optional | int | Default: 2
        failure_threshold: 2

      # ---------- Mooncake Store (only when store_pipeline is "Mooncake|Posix") ----------

      # Local hostname for Mooncake store.
      # Optional | string | Default: auto-detected by Mooncake SDK
      local_hostname: "127.0.0.1"

      # Mooncake master server address.
      # Optional | string | Default: 127.0.0.1:50088
      master_server_address: "127.0.0.1:50088"

      # Metadata server mode.
      # Optional | string | Default: P2PHANDSHAKE
      metadata_server: "P2PHANDSHAKE"

      # Transport protocol.
      # Optional | string | Default: ascend
      protocol: "ascend"

      # Global segment size (GB).
      # Optional | int | Default: 30
      global_segment_size_gb: 30

      # Number of replicas.
      # Optional | int | Default: 1
      replica_num: 1

      # Shared buffer capacity (GB) for Mooncake store.
      # Optional | int | Default: 0 (use built-in default)
      share_buffer_capacity_gb: 64

      # ---------- YuanRong Store (only when store_pipeline is "YuanRong|Posix") ----------

      # YuanRong service host address.
      # Required | string | User-configured
      yuanrong_host: "192.168.0.1"

      # YuanRong service port.
      # Required | int | 1-65535
      yuanrong_port: 18483

      # YuanRong namespace. Falls back to unique_id if not set.
      # Optional | string | User-configured
      yuanrong_namespace: "default"

      # Path to YuanRong resource log file.
      # Optional | string | User-configured
      yuanrong_resource_log_path: "/var/log/yuanrong.log"

      # Enable remote H2D (host-to-device) transfer.
      # Optional | bool | Default: true
      yuanrong_enable_remote_h2d: true

      # YuanRong operation timeout (ms).
      # Optional | int | Default: 60000
      yuanrong_timeout_ms: 60000

      # Waiting queue depth.
      # Optional | int | Default: 8192, >1
      yuanrong_waiting_queue_depth: 8192

      # Number of load worker threads.
      # Optional | int | Default: 4, >0
      yuanrong_load_worker_count: 4

      # Workers blocked waiting for vLLM prerequisite events.
      # Optional | int | Default: 2, >0
      yuanrong_dump_prerequisite_worker_count: 2

      # Posix cold recovery pipeline batch size.
      # Optional | int | Default: 32, >0
      yuanrong_recovery_batch_size: 32

      # Number of host buffers. If 0, auto-derived from recovery_batch_size
      # and host_buffer_capacity_gb.
      # Optional | int | Default: 0 (auto-derived)
      yuanrong_host_buffer_count: 0

      # Host buffer capacity per buffer (GB).
      # Optional | int | Default: 8
      yuanrong_host_buffer_capacity_gb: 8

      # Number of H2D streams.
      # Optional | int | Default: 4, >0
      yuanrong_h2d_stream_count: 4

      # Number of backfill worker threads.
      # Optional | int | Default: 1, >0
      yuanrong_backfill_worker_count: 1

      # Backfill queue depth.
      # Optional | int | Default: 128, >0
      yuanrong_backfill_queue_depth: 128

      # Maximum YuanRong buffers held by Posix per UCM process (GB).
      # Optional | int | Default: 1
      yuanrong_posix_max_inflight_gb: 1
```

:::

---

## Parameter Reference

### Top-Level Parameters

| Parameter | Required | Type | Value Range | Description |
|---|---|---|---|---|
| `use_layerwise` | Optional | bool | Default: `true` | Enable layer-wise (per-layer) load/save mode. Recommended `true`; DeepSeek V4 series recommends `false`. |
| `enable_event_sync` | Optional | bool | Default: `true` | Performance optimization switch. Recommended to enable. |
| `persist_token_threshold` | Optional | int | `0` | When request length < `persist_token_threshold`, UCM does not process the request. |
| `wa_dump_block_wise` | Optional | bool | `true` | Only used in FAWA connector. `true`: every block's WA cache is dumped (high frequency); `false`: only dump last block's WA cache of each chunk prefill (low frequency). |
| `load_tokens_threshold` | Optional | int | Default: `2048` | Minimum token threshold for triggering KV cache loading. Only effective for DeepSeek V4 series. When external hit tokens > `load_tokens_threshold`, triggers KV Cache loading. |
| `enable_record_traces` | Optional | bool | `false` | Record request information (timestamps, input length, output length, etc.). |
| `enable_metrics` | Optional | bool | Default: `true` | Whether to enable metrics collection. |
| `metrics_config_path` | Optional | string | User-configured | Custom metrics config file path. Enables UCM online monitoring via toolkit. Reference config: `examples/metrics/metrics_configs.yaml`. |
| `use_lite` | Optional | bool | `false` | Enable UCM Lite. Does not save/load KV Cache data, only saves and queries metadata. Used to evaluate KV Cache hit rate — no acceleration effect. |

### Pipeline & Storage

| Parameter | Required | Type | Value Range | Description |
|---|---|---|---|---|
| `store_pipeline` | **Required** | string | See [store_pipeline Valid Values](#store_pipeline-valid-values) | Pipeline name. Recommended: `Cache\|Posix`. |
| `storage_backends` | **Required** | string | User-configured, multiple mount points separated by `:` | Local directory or mount point. Multiple mount points are separated by colons. |
| `timeout_ms` | Optional | int | Default: `30000`, >0 | Timeout for memory/DRAM/SHM copies and disk read/write (ms). |

### Cache Store

| Parameter | Required | Type | Value Range | Description |
|---|---|---|---|---|
| `cache_buffer_capacity_gb` | Optional | int | See description | For GQA, default is 32GB DRAM per card. For MLA, default is 128GB shm space per node. Recommended to use defaults. |
| `cache_io_aggregation` | Optional | bool | Default: `false`, auto-enabled when `PLATFORM=ascend` and model is V4 | Enable IO aggregation H2D transfer. Only effective on A2 devices. |
| `cache_sdma_direct` | Optional | bool | Depends on build env: `true` when `PLATFORM=ascend-a3`, `false` otherwise | Enable SDMA H2D/D2H transfer. Only effective on A3 devices. Recommended to disable. |
| `cache_load_backend_only` | Optional | bool | Default: `false` | Force load from SSD even on cache hit. Test only. |
| `share_buffer_enable` | Optional | bool | MLA: default enabled; GQA: default disabled | Enable shared memory. MLA without shm or GQA with shm causes performance degradation. |

### Posix Store

| Parameter | Required | Type | Value Range | Description |
|---|---|---|---|---|
| `io_direct` | Optional | bool | Default: `true` | Enable Direct I/O (bypass OS page cache). `false`: uses PageCache; `true`: skips PageCache. |
| `posix_io_engine` | Optional | string | Default: `psync` | File I/O mode. `psync`: synchronous; `aio`: asynchronous, requires `io_direct=true`. |
| `posix_data_trans_concurrency` | Optional | int | Default: `128` | Read/write threads per card in `psync` mode. NFS over RDMA: 128/card. Not used in `aio` mode. |
| `posix_open_concurrency` | Optional | int | Default: `32` | File open threads in `aio` mode. Not applicable in `psync`. |
| `posix_commit_concurrency` | Optional | int | Default: `4` | File rename threads in `aio` mode. Not applicable in `psync`. |
| `posix_lookup_concurrency` | Optional | int | Default: `16` | Threads for checking file existence at mount point. |

### GC (Garbage Collection)

| Parameter | Required | Type | Value Range | Description |
|---|---|---|---|---|
| `posix_capacity_gb` | Optional | int | Default: `0` (no GC); must not exceed mounted filesystem available capacity | Max disk storage capacity (GB). Triggers GC when used >= `posix_capacity_gb * posix_gc_trigger_threshold_ratio`. In multi-instance deployments sharing the same filesystem, only one instance should enable GC; others should not. |
| `posix_gc_trigger_threshold_ratio` | Conditional | float | Default: `0.7`, 0~1. Not set when `posix_capacity_gb` is not configured | GC trigger threshold ratio. Used with `posix_capacity_gb`. |
| `posix_gc_recycle_percent` | Optional | float | Default: `0.1`, 0~1. Not set when `posix_capacity_gb` is not configured | Ratio of current capacity deleted per GC round. |
| `posix_gc_max_recycle_count_per_shard` | Optional | int | Default: `50000`, >0. Not recommended to modify. Not set when `posix_capacity_gb` is not configured | Max file deletion count per directory per GC round. |
| `posix_gc_shard_sample_ratio` | Optional | float | Default: `0.1`, 0~1. Not set when `posix_capacity_gb` is not configured | Sample 10% directories to estimate total capacity. |
| `posix_gc_check_interval_sec` | Optional | int | Default: `30`, >0. Not set when `posix_capacity_gb` is not configured | GC sampling and trigger interval. |
| `posix_gc_concurrency` | Optional | int | Default: `16`, >0. Not set when `posix_capacity_gb` is not configured | GC thread pool worker count. |
| `posix_gc_task_timeout_ms` | Optional | int | Default: `300000`, >0. Not set when `posix_capacity_gb` is not configured | Single directory task timeout watchdog. `0` = disabled. |
| `posix_gc_precise_mode` | Optional | bool | Default: `true`. Not set when `posix_capacity_gb` is not configured | `true`: precise mode (global coldest); `false`: performance mode (per-directory coldest). |

### store_health

| Parameter | Required | Type | Value Range | Description |
|---|---|---|---|---|
| `enabled` | Optional | bool | Default: `true` | Master switch for storage isolation. Adds circuit breaker for disk KV cache. |
| `health_check_interval_s` | Optional | int | Default: `10` | Disk health check interval (sec). Must be >0 and > `health_check_timeout_s`. |
| `health_check_timeout_s` | Optional | int | Default: `3` | Single probe timeout (sec). Must be >0 and < `health_check_interval_s`. |
| `health_window_size` | Optional | int | Default: `8` | Fault statistics window. Must be positive and >= `failure_threshold`. |
| `failure_threshold` | Optional | int | Default: `2` | Fault trigger threshold. Must be positive and <= `health_window_size`. |

### Mooncake Store

> Only applicable when `store_pipeline` is `Mooncake` or `Mooncake|Posix`.

| Parameter | Required | Type | Value Range | Description |
|---|---|---|---|---|
| `local_hostname` | Optional | string | Default: auto-detected by Mooncake SDK | Local hostname for Mooncake store. |
| `master_server_address` | Optional | string | Default: `127.0.0.1:50088` | Mooncake master server address. |
| `metadata_server` | Optional | string | Default: `P2PHANDSHAKE` | Metadata server mode. |
| `protocol` | Optional | string | Default: `ascend` | Transport protocol. |
| `global_segment_size_gb` | Optional | int | Default: `30` | Global segment size (GB). |
| `replica_num` | Optional | int | Default: `1` | Number of replicas. |
| `share_buffer_capacity_gb` | Optional | int | Default: `0` (use built-in default) | Shared buffer capacity (GB) for Mooncake store. |

### YuanRong Store

> Only applicable when `store_pipeline` is `YuanRong` or `YuanRong|Posix`.

| Parameter | Required | Type | Value Range | Description |
|---|---|---|---|---|
| `yuanrong_host` | **Required** | string | User-configured | YuanRong service host address. |
| `yuanrong_port` | **Required** | int | 1–65535 | YuanRong service port. |
| `yuanrong_namespace` | Optional | string | User-configured | YuanRong namespace. Falls back to `unique_id` if not set. |
| `yuanrong_resource_log_path` | Optional | string | User-configured | Path to YuanRong resource log file. |
| `yuanrong_enable_remote_h2d` | Optional | bool | Default: `true` | Enable remote H2D (host-to-device) transfer. |
| `yuanrong_timeout_ms` | Optional | int | Default: `60000` | YuanRong operation timeout (ms). |
| `yuanrong_waiting_queue_depth` | Optional | int | Default: `8192`, >1 | Waiting queue depth. |
| `yuanrong_load_worker_count` | Optional | int | Default: `4`, >0 | Number of load worker threads. |
| `yuanrong_dump_prerequisite_worker_count` | Optional | int | Default: `2`, >0 | Workers blocked waiting for vLLM prerequisite events. |
| `yuanrong_recovery_batch_size` | Optional | int | Default: `32`, >0 | Posix cold recovery pipeline batch size. |
| `yuanrong_host_buffer_count` | Optional | int | Default: `0` (auto-derived) | Number of host buffers. If `0`, auto-derived from `recovery_batch_size` and `host_buffer_capacity_gb`. |
| `yuanrong_host_buffer_capacity_gb` | Optional | int | Default: `8` | Host buffer capacity per buffer (GB). |
| `yuanrong_h2d_stream_count` | Optional | int | Default: `4`, >0 | Number of H2D streams. |
| `yuanrong_backfill_worker_count` | Optional | int | Default: `1`, >0 | Number of backfill worker threads. |
| `yuanrong_backfill_queue_depth` | Optional | int | Default: `128`, >0 | Backfill queue depth. |
| `yuanrong_posix_max_inflight_gb` | Optional | int | Default: `1` | Maximum YuanRong buffers held by Posix per UCM process (GB). |

---

## store_pipeline Valid Values

| Value | Description |
|---|---|
| `Cache\|Posix` | Normal use case |
| `Cache\|Empty` | MLA pure Cache test. No hit once evicted from the Cache layer. |
| `Cache\|Fake` | MLA/GQA pure Cache test. Fake stores block metadata, so it may still hit after eviction - precision risk. |
| `Empty` | All Store interfaces are empty implementations; the engine never hits. |
| `Fake` | No actual load/dump (load and dump are empty implementations), but stores metadata so lookup can hit. Tests UCM peak performance. |
| `Mooncake` | Mooncake memory pool. vllm-ascend only. |
| `Mooncake\|Posix` | Mooncake memory pool with disk persistence. |
| `YuanRong` | YuanRong memory pool. |
| `YuanRong\|Posix` | YuanRong memory pool with disk persistence. |