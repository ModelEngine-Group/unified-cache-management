# UCM Metrics Reference

## 1. Metrics Exported by Default

The tables below use the default `ucm:` prefix. The default configuration contains 78 Counters, 14 Gauges, and 67 Histograms.

See [UCM Health Metrics](health_metrics.md) for Store health metrics and recommended aggregation.

### 1.1 Connector

#### Counters

| Metric                                  | Description                                                       |
| --------------------------------------- | ----------------------------------------------------------------- |
| `ucm:load_bytes_total`                | Cumulative bytes loaded by all`start_load_kv` calls             |
| `ucm:save_bytes_total`                | Cumulative bytes saved by all`wait_for_save` calls              |
| `ucm:total_prefix_query_tokens_total` | Total prefix-cache query tokens observed by the UCM Connector     |
| `ucm:gpu_hbm_hit_tokens_total`        | Prefix tokens already found in GPU/HBM before UCM Lookup          |
| `ucm:ucm_hit_tokens_total`            | Prefix tokens hit by the UCM Connector                            |
| `ucm:total_prefix_query_blocks_total` | Total complete prefix blocks queried by the UCM Connector         |
| `ucm:gpu_hbm_hit_blocks_total`        | Complete prefix blocks already found in GPU/HBM before UCM Lookup |

#### Gauges

No Connector-specific Gauges are exported by default.

#### Histograms

| Metric                                                    | Description                                                                |
| --------------------------------------------------------- | -------------------------------------------------------------------------- |
| `ucm:save_duration`                                     | Duration from entering`wait_for_save` until asynchronous Dump completion |
| `ucm:save_completion_wait_duration`                     | Time actually blocked while confirming asynchronous Dump completion        |
| `ucm:interval_lookup_hit_rates`                         | Per-request UCM Lookup hit-rate distribution                               |
| `ucm:connector_get_block_size_duration_ms`              | Duration of Connector interface`get_block_size`                          |
| `ucm:connector_get_kv_connector_stats_duration_ms`      | Duration of Connector interface`get_kv_connector_stats`                  |
| `ucm:connector_get_num_new_matched_tokens_duration_ms`  | Duration of Connector interface`get_num_new_matched_tokens`              |
| `ucm:connector_update_state_after_alloc_duration_ms`    | Duration of Connector interface`update_state_after_alloc`                |
| `ucm:connector_register_kv_caches_duration_ms`          | Duration of Connector interface`register_kv_caches`                      |
| `ucm:connector_build_connector_meta_duration_ms`        | Duration of Connector interface`build_connector_meta`                    |
| `ucm:connector_bind_connector_metadata_duration_ms`     | Duration of Connector interface`bind_connector_metadata`                 |
| `ucm:connector_handle_preemptions_duration_ms`          | Duration of Connector interface`handle_preemptions`                      |
| `ucm:connector_has_connector_metadata_duration_ms`      | Duration of Connector interface`has_connector_metadata`                  |
| `ucm:connector_start_load_kv_duration_ms`               | Duration of Connector interface`start_load_kv`                           |
| `ucm:connector_wait_for_layer_load_duration_ms`         | Duration of Connector interface`wait_for_layer_load`                     |
| `ucm:connector_save_kv_layer_duration_ms`               | Duration of Connector interface`save_kv_layer`                           |
| `ucm:connector_wait_for_save_duration_ms`               | Duration of Connector interface`wait_for_save`                           |
| `ucm:connector_request_finished_all_groups_duration_ms` | Duration of Connector interface`request_finished_all_groups`             |
| `ucm:connector_request_finished_duration_ms`            | Duration of Connector interface`request_finished`                        |
| `ucm:connector_get_finished_duration_ms`                | Duration of Connector interface`get_finished`                            |
| `ucm:connector_build_connector_worker_meta_duration_ms` | Duration of Connector interface`build_connector_worker_meta`             |
| `ucm:connector_update_connector_output_duration_ms`     | Duration of Connector interface`update_connector_output`                 |
| `ucm:connector_clear_connector_metadata_duration_ms`    | Duration of Connector interface`clear_connector_metadata`                |
| `ucm:layerwise_layer_load_duration_ms`                  | Per-layer wall-clock time from layer load start to `wait_for_layer_load` return |
| `ucm:layerwise_batch_load_duration_sum_ms`              | Sum of per-layer load durations within one Layerwise batch               |

### 1.2 Cache Store

#### Counters

| Metric                                        | Description                                                                |
| --------------------------------------------- | -------------------------------------------------------------------------- |
| `ucm:cache_lookup_hit_blocks_total`         | Blocks served directly by Cache Lookup without descending to the backend   |
| `ucm:cache_lookup_miss_blocks_total`        | Blocks missed by Cache Lookup and passed to the backend                    |
| `ucm:cache_load_shards_total`               | Total shards whose Cache buffer state was inspected during Load            |
| `ucm:cache_load_wait_shards_total`          | Shards whose Cache buffer was not Ready when acquired and required waiting |
| `ucm:cache_load_backend_shards_total`       | Shards that descend to the backend during Cache buffer allocation          |
| `ucm:cache_load_success_shards_total`       | Shards successfully loaded from an already-ready Cache buffer to device    |
| `ucm:cache_posix_load_success_shards_total` | Shards successfully loaded to device after waiting for Posix to fill Cache |
| `ucm:cache_dump_shards_total`               | Total shard descriptors processed by Cache Dump, including failed tasks    |
| `ucm:cache_dump_backend_shards_total`       | Owner shards actually written to the backend                               |
| `ucm:cache_load_bytes_total`                | Cumulative bytes loaded through the Cache stage                            |
| `ucm:cache_dump_bytes_total`                | Cumulative bytes dumped through the Cache stage                            |

For a Cache | Posix pipeline, the Cache load share is `(total shards - wait shards) / total shards`, and the Posix load share is `wait shards / total shards`. Grafana and Metrics View apply these shares to the external-cache hit rate.

#### Gauges

No Cache Store-specific Gauges are exported by default.

#### Histograms

| Metric                                        | Description                                                                              |
| --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `ucm:cache_lookup_duration_ms`              | Wall-clock time of one Cache buffer`Lookup`/`LookupOnPrefix` call                    |
| `ucm:cache_lookup_backend_duration_ms`      | Backend Lookup wall-clock time when there is no buffer or the buffer misses              |
| `ucm:cache_load_duration_ms`                | End-to-end Cache-stage Load task duration                                                |
| `ucm:cache_dump_duration_ms`                | End-to-end Cache-stage Dump task duration                                                |
| `ucm:cache_load_bandwidth_gbps`             | Effective bandwidth over the complete Cache Load task lifecycle                          |
| `ucm:cache_dump_bandwidth_gbps`             | Effective bandwidth over the complete Cache Dump task lifecycle                          |
| `ucm:cache_load_queue_wait_duration_ms`     | Time a Cache Load task waits before a dispatch worker picks it up                        |
| `ucm:cache_dump_queue_wait_duration_ms`     | Time a Cache Dump task waits before a dispatch worker picks it up                        |
| `ucm:cache_load_backend_submit_duration_ms` | Time to allocate a Cache buffer and synchronously submit the backend Load                |
| `ucm:cache_shard_backend_wait_ms`           | Time one shard waits for the backend to become ready before H2D submission               |
| `ucm:cache_h2d_submit_ms`                   | CPU overhead of one asynchronous shard H2D submission, excluding transfer time           |
| `ucm:cache_h2d_sync_ms`                     | Remaining H2D stream drain time after the final shard submission                         |
| `ucm:cache_dump_mkbuf_duration_ms`          | Cache Dump buffer allocation/reuse and asynchronous D2H submission time                  |
| `ucm:cache_dump_prereq_wait_ms`             | Time waiting for the layer KV-ready compute event before D2H starts                      |
| `ucm:cache_d2h_duration_ms`                 | Cache Dump stream synchronization time, including prerequisite compute wait and D2H copy |
| `ucm:cache_dump_backend_submit_duration_ms` | Time to synchronously submit the buffer to the lower Store                               |
| `ucm:cache_dump_backend_wait_duration_ms`   | Time waiting for the lower Store to complete the write                                   |

### 1.3 Posix Store

#### Counters

| Metric                                  | Description                                                 |
| --------------------------------------- | ----------------------------------------------------------- |
| `ucm:posix_s2h_bytes_total`           | Cumulative bytes read from Posix storage into host buffers  |
| `ucm:posix_h2s_bytes_total`           | Cumulative bytes written from host buffers to Posix storage |
| `ucm:posix_lookup_query_blocks_total` | Total blocks submitted to Posix Lookup                      |
| `ucm:posix_lookup_hit_blocks_total`   | Blocks found by Posix Lookup                                |

#### Gauges

| Metric                             | Description                                                    |
| ---------------------------------- | -------------------------------------------------------------- |
| `ucm:posix_store_used_bytes`     | Estimated logical Posix Store usage in bytes from GC sampling  |
| `ucm:posix_store_capacity_bytes` | Configured logical Posix Store capacity in bytes               |
| `ucm:posix_store_usage_ratio`    | Estimated logical Posix Store usage ratio                      |
| `ucm:posix_store_health`         | Effective circuit-breaker state: 1 is available and 0 is fused |
| `ucm:posix_gc_running`           | Garbage collection state: 1 is running and 0 is idle           |

#### Histograms

| Metric                                    | Description                                                                         |
| ----------------------------------------- | ----------------------------------------------------------------------------------- |
| `ucm:posix_load_task_duration_ms`       | End-to-end Posix Load task duration from submission until the final shard completes |
| `ucm:posix_dump_task_duration_ms`       | End-to-end Posix Dump task duration from submission until the final shard completes |
| `ucm:posix_s2h_bandwidth_gbps`          | Per-task Posix read bandwidth                                                       |
| `ucm:posix_h2s_bandwidth_gbps`          | Per-task Posix write bandwidth                                                      |
| `ucm:posix_load_queue_wait_duration_ms` | Time a Posix Load task waits before the first worker picks it up                    |
| `ucm:posix_dump_queue_wait_duration_ms` | Time a Posix Dump task waits before the first worker picks it up                    |

### 1.4 YuanRong Store

#### Counters

| Metric                                                         | Description                                                                   |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `ucm:yuanrong_load_success_shards_total`                     | Shards successfully loaded from YuanRong to device                            |
| `ucm:yuanrong_lookup_miss_posix_load_success_shards_total`   | Shards successfully loaded from Posix after a YuanRong Lookup miss            |
| `ucm:yuanrong_load_fallback_posix_load_success_shards_total` | Shards successfully loaded from Posix after a YuanRong Load failure           |
| `ucm:yuanrong_local_dram_load_hits_total`                    | Estimated YuanRong local DRAM Get hits forwarded from`kv_resource.log`      |
| `ucm:yuanrong_remote_load_hits_total`                        | Estimated YuanRong remote-worker Get hits forwarded from`kv_resource.log`   |
| `ucm:yuanrong_local_ssd_load_hits_total`                     | Estimated YuanRong local spill-SSD Get hits forwarded from`kv_resource.log` |
| `ucm:yuanrong_l2_load_hits_total`                            | YuanRong L2 persistence Get hits forwarded from`kv_resource.log`            |

#### Gauges

| Metric                                                      | Description                                                            |
| ----------------------------------------------------------- | ---------------------------------------------------------------------- |
| `ucm:yuanrong_dram_used_bytes`                            | YuanRong physical shared-memory usage in bytes                         |
| `ucm:yuanrong_dram_capacity_bytes`                        | YuanRong shared-memory capacity in bytes                               |
| `ucm:yuanrong_dram_usage_ratio`                           | YuanRong physical shared-memory usage ratio                            |
| `ucm:yuanrong_ssd_used_bytes`                             | YuanRong physical spill-disk usage in bytes                            |
| `ucm:yuanrong_ssd_capacity_bytes`                         | YuanRong spill-disk capacity in bytes                                  |
| `ucm:yuanrong_ssd_usage_ratio`                            | YuanRong physical spill-disk usage ratio                               |
| `ucm:yuanrong_resource_log_last_update_timestamp_seconds` | Unix timestamp of the latest YuanRong resource snapshot parsed by UCM  |
| `ucm:yuanrong_resource_log_reporter_leader`               | Whether this UCM process is the host YuanRong resource reporter leader |

#### Histograms

No YuanRong-specific Histograms are exported by default.

### 1.5 Mooncake Store

#### Counters

| Metric                                      | Description                                                                 |
| ------------------------------------------- | --------------------------------------------------------------------------- |
| `ucm:mooncake_load_blocks_total`          | Total blocks processed by the Mooncake Load stage                           |
| `ucm:mooncake_dump_blocks_total`          | Total blocks processed by the Mooncake Dump stage                           |
| `ucm:mooncake_lookup_hit_blocks_total`    | Blocks found directly by Mooncake Lookup before descending to the backend   |
| `ucm:mooncake_load_bytes_total`           | Cumulative bytes loaded through the Mooncake stage                          |
| `ucm:mooncake_dump_bytes_total`           | Cumulative bytes dumped through the Mooncake stage                          |
| `ucm:mooncake_load_hit_shards_total`      | Load shards served directly by Mooncake                                     |
| `ucm:mooncake_load_miss_shards_total`     | Load shards that miss Mooncake and descend to the backend or are recomputed |
| `ucm:mooncake_load_backend_shards_total`  | Load shards submitted to the backend after a Mooncake miss                  |
| `ucm:mooncake_dump_existing_shards_total` | Dump shards already present in Mooncake                                     |
| `ucm:mooncake_dump_missing_shards_total`  | Missing Dump shards written to Mooncake                                     |
| `ucm:mooncake_dump_backend_shards_total`  | Dump shards archived to the backend                                         |
| `ucm:mooncake_h2d_bytes_total`            | Cumulative bytes copied from host to device by Mooncake                     |
| `ucm:mooncake_d2h_bytes_total`            | Cumulative bytes copied from device to host by Mooncake                     |

#### Gauges

| Metric                        | Description                                                    |
| ----------------------------- | -------------------------------------------------------------- |
| `ucm:mooncake_store_health` | Effective circuit-breaker state: 1 is available and 0 is fused |

#### Histograms

| Metric                                           | Description                                                           |
| ------------------------------------------------ | --------------------------------------------------------------------- |
| `ucm:mooncake_load_duration_ms`                | End-to-end Mooncake Load task duration                                |
| `ucm:mooncake_dump_duration_ms`                | End-to-end Mooncake Dump task duration                                |
| `ucm:mooncake_load_bandwidth_gbps`             | Effective Mooncake-stage Load bandwidth                               |
| `ucm:mooncake_dump_bandwidth_gbps`             | Effective Mooncake-stage Dump bandwidth                               |
| `ucm:mooncake_load_queue_wait_duration_ms`     | Time a Mooncake Load task waits before a dispatch worker picks it up  |
| `ucm:mooncake_dump_queue_wait_duration_ms`     | Time a Mooncake Dump task waits before a dispatch worker picks it up  |
| `ucm:mooncake_get_duration_ms`                 | Mooncake batch-get duration on the Load path                          |
| `ucm:mooncake_exists_duration_ms`              | Mooncake batch-exists check duration on the Dump path                 |
| `ucm:mooncake_put_duration_ms`                 | Mooncake batch-put duration on the Dump path                          |
| `ucm:mooncake_load_backend_submit_duration_ms` | Time to submit a backend Load after a Mooncake miss                   |
| `ucm:mooncake_backend_load_wait_duration_ms`   | Time waiting for the backend to load missing shards                   |
| `ucm:mooncake_h2d_duration_ms`                 | Mooncake Load H2D stream drain time                                   |
| `ucm:mooncake_dump_prereq_wait_ms`             | Time waiting for the prerequisite compute event before a Mooncake put |
| `ucm:mooncake_d2h_duration_ms`                 | Mooncake D2H stream drain time required for backend archival          |
| `ucm:mooncake_dump_backend_submit_duration_ms` | Time to submit a backend Dump after the D2H archival copy             |
| `ucm:mooncake_dump_backend_wait_duration_ms`   | Time waiting for backend archival to complete                         |

## 2. Raw Metrics Usage

### 2.1 Hit Rate

Use a layered calculation instead of calculating each Store hit rate independently. First calculate the total hit rate at the boundary of two adjacent cache layers. Then split that total according to the actual load-source ratio reported by those layers. This keeps the numerator and denominator semantics consistent across the hierarchy and provides the most accurate tier-level estimate.

For example, the external-cache hit rate can be calculated from vLLM token Counters:

```text
external_cache_hit_rate = external_hit / external_query
                          * (1 - hbm_hit / hbm_query)
```

For a Cache | Posix pipeline, split the external-cache hit rate using shard Counters:

```text
cache_share = (cache_load_shards - cache_load_wait_shards)
              / cache_load_shards
posix_share = cache_load_wait_shards / cache_load_shards

cache_hit_rate = external_cache_hit_rate * cache_share
posix_hit_rate = external_cache_hit_rate * posix_share
```

Apply the same approach to other layered Stores: calculate their combined hit rate first, then split it using Counters that represent the actual load source. Use `rate()` or `increase()` over the same time range for every Counter in a formula, and do not mix token, block, and shard counts within the same ratio.

### 2.2 Bandwidth

UCM exposes two useful bandwidth views with different meanings:

- The `*_bandwidth_gbps` Histograms record effective bandwidth for individual tasks. They show instantaneous task speed and its distribution, such as p50, p90, and p99, but they do not represent total system throughput.
- Average system bandwidth should be calculated as total transferred bytes divided by elapsed time. It includes concurrency and idle periods, so it represents the overall throughput observed during the selected time range.

Use a cumulative byte Counter for the required Store and transfer direction. For example:

```promql
sum(rate(ucm:cache_load_bytes_total[$__rate_interval])) / 1e9
```

Equivalently, for a fixed time range:

```text
average_bandwidth_GBps = increase(transferred_bytes_total) / elapsed_seconds / 1e9
```

Sum byte rates across workers before converting to GB/s. Keep Load and Dump, or read and write, as separate series. Do not average per-task bandwidth values to estimate system throughput.
