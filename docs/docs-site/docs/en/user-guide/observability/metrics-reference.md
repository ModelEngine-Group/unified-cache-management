# Metrics Reference

UCM metrics record KV-cache lookups, loads, saves and storage state. Use them to understand **whether cached data is being reused, how fast data moves, where time is spent and whether storage is available**. Start with the question you need to answer, then consult the full catalog below.

| Question | Start with | How to read it |
| --- | --- | --- |
| Is caching helping? | Connector query and hit token counts, plus loaded bytes | Over the same time window, hit tokens divided by query tokens show the contribution of reusable prefixes; loaded bytes confirm transfer activity. Compare engine TTFT to determine whether time to first token improves |
| How fast does data move? | The active Store's `*_bytes_total` and `*_bandwidth_gbps` | The byte Counter's per-second rate shows aggregate throughput; the bandwidth Histogram shows individual task speed. Low throughput under light load does not necessarily mean slow storage |
| Where do cache operations spend time? | `*_duration_ms`, then queuing, backend waits and H2D/D2H timings | Start with the whole task, then distinguish queuing, lower-store waits and host/device copies. Compare similar task sizes and concurrency |
| Is storage nearly full or unavailable? | `*_usage_ratio`, `*_store_health` and, for Posix, `posix_gc_running` | Usage shows capacity pressure, health shows whether new operations are allowed, and GC shows reclamation activity. High occupancy and backend failure are different conditions |

**Check the metric type before interpreting its value.** Counters, Gauges and Histograms need different queries:

| Type | What it records | Reading and example |
| --- | --- | --- |
| Counter | A cumulative quantity since process startup, usually ending in `_total`; it can reset on restart | Use `rate(...[5m])` for the average per-second rate over five minutes and `increase(...[5m])` for the increase during that window. `load_bytes_total` alone is cumulative loaded bytes, not current bandwidth |
| Gauge | The latest reported capacity, ratio or state; it can rise or fall | Read the value and its trend directly. For example, `posix_store_usage_ratio = 0.8` means estimated logical occupancy is 80%; `posix_store_health = 0` means the circuit breaker is open |
| Histogram | Samples of operation durations or bandwidth | Use the mean for the overall level and quantiles for the distribution. A duration p95 of 20 ms means approximately 95% of sampled operations took no more than 20 ms; it is not the duration of each user request |

Check units too: `*_bytes*` uses bytes and `*_duration_ms` uses milliseconds. **`save_duration` and `save_completion_wait_duration` also use milliseconds**, despite lacking an `_ms` suffix. `*_bandwidth_gbps` actually uses **GB/s (one billion bytes per second)**. H2D means host memory to device memory, and D2H is the reverse; Posix S2H/H2S describe reads and writes between storage and host memory.

Compare the same model, service and time window. Keep `instance`, `engine` and `worker_rank` labels when diagnosing a problem. A KV block passing through the Connector and several Stores is counted at each stage, so adding their byte counts does not give a unique data volume.

Names use the `ucm:` prefix by default. A series may appear only after the metric is enabled and its code path executes; missing does not mean zero. The vLLM connector path also depends on statistics being synchronized during inference, so idle values may remain at their last reported state.

See [metrics setup](metrics.md) for collection, [metrics development](../../developer-guide/add-metrics.md) for the export path and [storage health](health-metrics.md) for health interpretation.

## 1. Metric catalog

### 1.1 Connector

The Connector joins the inference engine to UCM. Start here to understand **how much prefix data can be reused and how long engine calls into the cache take**. Query tokens form the counting base, HBM hits describe prefixes already covered on the device, and UCM hits describe additional external prefixes. These are not request counts.

For slow saves, compare `save_duration` with `save_completion_wait_duration`: the former spans the asynchronous save, while the latter measures actual blocking when confirming completion. `connector_*_duration_ms` measures interface calls; transfers may continue after asynchronous calls return, so interface time does not replace full I/O time.

#### Counters

| Metric                                  | Description                                                       |
| --------------------------------------- | ----------------------------------------------------------------- |
| `ucm:load_bytes_total`                | Cumulative bytes successfully loaded through Connector transfer paths |
| `ucm:save_bytes_total`                | Cumulative bytes successfully submitted through Connector save paths  |
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
| `ucm:save_duration` | Time from entering `wait_for_save` until asynchronous Dump completion, in ms; use it to track overall save duration |
| `ucm:save_completion_wait_duration` | Time actually blocked while confirming asynchronous Dump completion, in ms; use it to track the remaining wait imposed on the caller |
| `ucm:interval_lookup_hit_rates`                         | Per-request UCM Lookup hit-rate distribution[^export-scope] |
| `ucm:connector_get_block_size_duration_ms`              | Duration of Connector interface `get_block_size`                          |
| `ucm:connector_get_kv_connector_stats_duration_ms`      | Duration of Connector interface `get_kv_connector_stats`                  |
| `ucm:connector_get_num_new_matched_tokens_duration_ms`  | Duration of Connector interface `get_num_new_matched_tokens`              |
| `ucm:connector_update_state_after_alloc_duration_ms`    | Duration of Connector interface `update_state_after_alloc`                |
| `ucm:connector_register_kv_caches_duration_ms`          | Duration of Connector interface `register_kv_caches`                      |
| `ucm:connector_build_connector_meta_duration_ms`        | Duration of Connector interface `build_connector_meta`                    |
| `ucm:connector_bind_connector_metadata_duration_ms`     | Duration of Connector interface `bind_connector_metadata`                 |
| `ucm:connector_handle_preemptions_duration_ms`          | Duration of Connector interface `handle_preemptions`                      |
| `ucm:connector_has_connector_metadata_duration_ms`      | Duration of Connector interface `has_connector_metadata`                  |
| `ucm:connector_start_load_kv_duration_ms`               | Duration of Connector interface `start_load_kv`                           |
| `ucm:connector_wait_for_layer_load_duration_ms`         | Duration of Connector interface `wait_for_layer_load`                     |
| `ucm:connector_save_kv_layer_duration_ms`               | Duration of Connector interface `save_kv_layer`                           |
| `ucm:connector_wait_for_save_duration_ms`               | Duration of Connector interface `wait_for_save`                           |
| `ucm:connector_request_finished_all_groups_duration_ms` | Duration of Connector interface `request_finished_all_groups`             |
| `ucm:connector_request_finished_duration_ms`            | Duration of Connector interface `request_finished`                        |
| `ucm:connector_get_finished_duration_ms`                | Duration of Connector interface `get_finished`                            |
| `ucm:connector_build_connector_worker_meta_duration_ms` | Duration of Connector interface `build_connector_worker_meta`             |
| `ucm:connector_update_connector_output_duration_ms`     | Duration of Connector interface `update_connector_output`                 |
| `ucm:connector_clear_connector_metadata_duration_ms`    | Duration of Connector interface `clear_connector_metadata`                |
| `ucm:layerwise_layer_load_duration_ms`                  | Per-layer wall-clock time from layer load start to `wait_for_layer_load` return |
| `ucm:layerwise_batch_load_duration_sum_ms`              | Sum of per-layer load durations within one Layerwise batch               |

### 1.2 Cache Store

Cache Store uses host buffers between the device and lower storage. Use it to understand **whether a load finds ready data immediately or waits for the backend**. Start with the increase in `cache_load_wait_shards_total` relative to `cache_load_shards_total`, then check successful-load shard counts for completion. A rising wait share means fewer shards are immediately ready; it does not directly imply more disk reads.

When loads slow down, compare queuing, backend-ready waits and H2D synchronization. For saves, compare prerequisite compute waits, D2H and backend-write waits. These timings can overlap and mix shard and task granularity, so adding them does not reconstruct total duration.

#### Counters

| Metric                                        | Description                                                                |
| --------------------------------------------- | -------------------------------------------------------------------------- |
| `ucm:cache_lookup_hit_blocks_total`         | Blocks served directly by Cache Lookup without descending to the backend   |
| `ucm:cache_lookup_miss_blocks_total`        | Blocks missed by Cache Lookup and passed to the backend                    |
| `ucm:cache_load_shards_total`               | Total shards whose Cache buffer state was inspected during Load            |
| `ucm:cache_load_wait_shards_total`          | Shards whose Cache buffer was not Ready when acquired and required waiting |
| `ucm:cache_load_backend_shards_total`       | Shards that descend to the backend during Cache buffer allocation          |
| `ucm:cache_load_success_shards_total`       | Shards successfully loaded from an already-ready Cache buffer to device    |
| `ucm:cache_posix_load_success_shards_total` | Shards successfully loaded to device after waiting for Posix to fill Cache[^cache-source] |
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
| `ucm:cache_lookup_duration_ms`              | Cache buffer scan time for `Lookup`, `LookupOnPrefix`, or `LookupOnReverse`; no samples are produced when shared memory is disabled because Lookup goes directly to the lower Store |
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

Posix Store reads and writes KV data through the filesystem. Use it to understand **file-backend hits, read/write speed and capacity pressure**. The ratio of increases in `posix_lookup_hit_blocks_total` and `posix_lookup_query_blocks_total` describes only blocks actually queried at the Posix layer.

For slow I/O, compare task duration with queue wait. For capacity, view logical usage alongside `posix_gc_running`. Running GC is normal reclamation activity; `posix_store_health = 0` means the circuit breaker is blocking new operations. Logical usage is estimated from GC samples and does not replace checking host filesystem free space.

#### Counters

| Metric                                  | Description                                                 |
| --------------------------------------- | ----------------------------------------------------------- |
| `ucm:posix_s2h_bytes_total`           | Cumulative bytes read from Posix storage into host buffers[^posix-bytes] |
| `ucm:posix_h2s_bytes_total`           | Cumulative bytes written from host buffers to Posix storage[^posix-bytes] |
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

YuanRong metrics explain **which tier serves loads, whether loads fall back to Posix, and shared-memory and spill-disk occupancy**. An increase in `lookup_miss_posix_load_success` means Posix supplied data after a lookup miss. An increase in `load_fallback_posix_load_success` means fallback succeeded after a load failure; investigate the associated error logs.

DRAM, remote-worker and SSD hit estimates come from resource logs, so do not divide them by Connector token counts. Before reading capacity or source distributions, check that `yuanrong_resource_log_last_update_timestamp_seconds` is advancing. `reporter_leader = 1` identifies the reporting process; it is not a backend health signal.

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

Mooncake metrics explain **whether Mooncake supplies the data, whether loads descend to a backend and which step slows transfers**. Compare increases in `mooncake_load_hit_shards_total` and `mooncake_load_miss_shards_total`, then check `mooncake_load_backend_shards_total` for lower-store load activity.

For slow loads, inspect queuing, batch-get, backend waits and H2D. For slow saves, inspect prerequisite compute waits, batch-put and backend archival. Increasing `mooncake_dump_existing_shards_total` means the data already existed at save time, not a failure by itself. If the health Gauge is zero, follow [storage health](health-metrics.md) to investigate the open circuit breaker.

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

Run the PromQL examples in Prometheus or against a Prometheus data source in Grafana. They assume the scrape job is named `vllm`; change `job` to match your deployment. If a job contains several models or services, add label filters before aggregating. Examples use the last five minutes; in Grafana, `[5m]` can be replaced with `[$__rate_interval]`.

### 2.1 Hit Rate

Start with the direct question: **of the prefix tokens queried by the Connector, how many had reusable data found by UCM?**

```promql
sum(rate(ucm:ucm_hit_tokens_total{job="vllm"}[5m]))
/
sum(rate(ucm:total_prefix_query_tokens_total{job="vllm"}[5m]))
```

A result of `0.3` means UCM hit tokens account for 30% of Connector query tokens in this window. It does not mean 30% of requests fully hit or latency fell by 30%. The denominator includes tokens already covered by HBM, so this is UCM's hit contribution to all queried prefixes. With no queries in the window, the ratio is undefined and should be shown as no data.

Use a layered calculation instead of calculating each Store hit rate independently. First calculate the total hit rate at the boundary of two adjacent cache layers. Then split that total according to the actual load-source ratio reported by those layers. This keeps the numerator and denominator semantics consistent across the hierarchy and provides the most accurate tier-level estimate.

For example, the external-cache hit rate can be calculated from vLLM token Counters:

```text
external_cache_hit_rate = external_hit / external_query
                          * (1 - hbm_hit / hbm_query)
```

Here `external_query` counts the tokens queried after HBM misses. These are formula symbols, not exported metric names. Do not substitute `total_prefix_query_tokens_total`, which includes the HBM-covered part, for that denominator.

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
sum(rate(ucm:cache_load_bytes_total{job="vllm"}[5m])) / 1e9
```

Equivalently, for a fixed time range:

```text
average_bandwidth_GBps = increase(transferred_bytes_total) / elapsed_seconds / 1e9
```

Sum byte rates across workers before converting to GB/s. Keep Load and Dump, or read and write, as separate series. Do not average per-task bandwidth values to estimate system throughput.

A result of `2` means the average counted byte rate through the Cache load stage was 2 GB/s over five minutes. Low bandwidth quantiles, such as p10, help identify slow tasks. Bandwidth p99 describes the fast end of the distribution, not a latency tail.

### 2.3 Duration and slow tasks

For Cache Load, a Histogram's `_count` is the number of sampled tasks, `_sum` is their accumulated duration, and `_bucket{le="20"}` counts samples taking no more than 20 ms. Mean task duration over five minutes is:

```promql
sum(rate(ucm:cache_load_duration_ms_sum{job="vllm"}[5m]))
/
sum(rate(ucm:cache_load_duration_ms_count{job="vllm"}[5m]))
```

To inspect the slower end of the distribution, query p95:

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(ucm:cache_load_duration_ms_bucket{job="vllm"}[5m]))
)
```

Both results are in ms. A stable mean with a rising p95 indicates deterioration at the slower end. Account for task-size and load changes, then inspect the stage's queuing, backend waits and copy timings. Quantiles are bucket estimates; merge buckets across workers before calculating p95, rather than averaging worker p95 values. See the [Prometheus Histogram guide](https://prometheus.io/docs/practices/histograms/) for the calculation.

### 2.4 Capacity and health

Query Gauges directly, retaining individual series to identify their processes. For example, view Posix logical occupancy as a percentage:

```promql
100 * ucm:posix_store_usage_ratio{job="vllm"}
```

A result of `80` means that series last reported 80% logical occupancy. If it remains near capacity, check `ucm:posix_gc_running` and filesystem free space to see whether reclamation keeps up with writes.

```promql
ucm:posix_store_health{job="vllm"}
```

`1` allows new cache operations and `0` means the circuit breaker is open. Do not sum process values and interpret the result as one health state. Confirm statistics are still updating and correlate probe results; see [storage health](health-metrics.md) for the full interpretation.

## 3. Runtime interpretation notes

[^export-scope]: The current `vllm_connector` consumer excludes `interval_lookup_hit_rates` in `ucm/metrics_config.py`. The native metric remains in this catalog; do not expect it on the default vLLM endpoint.
[^cache-source]: `cache_posix_load_success_shards_total` follows the Cache buffer's not-ready state at dispatch. The lower Store need not be Posix. For Cache/Posix, source-share formulas estimate tier contribution; correlate backend activity and completed transfers rather than treating a wait as a physical disk read.
[^posix-bytes]: Current Posix counters accumulate nominal task bytes at completion, including error or aborted paths. Check errors alongside byte rates; physical storage throughput requires filesystem or device observations.
