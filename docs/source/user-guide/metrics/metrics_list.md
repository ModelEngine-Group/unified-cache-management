# UCM Metrics Reference

## 1. Metrics Exported by Default

The tables below use the default `ucm:` prefix. The default configuration contains 164 Counters, 37 Gauges, and 133 Histograms.

See [UCM Health Metrics](health_metrics.md) for Store health metrics and recommended aggregation.

### 1.1 Connector

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
| `ucm:save_duration`                                     | Duration from entering `wait_for_save` until asynchronous Dump completion |
| `ucm:save_completion_wait_duration`                     | Time actually blocked while confirming asynchronous Dump completion        |
| `ucm:interval_lookup_hit_rates`                         | Per-request UCM Lookup hit-rate distribution                               |
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

### 1.6 ASU / KV metrics

ASU forwards these 37 KV metrics (23 Counters and 14 Histograms) into the UCM collector. Names below use the default `ucm:` prefix. Submission counts describe KV API calls and entries, not unique inference requests or successful asynchronous completions.

The generator's `--check` validates KV table names, type sections, duplicates, omissions, and counts in the English, Chinese, and legacy references. Descriptions and units remain manually maintained; `--sync-ucm` does not rewrite these tables.

**Units:** all `kv_*_duration_seconds` observations and bucket boundaries are in seconds. Existing `*_duration_ms` metrics, `save_duration` and `save_completion_wait_duration` use milliseconds. Multiply a KV duration mean or quantile by `1000` when displaying milliseconds; do not combine seconds and milliseconds without conversion.

The two `kv_transport_node_task_*` histograms retain `node_id` in standalone mode. The UCM adapter aggregates their observations across nodes and does not expose that node label; other exporter labels still apply.

#### Installation switches

ASU backend installation is controlled only by the store-level `ucm_connectors[].ucm_connector_config.enable_metrics` switch (default `true`). Direct, HMA, HLA, and HLA LayerWise connectors pass the store configuration through without rewriting this switch. The top-level `enable_metrics`, effective metrics catalog, and consumer settings retain their existing UCM initialization and export behavior; they do not prevent ASU backend installation. Custom catalogs must contain the KV definitions to collect these metrics.

Installing the backend does not initialize the UCM collector or start an exporter. When the collector is not initialized, UCM ignores forwarded updates. If another component has already initialized the process-wide collector and export path, these samples may still be collected and exported. The store installation switch does not provide per-store filtering for KV metrics.

To opt out of installation for an ASU store, add `enable_metrics: false` beside `store_pipeline: "ASU"` in its existing `ucm_connector_config`. For KV metrics, this setting controls installation, not per-store filtering or shutdown: the backend is process-wide, and setting it to `false` does not uninstall a backend already installed by another store. Direct C++ callers only have the store installation switch and must arrange UCM metric registration and draining themselves.

Metrics installation is best-effort: failure emits a rate-limited UCM ERROR log and Store initialization continues without installing UCM metrics. An existing backend is left unchanged. Installation failures are not cached permanently; a later Store `Setup()` can retry. After all metric writers have stopped and `kv::metrics::Shutdown()` has completed, a later setup can reinstall the retained UCM adapter. Concurrent shutdown and writes, or switching backend types with live cached bindings, remain unsupported. Incompatible cached bindings and failed updates are skipped with rate-limited ERROR logs. Installation and update exceptions are caught by the adapter; logging uses the standard UCM logger without a separate failure fallback.

The adapter uses the existing UCM `UpdateStats()` API. Samples are silently skipped when the collector is not initialized or the metric is unregistered. Later initialization and registration restore sampling through the same cached metric; previously skipped samples are not replayed. Use `--ucm-config` to check the completeness of a custom catalog.

#### Counters

| Metric | Unit | Description |
| --- | --- | --- |
| `ucm:kv_client_query_requests_total` | count | Total KV client query submissions |
| `ucm:kv_client_query_entries_total` | keys / entries | Total keys submitted to KV client query |
| `ucm:kv_client_query_errors_total` | count | Total failed KV client query submissions |
| `ucm:kv_client_load_requests_total` | count | Total KV client load submissions |
| `ucm:kv_client_load_entries_total` | keys / entries | Total entries submitted to KV client load |
| `ucm:kv_client_load_errors_total` | count | Total failed KV client load submissions |
| `ucm:kv_client_store_requests_total` | count | Total KV client store submissions |
| `ucm:kv_client_store_entries_total` | keys / entries | Total entries submitted to KV client store |
| `ucm:kv_client_store_errors_total` | count | Total failed KV client store submissions |
| `ucm:kv_client_batch_load_requests_total` | count | Total KV client batch-load submissions |
| `ucm:kv_client_batch_load_entries_total` | keys / entries | Total entries submitted to KV client batch-load |
| `ucm:kv_client_batch_load_errors_total` | count | Total failed KV client batch-load submissions |
| `ucm:kv_client_batch_store_requests_total` | count | Total KV client batch-store submissions |
| `ucm:kv_client_batch_store_entries_total` | keys / entries | Total entries submitted to KV client batch-store |
| `ucm:kv_client_batch_store_errors_total` | count | Total failed KV client batch-store submissions |
| `ucm:kv_client_delete_requests_total` | count | Total KV client delete submissions |
| `ucm:kv_client_delete_entries_total` | keys / entries | Total keys submitted to KV client delete |
| `ucm:kv_client_delete_errors_total` | count | Total failed KV client delete submissions |
| `ucm:kv_client_wait_requests_total` | count | Total KV client wait calls |
| `ucm:kv_client_wait_errors_total` | count | Total failed KV client wait calls |
| `ucm:kv_transport_task_completion_timeouts_total` | count | Total KV transport task completion timeouts |
| `ucm:kv_transport_task_io_timeouts_total` | count | Total KV transport sub-batch IO timeouts |
| `ucm:kv_transport_task_connection_errors_total` | count | Total KV transport connection errors |

#### Histograms

| Metric | Unit | Description |
| --- | --- | --- |
| `ucm:kv_client_task_enqueue_duration_seconds` | seconds | KV client API-to-enqueue duration |
| `ucm:kv_client_task_queue_duration_seconds` | seconds | KV client task queue duration |
| `ucm:kv_client_task_process_duration_seconds` | seconds | KV client task processing duration |
| `ucm:kv_client_task_send_duration_seconds` | seconds | KV client duration until all Send calls return |
| `ucm:kv_client_task_e2e_duration_seconds` | seconds | KV client task end-to-end duration |
| `ucm:kv_transport_task_pre_send_duration_seconds` | seconds | KV transport duration before Send |
| `ucm:kv_transport_task_queue_duration_seconds` | seconds | KV transport task queue duration |
| `ucm:kv_transport_task_process_duration_seconds` | seconds | KV transport task processing duration |
| `ucm:kv_transport_task_send_duration_seconds` | seconds | KV transport duration until Send returns |
| `ucm:kv_transport_task_send_call_duration_seconds` | seconds | KV transport provider Send call duration |
| `ucm:kv_transport_task_completion_duration_seconds` | seconds | KV transport completion duration after Send |
| `ucm:kv_transport_node_task_send_duration_seconds` | seconds | KV transport node task send duration; standalone labels by node_id, UCM aggregates across nodes |
| `ucm:kv_transport_node_task_completion_duration_seconds` | seconds | KV transport node task completion duration; standalone labels by node_id, UCM aggregates across nodes |
| `ucm:kv_transport_task_e2e_duration_seconds` | seconds | KV transport task end-to-end duration from submit to completion |

#### Updating KV definitions

Edit `kv_semantics/metrics/config/kv_metrics.yaml`, then run from the repository root:

```bash
python kv_semantics/metrics/tools/generate_kv_metrics.py --sync-ucm
python kv_semantics/metrics/tools/generate_kv_metrics.py --check
```

Commit the source, generated C++ descriptor header, and generated KV blocks in `examples/metrics/metrics_configs.yaml` and `ucm/default_metrics_config.py` together. `--check` detects stale outputs without writing files.

When adding, removing, or changing the type of a KV metric, also update its table row and count summary in both language versions of `metrics-reference.md` and this file. `--check` rejects missing, duplicate, obsolete, or incorrectly grouped KV rows and stale count summaries. The default catalog totals above are checked against all metric lists in `ucm/default_metrics_config.py`, including non-KV metrics. Descriptions and units still require manual review; `--sync-ucm` does not overwrite documentation.

For custom deployment catalogs, synchronize the KV definitions and run `python kv_semantics/metrics/tools/generate_kv_metrics.py --ucm-config /etc/ucm/metrics_configs.yaml` separately (requires PyYAML). Pass the metrics catalog, not the serving configuration. Every KV definition must appear exactly once with matching type, documentation text and buckets; extra custom metrics are allowed. `--ucm-config` cannot be combined with `--check` or `--sync-ucm`. Use unquoted decimal bucket values such as `0.00001`, because YAML 1.1 may parse `1e-5` as a string.

### 1.7 Native AsuStore metrics

These 20 metrics (16 Counters and 4 Histograms) use native UCM `UpdateStats()` at the AsuStore interface, independently of the 37 KV metrics above. The existing store-level `enable_metrics` switch controls this instance's `asu_*` observations; its shared KV backend installation behavior is unchanged. Registration and export still use the outer UCM configuration; custom catalogs must include the desired definitions.

The three lookup methods share call/error counts and duration. Ordinary lookup counts hits in the returned vector; prefix lookup counts the returned usable prefix length. Compute their hit ratios separately. Error returns do not contribute queried or hit blocks; timeout fallbacks contribute queried blocks with zero hits. Reverse lookup reports call results and duration only, without interpreting its returned position as a hit count.

Load/Dump returning a handle means successful submission; payload bytes are the sum of submitted KV buffer sizes after ASU alignment. Wait counts each call separately, and timeout counts are a subset of wait errors. These are not final task counts, successfully completed bytes, or full asynchronous task durations. Durations use milliseconds and include error returns; no resource sampling thread is added.

#### Counters

| Metric | Unit | Description |
| --- | --- | --- |
| `ucm:asu_lookup_requests_total` | count | AsuStore Lookup, prefix lookup and reverse lookup calls |
| `ucm:asu_lookup_errors_total` | count | Failed AsuStore lookup calls; excludes timeout fallback to misses |
| `ucm:asu_lookup_query_blocks_total` | blocks | Blocks queried by normally returned Lookup calls, including timeout fallback |
| `ucm:asu_lookup_hit_blocks_total` | blocks | Hit blocks returned by Lookup; timeout fallback contributes zero hits |
| `ucm:asu_lookup_prefix_query_blocks_total` | blocks | Blocks queried by normally returned prefix lookup calls, including timeout fallback |
| `ucm:asu_lookup_prefix_hit_blocks_total` | blocks | Usable prefix blocks returned by prefix lookup; timeout fallback contributes zero hits |
| `ucm:asu_lookup_timeout_fallback_total` | count | AsuStore lookup calls returning cache misses after query timeout |
| `ucm:asu_load_requests_total` | count | AsuStore Load calls, including validation and submission failures |
| `ucm:asu_load_submit_errors_total` | count | Failed AsuStore Load submissions, including buffer validation failures |
| `ucm:asu_load_submitted_bytes_total` | bytes | Payload bytes accepted by AsuStore Load; not successfully completed bytes |
| `ucm:asu_dump_requests_total` | count | AsuStore Dump calls, including validation and submission failures |
| `ucm:asu_dump_submit_errors_total` | count | Failed AsuStore Dump submissions, including buffer validation failures |
| `ucm:asu_dump_submitted_bytes_total` | bytes | Payload bytes accepted by AsuStore Dump; not successfully completed bytes |
| `ucm:asu_wait_requests_total` | count | AsuStore Wait calls; repeated waits are counted separately |
| `ucm:asu_wait_errors_total` | count | AsuStore Wait calls returning errors, including timeouts and task failures |
| `ucm:asu_wait_timeouts_total` | count | AsuStore Wait calls returning timeout; a subset of wait errors |

#### Histograms

| Metric | Unit | Description |
| --- | --- | --- |
| `ucm:asu_lookup_duration_ms` | ms | AsuStore lookup call duration including key conversion, query wait and result processing (ms) |
| `ucm:asu_load_submit_duration_ms` | ms | AsuStore Load submission duration including buffer conversion and failed submissions (ms) |
| `ucm:asu_dump_submit_duration_ms` | ms | AsuStore Dump submission duration including buffer conversion and failed submissions (ms) |
| `ucm:asu_wait_duration_ms` | ms | AsuStore Wait call duration including failed and timed-out waits (ms) |

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
