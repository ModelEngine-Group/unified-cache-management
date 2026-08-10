# UCM Metrics Reference

## 3. Metrics Exported by Default

The tables below use the default `ucm:` prefix. The default configuration contains 67 Counters, 4 Gauges, and 70 Histograms. Unless a metric name or description states otherwise, duration metrics use milliseconds, bandwidth metrics use GB/s, and accumulated byte counts use bytes.

### 3.1 Counters

#### Cache Store

| Metric | Description |
| --- | --- |
| `ucm:cache_lookup_hit_blocks_total` | Blocks served directly by Cache Lookup without descending to the backend |
| `ucm:cache_lookup_miss_blocks_total` | Blocks missed by Cache Lookup and passed to the backend |
| `ucm:cache_load_blocks_total` | Total blocks processed by Cache Load |
| `ucm:cache_dump_blocks_total` | Total blocks processed by Cache Dump |
| `ucm:cache_load_shards_total` | Total shards dispatched by Cache Load |
| `ucm:cache_load_backend_shards_total` | Shards that actually descend to the backend during Cache buffer allocation |
| `ucm:cache_dump_shards_total` | Total shards dispatched by Cache Dump |
| `ucm:cache_dump_backend_shards_total` | Owner shards actually written to the backend |
| `ucm:cache_load_queue_full_total` | Cache Load submissions rejected because the waiting queue is full |
| `ucm:cache_dump_queue_full_total` | Cache Dump submissions rejected because the waiting queue is full |
| `ucm:cache_backend_load_submit_errors_total` | Failures when Cache submits a Load to the backend |
| `ucm:cache_backend_load_wait_errors_total` | Failures while Cache waits for a backend Load |
| `ucm:cache_backend_dump_submit_errors_total` | Failures when Cache submits a Dump to the backend |
| `ucm:cache_backend_dump_wait_errors_total` | Failures while Cache waits for a backend Dump |
| `ucm:cache_h2d_errors_total` | Cache H2D transfer or stream synchronization failures |
| `ucm:cache_d2h_errors_total` | Cache D2H transfer, event wait, or stream synchronization failures |
| `ucm:cache_load_bytes_total` | Cumulative bytes loaded through the Cache stage |
| `ucm:cache_dump_bytes_total` | Cumulative bytes dumped through the Cache stage |

#### Posix Store

| Metric | Description |
| --- | --- |
| `ucm:posix_s2h_bytes_total` | Cumulative bytes read from Posix storage into host buffers |
| `ucm:posix_h2s_bytes_total` | Cumulative bytes written from host buffers to Posix storage |
| `ucm:posix_lookup_query_blocks_total` | Total blocks submitted to Posix Lookup |
| `ucm:posix_lookup_hit_blocks_total` | Blocks found by Posix Lookup |
| `ucm:posix_healthy_count_total` | Successful Posix health probes |
| `ucm:posix_unhealthy_count_total` | Failed Posix health probes |
| `ucm:posix_aio_timeout_total` | Posix AIO task or submission timeouts |
| `ucm:posix_io_timeout_total` | Posix synchronous I/O worker task timeouts |
| `ucm:posix_open_errors_total` | Posix file-open failures |
| `ucm:posix_io_errors_total` | Posix read, write, or AIO completion failures |

#### Mooncake Store

| Metric | Description |
| --- | --- |
| `ucm:mooncake_load_blocks_total` | Total blocks processed by the Mooncake Load stage |
| `ucm:mooncake_dump_blocks_total` | Total blocks processed by the Mooncake Dump stage |
| `ucm:mooncake_lookup_hit_blocks_total` | Blocks found directly by Mooncake Lookup before descending to the backend |
| `ucm:mooncake_healthy_count_total` | Successful Mooncake health probes |
| `ucm:mooncake_unhealthy_count_total` | Failed Mooncake health probes |
| `ucm:mooncake_load_bytes_total` | Cumulative bytes loaded through the Mooncake stage |
| `ucm:mooncake_dump_bytes_total` | Cumulative bytes dumped through the Mooncake stage |
| `ucm:mooncake_load_hit_shards_total` | Load shards served directly by Mooncake |
| `ucm:mooncake_load_miss_shards_total` | Load shards that miss Mooncake and descend to the backend or are recomputed |
| `ucm:mooncake_load_backend_shards_total` | Load shards submitted to the backend after a Mooncake miss |
| `ucm:mooncake_dump_existing_shards_total` | Dump shards already present in Mooncake |
| `ucm:mooncake_dump_missing_shards_total` | Missing Dump shards written to Mooncake |
| `ucm:mooncake_dump_backend_shards_total` | Dump shards archived to the backend |
| `ucm:mooncake_load_queue_full_total` | Mooncake Load submissions rejected because the waiting queue is full |
| `ucm:mooncake_dump_queue_full_total` | Mooncake Dump submissions rejected because the waiting queue is full |
| `ucm:mooncake_get_errors_total` | Mooncake batch-get failures |
| `ucm:mooncake_put_errors_total` | Mooncake batch-put failures |
| `ucm:mooncake_backend_load_submit_errors_total` | Mooncake backend Load submission failures |
| `ucm:mooncake_backend_load_wait_errors_total` | Failures while waiting for a Mooncake backend Load |
| `ucm:mooncake_backend_dump_submit_errors_total` | Mooncake backend Dump submission failures |
| `ucm:mooncake_backend_dump_wait_errors_total` | Failures while waiting for a Mooncake backend Dump |
| `ucm:mooncake_h2d_errors_total` | Mooncake H2D transfer or synchronization failures |
| `ucm:mooncake_d2h_errors_total` | Mooncake D2H transfer, event wait, or synchronization failures |
| `ucm:mooncake_h2d_bytes_total` | Cumulative bytes copied from host to device by Mooncake |
| `ucm:mooncake_d2h_bytes_total` | Cumulative bytes copied from device to host by Mooncake |

#### Connector

| Metric | Description |
| --- | --- |
| `ucm:load_bytes_total` | Cumulative bytes loaded by all `start_load_kv` calls |
| `ucm:save_bytes_total` | Cumulative bytes saved by all `wait_for_save` calls |
| `ucm:total_prefix_query_tokens_total` | Total prefix-cache query tokens observed by the UCM connector |
| `ucm:gpu_hbm_hit_tokens_total` | Prefix tokens already found in GPU/HBM before UCM Lookup |
| `ucm:ucm_hit_tokens_total` | Prefix tokens hit by the UCM connector |
| `ucm:total_prefix_query_blocks_total` | Total complete prefix blocks queried by the UCM connector |
| `ucm:gpu_hbm_hit_blocks_total` | Complete prefix blocks already found in GPU/HBM before UCM Lookup |
| `ucm:connector_lookup_errors_total` | Connector Lookup errors handled as cache misses |
| `ucm:connector_load_submit_errors_total` | Connector Load submission failures |
| `ucm:connector_load_wait_errors_total` | Connector Load wait failures |
| `ucm:connector_load_invalid_requests_total` | Events in which a Load failure invalidates request blocks |
| `ucm:connector_load_invalid_blocks_total` | Newly invalidated vLLM block IDs caused by Load failures |
| `ucm:connector_dump_submit_errors_total` | Connector Dump submission failures |
| `ucm:connector_dump_wait_errors_total` | Connector Dump wait failures |

### 3.2 Gauges

See [UCM Health Metrics](health_metrics.md) for Store health Counters, Gauges, and recommended aggregation.

| Metric | Description |
| --- | --- |
| `ucm:cache_lookup_hit_rate` | Instantaneous hit rate of the most recent Cache Lookup |
| `ucm:posix_store_health` | Effective Posix circuit-breaker state: 1 is available and 0 is fused |
| `ucm:mooncake_store_health` | Effective Mooncake circuit-breaker state: 1 is available and 0 is fused |
| `ucm:posix_gc_running` | Posix garbage collection state: 1 is running and 0 is idle |

### 3.3 Histograms

#### Connector Base Metrics

| Metric | Description |
| --- | --- |
| `ucm:load_requests_num` | Requests involved in one UCM Load |
| `ucm:load_blocks_num` | Blocks involved in one UCM Load |
| `ucm:load_duration` | UCM Connector Load duration |
| `ucm:load_speed` | UCM Connector Load throughput in GB/s |
| `ucm:save_requests_num` | Requests involved in one UCM Save |
| `ucm:save_blocks_num` | Blocks involved in one UCM Save |
| `ucm:save_duration` | Duration from entering `wait_for_save` until asynchronous Dump completion |
| `ucm:save_completion_wait_duration` | Time actually blocked while confirming asynchronous Dump completion |
| `ucm:interval_lookup_hit_rates` | Per-request UCM Lookup hit-rate distribution |

#### Cache Store

| Metric | Description |
| --- | --- |
| `ucm:cache_lookup_duration_ms` | Wall-clock time of one Cache buffer `Lookup`/`LookupOnPrefix` call |
| `ucm:cache_lookup_backend_duration_ms` | Backend Lookup wall-clock time when there is no buffer or the buffer misses |
| `ucm:cache_load_duration_ms` | End-to-end Cache-stage Load task duration |
| `ucm:cache_dump_duration_ms` | End-to-end Cache-stage Dump task duration |
| `ucm:cache_load_bandwidth_gbps` | Effective bandwidth over the complete Cache Load task lifecycle |
| `ucm:cache_dump_bandwidth_gbps` | Effective bandwidth over the complete Cache Dump task lifecycle |
| `ucm:cache_load_queue_wait_duration_ms` | Time a Cache Load task waits before a dispatch worker picks it up |
| `ucm:cache_dump_queue_wait_duration_ms` | Time a Cache Dump task waits before a dispatch worker picks it up |
| `ucm:cache_load_backend_submit_duration_ms` | Time to allocate a Cache buffer and synchronously submit the backend Load |
| `ucm:cache_shard_backend_wait_ms` | Time one shard waits for the backend to become ready before H2D submission |
| `ucm:cache_h2d_submit_ms` | CPU overhead of one asynchronous shard H2D submission, excluding transfer time |
| `ucm:cache_h2d_sync_ms` | Remaining H2D stream drain time after the final shard submission |
| `ucm:cache_dump_mkbuf_duration_ms` | Cache Dump buffer allocation/reuse and asynchronous D2H submission time |
| `ucm:cache_dump_prereq_wait_ms` | Time waiting for the layer KV-ready compute event before D2H starts |
| `ucm:cache_d2h_duration_ms` | Cache Dump stream synchronization time, including prerequisite compute wait and D2H copy |
| `ucm:cache_dump_backend_submit_duration_ms` | Time to synchronously submit the buffer to the lower Store |
| `ucm:cache_dump_backend_wait_duration_ms` | Time waiting for the lower Store to complete the write |

#### Posix Store

| Metric | Description |
| --- | --- |
| `ucm:posix_load_task_duration_ms` | End-to-end Posix Load task duration from submission until the final shard completes |
| `ucm:posix_dump_task_duration_ms` | End-to-end Posix Dump task duration from submission until the final shard completes |
| `ucm:posix_s2h_bandwidth_gbps` | Per-task Posix read bandwidth |
| `ucm:posix_h2s_bandwidth_gbps` | Per-task Posix write bandwidth |
| `ucm:posix_load_queue_wait_duration_ms` | Time a Posix Load task waits before the first worker picks it up |
| `ucm:posix_dump_queue_wait_duration_ms` | Time a Posix Dump task waits before the first worker picks it up |

#### Mooncake Store

| Metric | Description |
| --- | --- |
| `ucm:mooncake_load_duration_ms` | End-to-end Mooncake Load task duration |
| `ucm:mooncake_dump_duration_ms` | End-to-end Mooncake Dump task duration |
| `ucm:mooncake_load_bandwidth_gbps` | Effective Mooncake-stage Load bandwidth |
| `ucm:mooncake_dump_bandwidth_gbps` | Effective Mooncake-stage Dump bandwidth |
| `ucm:mooncake_load_queue_wait_duration_ms` | Time a Mooncake Load task waits before a dispatch worker picks it up |
| `ucm:mooncake_dump_queue_wait_duration_ms` | Time a Mooncake Dump task waits before a dispatch worker picks it up |
| `ucm:mooncake_get_duration_ms` | Mooncake batch-get duration on the Load path |
| `ucm:mooncake_exists_duration_ms` | Mooncake batch-exists check duration on the Dump path |
| `ucm:mooncake_put_duration_ms` | Mooncake batch-put duration on the Dump path |
| `ucm:mooncake_load_backend_submit_duration_ms` | Time to submit a backend Load after a Mooncake miss |
| `ucm:mooncake_backend_load_wait_duration_ms` | Time waiting for the backend to load missing shards |
| `ucm:mooncake_h2d_duration_ms` | Mooncake Load H2D stream drain time |
| `ucm:mooncake_dump_prereq_wait_ms` | Time waiting for the prerequisite compute event before a Mooncake put |
| `ucm:mooncake_d2h_duration_ms` | Mooncake D2H stream drain time required for backend archival |
| `ucm:mooncake_dump_backend_submit_duration_ms` | Time to submit a backend Dump after the D2H archival copy |
| `ucm:mooncake_dump_backend_wait_duration_ms` | Time waiting for backend archival to complete |

#### Layerwise

| Metric | Description |
| --- | --- |
| `ucm:layerwise_batch_total_ms` | Total batch wall-clock time from entering `start_load_kv` until `wait_for_save` returns |
| `ucm:layerwise_batch_total_load_only_ms` | Total wall-clock time of a load-only layerwise batch |
| `ucm:layerwise_batch_total_save_only_ms` | Total wall-clock time of a save-only layerwise batch |
| `ucm:layerwise_batch_total_load_save_ms` | Total wall-clock time of a layerwise batch containing Load and Save |
| `ucm:layerwise_batch_total_no_transfer_ms` | Total wall-clock time of a layerwise batch with no Load or Save transfer |
| `ucm:layerwise_batch_load_wait_total_load_only_ms` | Sum of all `wait_for_layer_load` blocking time in a load-only batch |
| `ucm:layerwise_batch_load_wait_total_load_save_ms` | Sum of all `wait_for_layer_load` blocking time in a load-and-save batch |
| `ucm:layerwise_batch_save_tail_save_only_ms` | `wait_for_save` tail duration in a save-only batch |
| `ucm:layerwise_batch_save_tail_load_save_ms` | `wait_for_save` tail duration in a load-and-save batch |
| `ucm:layerwise_wait_blocking_ms` | Time blocked by one `wait_for_layer_load`; values near zero indicate good overlap |
| `ucm:layerwise_wait_tasks_count` | Request Load tasks awaited by one layer wait |
| `ucm:layerwise_inter_wait_interval_ms` | Interval between consecutive `wait_for_layer_load` calls |
| `ucm:layerwise_next_layer_submit_ms` | Time to submit the next layer's Load task in `wait_for_layer_load` |
| `ucm:layerwise_first_layer_submit_ms` | Time to submit the first layer's Load task in `start_load_kv` |
| `ucm:layerwise_first_layer_requests` | Requests whose first-layer Load is submitted in `start_load_kv` |
| `ucm:layerwise_save_submit_ms` | Time to submit one layer's Dump task in `save_kv_layer` |
| `ucm:layerwise_save_tail_total_ms` | Compatibility metric; Layerwise no longer waits for Dump completion in `wait_for_save` |

#### FAWA

| Metric | Description |
| --- | --- |
| `ucm:fawa_scheduler_lookup_external_hit_blocks_ms` | Scheduler Store Lookup duration |
| `ucm:fawa_scheduler_get_num_new_matched_tokens_ms` | Total Store Lookup and block-hash generation duration |
| `ucm:fawa_worker_wait_wait_all_load_task_ms` | Worker Store Load wait duration |
| `ucm:fawa_worker_start_load_kv_ms` | Worker Store Load task construction and submission duration |
| `ucm:fawa_worker_wait_for_save_ms` | Worker Store Dump duration |
