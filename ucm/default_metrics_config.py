#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# This file is generated from examples/metrics/metrics_configs.yaml.
# Update the YAML first, then regenerate this file.
#

from copy import deepcopy
from typing import Any

# fmt: off
_COUNTER_METRICS = [
    (
        "cache_lookup_hit_blocks_total",
        "Number of lookup hits served by the Cache stage (no descent to backend)",
    ),
    (
        "cache_lookup_miss_blocks_total",
        "Number of lookup misses at the Cache stage (had to query the backend)",
    ),
    (
        "cache_load_shards_total",
        "Total shards whose Cache buffer state was inspected during load",
    ),
    (
        "cache_load_wait_shards_total",
        "Shards whose Cache buffer was not ready when acquired and required waiting",
    ),
    (
        "cache_load_backend_shards_total",
        (
            "Shards that descended to the backend on load (true cache miss at the "
            "buffer-allocation stage; aka backend-load count)"
        ),
    ),
    (
        "cache_load_success_shards_total",
        "Shards successfully loaded from an already-ready Cache buffer to device",
    ),
    (
        "cache_posix_load_success_shards_total",
        "Shards successfully loaded to device after waiting for Posix to fill Cache",
    ),
    (
        "cache_load_failed_shards_total",
        "Cache load shards that did not complete device delivery",
    ),
    (
        "cache_dump_shards_total",
        "Total shard descriptors processed by Cache dump, including failed tasks",
    ),
    (
        "cache_dump_backend_shards_total",
        (
            "Shards actually pushed to backend on dump (excludes !handle.Owner() skips "
            "in shared-buffer scenario)"
        ),
    ),
    (
        "cache_load_queue_full_total",
        "Number of Cache load submissions rejected because the waiting queue was full",
    ),
    (
        "cache_dump_queue_full_total",
        "Number of Cache dump submissions rejected because the waiting queue was full",
    ),
    (
        "cache_backend_load_submit_errors_total",
        "Number of Cache load backend submit failures",
    ),
    (
        "cache_backend_load_wait_errors_total",
        "Number of Cache load backend wait failures",
    ),
    (
        "cache_backend_dump_submit_errors_total",
        "Number of Cache dump backend submit failures",
    ),
    (
        "cache_backend_dump_wait_errors_total",
        "Number of Cache dump backend wait failures",
    ),
    (
        "cache_h2d_errors_total",
        "Number of Cache host-to-device transfer or sync failures",
    ),
    (
        "cache_d2h_errors_total",
        "Number of Cache device-to-host transfer, event wait, or sync failures",
    ),
    (
        "cache_load_bytes_total",
        "Total bytes loaded through the Cache stage (per-task size summed)",
    ),
    (
        "cache_dump_bytes_total",
        "Total bytes dumped through the Cache stage (per-task size summed)",
    ),
    (
        "posix_s2h_bytes_total",
        (
            "Total bytes transferred from posix storage to host buffer (load path, "
            "summed per completed task)"
        ),
    ),
    (
        "posix_h2s_bytes_total",
        (
            "Total bytes transferred from host buffer to posix storage (dump path, "
            "summed per completed task)"
        ),
    ),
    (
        "posix_lookup_query_blocks_total",
        "Total blocks submitted to Posix lookup",
    ),
    (
        "posix_lookup_hit_blocks_total",
        "Blocks found by Posix lookup",
    ),
    (
        "posix_healthy_count_total",
        "Number of successful Posix health probes",
    ),
    (
        "posix_unhealthy_count_total",
        "Number of failed Posix health probes",
    ),
    (
        "posix_aio_timeout_total",
        "Number of Posix AIO task or submit timeouts",
    ),
    (
        "posix_io_timeout_total",
        "Number of Posix synchronous worker task timeouts",
    ),
    (
        "posix_open_errors_total",
        "Number of Posix open failures",
    ),
    (
        "posix_io_errors_total",
        "Number of Posix read, write, or AIO completion failures",
    ),
    (
        "yuanrong_load_success_shards_total",
        "Shards successfully loaded from YuanRong to device",
    ),
    (
        "yuanrong_lookup_miss_posix_load_success_shards_total",
        "Shards successfully loaded from Posix after YuanRong lookup miss",
    ),
    (
        "yuanrong_load_fallback_posix_load_success_shards_total",
        "Shards successfully loaded from Posix after YuanRong load failure",
    ),
    (
        "yuanrong_load_failed_shards_total",
        "YuanRong pipeline load shards that did not complete device delivery",
    ),
    (
        "yuanrong_local_dram_load_hits_total",
        "Estimated YuanRong local DRAM Get hits forwarded from kv_resource.log",
    ),
    (
        "yuanrong_remote_load_hits_total",
        "Estimated YuanRong remote worker Get hits forwarded from kv_resource.log",
    ),
    (
        "yuanrong_local_ssd_load_hits_total",
        "Estimated YuanRong local spill SSD Get hits forwarded from kv_resource.log",
    ),
    (
        "yuanrong_l2_load_hits_total",
        "YuanRong L2 persistence Get hits forwarded from kv_resource.log",
    ),
    (
        "yuanrong_resource_log_read_errors_total",
        "Number of failures opening, reading, or parsing YuanRong kv_resource.log",
    ),
    (
        "mooncake_load_blocks_total",
        "Total blocks loaded through the Mooncake stage",
    ),
    (
        "mooncake_dump_blocks_total",
        "Total blocks dumped through the Mooncake stage",
    ),
    (
        "mooncake_lookup_hit_blocks_total",
        "Blocks found directly by Mooncake lookup before backend descent",
    ),
    (
        "mooncake_healthy_count_total",
        "Number of successful Mooncake health probes",
    ),
    (
        "mooncake_unhealthy_count_total",
        "Number of failed Mooncake health probes",
    ),
    (
        "mooncake_load_bytes_total",
        "Total bytes loaded through the Mooncake stage",
    ),
    (
        "mooncake_dump_bytes_total",
        "Total bytes dumped through the Mooncake stage",
    ),
    (
        "mooncake_load_hit_shards_total",
        "Mooncake load shards served directly from Mooncake",
    ),
    (
        "mooncake_load_miss_shards_total",
        "Mooncake load shards that missed and descended to backend or recompute",
    ),
    (
        "mooncake_load_backend_shards_total",
        "Mooncake load shards submitted to the backend after a Mooncake miss",
    ),
    (
        "mooncake_dump_existing_shards_total",
        "Mooncake dump shards already present in Mooncake",
    ),
    (
        "mooncake_dump_missing_shards_total",
        "Mooncake dump shards written to Mooncake because they were missing",
    ),
    (
        "mooncake_dump_backend_shards_total",
        "Mooncake dump shards archived to the backend",
    ),
    (
        "mooncake_load_queue_full_total",
        (
            "Number of Mooncake load submissions rejected because the waiting queue was "
            "full"
        ),
    ),
    (
        "mooncake_dump_queue_full_total",
        (
            "Number of Mooncake dump submissions rejected because the waiting queue was "
            "full"
        ),
    ),
    (
        "mooncake_get_errors_total",
        "Number of Mooncake batch get failures",
    ),
    (
        "mooncake_put_errors_total",
        "Number of Mooncake batch put failures",
    ),
    (
        "mooncake_backend_load_submit_errors_total",
        "Number of Mooncake backend load submit failures",
    ),
    (
        "mooncake_backend_load_wait_errors_total",
        "Number of Mooncake backend load wait failures",
    ),
    (
        "mooncake_backend_dump_submit_errors_total",
        "Number of Mooncake backend dump submit failures",
    ),
    (
        "mooncake_backend_dump_wait_errors_total",
        "Number of Mooncake backend dump wait failures",
    ),
    (
        "mooncake_h2d_errors_total",
        "Number of Mooncake H2D transfer or sync failures",
    ),
    (
        "mooncake_d2h_errors_total",
        "Number of Mooncake D2H transfer, event wait, or sync failures",
    ),
    (
        "mooncake_h2d_bytes_total",
        "Total Mooncake bytes copied from host to device",
    ),
    (
        "mooncake_d2h_bytes_total",
        "Total Mooncake bytes copied from device to host",
    ),
    (
        "load_bytes_total",
        (
            "Total bytes loaded through the UCM connector (summed across all "
            "start_load_kv calls)"
        ),
    ),
    (
        "save_bytes_total",
        (
            "Total bytes saved through the UCM connector (summed across all "
            "wait_for_save calls)"
        ),
    ),
    (
        "total_prefix_query_tokens_total",
        "Total prefix cache query tokens observed by the UCM connector",
    ),
    (
        "gpu_hbm_hit_tokens_total",
        "Prefix cache tokens already hit in GPU or HBM before UCM lookup",
    ),
    (
        "ucm_hit_tokens_total",
        "Prefix cache tokens hit by the UCM connector",
    ),
    (
        "total_prefix_query_blocks_total",
        "Total full prefix blocks queried through the UCM connector",
    ),
    (
        "gpu_hbm_hit_blocks_total",
        "Full prefix blocks already hit in GPU or HBM before UCM lookup",
    ),
    (
        "connector_lookup_errors_total",
        "Number of connector lookup errors treated as cache misses",
    ),
    (
        "connector_load_submit_errors_total",
        "Number of connector load submit failures",
    ),
    (
        "connector_load_wait_errors_total",
        "Number of connector load wait failures",
    ),
    (
        "connector_load_invalid_requests_total",
        "Number of connector load failure events that invalidated request blocks",
    ),
    (
        "connector_load_invalid_blocks_total",
        "Number of newly invalidated vLLM block ids caused by connector load failures",
    ),
    (
        "connector_dump_submit_errors_total",
        "Number of connector dump submit failures",
    ),
    (
        "connector_dump_wait_errors_total",
        "Number of connector dump wait failures",
    ),
]
_GAUGE_METRICS = [
    (
        "yuanrong_dram_used_bytes",
        "YuanRong physical shared-memory usage in bytes",
        {"multiprocess_mode": 'livemostrecent'},
    ),
    (
        "yuanrong_dram_capacity_bytes",
        "YuanRong shared-memory capacity in bytes",
        {"multiprocess_mode": 'livemostrecent'},
    ),
    (
        "yuanrong_dram_usage_ratio",
        "YuanRong physical shared-memory usage ratio",
        {"multiprocess_mode": 'livemostrecent'},
    ),
    (
        "yuanrong_ssd_used_bytes",
        "YuanRong physical spill-disk usage in bytes",
        {"multiprocess_mode": 'livemostrecent'},
    ),
    (
        "yuanrong_ssd_capacity_bytes",
        "YuanRong spill-disk capacity in bytes",
        {"multiprocess_mode": 'livemostrecent'},
    ),
    (
        "yuanrong_ssd_usage_ratio",
        "YuanRong physical spill-disk usage ratio",
        {"multiprocess_mode": 'livemostrecent'},
    ),
    (
        "yuanrong_resource_log_last_update_timestamp_seconds",
        "Unix timestamp of the latest YuanRong resource snapshot parsed by UCM",
        {"multiprocess_mode": 'livemostrecent'},
    ),
    (
        "yuanrong_resource_log_reporter_leader",
        "Whether this UCM process is the host YuanRong resource reporter leader",
        {"multiprocess_mode": 'livemostrecent'},
    ),
    (
        "posix_store_used_bytes",
        "Estimated logical Posix Store usage in bytes from GC sampling",
        {"multiprocess_mode": 'livemostrecent'},
    ),
    (
        "posix_store_capacity_bytes",
        "Configured logical Posix Store capacity in bytes",
        {"multiprocess_mode": 'livemostrecent'},
    ),
    (
        "posix_store_usage_ratio",
        "Estimated logical Posix Store usage ratio",
        {"multiprocess_mode": 'livemostrecent'},
    ),
    (
        "posix_store_health",
        "Effective Posix health breaker state, where 1 is enabled and 0 is fused",
        {"multiprocess_mode": 'livemostrecent'},
    ),
    (
        "mooncake_store_health",
        "Effective Mooncake health breaker state, where 1 is enabled and 0 is fused",
        {"multiprocess_mode": 'livemostrecent'},
    ),
    (
        "posix_gc_running",
        "Posix garbage collection state, where 1 is running and 0 is idle",
        {"multiprocess_mode": 'livemostrecent'},
    ),
]
_CONNECTOR_INTERFACE_METHODS = [
    "get_block_size",
    "get_kv_connector_stats",
    "get_num_new_matched_tokens",
    "update_state_after_alloc",
    "register_kv_caches",
    "build_connector_meta",
    "bind_connector_metadata",
    "handle_preemptions",
    "has_connector_metadata",
    "start_load_kv",
    "wait_for_layer_load",
    "save_kv_layer",
    "wait_for_save",
    "request_finished_all_groups",
    "request_finished",
    "get_finished",
    "build_connector_worker_meta",
    "update_connector_output",
    "clear_connector_metadata",
    "get_block_ids_with_load_errors",
]
_CONNECTOR_INTERFACE_DURATION_BUCKETS = [
    0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
    2000, 5000, 10000,
]
_HISTOGRAM_METRICS = [
    (
        "save_duration",
        "Time from UCM connector wait_for_save entry to async dump task completion (ms)",
        [0, 50, 100, 150, 200, 250, 300, 350, 400, 550, 600, 750, 800, 850, 900, 950, 1000],
    ),
    (
        "save_completion_wait_duration",
        "Time spent blocked while confirming async UCM connector dump completion (ms)",
        [0, 1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 550, 600, 750, 800, 850, 900, 950, 1000],
    ),
    (
        "interval_lookup_hit_rates",
        "Hit rates of ucm lookup requests",
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ),
    (
        "cache_lookup_duration_ms",
        "Cache buffer lookup wall-clock time per `Lookup` / `LookupOnPrefix` call (ms)",
        [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
    ),
    (
        "cache_lookup_backend_duration_ms",
        (
            "Backend lookup wall-clock time when descending due to no buffer or buffer "
            "miss (ms)"
        ),
        [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "cache_load_duration_ms",
        "End-to-end Cache stage load task duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "cache_dump_duration_ms",
        "End-to-end Cache stage dump task duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "cache_load_bandwidth_gbps",
        "Cache stage effective load bandwidth (GB/s)",
        [0.5, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
    ),
    (
        "cache_dump_bandwidth_gbps",
        "Cache stage effective dump bandwidth (GB/s)",
        [0.5, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
    ),
    (
        "cache_load_queue_wait_duration_ms",
        "Time a Cache load task spent queued before dispatch worker pickup (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "cache_dump_queue_wait_duration_ms",
        "Time a Cache dump task spent queued before dispatch worker pickup (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "cache_load_backend_submit_duration_ms",
        (
            "Cache load backend submit duration: buffer allocation plus synchronous "
            "backend load submission (ms)."
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "cache_shard_backend_wait_ms",
        (
            "Cache load per-shard time spent in WaitBackendTaskReady before H2D submit "
            "(ms). This is not a task-level duration."
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "cache_h2d_submit_ms",
        (
            "Cache load per-shard H2D async submit CPU cost after backend wait (ms). "
            "Submission only; NOT the actual transfer time (see cache_h2d_sync_ms)."
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "cache_h2d_sync_ms",
        (
            "Cache load residual H2D stream drain after the last shard submit (ms). "
            "Large => H2D copy is the bottleneck; ~0 with large "
            "cache_shard_backend_wait_ms => storage read is the bottleneck."
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "cache_dump_mkbuf_duration_ms",
        (
            "Cache dump mk_buf phase: buffer allocation/reuse + D2H async submit before "
            "stream sync (ms)"
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "cache_dump_prereq_wait_ms",
        (
            "Cache dump time waiting for the prerequisite compute event (layer KV "
            "ready) to fire before D2H can start (ms). Large => dump is compute-gated, "
            "not copy-gated."
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "cache_d2h_duration_ms",
        (
            "Cache dump stream synchronize duration including prerequisite compute wait "
            "and D2H copy (ms). Use cache_dump_prereq_wait_ms to estimate the "
            "compute-gated portion."
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "cache_dump_backend_submit_duration_ms",
        (
            "Cache dump backend submit duration: synchronous time to pass buffers to "
            "the lower tier (ms). Does NOT include the lower tier's actual write time."
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "cache_dump_backend_wait_duration_ms",
        (
            "Cache dump time waiting for the lower tier to finish writing a dumped task "
            "(ms). Large => storage write is the bottleneck."
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    ),
    (
        "posix_load_task_duration_ms",
        (
            "End-to-end Posix load task duration (ms): submit to last shard finished, "
            "task-level (compare with cache_load_duration_ms)"
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "posix_dump_task_duration_ms",
        (
            "End-to-end Posix dump task duration (ms): submit to last shard finished, "
            "task-level (compare with cache_dump_duration_ms)"
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "posix_s2h_bandwidth_gbps",
        "Posix stage read bandwidth per task (GB/s) = totalBytes / task_wallclock",
        [0.05, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24, 32],
    ),
    (
        "posix_h2s_bandwidth_gbps",
        "Posix stage write bandwidth per task (GB/s) = totalBytes / task_wallclock",
        [0.05, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24, 32],
    ),
    (
        "posix_load_queue_wait_duration_ms",
        "Time a Posix load task spent queued before first worker pickup (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "posix_dump_queue_wait_duration_ms",
        "Time a Posix dump task spent queued before first worker pickup (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "mooncake_load_duration_ms",
        "End-to-end Mooncake load task duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "mooncake_dump_duration_ms",
        "End-to-end Mooncake dump task duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "mooncake_load_bandwidth_gbps",
        "Mooncake stage effective load bandwidth (GB/s)",
        [0.5, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
    ),
    (
        "mooncake_dump_bandwidth_gbps",
        "Mooncake stage effective dump bandwidth (GB/s)",
        [0.5, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
    ),
    (
        "mooncake_load_queue_wait_duration_ms",
        "Time a Mooncake load task spent queued before dispatch worker pickup (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "mooncake_dump_queue_wait_duration_ms",
        "Time a Mooncake dump task spent queued before dispatch worker pickup (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "mooncake_get_duration_ms",
        "Mooncake batch get duration on the load path (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "mooncake_exists_duration_ms",
        "Mooncake batch exists check duration on the dump path (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "mooncake_put_duration_ms",
        "Mooncake batch put duration on the dump path (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    ),
    (
        "mooncake_load_backend_submit_duration_ms",
        "Mooncake load backend submit duration after Mooncake miss (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "mooncake_backend_load_wait_duration_ms",
        "Mooncake load time waiting for the backend to finish a missed shard (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "mooncake_h2d_duration_ms",
        "Mooncake load H2D stream drain duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "mooncake_dump_prereq_wait_ms",
        "Mooncake dump time waiting for prerequisite compute event before put (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "mooncake_d2h_duration_ms",
        "Mooncake dump D2H stream drain duration for backend archive (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "mooncake_dump_backend_submit_duration_ms",
        "Mooncake dump backend submit duration after D2H archive copy (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "mooncake_dump_backend_wait_duration_ms",
        "Mooncake dump time waiting for backend archive completion (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    ),
    *[
        (
            f"connector_{method}_duration_ms",
            f"Wall-clock duration of UCMConnector.{method} invoked by vLLM (ms)",
            _CONNECTOR_INTERFACE_DURATION_BUCKETS,
        )
        for method in _CONNECTOR_INTERFACE_METHODS
    ],
    (
        "layerwise_layer_load_duration_ms",
        "Layerwise per-layer wall-clock time from layer load start to wait_for_layer_load return (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "layerwise_batch_load_duration_sum_ms",
        "Sum of per-layer load durations within one Layerwise batch (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
]


def _metric(name: str, documentation: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "documentation": documentation, **extra}


def _histogram_metric(
    name: str, documentation: str, buckets: list[int | float]
) -> dict[str, Any]:
    return _metric(name, documentation, buckets=buckets)


DEFAULT_METRICS_CONFIG: dict[str, Any] = {
    "log_interval": 5,
    "vllm_connector_prefix": "ucm:",
    "consumers": {"vllm_connector": True},
    "counter": [
        _metric(name, documentation) for name, documentation in _COUNTER_METRICS
    ],
    "gauge": [
        _metric(name, documentation, **extra)
        for name, documentation, extra in _GAUGE_METRICS
    ],
    "histogram": [
        _histogram_metric(name, documentation, buckets)
        for name, documentation, buckets in _HISTOGRAM_METRICS
    ],
}


def get_default_metrics_config() -> dict[str, Any]:
    return deepcopy(DEFAULT_METRICS_CONFIG)
