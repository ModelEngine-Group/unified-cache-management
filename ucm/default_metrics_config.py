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
    (
        "dramstore_lookup_tasks_submitted_total",
        "DramStore lookup: tasks submitted",
    ),
    (
        "dramstore_lookup_tasks_rejected_total",
        "DramStore lookup: tasks rejected",
    ),
    (
        "dramstore_lookup_tasks_succeeded_total",
        "DramStore lookup: tasks succeeded",
    ),
    (
        "dramstore_lookup_tasks_failed_total",
        "DramStore lookup: tasks failed",
    ),
    (
        "dramstore_lookup_task_timeouts_total",
        "DramStore lookup: task timeouts",
    ),
    (
        "dramstore_lookup_requests_completed_total",
        "DramStore lookup: requests completed",
    ),
    (
        "dramstore_lookup_requests_failed_total",
        "DramStore lookup: requests failed",
    ),
    (
        "dramstore_lookup_request_timeouts_total",
        "DramStore lookup: requests completed with Timeout status",
    ),
    (
        "dramstore_dump_tasks_submitted_total",
        "DramStore dump: tasks submitted",
    ),
    (
        "dramstore_dump_tasks_rejected_total",
        "DramStore dump: tasks rejected",
    ),
    (
        "dramstore_dump_tasks_succeeded_total",
        "DramStore dump: tasks succeeded",
    ),
    (
        "dramstore_dump_tasks_failed_total",
        "DramStore dump: tasks failed",
    ),
    (
        "dramstore_dump_task_timeouts_total",
        "DramStore dump: task timeouts",
    ),
    (
        "dramstore_dump_requests_completed_total",
        "DramStore dump: requests completed",
    ),
    (
        "dramstore_dump_requests_failed_total",
        "DramStore dump: requests failed",
    ),
    (
        "dramstore_dump_request_timeouts_total",
        "DramStore dump: requests completed with Timeout status",
    ),
    (
        "dramstore_load_tasks_submitted_total",
        "DramStore load: tasks submitted",
    ),
    (
        "dramstore_load_tasks_rejected_total",
        "DramStore load: tasks rejected",
    ),
    (
        "dramstore_load_tasks_succeeded_total",
        "DramStore load: tasks succeeded",
    ),
    (
        "dramstore_load_tasks_failed_total",
        "DramStore load: tasks failed",
    ),
    (
        "dramstore_load_task_timeouts_total",
        "DramStore load: task timeouts",
    ),
    (
        "dramstore_load_requests_completed_total",
        "DramStore load: requests completed",
    ),
    (
        "dramstore_load_requests_failed_total",
        "DramStore load: requests failed",
    ),
    (
        "dramstore_load_request_timeouts_total",
        "DramStore load: requests completed with Timeout status",
    ),
    (
        "dramstore_connect_attempts_total",
        "DramStore connect attempts",
    ),
    (
        "dramstore_connect_failures_total",
        "DramStore connect failures",
    ),
    (
        "dramstore_fence_attempts_total",
        "DramStore fence attempts",
    ),
    (
        "dramstore_fence_failures_total",
        "DramStore fence failures",
    ),
    (
        "dramstore_fence_timeout_triggers_total",
        "DramStore node fencing episodes triggered by request timeouts; excludes fence retries",
    ),
    (
        "dramstore_reply_slot_nospace_total",
        "DramStore reply slot acquisition failures due to exhausted capacity",
    ),
    (
        "dramstore_stale_replies_total",
        "DramStore stale replies",
    ),
    # -- DramPool server observability ----------------------------------------
    (
        "drampool_dump_requests_total",
        "Total DUMP requests received and processed by DramPool",
    ),
    (
        "drampool_load_requests_total",
        "Total LOAD requests received and processed by DramPool",
    ),
    (
        "drampool_lookup_requests_total",
        "Total LOOKUP requests received and processed by DramPool",
    ),
    (
        "drampool_dump_nospace_failures_total",
        "DUMP entries whose buffer allocation finally failed with NoSpace after eviction retries (registration failures excluded)",
    ),
    (
        "drampool_dump_failed_entries_total",
        "Total DUMP entries that settled with a Failed result (all failure paths merged)",
    ),
    (
        "drampool_load_miss_entries_total",
        "LOAD entries that missed stored data (LoadBegin failure or request longer than stored length)",
    ),
    (
        "drampool_lookup_miss_entries_total",
        "LOOKUP entries that do not exist or are not READY",
    ),
    (
        "drampool_transfer_failures_total",
        "Data transfers that ended in a non-Completed terminal state (DUMP and LOAD merged)",
    ),
    (
        "drampool_response_failures_total",
        "Response return-path failures (local submit failures plus write-back transfer failures, DUMP and LOAD merged)",
    ),
    (
        "drampool_submit_failures_total",
        "Data transfer submission failures (ExecuteAsync failure or invalid handle)",
    ),
    (
        "drampool_queue_request_full_total",
        "requestQueue TryPush failures because the queue was full",
    ),
    (
        "drampool_queue_completion_full_total",
        "completionQueue full events that forced SubmitCompletion to spin-wait",
    ),
    (
        "drampool_queue_response_buffer_retry_total",
        "Response flag-buffer NoSpace events where SubmitResponse parked the request for retry",
    ),
    (
        "drampool_resource_log_read_errors_total",
        "Number of failures opening, reading, or parsing the DramPool resource log",
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
    (
        "dramstore_scheduler_request_queue_size",
        "Sampled queued Requests across scheduler runners, excluding batches already dequeued",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "dramstore_scheduler_event_queue_size",
        "Sampled queued NodeEvents across scheduler runners, excluding batches already dequeued",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "dramstore_task_queue_size",
        "Sampled TaskManager submission queue occupancy (tasks)",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "dramstore_task_queue_capacity",
        "TaskManager submission queue capacity (tasks)",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "dramstore_completion_queue_size",
        "Sampled TaskManager completion queue occupancy (request completions)",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "dramstore_completion_queue_capacity",
        "TaskManager completion queue capacity (request completions)",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "dramstore_tasks_active",
        "Sampled tasks awaiting completion, excluding queued submissions and retained results",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "dramstore_io_entries_used",
        "Sampled entries reserved by active tasks",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "dramstore_io_entries_capacity",
        "Maximum entries reserved by active tasks",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "dramstore_reply_buffer_used_bytes",
        "Sampled leased reply slot bytes including alignment, not payload bytes",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "dramstore_transport_queue_size",
        "Sampled aggregate queued Transmit and Connect admission occupancy",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "dramstore_transport_queue_capacity",
        "Aggregate Transmit and Connect admission capacity",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "dramstore_transport_fence_queue_size",
        "Sampled aggregate queued Fence admission occupancy",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "dramstore_transport_fence_queue_capacity",
        "Aggregate reserved Fence admission capacity",
        {"multiprocess_mode": "livemostrecent"},
    ),
    # -- DramPool server observability ----------------------------------------
    (
        "drampool_metadata_entry_count",
        "Current number of cached entries (blocks) in the pool, summed over all shards",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "drampool_buffer_pool_usage_ratio",
        "Used-slot ratio of the data pools, aggregated over all configured block sizes",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "drampool_flag_pool_usage_ratio",
        "Used-slot ratio of the flag response-buffer pool",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "drampool_queue_request_size",
        "Current number of requests queued in requestQueue",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "drampool_queue_completion_size",
        "Current number of completion records queued in completionQueue",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "drampool_queue_completion_inflight",
        "Completion records currently in flight in the Poller pending window",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "drampool_queue_request_capacity",
        "Configured requestQueue depth (requests)",
        {"multiprocess_mode": "livemostrecent"},
    ),
    (
        "drampool_queue_completion_capacity",
        "Configured completionQueue depth (records)",
        {"multiprocess_mode": "livemostrecent"},
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
    (
        "dramstore_lookup_duration_ms",
        "End-to-end DramStore lookup duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "dramstore_lookup_task_queue_duration_ms",
        "DramStore lookup task queue duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "dramstore_lookup_task_to_request_duration_ms",
        "DramStore lookup duration from task admission through request construction (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "dramstore_lookup_request_duration_ms",
        "DramStore lookup request duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "dramstore_lookup_request_queue_duration_ms",
        "DramStore lookup request NodeScheduler queue duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "dramstore_lookup_request_pending_duration_ms",
        "DramStore lookup request NodeActor pending duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "dramstore_lookup_request_setup_duration_ms",
        "DramStore lookup request successful setup duration including reply-slot allocation and encoding (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "dramstore_lookup_request_transport_queue_duration_ms",
        "DramStore lookup request TransportExecutor queue duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "dramstore_lookup_request_transport_send_duration_ms",
        "DramStore lookup request TCP send duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    ),
    (
        "dramstore_lookup_request_remote_duration_ms",
        "DramStore lookup request remote processing and reply duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "dramstore_dump_duration_ms",
        "End-to-end DramStore dump duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "dramstore_dump_task_queue_duration_ms",
        "DramStore dump task queue duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "dramstore_dump_task_to_request_duration_ms",
        "DramStore dump duration from task admission through request construction (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "dramstore_dump_request_duration_ms",
        "DramStore dump request duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "dramstore_dump_request_queue_duration_ms",
        "DramStore dump request NodeScheduler queue duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "dramstore_dump_request_pending_duration_ms",
        "DramStore dump request NodeActor pending duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "dramstore_dump_request_setup_duration_ms",
        "DramStore dump request successful setup duration including reply-slot allocation and encoding (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "dramstore_dump_request_transport_queue_duration_ms",
        "DramStore dump request TransportExecutor queue duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "dramstore_dump_request_transport_send_duration_ms",
        "DramStore dump request TCP send duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    ),
    (
        "dramstore_dump_request_remote_duration_ms",
        "DramStore dump request remote processing and reply duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "dramstore_load_duration_ms",
        "End-to-end DramStore load duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "dramstore_load_task_queue_duration_ms",
        "DramStore load task queue duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "dramstore_load_task_to_request_duration_ms",
        "DramStore load duration from task admission through request construction (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "dramstore_load_request_duration_ms",
        "DramStore load request duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "dramstore_load_request_queue_duration_ms",
        "DramStore load request NodeScheduler queue duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "dramstore_load_request_pending_duration_ms",
        "DramStore load request NodeActor pending duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "dramstore_load_request_setup_duration_ms",
        "DramStore load request successful setup duration including reply-slot allocation and encoding (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "dramstore_load_request_transport_queue_duration_ms",
        "DramStore load request TransportExecutor queue duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "dramstore_load_request_transport_send_duration_ms",
        "DramStore load request TCP send duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    ),
    (
        "dramstore_load_request_remote_duration_ms",
        "DramStore load request remote processing and reply duration (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "dramstore_dump_prerequisite_duration_ms",
        "DUMP prerequisite event wait before task admission (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    # -- DramPool server observability ----------------------------------------
    (
        "drampool_dump_prepare_duration_ms",
        "DUMP local preparation duration per batch: StoreBegin per entry, buffer allocation, eviction retries, and transfer submission (ms, successful path only; split into dump_metadata and dump_submit)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "drampool_dump_metadata_duration_ms",
        "DUMP metadata-processing segment per batch: StoreBegin per entry, buffer allocation, eviction retries, and transfer-item building, excluding the ExecuteAsync submission (ms, successful path only)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "drampool_dump_submit_duration_ms",
        "DUMP data-transfer ExecuteAsync submission duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "drampool_load_prepare_duration_ms",
        "LOAD local preparation duration per batch: LoadBegin per entry, length validation, and transfer submission (ms, successful path only; split into load_metadata and load_submit)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "drampool_load_metadata_duration_ms",
        "LOAD metadata-processing segment per batch: LoadBegin per entry, length validation, and transfer-item building, excluding the ExecuteAsync submission (ms, successful path only)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "drampool_load_submit_duration_ms",
        "LOAD data-transfer ExecuteAsync submission duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "drampool_lookup_scan_duration_ms",
        "Per-batch LOOKUP metadata scan duration: per-entry existence checks plus result filling (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "drampool_dump_transfer_duration_ms",
        "DUMP data transfer duration from ExecuteAsync return to terminal-state observation, including HiXL transfer, polling delay, and GetStatus (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "drampool_load_transfer_duration_ms",
        "LOAD data transfer duration from ExecuteAsync return to terminal-state observation, including HiXL transfer, polling delay, and GetStatus (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "drampool_get_status_duration_ms",
        "GetStatus() execution duration when the Poller queries data or response transfers (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "drampool_response_submit_duration_ms",
        "Response write-back ExecuteAsync submission duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "drampool_response_rtt_ms",
        "Response round-trip duration from response-transfer ExecuteAsync return to write-back completion in client memory, including polling delay and GetStatus but excluding the submission itself (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "drampool_metadata_storeend_duration_ms",
        "Per-entry StoreEnd settlement duration after DUMP transfer completion (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "drampool_metadata_loadend_duration_ms",
        "Per-entry LoadEnd reference-release duration (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "drampool_metadata_evict_gc_duration_ms",
        "Duration of one background GC eviction sweep across all shards (ms)",
        [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "drampool_queue_request_enqueue_wait_ms",
        "Per-request wait from preparation to successful requestQueue TryPush, including full-queue retries (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500],
    ),
    (
        "drampool_queue_request_residence_ms",
        "Per-request residence in requestQueue from the successful TryPush to the TaskWorker dequeue, excluding the TryPush wait (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ),
    (
        "drampool_dump_batch_total_duration_ms",
        "End-to-end DUMP batch duration across the full server-side request lifecycle: from requestQueue push to response-transfer terminal state, or the record leaving the Poller on a permanent response failure (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "drampool_load_batch_total_duration_ms",
        "End-to-end LOAD batch duration across the full server-side request lifecycle: from requestQueue push to response-transfer terminal state, or the record leaving the Poller on a permanent response failure (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "drampool_lookup_batch_total_duration_ms",
        "End-to-end LOOKUP batch duration across the full server-side request lifecycle: from requestQueue push to response-transfer terminal state, or the record leaving the Poller on a permanent response failure (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, 1000],
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
