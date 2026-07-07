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
        "cache_load_blocks_total",
        "Total blocks loaded by the Cache stage",
    ),
    (
        "cache_dump_blocks_total",
        "Total blocks dumped by the Cache stage",
    ),
    (
        "cache_load_shards_total",
        "Total shards dispatched by Cache load (num blocks * shards_per_block)",
    ),
    (
        "cache_load_backend_shards_total",
        (
            "Shards that descended to the backend on load (true cache miss at the "
            "buffer-allocation stage; aka backend-load count)"
        ),
    ),
    (
        "cache_dump_shards_total",
        "Total shards dispatched by Cache dump (mirror of cache_load_shards_total)",
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
        "mooncake_load_blocks_total",
        "Total blocks loaded through the Mooncake stage",
    ),
    (
        "mooncake_dump_blocks_total",
        "Total blocks dumped through the Mooncake stage",
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
        "cache_lookup_hit_rate",
        "Instantaneous Cache stage hit rate from the most recent lookup call",
        {"multiprocess_mode": 'livemostrecent'},
    ),
]
_HISTOGRAM_METRICS = [
    (
        "load_requests_num",
        "Number of requests loaded from ucm",
        [1, 5, 10, 20, 50, 100, 200, 500, 1000],
    ),
    (
        "load_blocks_num",
        "Number of blocks loaded from ucm",
        [0, 50, 100, 150, 200, 250, 300, 350, 400, 550, 600, 750, 800, 850, 900, 950, 1000],
    ),
    (
        "load_duration",
        "Time to load from ucm (ms)",
        [0, 50, 100, 150, 200, 250, 300, 350, 400, 550, 600, 750, 800, 850, 900, 950, 1000],
    ),
    (
        "load_speed",
        "Speed of loading from ucm (GB/s)",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 50, 60, 70, 80, 90, 100],
    ),
    (
        "save_requests_num",
        "Number of requests saved to ucm",
        [1, 5, 10, 20, 50, 100, 200, 500, 1000],
    ),
    (
        "save_blocks_num",
        "Number of blocks saved to ucm",
        [0, 50, 100, 150, 200, 250, 300, 350, 400, 550, 600, 750, 800, 850, 900, 950, 1000],
    ),
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
    (
        "layerwise_batch_total_ms",
        (
            "Layerwise batch wall-clock time from start_load_kv entry to wait_for_save "
            "return (ms)"
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "layerwise_batch_total_load_only_ms",
        (
            "Layerwise load-only batch wall-clock time from start_load_kv entry to "
            "wait_for_save return (ms)"
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "layerwise_batch_total_save_only_ms",
        (
            "Layerwise save-only batch wall-clock time from start_load_kv entry to "
            "wait_for_save return (ms)"
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "layerwise_batch_total_load_save_ms",
        (
            "Layerwise load-and-save batch wall-clock time from start_load_kv entry to "
            "wait_for_save return (ms)"
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "layerwise_batch_total_no_transfer_ms",
        "Layerwise batch wall-clock time with neither load nor save work (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "layerwise_batch_load_wait_total_load_only_ms",
        (
            "Total wait_for_layer_load blocking time accumulated within one load-only "
            "layerwise batch (ms)"
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "layerwise_batch_load_wait_total_load_save_ms",
        (
            "Total wait_for_layer_load blocking time accumulated within one "
            "load-and-save layerwise batch (ms)"
        ),
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    ),
    (
        "layerwise_batch_save_tail_save_only_ms",
        "wait_for_save tail duration within one save-only layerwise batch (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    ),
    (
        "layerwise_batch_save_tail_load_save_ms",
        "wait_for_save tail duration within one load-and-save layerwise batch (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    ),
    (
        "layerwise_wait_blocking_ms",
        "Time wait_for_layer_load blocked before returning (ms). Near 0 = good overlap.",
        [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "layerwise_wait_tasks_count",
        "Number of per-request load tasks awaited in a single layer wait",
        [0, 1, 2, 4, 8, 16, 32, 64],
    ),
    (
        "layerwise_inter_wait_interval_ms",
        "Interval between consecutive wait_for_layer_load calls (~forward time) (ms)",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    ),
    (
        "layerwise_next_layer_submit_ms",
        "Time to submit next layer's load tasks inside wait_for_layer_load (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50],
    ),
    (
        "layerwise_first_layer_submit_ms",
        (
            "Time to submit first layer load tasks during start_load_kv - TTFT critical "
            "(ms)"
        ),
        [0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
    ),
    (
        "layerwise_first_layer_requests",
        "Number of requests whose first-layer load was submitted in start_load_kv",
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
    ),
    (
        "layerwise_save_submit_ms",
        "Time to submit one layer's dump task in save_kv_layer (ms)",
        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50],
    ),
    (
        "layerwise_save_tail_total_ms",
        "Legacy metric; LayerWise no longer waits for dump completion in wait_for_save",
        [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    ),
    (
        "fawa_scheduler_lookup_external_hit_blocks_ms",
        "store lookup latency",
        [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50],
    ),
    (
        "fawa_scheduler_get_num_new_matched_tokens_ms",
        "store lookup latency + generate block hash latency",
        [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50],
    ),
    (
        "fawa_worker_wait_wait_all_load_task_ms",
        "store load latency",
        [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1250, 1500, 1750, 2000],
    ),
    (
        "fawa_worker_start_load_kv_ms",
        "store load task latency + generate task latency",
        [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1250, 1500, 1750, 2000],
    ),
    (
        "fawa_worker_wait_for_save_ms",
        "store dump task latency",
        [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500],
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
