"""Built-in SGLang metrics for HiCache and direct Linker adapters.

Common definitions are copied from UCM defaults; vLLM configuration is never mutated.
Keep these additions in sync with examples/metrics/metrics_configs_sglang.yaml.
"""

from ucm.default_metrics_config import get_default_metrics_config


def get_linker_metric_definitions() -> dict:
    """Bounded per-worker series; no request IDs, layers or pool IDs as labels."""
    counters = {
        "lookup_requests_total": "Owner lookup attempts; excludes neutral non-owner responses",
        "lookup_errors_total": "Failed owner lookup attempts",
        "lookup_query_pages_total": "Full requested KV pages on successful owner lookups, including resumed prefixes; repeated lookups count again",
        "lookup_hit_pages_total": "Maximum locally restorable KV page boundary on successful owner lookups; before cross-rank intersection, not final token reuse",
        "lookup_resumed_total": "Owner lookup attempts with a previous hit boundary",
        "lookup_fallback_total": "Per-group prefix probes restarted at page zero after the cached boundary disappeared",
        "lookup_skipped_total": "Neutral lookup responses on non-owner ranks; not actual storage hits",
        "load_requests_total": "Requests accepted into the scheduler's pending load set",
        "load_cancelled_requests_total": "Pending requests removed by cancellation or reset before batching",
        "load_ready_layers_total": "Logical layers published ready across load batches; not physical Store tasks",
        "offload_skipped_total": "Offload calls acknowledged on non-owner ranks without storage I/O",
        "load_check_errors_total": "Physical load tasks with a Check failure; counted separately from Wait failures",
        "dump_check_errors_total": "Physical dump tasks with a Check failure; counted separately from Wait failures",
    }
    histograms = [
        {
            "name": "sglang_linker_lookup_duration_ms",
            "documentation": "SGLang linker: Owner adapter lookup duration including failed attempts (ms)",
            "buckets": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
        }
    ]
    for operation, unit in (
        ("load", "load batches"),
        ("offload", "owner offload nodes"),
    ):
        counters.update(
            {
                f"{operation}_operations_total": f"Enqueued {unit}",
                f"{operation}_started_total": f"Dequeued {unit}, before event synchronization and preparation",
                f"{operation}_completed_total": f"Finished {unit}, including failures; before ordered publication",
                f"{operation}_errors_total": f"Failed {unit}, including preparation, submission and completion errors",
            }
        )
        for suffix, description in (
            (
                "queue_duration_ms",
                "Enqueue to worker dequeue time; excludes scheduler pending-load residence",
            ),
            (
                "duration_ms",
                "Enqueue to worker completion time including queueing, preparation and failures; excludes ordered-publication delay and downstream background persistence",
            ),
        ):
            histograms.append(
                {
                    "name": f"sglang_linker_{operation}_{suffix}",
                    "documentation": f"SGLang linker {operation}: {description} (ms)",
                    "buckets": [
                        0.01,
                        0.1,
                        0.5,
                        1,
                        5,
                        10,
                        50,
                        100,
                        500,
                        1000,
                        5000,
                        10000,
                    ],
                }
            )
    return {
        "counter": [
            {"name": f"sglang_linker_{name}", "documentation": f"SGLang linker: {doc}"}
            for name, doc in counters.items()
        ],
        "histogram": histograms,
    }


def get_default_sglang_metrics_config() -> dict:
    """Return independent definitions without reading any YAML at runtime."""
    config = get_default_metrics_config()
    # Standalone KV exporters use kv: aliases. SGLang's shared endpoint keeps
    # the existing ucm: namespace used by its dashboards and explicit YAML.
    for kind in ("counter", "gauge", "histogram"):
        for definition in config[kind]:
            if str(definition.get("prometheus_name", "")).startswith("kv:"):
                definition.pop("prometheus_name")
    config.pop("vllm_connector_prefix", None)
    config["multiproc_prefix"] = "ucm:"
    config["consumers"] = {"multiproc": True, "vllm_connector": False}
    config["counter"] = [
        {
            "name": "sglang_connector_load_requests_total",
            "documentation": "SGLang connector load: Physical storage transfer calls attempted, per worker and pool or layer",
        },
        {
            "name": "sglang_connector_load_bytes_total",
            "documentation": "SGLang connector load: Bytes transferred by successful storage calls, local to this worker",
        },
        {
            "name": "sglang_connector_load_submit_errors_total",
            "documentation": "SGLang connector load: Storage submission failures",
        },
        {
            "name": "sglang_connector_load_wait_errors_total",
            "documentation": "SGLang connector load: Storage completion wait failures",
        },
        {
            "name": "sglang_connector_dump_requests_total",
            "documentation": "SGLang connector dump: Physical storage transfer calls attempted, per worker and pool or layer",
        },
        {
            "name": "sglang_connector_dump_bytes_total",
            "documentation": "SGLang connector dump: Bytes transferred by successful storage calls, local to this worker",
        },
        {
            "name": "sglang_connector_dump_submit_errors_total",
            "documentation": "SGLang connector dump: Storage submission failures",
        },
        {
            "name": "sglang_connector_dump_wait_errors_total",
            "documentation": "SGLang connector dump: Storage completion wait failures",
        },
    ] + config["counter"]
    config["histogram"] = [
        {
            "name": "sglang_connector_load_pages",
            "documentation": "SGLang connector load: Successful storage batch size in pool pages; not deduplicated across TP/PP, pools or layers",
            "buckets": [1, 5, 10, 50, 100, 128, 256, 512],
        },
        {
            "name": "sglang_connector_load_duration_ms",
            "documentation": "SGLang connector load: Storage submission-to-completion duration including failed attempts (ms); excludes buffer preparation and restoration",
            "buckets": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
        },
        {
            "name": "sglang_connector_load_bandwidth_gbps",
            "documentation": "SGLang connector load: Successful call bytes divided by storage submission-to-completion time (GB/s, decimal bytes); not device-to-storage end-to-end bandwidth",
            "buckets": [0.1, 0.5, 1, 5, 10, 20, 50, 100],
        },
        {
            "name": "sglang_connector_dump_pages",
            "documentation": "SGLang connector dump: Successful storage batch size in pool pages; not deduplicated across TP/PP, pools or layers",
            "buckets": [1, 5, 10, 50, 100, 128, 256, 512],
        },
        {
            "name": "sglang_connector_dump_duration_ms",
            "documentation": "SGLang connector dump: Storage submission-to-completion duration including failed attempts (ms); excludes buffer preparation and restoration",
            "buckets": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
        },
        {
            "name": "sglang_connector_dump_bandwidth_gbps",
            "documentation": "SGLang connector dump: Successful call bytes divided by storage submission-to-completion time (GB/s, decimal bytes); not device-to-storage end-to-end bandwidth",
            "buckets": [0.1, 0.5, 1, 5, 10, 20, 50, 100],
        },
    ] + config["histogram"]
    for name, documentation in (
        (
            "load_layer_duration_ms",
            "Per logical layer submission through worker wait and cleanup, including failures and ordered-reaping delay (ms)",
        ),
    ):
        config["histogram"].append(
            {
                "name": f"sglang_linker_{name}",
                "documentation": documentation,
                "dynamic_labels": ["layer"],
                "buckets": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
            }
        )
    for kind, definitions in get_linker_metric_definitions().items():
        config[kind].extend(definitions)

    # Nested interface samples overlap; do not sum them into end-to-end time.
    for name, documentation in (
        (
            "sglang_linker_interface_entry_layer_rows_duration_ms",
            "Synchronous linker._entry_layer_rows call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_linker_interface_expand_duration_ms",
            "Synchronous linker._expand call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_linker_interface_page_transfers_duration_ms",
            "Synchronous linker._page_transfers call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_linker_interface_pool_metadata_duration_ms",
            "Synchronous linker._pool_metadata call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_linker_interface_select_store_ranks_duration_ms",
            "Synchronous linker._select_store_ranks call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_linker_interface_wait_transfer_events_duration_ms",
            "Synchronous linker._wait_transfer_events call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_linker_interface_finish_layer_wise_loading_duration_ms",
            "Synchronous linker.finish_layer_wise_loading call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_linker_interface_load_duration_ms",
            "Synchronous linker.load call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_linker_interface_lookup_duration_ms",
            "Synchronous linker.lookup call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_linker_interface_offload_duration_ms",
            "Synchronous linker.offload call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_linker_interface_start_layer_wise_loading_duration_ms",
            "Synchronous linker.start_layer_wise_loading call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_adapter_interface_block_ids_duration_ms",
            "Synchronous adapter._block_ids call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_adapter_interface_ensure_store_schema_duration_ms",
            "Synchronous adapter._ensure_store_schema call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_adapter_interface_estimate_gc_capacities_duration_ms",
            "Synchronous adapter._estimate_gc_capacities call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_adapter_interface_prepare_transfers_duration_ms",
            "Synchronous adapter._prepare_transfers call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_adapter_interface_submit_dump_duration_ms",
            "Synchronous adapter._submit_dump call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_adapter_interface_transfer_layout_duration_ms",
            "Synchronous adapter._transfer_layout call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_adapter_interface_lookup_duration_ms",
            "Synchronous adapter.lookup call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_adapter_interface_prepare_load_duration_ms",
            "Synchronous adapter.prepare_load call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_adapter_interface_register_device_pools_duration_ms",
            "Synchronous adapter.register_device_pools call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_adapter_interface_submit_offload_duration_ms",
            "Synchronous adapter.submit_offload call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_adapter_interface_submit_prepared_load_layer_duration_ms",
            "Synchronous adapter.submit_prepared_load_layer call duration including failures; not asynchronous I/O latency (ms)",
        ),
        (
            "sglang_linker_load_prepare_duration_ms",
            "Worker/lookup load_prepare phase wall-clock duration including failures (ms)",
        ),
        (
            "sglang_linker_offload_prepare_duration_ms",
            "Worker/lookup offload_prepare phase wall-clock duration including failures (ms)",
        ),
        (
            "sglang_linker_offload_io_duration_ms",
            "Worker/lookup offload_io phase wall-clock duration including failures (ms)",
        ),
        (
            "sglang_linker_lookup_expand_duration_ms",
            "Completed lookup_expand phase wall-clock duration (ms); failures are covered by interface timing",
        ),
        (
            "sglang_linker_lookup_prepare_duration_ms",
            "Completed lookup_prepare phase wall-clock duration (ms); failures are covered by interface timing",
        ),
        (
            "sglang_linker_lookup_result_duration_ms",
            "Completed lookup_result phase wall-clock duration (ms); failures are covered by interface timing",
        ),
    ):
        config["histogram"].append(
            {
                "name": name,
                "documentation": documentation,
                "buckets": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
            }
        )
    config["histogram"].append(
        {
            "name": "sglang_linker_load_layer_wait_duration_ms",
            "documentation": "Consumer wait_until call duration by logical layer, including ready callbacks, cleanup and failures; not pure GPU stall time (ms)",
            "dynamic_labels": ["layer"],
            "buckets": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
        }
    )
    return config
