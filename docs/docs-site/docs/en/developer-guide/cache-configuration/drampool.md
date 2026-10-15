# DramPool

[English](drampool.md) | [中文](../../../zh/developer-guide/cache-configuration/drampool.md)

DramPool organizes host DRAM into a KV cache pool shared by multiple inference instances. Inference instances access DramPool through DramStore. Currently, only HIXL-based data transfer is supported.

## Quick Start

### Environment and Prerequisites

- Devices: Ascend A3 or later.
- CANN: 9.1.0 or later.
- Drivers and communication libraries: Ascend drivers and HIXL libraries compatible with the devices and CANN version.
- Operating system: Linux.
- Network: RDMA networking is required between DramPool and inference nodes.
- Host memory: Available capacity must meet the memory requirements of the KV cache pool and runtime buffers.

### Build

Load the CANN environment and build UCM for the A3 platform from the UCM repository root. These commands build and install both DramStore and DramPool:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PLATFORM=ascend-a3
pip install -v -e . --no-build-isolation
```

With the editable installation above, the build artifacts are located at:

- `ucm/store/dram/libdramstore.so`: The DramStore client library. When `store_pipeline: "Dram"` is selected, it is loaded by `UcmPipelineStore`.
- `bin/drampool`: The standalone DramPool server executable.

### Configuration

#### DramPool Configuration

The main DramPool configuration options are listed below.

| Option | Description |
| --- | --- |
| `transport.device_ids` | Device IDs used for data transfer. |
| `transport.hixl.listen_port` | HIXL communication listening port. |
| `transport.hixl.enable_hixl_cs` | Whether to enable HIXL CS. |
| `health.port` | Health check service port. Set to `0` to disable. |
| `logger.dir` | Log output directory. |

For a complete configuration example, see [drampool.yaml](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/drampool.yaml).

#### DramStore Configuration

The main DramStore configuration options are listed below.

| Option | Description |
| --- | --- |
| `ucm_connector_name` | UCM connector name. DramStore uses `UcmPipelineStore`. |
| `store_pipeline` | Storage pipeline. `Dram` selects DramStore. |
| `local_addr` | Required. DramStore local service IP and base port in `IP:port` format. The actual port is offset by the physical device ID and scheduler/worker role. Reserve non-overlapping port ranges for each instance. |
| `peer_addrs` | Required. A non-empty list of DramPool service addresses, matching `--addr` for each DramPool instance. |
| `hixl_listen_port` | Local HIXL base port, default `36666`. The actual port is offset by the physical device ID and scheduler/worker role. Reserve non-overlapping port ranges for each instance. |
| `enable_hixl_cs` | Whether to enable HIXL CS, default `false`. |
| `router_type` | Routing strategy. Supports `ring_hash`, `maglev`, `contiguous_block_affinity`, and `batch_topk_affinity`. Default: `ring_hash`. |
| `enable_event_sync` | Whether to enable compute-stream event synchronization, default `true`. When enabled, waits for compute-stream events before remote access to KV memory. |
| `use_layerwise` | Whether to enable layerwise KV cache transfer, default `true`. |

For a complete configuration example, see [ucm_config_dramstore.yaml](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/ucm_config_dramstore.yaml).

### Startup

#### Start DramPool

Run the following command from the UCM repository root to start DramPool:

```bash
./bin/drampool \
  --addr 192.168.1.10:9000 \
  --pool-size-gb 64 \
  --kvcache-block-sizes 262144 \
  --config /absolute/path/to/drampool.yaml
```

Startup parameters are described below.

| Parameter | Description |
| --- | --- |
| `--addr` | DramPool control service address, which must be reachable by all DramStore clients. |
| `--pool-size-gb` | KV cache pool capacity in GiB. Reserve host memory for the operating system and runtime buffers. |
| `--kvcache-block-sizes` | Supported KV cache block sizes in bytes, matching the tensor sizes transferred by DramStore. Separate multiple values with spaces. This differs from vLLM's `block_size`, which is measured in tokens. |
| `--kvcache-block-proportions` | Capacity weights for the block sizes. Capacity is divided equally by default. The number of weights must match the number of block sizes. |
| `--ttl-minutes` | Absolute cache block lifetime in minutes, default `120`. |
| `--config` | Runtime configuration file path, default `examples/drampool.yaml`, resolved relative to the current working directory. |

The following message is logged when service startup completes:

```text
DramPool service ready
```

When health checks are enabled (`health.port` is nonzero), check service status with the following command. The IP matches `--addr`, and the port matches `health.port`:

```bash
curl --fail http://192.168.1.10:9100/health
```

To stop the service, send `SIGINT` or `SIGTERM` to the DramPool process.

#### Start the Inference Service

After DramPool is ready, run the following command to start a single-device inference service:

```bash
vllm serve <model-path> \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --kv-transfer-config '{"kv_connector":"UCMConnector","kv_role":"kv_both","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/etc/ucm/dramstore.yaml"}}'
```

For model and inference parameters, see the [vLLM startup parameter documentation](https://docs.vllm.ai/en/latest/cli/serve/). UCM integration options in `--kv-transfer-config` are listed below.

| Option | Description |
| --- | --- |
| `kv_connector` | UCM connector name. Use `UCMConnector`. |
| `kv_role` | KV cache transfer role. `kv_both` supports both cache load and dump. |
| `kv_connector_module_path` | UCM connector module path. Use `ucm.integration.vllm.ucm_connector`. |
| `kv_connector_extra_config.UCM_CONFIG_FILE` | DramStore configuration file path. |

## Metrics

DramStore and DramPool metrics are exported through vLLM's `/metrics` endpoint.

### Enable Metrics

UCM metrics are enabled by default. They can also be explicitly enabled in the DramStore configuration file:

```yaml
enable_metrics: true
```

#### DramPool Configuration

Configure the following in `drampool.yaml`:

```yaml
metrics:
  enabled: true
  output_dir: ./logs
  interval_ms: 10000
```

| Option | Description |
| --- | --- |
| `metrics.enabled` | Whether to enable metrics snapshot export. |
| `metrics.output_dir` | Snapshot output directory. Uses `logger.dir` when empty. The configuration above writes the snapshot to `./logs/drampool_metrics.json`. |
| `metrics.interval_ms` | Snapshot export interval in milliseconds. Must be greater than `0` when enabled. |

#### DramStore Configuration

Configure the following in `ucm_config_dramstore.yaml`:

```yaml
drampool_resource_log_path: "./logs/drampool_metrics.json"
drampool_resource_metrics_enable: true
drampool_resource_metrics_interval_sec: 15
```

| Option | Description |
| --- | --- |
| `drampool_resource_log_path` | DramPool metrics snapshot file path. Relative paths are resolved against the process working directory. Share or mount the snapshot directory when deploying across hosts or containers. |
| `drampool_resource_metrics_enable` | Whether to enable snapshot reading. Enabled by default when a non-empty path is configured. |
| `drampool_resource_metrics_interval_sec` | Snapshot reading interval in seconds, default `15`. |

### Access Metrics

After starting the inference service and sending inference requests, run the following command to view metrics:

```bash
curl http://<vllm-ip>:<vllm-port>/metrics \
  | grep -E '^ucm:(dramstore_|drampool_)'
```

`dramstore_` and `drampool_` identify DramStore metrics and imported DramPool metrics, respectively. Both are exported through vLLM's `/metrics` endpoint, and updates depend on vLLM collecting connector statistics, so metrics may not refresh without inference requests; DramPool does not provide a standalone HTTP `/metrics` endpoint, and `health.port` is used only for health checks.

### Metric Descriptions

#### DramStore Metrics

| Metric | Type | Meaning and Observation |
| --- | --- | --- |
| `dramstore_lookup_duration_ms` | Histogram | Cache lookup duration in milliseconds. Monitor lookup latency and variation. |
| `dramstore_dump_duration_ms` | Histogram | KV cache dump duration in milliseconds. Monitor dump latency. |
| `dramstore_load_duration_ms` | Histogram | KV cache load duration in milliseconds. Monitor load latency. |
| `dramstore_lookup_tasks_failed_total` | Counter | Failed cache lookup tasks. Monitor the frequency of lookup errors. |
| `dramstore_dump_tasks_failed_total` | Counter | Failed cache dump tasks. Monitor the frequency of dump errors. |
| `dramstore_load_tasks_failed_total` | Counter | Failed cache load tasks. Monitor the frequency of load errors. |
| `dramstore_dump_task_timeouts_total` | Counter | Timed-out cache dump tasks. Monitor the frequency of dump timeouts. |
| `dramstore_load_task_timeouts_total` | Counter | Timed-out cache load tasks. Monitor the frequency of load timeouts. |
| `dramstore_task_queue_size` | Gauge | Tasks waiting for processing. Monitor task backlog. |
| `dramstore_connect_failures_total` | Counter | Failed DramPool connections. Monitor service connection errors. |

#### DramPool Metrics

| Metric | Type | Meaning and Observation |
| --- | --- | --- |
| `drampool_buffer_pool_usage_ratio` | Gauge | Cache pool usage ratio, separated by cache block size. Monitor capacity pressure for each block size. |
| `drampool_metadata_entry_count` | Gauge | Total cached entries. Monitor changes in cache size. |
| `drampool_dump_nospace_failures_total` | Counter | Entries whose dump failed due to insufficient cache space. Monitor cache capacity shortages. |
| `drampool_lookup_miss_entries_total` | Counter | Entries missed during cache lookup. Monitor cache misses. |
| `drampool_transfer_failures_total` | Counter | Failed data transfers during cache dump and load. Monitor transfer errors. |
| `drampool_response_failures_total` | Counter | Failed responses to DramStore. Monitor service response errors. |
| `drampool_queue_request_size` | Gauge | Requests waiting for processing. Monitor request backlog. |
| `drampool_lookup_batch_total_duration_ms` | Histogram | Server-side processing duration per batch of cache lookup requests in milliseconds. Monitor lookup latency. |
| `drampool_dump_batch_total_duration_ms` | Histogram | Server-side processing duration per batch of cache dump requests in milliseconds. Monitor dump latency. |
| `drampool_load_batch_total_duration_ms` | Histogram | Server-side processing duration per batch of cache load requests in milliseconds. Monitor load latency. |

### Prometheus and Grafana Integration

Prometheus collects DramStore and DramPool metrics through vLLM's `/metrics` endpoint. For deployment instructions, see [Metrics Integration](https://ucm.readthedocs.io/en/latest/user-guide/observability/metrics/).

Import the [Dram dashboard](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/metrics/grafana_dram.json) into Grafana and select the corresponding Prometheus data source.

## Performance

### Test Environment

- Test model: DeepSeek-V3.1-Terminus-w8a8.
- Hardware: A single Ascend A3 host with 16 devices.
- Test tool: AISBench.

### Test Description

The following results show UCM performance with DramPool in the prefix caching scenario. Input lengths are 4,000, 8,000, 16,000, and 32,000 tokens; maximum concurrency levels are 1, 4, 16, and 32; and the average output length is 1,000 tokens. The table compares TTFT (time to first token) and speedup for the baseline, hit - 50, and hit - 90 scenarios. TTFT speedup is calculated as **(baseline TTFT / scenario TTFT − 1) × 100%**, rounded to one decimal place.

| **Input Length** | **Requests** | **Maximum Concurrency** | **Baseline TTFT** | **hit - 50 TTFT** | **TTFT Speedup (%)** | **hit - 90 TTFT** | **TTFT Speedup (%)** |
| -----------: | ---------: | -------------: | ------------: | ----------------: | ----------------: | ----------------: | ----------------: |
|            4 000 |            4 |              1 |             893.8 |             702.5 |           **+27.2%** |             639.2 |           **+39.8%** |
|            4 000 |           16 |              4 |            1092.5 |            1020.0 |            **+7.1%** |             922.1 |           **+18.5%** |
|            4 000 |           64 |             16 |            1789.4 |            1731.8 |            **+3.3%** |            1697.7 |            **+5.4%** |
|            4 000 |          128 |             32 |            2491.3 |            2076.5 |           **+20.0%** |            2151.3 |           **+15.8%** |
|            8 000 |            4 |              1 |            1716.2 |            1111.7 |           **+54.4%** |             671.7 |          **+155.5%** |
|            8 000 |           16 |              4 |            2105.9 |            1707.8 |           **+23.3%** |             986.0 |          **+113.6%** |
|            8 000 |           64 |             16 |            3919.6 |            2774.0 |           **+41.3%** |            1768.6 |          **+121.6%** |
|            8 000 |          128 |             32 |            5939.1 |            3243.9 |           **+83.1%** |            2383.2 |          **+149.2%** |
|           16 000 |            4 |              1 |            3949.0 |            2321.5 |           **+70.1%** |             801.9 |          **+392.5%** |
|           16 000 |           16 |              4 |            5422.5 |            3318.2 |           **+63.4%** |            1242.9 |          **+336.3%** |
|           16 000 |           64 |             16 |            8579.4 |            4977.5 |           **+72.4%** |            2264.1 |          **+278.9%** |
|           16 000 |          128 |             32 |           65085.5 |           29686.6 |          **+119.2%** |           18672.3 |          **+248.6%** |
|           32 000 |            4 |              1 |            9781.7 |            6070.6 |           **+61.1%** |            2055.6 |          **+375.9%** |
|           32 000 |           16 |              4 |           12574.2 |            8200.6 |           **+53.3%** |            2686.5 |          **+368.1%** |
|           32 000 |           64 |             16 |           46070.9 |           37792.1 |           **+21.9%** |           22434.8 |          **+105.4%** |
|           32 000 |          128 |             32 |          178751.1 |          149170.5 |           **+19.8%** |           96544.9 |           **+85.1%** |
