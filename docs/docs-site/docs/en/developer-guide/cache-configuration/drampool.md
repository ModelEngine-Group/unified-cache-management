# DramPool

[English](../../../en/developer-guide/cache-configuration/drampool.md) | [中文](../../../zh/developer-guide/cache-configuration/drampool.md)

DramPool pools host DRAM for KV cache reuse across inference instances. DramStore is the UCM client that reads and writes KV blocks in the pool through HIXL. Cached blocks expire after a configured lifetime; restarting DramPool clears its in-memory cache.

## Prerequisites

- Ascend A3 or later hardware, with **CANN 9.1.0 or later** and compatible drivers and HIXL libraries.
- A Linux environment with RDMA networking between DramPool and inference nodes, and enough available host memory for the configured pool and runtime buffers.
- UCM and a compatible vLLM-Ascend installation. See [Quickstart-vLLM-Ascend](../../user-guide/quick_start/index.md#vllm-ascend) for inference integration.

## Launch DramPool

### Build

From the UCM repository root, load the CANN environment and build UCM for A3. This command also builds and installs DramStore and DramPool:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PLATFORM=ascend-a3
pip install -v -e . --no-build-isolation
```

With this editable installation, the outputs are:

- `ucm/store/dram/libdramstore.so`: the DramStore client library, loaded by `UcmPipelineStore` when `store_pipeline: "Dram"` is selected.
- `bin/drampool`: the standalone DramPool server executable.

Both targets are part of UCM's Ascend build and require no separate build command. Deploy the client library with UCM on inference nodes and run the server executable on pool nodes.

### Configure and start

Copy [examples/drampool.yaml](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/drampool.yaml) to a runtime configuration file. Set `transport.device_ids` to available device IDs. The service address is supplied through `--addr`; the pool does not require a client endpoint list.

For example, the DramPool entry can be configured as:

```yaml
transport:
  device_ids: [0]
  manager_max_threads: 4
  hixl:
    listen_port: 26666
    enable_hixl_cs: false
health:
  port: 9100
```

Merge these settings into the copied file. Run the executable from the repository root; the editable build installs it under `bin/`.

Choose the supported tensor sizes before starting the pool. DramPool must support each distinct byte size in DramStore's `tensor_size_list`, not their sum. For standard GQA with separate K/V tensors, each size is `tokens per block × KV heads per worker × head dimension × bytes per element`. Qwen3-0.6B with TP=1, block size 128 and BF16 uses `128 × 8 × 128 × 2 = 262144` bytes, as in the command below. Other models and layouts require their actual tensor sizes; confirm them in the worker's `create UcmPipelineStore with config` log.

```bash
./bin/drampool \
  --addr 192.168.1.10:9000 \
  --pool-size-gb 64 \
  --kvcache-block-sizes 262144 \
  --config /absolute/path/to/drampool.yaml
```

Replace the IP address, capacity and block size with your deployment values. Ensure connectivity between DramPool and DramStore service addresses and HIXL communication ports.

| Parameter | Description |
| --- | --- |
| `--addr` | DramPool control address, reachable by every DramStore client. |
| `--pool-size-gb` | Host memory contributed by this process, in GiB. Leave memory for the OS and runtime buffers. |
| `--kvcache-block-sizes` | Supported KV block sizes in **bytes**, matching the blocks sent by DramStore; this is not vLLM's token `block_size`. Multiple sizes are space-separated. |
| `--kvcache-block-proportions` | Optional relative capacity shares, one per block size; equal shares by default. |
| `--ttl-minutes` | Optional absolute block lifetime, default `120` minutes. |
| `--config` | Runtime YAML path; default `examples/drampool.yaml`, relative to the working directory. |

Wait for `DramPool service ready` in the logs. With `health.port` enabled, check readiness with:

```bash
curl --fail http://192.168.1.10:9100/health
```

Send `SIGINT` or `SIGTERM` to stop DramPool gracefully.

### Configure DramStore and start inference

Copy [examples/ucm_config_dramstore.yaml](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/ucm_config_dramstore.yaml) as the inference-side configuration, for example `/etc/ucm/dramstore.yaml`. Set the local service address and DramPool addresses for your deployment:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Dram"
      local_addr: "192.168.1.20:9001"
      peer_addrs: ["192.168.1.10:9000"]
      hixl_listen_port: 36666
      enable_hixl_cs: false
      router_type: "ring_hash"

enable_event_sync: true
use_layerwise: true
```

| Parameter | Description |
| --- | --- |
| `local_addr` | DramStore's local service IP and base port. Use the inference node's IP and a distinct base port for each inference instance. |
| `peer_addrs` | DramPool service addresses, matching each pool's `--addr`. Multiple pools can be listed. |
| `hixl_listen_port` | Local HIXL base port, default `36666`. Actual ports are offset by physical device ID and scheduler/worker role; reserve a non-overlapping range for each instance. |
| `enable_hixl_cs` | Whether to enable HIXL CS, default `false`. |
| `router_type` | Routing strategy, default `ring_hash`. |

The connector supplies `device_id`, `tensor_size_list` and KV buffer addresses at runtime; do not fill them manually. Keep `enable_event_sync: true` so dumping waits for the compute-stream event before remote access to KV memory.

After DramPool is ready and vLLM-Ascend integration is configured, start a single-device inference instance:

```bash
vllm serve <model-path> \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --kv-transfer-config '{"kv_connector":"UCMConnector","kv_role":"kv_both","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/etc/ucm/dramstore.yaml"}}'
```

Replace the model and configuration paths and adjust inference options for your deployment.

When both components run on the same host and DramPool uses a single NPU, select different physical devices for the pool and inference process. For example, use pool `device_ids: [0]` and set `ASCEND_RT_VISIBLE_DEVICES=1` for inference.

To verify external reuse, send a prompt spanning several complete blocks and check that DramStore DUMP tasks succeed. Keep DramPool running, restart the inference process with the same configuration, and replay the prompt; confirm external hits and successful LOAD tasks. See [Verify external cache](../../user-guide/observability/verify-cache.md). A successful inference response alone does not establish cache reuse.

### Metrics

DramStore operation metrics and DramPool metrics are exposed through vLLM's `/metrics` endpoint. Enable DramPool snapshot export and configure the DramStore-side reader as follows.

**DramPool**

Adjust the `metrics` section in the copied `examples/drampool.yaml` for your deployment:

```yaml
metrics:
  enabled: true
  output_dir: /var/log/ucm/drampool
  interval_ms: 10000
```

The daemon atomically replaces `drampool_metrics.json` in `output_dir` every 10 seconds. Set `metrics.enabled: false` to disable snapshot collection and export. The example YAML enables export; omitting `metrics.enabled` defaults to `false`. An empty `output_dir` uses `logger.dir`, and `interval_ms` must be positive when enabled.

**DramStore**

Merge the following into the inference-side `ucm_connector_config` for the `Dram` pipeline:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Dram"
      # Keep the other DramStore connection settings.
      drampool_resource_metrics_enable: true
      drampool_resource_log_path: /var/log/ucm/drampool/drampool_metrics.json
      drampool_resource_metrics_interval_sec: 15
```

These three options control the client-side reader of DramPool snapshots, not DramStore's own operation metrics. Setting a nonempty path enables the reader by default; set `drampool_resource_metrics_enable: false` to disable it. The read interval defaults to 15 seconds, and the reader runs only in the scheduler process (`device_id < 0`). The snapshot must be visible and readable from that process; mount the output directory into it if the pool and inference run in different containers or hosts. The path refers to the inference process's filesystem.

Send an inference request, then verify DramStore and imported DramPool metrics through vLLM:

```bash
curl http://<vllm-ip>:<vllm-port>/metrics | grep -E '^ucm:(dramstore_|drampool_)'
```

DramPool exports a file snapshot rather than an HTTP `/metrics` endpoint; `health.port` serves only `/health`. Use [the Dram dashboard](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/metrics/grafana_dram.json) to inspect both components. Metrics refresh through vLLM's connector-statistics collection during inference; see [UCM Metrics Observability](../../user-guide/observability/metrics.md) for Prometheus and Grafana setup.

## Performance

### Overview

The following table compares TTFT (time to first token) for DramPool prefix-cache reuse across input lengths of 4,000, 8,000, 16,000 and 32,000 tokens, at maximum concurrency levels of 1, 4, 16 and 32. Average output length is 1,000 tokens. Results are shown for `Baseline`, `hit - 50` and `hit - 90`.

TTFT speedup is calculated as **(Baseline TTFT / Scenario TTFT − 1) × 100%**. Percentages are rounded to one decimal place; they express the gain in inverse TTFT, rather than the percentage reduction in latency.

| **Input length** | **Requests** | **Concurrent** | **Baseline TTFT** | **hit - 50 TTFT** | **TTFT Speedup (%)** | **hit - 90 TTFT** | **TTFT Speedup (%)** |
| ---------------: | -----------: | -------------: | ----------------: | ----------------: | -------------------: | ----------------: | -------------------: |
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

Across these runs, TTFT speedup is **3.3%–119.2%** for `hit - 50` and **5.4%–392.5%** for `hit - 90`. At input length 4,000 and concurrency 32, `hit - 90` has higher TTFT than `hit - 50`.
