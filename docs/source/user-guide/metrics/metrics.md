# Observability

UCM exports metrics through the vLLM `/metrics` endpoint. The metrics are
registered from `examples/metrics/metrics_configs.yaml`, accumulated inside UCM,
and exposed through `prometheus_client` in Prometheus multiprocess mode.

## How Metrics Flow

1. `metrics_configs.yaml` defines counters, gauges, and histograms.
2. `PrometheusStatsLogger` creates matching `prometheus_client` metrics with
   `model_name` and `worker_id` labels.
3. Histogram bucket boundaries are taken from the Python Prometheus histogram
   and registered into the C++ metrics library.
4. UCM code calls `UpdateStats()` on the hot path.
5. The C++ metrics library records counter, gauge, and histogram bucket deltas in
   per-thread double buffers.
6. Every `log_interval` seconds, the observability thread calls
   `get_all_stats_and_clear()` and applies the deltas to `prometheus_client`.
7. vLLM exposes the resulting cumulative Prometheus series through `/metrics`.

Histograms are bucketed at update time. UCM no longer stores raw histogram
sample vectors, so there is no `histogram_max_length` setting and no histogram
sample dropping caused by a vector length cap. The `+Inf` bucket is added
automatically when the metric is registered.

## Quick Start

### 1. Configure UCM Metrics

Set the Prometheus multiprocess directory before starting vLLM:

```bash
export PROMETHEUS_MULTIPROC_DIR=/vllm-workspace
```

In the UCM config file used by vLLM, set `metrics_config_path` to the metrics
configuration file you want to use, for example:

```yaml
metrics_config_path: "/vllm-workspace/unified-cache-management/examples/metrics/metrics_configs.yaml"
```

Then start vLLM with the UCM connector:

```bash
export CUDA_VISIBLE_DEVICES=0
vllm serve /home/models/Qwen2.5-14B-Instruct \
    --max-model-len 5000 \
    --tensor-parallel-size 1 \
    --gpu_memory_utilization 0.87 \
    --trust-remote-code \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --enforce-eager \
    --max-num-batched-tokens 40000 \
    --max-num-seqs 10 \
    --host 0.0.0.0 \
    --port 8000 \
    --kv-transfer-config \
    '{
        "kv_connector": "UCMConnector",
        "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config.yaml"
        }
    }'
```

Run a benchmark if you want traffic for the metrics:

```bash
vllm bench serve \
    --backend vllm \
    --model /home/models/Qwen2.5-14B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --dataset-name random \
    --num-prompts 20 \
    --random-input-len 200 \
    --random-output-len 10 \
    --request-rate 1 \
    --ignore-eos
```

Check that UCM metrics are present:

```bash
curl http://<vllm-worker-ip>:8000/metrics | grep ucm:
```

Prometheus multiprocess `.db` files should also appear in
`$PROMETHEUS_MULTIPROC_DIR`.

### 2. Start Prometheus and Grafana

Create `docker-compose.yaml`:

```yaml
version: "3"

services:
  prometheus:
    image: prom/prometheus:latest
    extra_hosts:
      - "host.docker.internal:host-gateway"
    ports:
      - "9090:9090"
    volumes:
      - ${PWD}/prometheus.yaml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    depends_on:
      - prometheus
    ports:
      - "3000:3000"
```

Create `prometheus.yaml`:

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 30s

scrape_configs:
  - job_name: vllm
    static_configs:
      - targets:
          - "host.docker.internal:8000"
```

Make sure the target port matches the vLLM service port. Then start the stack:

```bash
docker compose up
```

### 3. Import Grafana Dashboards

Open `http://<your-host>:3000`, add a Prometheus data source pointing to
`http://prometheus:9090`, then import the dashboard JSONs you need:

| File | Use case |
|------|----------|
| `examples/metrics/grafana_connector.json` | Connector-level activity: hit rate, request/block sizes, end-to-end load/save durations and speeds. |
| `examples/metrics/grafana_pipeline_store.json` | Cache Store and Posix Store diagnosis: task breakdowns, queue wait, transfer duration, bandwidth, backend load ratio. |
| `examples/metrics/grafana_layerwise.json` | `use_layerwise=true` diagnosis: `wait_for_layer_load()` blocking, inter-call interval, submit diagnostics, `wait_for_save()` tail. |

The dashboards share the `ucm` Grafana tag. After importing them, the dashboard
header contains an "Other UCM dashboards" dropdown that links between UCM
dashboards while preserving the time range and `model_name` value.

## Dashboard View and Worker Selectors

Each dashboard has a `View` selector and a `worker_id` selector:

- **Aggregated**: default service-level view. Worker labels are collapsed.
- **Per Worker**: split panels by `worker_id` for worker-specific diagnosis.
- **worker_id**: defaults to **All**. Select a specific worker ID to filter all
  UCM panels to that worker only.

Heatmap panels and panels grouped by another dimension may ignore the `View`
selector because their grouping is already defined by the panel. They still use
the `worker_id` filter.

## Metrics Configuration

Metrics are configured in `examples/metrics/metrics_configs.yaml`:

```yaml
log_interval: 5
multiproc_dir: "/vllm-workspace"
metric_prefix: "ucm:"

counter:
  - name: "cache_load_bytes_total"
    documentation: "Total bytes loaded through the Cache stage"

gauge:
  - name: "cache_lookup_hit_rate"
    documentation: "Instantaneous Cache stage hit rate from the most recent lookup call"
    multiprocess_mode: "livemostrecent"

histogram:
  - name: "cache_load_duration_ms"
    documentation: "End-to-end Cache stage load task duration (ms)"
    buckets: [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
```

Metric names are exported with the configured prefix. For example,
`cache_load_duration_ms` becomes `ucm:cache_load_duration_ms`. Prometheus also
exports histogram helper series such as `_bucket`, `_sum`, and `_count`.

Counter values are increments. Gauge values replace the current value.
Histogram values are observations that are immediately assigned to configured
buckets in the C++ metrics library.

## Available Metrics

The default metrics configuration contains the following UCM metrics.

### Counters

| Metric | Description |
|--------|-------------|
| `ucm:cache_lookup_hit_blocks_total` | Number of lookup hits served by the Cache stage. |
| `ucm:cache_lookup_miss_blocks_total` | Number of lookup misses at the Cache stage. |
| `ucm:cache_load_blocks_total` | Total blocks loaded by the Cache stage. |
| `ucm:cache_dump_blocks_total` | Total blocks dumped by the Cache stage. |
| `ucm:cache_load_shards_total` | Total shards dispatched by Cache load. |
| `ucm:cache_load_backend_shards_total` | Shards that descended to the backend on load. |
| `ucm:cache_dump_shards_total` | Total shards dispatched by Cache dump. |
| `ucm:cache_dump_backend_shards_total` | Shards actually pushed to backend on dump. |
| `ucm:cache_load_bytes_total` | Total bytes loaded through the Cache stage. |
| `ucm:cache_dump_bytes_total` | Total bytes dumped through the Cache stage. |
| `ucm:posix_s2h_bytes_total` | Total bytes transferred from Posix storage to host buffer. |
| `ucm:posix_h2s_bytes_total` | Total bytes transferred from host buffer to Posix storage. |
| `ucm:load_bytes_total` | Total bytes loaded through the UCM connector. |
| `ucm:save_bytes_total` | Total bytes saved through the UCM connector. |

### Gauges

| Metric | Description |
|--------|-------------|
| `ucm:cache_lookup_hit_rate` | Instantaneous Cache stage hit rate from the most recent lookup call. |

### Histograms

| Metric | Description |
|--------|-------------|
| `ucm:load_requests_num` | Number of requests loaded from UCM. |
| `ucm:load_blocks_num` | Number of blocks loaded from UCM. |
| `ucm:load_duration` | Time to load from UCM in milliseconds. |
| `ucm:load_speed` | Speed of loading from UCM in GB/s. |
| `ucm:save_requests_num` | Number of requests saved to UCM. |
| `ucm:save_blocks_num` | Number of blocks saved to UCM. |
| `ucm:save_duration` | Time to save to UCM in milliseconds. |
| `ucm:save_speed` | Speed of saving to UCM in GB/s. |
| `ucm:interval_lookup_hit_rates` | Hit rates of UCM lookup requests. |
| `ucm:cache_lookup_duration_ms` | Cache buffer lookup wall-clock time. |
| `ucm:cache_lookup_backend_duration_ms` | Backend lookup wall-clock time when descending due to no buffer or buffer miss. |
| `ucm:cache_load_duration_ms` | End-to-end Cache stage load task duration in milliseconds. |
| `ucm:cache_dump_duration_ms` | End-to-end Cache stage dump task duration in milliseconds. |
| `ucm:cache_load_bandwidth_gbps` | Cache stage effective load bandwidth in GB/s. |
| `ucm:cache_dump_bandwidth_gbps` | Cache stage effective dump bandwidth in GB/s. |
| `ucm:cache_load_queue_wait_duration_ms` | Time a Cache load task spent queued before dispatch worker pickup. |
| `ucm:cache_dump_queue_wait_duration_ms` | Time a Cache dump task spent queued before dispatch worker pickup. |
| `ucm:cache_load_dispatch_duration_ms` | Cache load dispatch cost: buffer allocation plus backend submission. |
| `ucm:cache_shard_backend_wait_ms` | Cache load per-shard `WaitBackendTaskReady()` duration. |
| `ucm:cache_shard_h2d_ms` | Cache load per-shard H2D async submit duration. |
| `ucm:cache_dump_mkbuf_duration_ms` | Cache dump mk_buf phase duration. |
| `ucm:cache_d2h_duration_ms` | Cache dump D2H stream sync phase duration. |
| `ucm:cache_dump_backend_submit_duration_ms` | Cache dump synchronous backend submit duration. |
| `ucm:posix_load_task_duration_ms` | End-to-end Posix load task duration. |
| `ucm:posix_dump_task_duration_ms` | End-to-end Posix dump task duration. |
| `ucm:posix_s2h_bandwidth_gbps` | Posix stage read bandwidth per task in GB/s. |
| `ucm:posix_h2s_bandwidth_gbps` | Posix stage write bandwidth per task in GB/s. |
| `ucm:posix_load_queue_wait_duration_ms` | Time a Posix load task spent queued before first worker pickup. |
| `ucm:posix_dump_queue_wait_duration_ms` | Time a Posix dump task spent queued before first worker pickup. |
| `ucm:layerwise_wait_blocking_ms` | Time `wait_for_layer_load()` blocked before returning. |
| `ucm:layerwise_wait_tasks_count` | Number of per-request load tasks awaited in a single layer wait. |
| `ucm:layerwise_inter_wait_interval_ms` | Interval between consecutive `wait_for_layer_load()` calls. |
| `ucm:layerwise_next_layer_submit_ms` | Time to submit next layer's load tasks inside `wait_for_layer_load()`. |
| `ucm:layerwise_first_layer_submit_ms` | Time to submit first layer load tasks during `start_load_kv`. |
| `ucm:layerwise_first_layer_requests` | Number of requests whose first-layer load was submitted in `start_load_kv`. |
| `ucm:layerwise_save_submit_ms` | Time to submit one layer's dump task in `save_kv_layer()`. |
| `ucm:layerwise_save_tail_total_ms` | Total `wait_for_save()` duration. |
