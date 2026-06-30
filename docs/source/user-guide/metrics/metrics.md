# Observability

UCM exports metrics through the vLLM `/metrics` endpoint. By default, the
built-in metrics config in `ucm/default_metrics_config.py` registers the full
metric set. If `metrics_config_path` is provided, that file becomes the
enable-list instead. Metrics are accumulated inside UCM and fanned out to the
enabled Python-side consumers.

## How Metrics Flow

1. The built-in default config, or a user-provided `metrics_configs.yaml`, defines
   counters, gauges, and histograms.
2. The Python metrics dispatcher drains the C++ metrics snapshot once and fans
   it out to the enabled `multiproc` and `vllm_connector` consumers.
3. `multiproc` creates `prometheus_client` metrics with `model_name` and
   `worker_id` labels. `vllm_connector` creates vLLM KV connector metrics with
   `model_name`, `engine`, and `worker_rank` labels.
4. Histogram bucket boundaries are taken from the Python Prometheus histogram
   and registered into the C++ metrics library.
5. UCM code calls `UpdateStats()` on the hot path.
6. The C++ metrics library records counter, gauge, and histogram bucket deltas in
   per-thread double buffers.
7. The dispatcher applies deltas to each enabled Python consumer without one
   consumer clearing the other's accumulated snapshot.
8. vLLM exposes the resulting cumulative Prometheus series through `/metrics`.

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

In the UCM config file used by vLLM, metrics are enabled by default. You can set
the switch explicitly:

```yaml
enable_metrics: true
```

To disable all UCM metrics:

```yaml
enable_metrics: false
```

To customize which metrics are registered, set `metrics_config_path` to the
metrics configuration file you want to use. When this path is omitted, UCM uses
the built-in default config:

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

Check that UCM vLLM connector metrics are present:

```bash
curl http://<vllm-worker-ip>:8000/metrics | grep 'ucm:'
```

If the `multiproc` consumer is enabled, Prometheus multiprocess `.db` files
should also appear in `$PROMETHEUS_MULTIPROC_DIR`.

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
| `examples/metrics/grafana_layerwise.json` | `use_layerwise=true` diagnosis: `wait_for_layer_load()` blocking, inter-call interval, and asynchronous dump submit diagnostics. |
| `examples/metrics/grafana_vllm.json` | vLLM service-side request latency, token throughput, scheduler state, and cache state. |

The dashboards share the `ucm` Grafana tag. After importing them, the dashboard
header contains an "Other UCM dashboards" dropdown that links between UCM
dashboards while preserving the time range and `model_name` value.

## Dashboard Job, View, and Worker Selectors

Each dashboard has a `job` selector. It defaults to **All** and uses regex
matching, so dashboards also work for metrics that do not carry a `job` label.

The UCM dashboards also have a `View` selector and a `worker_rank` selector:

- **Aggregated**: default service-level view. Worker labels are collapsed.
- **Per Worker**: split panels by `worker_rank` for worker-specific diagnosis.
- **worker_rank**: defaults to **All**. Select a specific worker rank to filter all
  UCM panels to that worker only.

Heatmap panels and panels grouped by another dimension may ignore the `View`
selector because their grouping is already defined by the panel. They still use
the `worker_rank` filter.

## Metrics Configuration

Metrics are configured in `examples/metrics/metrics_configs.yaml`:

```yaml
log_interval: 5
# multiproc_dir: "/vllm-workspace"
# multiproc_prefix: "ucm_multiproc:"
vllm_connector_prefix: "ucm:"

consumers:
  # multiproc: true
  vllm_connector: true

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

Metric names are exported per consumer. For example, `cache_load_duration_ms`
is exported as `ucm:cache_load_duration_ms` by the default `vllm_connector`
consumer. If `multiproc` is also enabled, use a separate prefix such as
`ucm_multiproc:` so both consumers do not register the same Prometheus metric.
Prometheus also exports histogram helper series such as `_bucket`, `_sum`,
and `_count`.

Counter values are increments. Gauge values replace the current value.
Histogram values are observations that are immediately assigned to configured
buckets in the C++ metrics library.

## Available Metrics

The default metrics configuration contains the following UCM metric names. The
table uses the default `vllm_connector_prefix`. UCM duration metrics are
exported in milliseconds.

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
| `ucm:save_duration` | Time from UCM connector `wait_for_save` entry to async dump task completion in milliseconds. |
| `ucm:save_completion_wait_duration` | Time spent blocked while confirming async UCM connector dump completion in milliseconds. |
| `ucm:interval_lookup_hit_rates` | Hit rates of UCM lookup requests. |
| `ucm:cache_lookup_duration_ms` | Cache buffer lookup wall-clock time. |
| `ucm:cache_lookup_backend_duration_ms` | Backend lookup wall-clock time when descending due to no buffer or buffer miss. |
| `ucm:cache_load_duration_ms` | End-to-end Cache stage load task duration in milliseconds. |
| `ucm:cache_dump_duration_ms` | End-to-end Cache stage dump task duration in milliseconds. |
| `ucm:cache_load_bandwidth_gbps` | Cache stage effective load throughput in GB/s over the whole task lifetime, including queue/backend waits. Not a DMA bandwidth (see `cache_h2d_bandwidth_gbps`). |
| `ucm:cache_dump_bandwidth_gbps` | Cache stage effective dump throughput in GB/s over the whole task lifetime, including queue and compute-event waits. Not a DMA bandwidth (see `cache_d2h_bandwidth_gbps`). |
| `ucm:cache_load_queue_wait_duration_ms` | Time a Cache load task spent queued before dispatch worker pickup. |
| `ucm:cache_dump_queue_wait_duration_ms` | Time a Cache dump task spent queued before dispatch worker pickup. |
| `ucm:cache_load_backend_submit_duration_ms` | Cache load backend submit duration: buffer allocation plus backend load submission. |
| `ucm:cache_shard_backend_wait_ms` | Cache load per-shard `WaitBackendTaskReady()` duration. |
| `ucm:cache_h2d_submit_ms` | Cache load per-shard H2D async submit CPU cost. Submission only, not the transfer. |
| `ucm:cache_h2d_sync_ms` | Cache load residual H2D stream drain after the last shard submit. Large values mean H2D copy is the bottleneck. |
| `ucm:cache_h2d_bandwidth_gbps` | Cache load pure H2D copy bandwidth, directly comparable to memcpy microbenchmarks. |
| `ucm:cache_dump_mkbuf_duration_ms` | Cache dump mk_buf phase duration (buffer allocation/reuse plus D2H async submit). |
| `ucm:cache_dump_prereq_wait_ms` | Cache dump wait for the prerequisite compute event before D2H can start. Large values mean dump is compute-gated. |
| `ucm:cache_d2h_duration_ms` | Cache dump pure D2H copy drain, compute-event wait excluded. |
| `ucm:cache_d2h_bandwidth_gbps` | Cache dump pure D2H copy bandwidth, directly comparable to memcpy microbenchmarks. |
| `ucm:cache_dump_backend_submit_duration_ms` | Cache dump synchronous backend submit duration. |
| `ucm:cache_dump_backend_wait_duration_ms` | Cache dump wait for the lower tier to finish writing. Large values mean storage write is the bottleneck. |
| `ucm:posix_load_task_duration_ms` | End-to-end Posix load task duration. |
| `ucm:posix_dump_task_duration_ms` | End-to-end Posix dump task duration. |
| `ucm:posix_s2h_bandwidth_gbps` | Posix stage read bandwidth per task in GB/s. |
| `ucm:posix_h2s_bandwidth_gbps` | Posix stage write bandwidth per task in GB/s. |
| `ucm:posix_load_queue_wait_duration_ms` | Time a Posix load task spent queued before first worker pickup. |
| `ucm:posix_dump_queue_wait_duration_ms` | Time a Posix dump task spent queued before first worker pickup. |
| `ucm:layerwise_batch_total_ms` | Layerwise batch wall-clock time from `start_load_kv()` entry to `wait_for_save()` return. |
| `ucm:layerwise_batch_total_load_only_ms` | Layerwise load-only batch wall-clock time. |
| `ucm:layerwise_batch_total_save_only_ms` | Layerwise save-only batch wall-clock time. |
| `ucm:layerwise_batch_total_load_save_ms` | Layerwise load-and-save batch wall-clock time. |
| `ucm:layerwise_batch_total_no_transfer_ms` | Layerwise batch wall-clock time with neither load nor save work. |
| `ucm:layerwise_batch_load_wait_total_load_only_ms` | Total `wait_for_layer_load()` blocking time accumulated within one load-only layerwise batch. |
| `ucm:layerwise_batch_load_wait_total_load_save_ms` | Total `wait_for_layer_load()` blocking time accumulated within one load-and-save layerwise batch. |
| `ucm:layerwise_batch_save_tail_save_only_ms` | `wait_for_save()` tail duration within one save-only layerwise batch. |
| `ucm:layerwise_batch_save_tail_load_save_ms` | `wait_for_save()` tail duration within one load-and-save layerwise batch. |
| `ucm:layerwise_wait_blocking_ms` | Time `wait_for_layer_load()` blocked before returning. |
| `ucm:layerwise_wait_tasks_count` | Number of per-request load tasks awaited in a single layer wait. |
| `ucm:layerwise_inter_wait_interval_ms` | Interval between consecutive `wait_for_layer_load()` calls. |
| `ucm:layerwise_next_layer_submit_ms` | Time to submit next layer's load tasks inside `wait_for_layer_load()`. |
| `ucm:layerwise_first_layer_submit_ms` | Time to submit first layer load tasks during `start_load_kv`. |
| `ucm:layerwise_first_layer_requests` | Number of requests whose first-layer load was submitted in `start_load_kv`. |
| `ucm:layerwise_save_submit_ms` | Time to submit one layer's dump task in `save_kv_layer()`. |
| `ucm:layerwise_save_tail_total_ms` | Legacy metric; LayerWise no longer waits for dump completion in `wait_for_save()`. |
