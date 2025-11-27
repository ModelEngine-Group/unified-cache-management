# Observability

UCM (Unified Cache Management) provides detailed metrics monitoring through Prometheus endpoints, allowing in-depth monitoring of cache performance and behavior. This document describes how to enable and configure observability from the embedded vLLM `/metrics` API endpoint.

---

## Quick Start Guide

### 1) On UCM Side

First, set the `PROMETHEUS_MULTIPROC_DIR` environment variable.

```bash
export PROMETHEUS_MULTIPROC_DIR=/vllm-workspace
```

Then, start the UCM service.

```bash
export CUDA_VISIBLE_DEVICES=0
vllm serve /home/models/Qwen2.5-14B-Instruct  \
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
**Note**: You can refer to the `ucm_config.yaml` file at https://github.com/ModelEngine-Group/unified-cache-management/tree/develop/examples to configure the `metrics_config_path` parameter.

You can use the `vllm bench serve` command to run benchmarks:

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

Once the HTTP server is running, you can access the UCM metrics at the `/metrics` endpoint.

```bash
curl http://$<vllm-worker-ip>:8000/metrics | grep ucm:
```

You will also find some `.db` files in the `$PROMETHEUS_MULTIPROC_DIR` directory, which are temporary files used by Prometheus.

### 2) Start Prometheus and Grafana with Docker Compose

#### Create Docker Compose Configuration Files

First, create the `docker-compose.yaml` file:

```yaml
# docker-compose.yaml
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

Then, create the `prometheus.yaml` configuration file:

```yaml
# prometheus.yaml
global:
  scrape_interval: 5s
  evaluation_interval: 30s

scrape_configs:
  - job_name: vllm
    static_configs:
      - targets:
          - 'host.docker.internal:8000'  
```

**Note**: Make sure the port number in `prometheus.yaml` matches the port number used when starting the vLLM service.

#### Start Services

Run the following command in the directory containing `docker-compose.yaml` and `prometheus.yaml`:

```bash
docker compose up
```

This will start Prometheus and Grafana services.

### 3) Configure Grafana Dashboard

#### Access Grafana

Navigate to `http://<your-host>:3000`. Log in with the default username (`admin`) and password (`admin`). You will be prompted to change the password on first login.

#### Add Prometheus Data Source

1. Navigate to `http://<your-host>:3000/connections/datasources/new` and select **Prometheus**.

2. On the Prometheus configuration page, add the Prometheus server URL in the **Connection** section. For this Docker Compose setup, Grafana and Prometheus run in separate containers, but Docker creates DNS names for each container. You can directly use `http://prometheus:9090`.

3. Click **Save & Test**. You should see a green checkmark showing "Successfully queried the Prometheus API."

#### Import Dashboard

1. Navigate to `http://<your-host>:3000/dashboard/import`.

2. Click **Upload JSON file**, then upload the `unified-cache-management/examples/metrics/grafana.json` file.

3. Select the Prometheus data source configured earlier.

4. Click **Import** to complete the import.

You should now be able to see the UCM monitoring dashboard with real-time visualization of all 9 metrics.

## Available Metrics

UCM exposes various metrics to monitor its performance. The following table lists all available metrics organized by category:

| Metric Name | Type | Description |
|------------|------|-------------|
| **Load Operation Metrics** | | |
| `ucm:load_requests_num` | Histogram | Number of requests loaded per `start_load_kv` call |
| `ucm:load_blocks_num` | Histogram | Number of blocks loaded per `start_load_kv` call |
| `ucm:load_duration` | Histogram | Time to load KV cache from UCM (milliseconds) |
| `ucm:load_speed` | Histogram | Speed of loading from UCM (GB/s) |
| **Save Operation Metrics** | | |
| `ucm:save_requests_num` | Histogram | Number of requests saved per `wait_for_save` call |
| `ucm:save_blocks_num` | Histogram | Number of blocks saved per `wait_for_save` call |
| `ucm:save_duration` | Histogram | Time to save to UCM (milliseconds) |
| `ucm:save_speed` | Histogram | Speed of saving to UCM (GB/s) |
| **Lookup Hit Rate Metrics** | | |
| `ucm:interval_lookup_hit_rates` | Histogram | Hit rate of UCM lookup requests |

## Prometheus Configuration

Metrics configuration is defined in the `unified-cache-management/examples/metrics/metrics_configs.yaml` file:

```yaml
log_interval: 5  # Interval in seconds for logging metrics

prometheus:
  multiproc_dir: "/vllm-workspace"  # Prometheus directory
  metric_prefix: "ucm:"  # Metric name prefix
  
  enabled_metrics:
    counters: true
    gauges: true
    histograms: true
  
  histograms:
    - name: "load_requests_num"
      documentation: "Number of requests loaded from ucm"
      buckets: [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    # ... other metric configurations
```

---

