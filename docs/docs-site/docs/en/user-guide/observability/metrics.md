# UCM Metrics Observability

UCM exports metrics through the vLLM connector and reuses vLLM's Prometheus `/metrics` endpoint. No separate exporter,
export mode, or service port is required.

We recommend using Prometheus to scrape vLLM metrics and Grafana to visualize the collected data.

Use a scrape and dashboard refresh interval of **at least 5 seconds** for UCM metrics. The Prometheus and Metrics-view
examples in this guide both use 5 seconds. A shorter interval usually does not make UCM metrics update faster.

The effective refresh frequency also depends on vLLM. UCM first accumulates metrics internally. New data is synchronized
to the Prometheus metrics exposed by vLLM only after vLLM processes a request and calls the connector's
`get_kv_connector_stats()` method. **When there are no inference requests, vLLM does not call this method and UCM metrics
do not update.**

## Enable or Disable Metrics

### Use the Built-in Configuration

UCM metrics are **enabled by default**. When `metrics_config_path` is omitted, UCM uses the complete built-in metric set.
Metrics can also be enabled explicitly:

```yaml
enable_metrics: true
```

To disable all UCM metrics:

```yaml
enable_metrics: false
```

### Use a Custom Configuration

To restrict the exported metric set or customize Histogram buckets, set the following top-level UCM options:

```yaml
enable_metrics: true
metrics_config_path: "/workspace/unified-cache-management/examples/metrics/metrics_configs.yaml"
```

Once `metrics_config_path` is set, the file becomes the metric enable-list. Only metrics defined in that file are registered.

The metrics file must exist and be readable by the vLLM process. Otherwise, UCM metrics are not exposed.

## Access Metrics

Start vLLM with the UCM connector and send at least one inference request. Then verify that UCM metrics are available:

```bash
curl http://<vllm-ip>:<vllm-port>/metrics | grep '^ucm:'
```

Most UCM metrics appear only after their corresponding code path has run. When there is no external-storage hit, only a
small subset of metrics may be present.

### UCM Metric Labels

Each metric exported through the vLLM connector carries these labels:

| Label | Meaning | Example |
| --- | --- | --- |
| `model_name` | Model name served by vLLM, taken from the vLLM model configuration | `Qwen3-32B` |
| `engine` | vLLM engine that produced the metric; distinguishes DP instances in the same service | `engine-0` |
| `worker_rank` | UCM process that produced the metric, corresponding to a TP instance; workers use their distributed rank and the scheduler uses `scheduler` | `0`, `1`, `scheduler` |

For example:

```text
ucm:cache_load_bytes_total{model_name="Qwen3-32B",engine="engine-0",worker_rank="0"} 1.048576e+08
```

## Prometheus and Grafana Integration

Prometheus and Grafana are the recommended combination for observing UCM. Prometheus periodically scrapes the vLLM
metrics endpoint, stores historical time-series data, and provides a query interface. Grafana queries Prometheus and
displays the metrics as dashboards.

### Configure Prometheus

If Prometheus already scrapes vLLM's `/metrics` endpoint, no additional scrape job is needed for UCM because both vLLM
and UCM metrics are exposed through the same endpoint.

Create `prometheus.yml` and configure Prometheus to scrape the vLLM service:

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 30s

scrape_configs:
  - job_name: vllm
    metrics_path: /metrics
    static_configs:
      - targets:
          - "<vllm-ip>:8000"
```

Set the scrape target to an engine address and port reachable from the Prometheus container, such as the quickstart host's actual IP and port 7800. Do not use the Prometheus container's own `127.0.0.1`.

From the host directory containing `prometheus.yml`, create the network and persistent volume, then start Prometheus:

```bash
docker network create ucm-monitoring
docker volume create prometheus-data

docker run -d \
  --name prometheus \
  --restart unless-stopped \
  --network ucm-monitoring \
  -p 9090:9090 \
  -v "$PWD/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  -v prometheus-data:/prometheus \
  prom/prometheus
```

Open `http://<prometheus-host>:9090/targets`, confirm the `vllm` target is **UP**, then query `ucm:` metrics. Reuse `ucm-monitoring` when the network already exists.

### Install Grafana

Create a persistent volume and start Grafana:

```bash
docker volume create grafana-data

docker run -d \
  --name grafana \
  --restart unless-stopped \
  --network ucm-monitoring \
  -p 3000:3000 \
  -v grafana-data:/var/lib/grafana \
  grafana/grafana
```

Open `http://<grafana-ip>:3000`. On the first login, use `admin` for both the username and password, then change the
password when prompted.

### Add the Prometheus Data Source

In Grafana, go to **Connections** → **Add new connection**, search for **Prometheus**, and configure:

- Prometheus server URL: `http://prometheus:9090`
- Authentication: **No authentication** for an unauthenticated local deployment
- Select **Save & test** and verify that Grafana can query Prometheus

The shared `ucm-monitoring` network resolves the hostname `prometheus`. For Grafana running elsewhere, use a Prometheus address reachable from that environment.

### Import UCM Dashboards

Go to **Dashboards** → **New** → **Import**, upload the required dashboard JSON file, select the Prometheus data source,
and click **Import**.

UCM provides these dashboards:

| File | Purpose |
| --- | --- |
| `examples/metrics/grafana_vllm.json` | vLLM request latency, token throughput, scheduler state, and cache state |
| `examples/metrics/grafana_store.json` | Store lookup, load, dump, bandwidth, and cache activity |
| `examples/metrics/grafana_connector.json` | Connector Lookup/Load/Save request counts, block counts, durations, throughput, and errors |

## Inspect metrics from a terminal

You can inspect the engine endpoint without Prometheus/Grafana. From the UCM checkout root in a terminal that can reach the endpoint, run:

```bash
python3 -m pip install -e toolkit
ucm-toolkit run metrics-view list-configs
ucm-toolkit run metrics-view check \
  --url http://127.0.0.1:7800/metrics \
  --config store
```

This inspects the current snapshot. See [metrics-view operations](../../toolkit/user/metrics-view.md) for background collection, time-window queries and MLA TP parameters.

## Reference

- [Metric definitions](metrics-reference.md)
- [Store health metrics](health-metrics.md)
