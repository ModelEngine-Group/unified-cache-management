# UCM Metrics Observability

UCM exports metrics through the vLLM connector and reuses vLLM's Prometheus `/metrics` endpoint. No separate exporter, export mode, or service port is required.

We recommend using Prometheus to scrape vLLM metrics and Grafana to visualize the collected data.

Use a scrape and dashboard refresh interval of **at least 5 seconds** for UCM metrics. The Prometheus and Metrics-view examples in this guide both use 5 seconds. A shorter interval usually does not make UCM metrics update faster.

The effective refresh frequency also depends on vLLM. UCM first accumulates metrics internally. New data is synchronized to the Prometheus metrics exposed by vLLM only after vLLM processes a request and calls the connector's `get_kv_connector_stats()` method. **When there are no inference requests, vLLM does not call this method and UCM metrics do not update.**

## Metrics workflow

The following example uses DP=2 and TP=1. Each DP contains one worker Store and one scheduler Store, and the two DPs process their own requests or batches.

```{mermaid}
sequenceDiagram
    autonumber
    participant C as Client
    participant A as vLLM API Server
    participant S0 as DP 0 Scheduler
    participant W0 as DP 0 Worker
    participant S1 as DP 1 Scheduler
    participant W1 as DP 1 Worker
    participant M as vLLM /metrics Endpoint<br/>(hosted by API Server)
    participant P as Prometheus

    C->>A: Send inference requests
    par Request or batch assigned to DP 0
        A->>S0: Schedule batch A
        S0->>W0: Run the model and UCM Lookup/Load/Save
        W0->>W0: Accumulate metrics inside UCM
        W0->>W0: vLLM calls get_kv_connector_stats() to collect accumulated UCM metrics
        W0-->>S0: Return worker metrics (worker_rank=0)
        S0->>S0: Return scheduler metrics (worker_rank=scheduler)
        S0-->>A: Report connector stats (engine=engine-0)
    and Request or batch assigned to DP 1
        A->>S1: Schedule batch B
        S1->>W1: Run the model and UCM Lookup/Load/Save
        W1->>W1: Accumulate metrics inside UCM
        W1->>W1: vLLM calls get_kv_connector_stats() to collect accumulated UCM metrics
        W1-->>S1: Return worker metrics (worker_rank=1)
        S1->>S1: Return scheduler metrics (worker_rank=scheduler)
        S1-->>A: Report connector stats (engine=engine-1)
    end
    A->>M: Update ucm:* series by metric type
    P->>M: GET /metrics
    M-->>P: Return vllm:* and ucm:* metrics
```

UCM accumulates Counters, Gauges, and Histograms in the process that performs each Lookup, Load, Save, or health probe. While processing requests, vLLM obtains the accumulated UCM metrics from the worker and scheduler connectors. These metrics return with each DP's engine stats, and vLLM Prometheus metrics write them to the corresponding series with the `model_name`, `engine`, and `worker_rank` labels.

The vLLM `/metrics` endpoint and Prometheus registry reside in the API Server process. Prometheus scrapes that endpoint directly; the API Server's HTTP route only returns data that has already been synchronized to the registry and does not call `get_kv_connector_stats()`. With no inference requests, UCM may still produce new data internally, but that data appears in `/metrics` only after the next vLLM request triggers synchronization.

## 1. Enable or Disable Metrics

### 1.1 Use the Built-in Configuration

UCM metrics are **enabled by default**. When `metrics_config_path` is omitted, UCM uses the complete built-in metric set. Metrics can also be enabled explicitly:

```yaml
enable_metrics: true
```

To disable all UCM metrics:

```yaml
enable_metrics: false
```

### 1.2 Use a Custom Configuration

To restrict the exported metric set or customize Histogram buckets, set the following top-level UCM options:

```yaml
enable_metrics: true
metrics_config_path: "/workspace/unified-cache-management/examples/metrics/metrics_configs.yaml"
```

Once `metrics_config_path` is set, the file becomes the metric enable-list. Only metrics defined in that file are registered.

The metrics file must exist and be readable by the vLLM process. Otherwise, UCM metrics are not exposed.

## 2. Access Metrics

Start vLLM with the UCM connector and send at least one inference request. Then verify that UCM metrics are available:

```bash
curl http://<vllm-ip>:<vllm-port>/metrics | grep '^ucm:'
```

Most UCM metrics appear only after their corresponding code path has run. When there is no external-storage hit, only a small subset of metrics may be present.

### 2.1 UCM Metric Labels

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

Prometheus adds the following target labels when it scrapes the endpoint:

| Label | Source | Meaning |
| --- | --- | --- |
| `job` | `job_name` in `prometheus.yml` | Scrape job name; `vllm` in this guide |
| `instance` | Prometheus scrape target | Scraped vLLM address and port, such as `10.0.0.8:8000` |

Because Prometheus adds `job` and `instance`, they are normally absent from the raw output of `curl /metrics`. Histogram `_bucket` series also carry the `le` label, which represents the bucket upper bound and is not a UCM business label.

## 3. Prometheus and Grafana

Prometheus and Grafana are the recommended combination for observing UCM. Prometheus periodically scrapes the vLLM metrics endpoint, stores historical time-series data, and provides a query interface. Grafana queries Prometheus and displays the metrics as dashboards.

### 3.1 Install and Configure Prometheus

If Prometheus already scrapes vLLM's `/metrics` endpoint, no additional scrape job is needed for UCM because both vLLM and UCM metrics are exposed through the same endpoint.

The following example installs Prometheus with Docker. For other installation methods, see the [Prometheus installation documentation](https://prometheus.io/docs/prometheus/latest/installation/).

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

Replace `<vllm-ip>:8000` with an address and port reachable from the Prometheus container. Do not use the container's own `127.0.0.1:8000`. If vLLM runs on the host, use the host's actual IP address. Docker Desktop users can also use `host.docker.internal:8000`.

Create a network and a persistent Prometheus volume:

```bash
docker network create ucm-monitoring
docker volume create prometheus-data
```

From the directory containing `prometheus.yml`, start Prometheus:

```bash
docker run -d \
  --name prometheus \
  --restart unless-stopped \
  --network ucm-monitoring \
  -p 9090:9090 \
  -v "$(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  -v prometheus-data:/prometheus \
  prom/prometheus
```

Open `http://<prometheus-ip>:9090/targets` and verify that the `vllm` target is **UP**. Then search for `vllm:` and `ucm:` on the Prometheus query page to verify that both metric families are available.

### 3.2 Install Grafana

The following example uses the official Grafana Docker image. For other installation methods, see the [Grafana installation documentation](https://grafana.com/docs/grafana/latest/setup-grafana/installation/).

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

Open `http://<grafana-ip>:3000`. On the first login, use `admin` for both the username and password, then change the password when prompted.

### 3.3 Add the Prometheus Data Source

In Grafana, go to **Connections** → **Add new connection**, search for **Prometheus**, and configure:

- Prometheus server URL: `http://prometheus:9090`
- Authentication: **No authentication** for an unauthenticated local deployment
- Select **Save & test** and verify that Grafana can query Prometheus

The hostname `prometheus` works because both containers joined the `ucm-monitoring` network. For other deployment layouts, use the Prometheus URL reachable from the Grafana service. The Prometheus data source is built into Grafana and requires no additional plugin.

### 3.4 Import UCM Dashboards

Go to **Dashboards** → **New** → **Import**, upload the required dashboard JSON file, select the Prometheus data source, and click **Import**.

UCM provides these dashboards:

| File | Purpose |
| --- | --- |
| `examples/metrics/grafana_vllm.json` | vLLM request latency, token throughput, scheduler state, cache state, Store health, and probe trends |
| `examples/metrics/grafana_connector.json` | Connector aggregate bandwidth, common interface duration, and Layerwise load duration |
| `examples/metrics/grafana_store.json` | Cache, Mooncake, and Posix Store performance metrics grouped by Store |

The `job` selector defaults to **All**. UCM dashboards also provide Aggregated/Per Worker views and a `worker_rank` filter.
In the Store dashboard, Cache and Posix are expanded by default, while Mooncake is collapsed.

Use these aggregation rules:

- For Counters, apply `rate()` or `increase()` to each series over the same time window, then sum across workers.
- For ratios, divide the aggregated numerator by the aggregated denominator. Do not calculate per-worker ratios and then take their arithmetic mean.
- Display Gauges by worker or aggregate them with `min`/`max`, depending on their semantics.
- For Histograms, aggregate `_bucket` series across workers before calculating percentiles.

## 4. Metrics-view

When Prometheus/Grafana or a graphical environment is unavailable, use the Metrics-view command-line tool from the UCM toolkit.

Metrics-view can inspect a single `/metrics` snapshot or collect samples in the background into SQLite and query a selected time window. It does not depend on Prometheus or Grafana.

Install the toolkit:

```bash
cd unified-cache-management
pip install -e toolkit
```

List built-in configurations:

```bash
ucm-toolkit run metrics-view list-configs
```

The built-in configurations correspond to the Grafana dashboards:

| Metrics View configuration | Grafana dashboard |
| --- | --- |
| `vllm` | `examples/metrics/grafana_vllm.json` |
| `connector` | `examples/metrics/grafana_connector.json` |
| `store` | `examples/metrics/grafana_store.json` |
| `metrics_lite` | A compact selection of commonly used metrics |

The `vllm`, `connector`, and `store` configurations expose every query shown by
the corresponding dashboard. A panel with one data series uses the panel title
as its name. For a panel with multiple series, the name is
`<panel title>: <series name>`.

### 4.1 Inspect the Current Snapshot

`check` fetches the current `/metrics` snapshot and displays cumulative results since service startup:

```bash
ucm-toolkit run metrics-view check \
  --url http://127.0.0.1:8000/metrics \
  --config vllm
```

GQA/MHA models use the default parameters. For MLA models, pass the actual service TP size. For example, for TP=8:

```bash
ucm-toolkit run metrics-view check \
  --url http://127.0.0.1:8000/metrics \
  --config metrics_lite \
  --config-param tp_size=8
```

The GB/s value reported by `check` is cumulative bytes divided by cumulative service uptime. It is not suitable for instantaneous bandwidth analysis; use background collection for bandwidth analysis.

### 4.2 Background Collection and Queries

Background collection does not require a query configuration. Pass `--url` multiple times to collect from multiple endpoints:

```bash
ucm-toolkit run metrics-view start \
  --url http://prefill:8000/metrics \
  --url http://decode:8000/metrics \
  --interval 5s
```

Each sample receives a `url=<full metrics URL>` label so that instances in a disaggregated prefill/decode deployment can be distinguished. A failure to scrape one URL does not block other URLs.

Check collection status or stop collection:

```bash
ucm-toolkit run metrics-view status
ucm-toolkit run metrics-view stop
```

Query the last 10 minutes and aggregate into one-minute intervals. Continue to pass the actual TP size for MLA models:

```bash
ucm-toolkit run metrics-view query \
  --window 10m \
  --aggr-by 1m \
  --config metrics_lite \
  --config-param tp_size=8
```

Filter by Prometheus labels:

```bash
ucm-toolkit run metrics-view query \
  --window 10m \
  --tag url=http://prefill:8000/metrics \
  --tag model_name=qwen
```

The default database, PID, and log files are `/tmp/ucm_metrics.db`, `/tmp/ucm_metrics.pid`, and `/tmp/terminal_metrics.log`. To clear the database:

```bash
ucm-toolkit run metrics-view clean
```

## 5. FAQ

### 5.1 No `ucm:` Metrics Appear in `/metrics`

Check the following:

1. `enable_metrics` is not set to false.
2. A custom `metrics_config_path` exists, is readable, and contains the required metrics.
3. vLLM has processed a request that exercises the corresponding UCM path.
4. `curl http://<vllm-ip>:<vllm-port>/metrics` can reach the service endpoint.

## Related Documentation

- [UCM Health Metrics](health_metrics.md): health probes, circuit-breaker state, and aggregation.
- [UCM Metrics Reference](metrics_list.md): metric names, types, semantics, and raw usage.

```{toctree}
:maxdepth: 1
:hidden:

health_metrics
metrics_list
```
